#!/usr/bin/env python3
"""V.7-equivalent IF baseline on v0.2 features.

Trains IsolationForest on a 200k stratified sample of v0.2 train-partition
features, scores all test-partition rows, joins to ``data/labels.csv`` via
time-window membership for ground-truth labels, computes AUROC + AP +
95% bootstrap CI, and writes metrics to ``artifacts/v0.2/diagnostics/``.

Run from the ``mamba-edge/`` working directory:

    python scripts/run_v0_2_if_baseline.py
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.features.v0_2_features import FEATURE_COLUMNS  # noqa: E402
from src.telemetry.tetragon_native_parser import parse_tetragon_ts  # noqa: E402

FEATURES_DIR = REPO_ROOT / "data" / "processed" / "v0.2-features"
TRAIN_DIR = FEATURES_DIR / "partition=train"
TEST_DIR = FEATURES_DIR / "partition=test"
LABELS_CSV = REPO_ROOT / "data" / "labels.csv"
DIAG_DIR = REPO_ROOT / "artifacts" / "v0.2" / "diagnostics"
METRICS_OUT = DIAG_DIR / "v0_2_if_baseline_metrics.json"

IDENTIFIER_COLUMNS = ("event_time", "proc_exec_id", "proc_pid")
FEATURE_ONLY_COLUMNS = tuple(c for c in FEATURE_COLUMNS if c not in IDENTIFIER_COLUMNS)

TRAIN_SAMPLE_SIZE = 200_000
N_ESTIMATORS = 100
RANDOM_STATE = 42
BOOTSTRAP_ITERS = 1_000
V07_FLOOR_AUROC = 0.710


def _git_head_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _list_parquet(root: Path) -> list[Path]:
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {root}")
    return files


def _load_features_matrix(files: list[Path], columns: tuple[str, ...]) -> pa.Table:
    """Read every file and concat into one Arrow Table.

    Reading the columns we need only (33 features) keeps the 16.97M-row train
    set well under 4 GB at u8/u16/i32/u32/u64 dtypes — verified against per-file
    metadata before commit.
    """
    tables: list[pa.Table] = []
    for fp in files:
        pf = pq.ParquetFile(fp)
        tables.append(pf.read(columns=list(columns)))
    return pa.concat_tables(tables)


def _arrow_to_float32(table: pa.Table) -> np.ndarray:
    """Cast a feature-only Arrow table to a dense float32 ndarray."""
    arrays = [table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names]
    return np.column_stack(arrays).astype(np.float32, copy=False)


def _sample_train_indices(n_total: int, k: int, seed: int) -> np.ndarray:
    """Pick ``k`` distinct indices in ``[0, n_total)`` with reproducible seeding.

    Uses ``random.sample`` (per the brief) to mirror V.7's unstratified protocol.
    """
    if k > n_total:
        raise ValueError(f"requested sample {k} exceeds population {n_total}")
    rng = random.Random(seed)
    return np.fromiter(rng.sample(range(n_total), k), dtype=np.int64)


def _load_labels(path: Path) -> list[tuple[int, int]]:
    """Parse ``labels.csv`` into a sorted list of half-open ``(start_ns, end_ns)``."""
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        intervals: list[tuple[int, int]] = []
        for row in reader:
            start_ns = parse_tetragon_ts(row["start_ts"])
            end_ns = parse_tetragon_ts(row["end_ts"])
            if start_ns is None or end_ns is None:
                raise ValueError(f"unparseable label row: {row}")
            intervals.append((start_ns, end_ns))
    intervals.sort()
    return intervals


def _label_test_rows(event_time_ns: np.ndarray, intervals: list[tuple[int, int]]) -> np.ndarray:
    """Return a boolean mask: True iff the timestamp falls in any half-open interval.

    Uses ``bisect`` over interval starts; for each timestamp, look up the
    largest start <= ts and check ts < that interval's end. Since intervals
    here don't overlap (verified separately), this is exact and O(n log m).
    """
    starts = np.array([s for s, _ in intervals], dtype=np.int64)
    ends = np.array([e for _, e in intervals], dtype=np.int64)
    # bisect_right(starts, ts) - 1 gives index of largest start <= ts (or -1).
    idx = np.searchsorted(starts, event_time_ns, side="right") - 1
    is_attack = np.zeros(event_time_ns.shape, dtype=bool)
    valid = idx >= 0
    valid_idx = idx[valid]
    is_attack[valid] = event_time_ns[valid] < ends[valid_idx]
    return is_attack


def _bootstrap_auroc_ci(
    y: np.ndarray,
    scores: np.ndarray,
    iters: int,
    seed: int,
) -> tuple[float, float]:
    """95% percentile bootstrap CI for AUROC over ``iters`` resamples."""
    rng = np.random.default_rng(seed)
    n = y.shape[0]
    aurocs = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        # Skip degenerate resample (all one class) by re-rolling.
        attempts = 0
        while (y_b.sum() == 0 or y_b.sum() == n) and attempts < 5:
            idx = rng.integers(0, n, size=n)
            y_b = y[idx]
            attempts += 1
        aurocs[i] = roc_auc_score(y_b, scores[idx])
    lo = float(np.percentile(aurocs, 2.5))
    hi = float(np.percentile(aurocs, 97.5))
    return lo, hi


def _decision_branch(auroc: float) -> str:
    if auroc >= 0.85:
        return "strong_signal"
    if auroc >= V07_FLOOR_AUROC:
        return "mixed"
    return "feature_design_wrong"


def _print_summary(payload: dict, metrics_out: Path) -> None:
    m = payload["metrics"]
    print("\n=== v0.2 IF baseline (V.7-equivalent on v0.2 features) ===")
    print(f"  n_estimators           : {payload['model']['n_estimators']:>12}")
    print(f"  train sample           : {payload['train_sample_size']:>12,}  / {payload['train_total']:,} rows")
    print(f"  test rows scored       : {payload['test_total']:>12,}  ({payload['test_attack_count']:,} attack, "
          f"{payload['test_benign_count']:,} benign — {payload['test_attack_pct']:.1%} attack)")
    print(f"  AUROC                  : {m['auroc']:>12.4f}  (95% CI {m['auroc_95ci_lo']:.4f}–{m['auroc_95ci_hi']:.4f})")
    print(f"  AP                     : {m['ap']:>12.4f}  (random baseline {payload['test_attack_pct']:.4f})")
    print(f"  V.7 floor (v0.1 parquet): {V07_FLOOR_AUROC:>12.4f}")
    delta = m["auroc"] - V07_FLOOR_AUROC
    print(f"  v0.2 lift over V.7     : {delta:+12.4f}")
    print(f"  decision branch        : {payload['decision_branch']}")
    print(f"  wall time              : {payload['wall_time_seconds']:.1f}s")
    print(f"\nWrote metrics to: {metrics_out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=BOOTSTRAP_ITERS,
        help=f"bootstrap resamples for AUROC CI (default {BOOTSTRAP_ITERS})",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=N_ESTIMATORS,
        help=f"IsolationForest n_estimators (default {N_ESTIMATORS}; V.7 used 200)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="IsolationForest n_jobs (default -1 = all cores; use 4 to coexist with other jobs)",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="suffix appended to metrics filename (e.g. '_n200' for non-overwriting reruns)",
    )
    args = parser.parse_args(argv)
    n_estimators = args.n_estimators
    metrics_out = (
        DIAG_DIR / f"v0_2_if_baseline_metrics{args.output_suffix}.json"
        if args.output_suffix
        else METRICS_OUT
    )

    t0 = time.perf_counter()

    train_files = _list_parquet(TRAIN_DIR)
    test_files = _list_parquet(TEST_DIR)
    print(f"train parquet files: {len(train_files)}")
    print(f"test  parquet files: {len(test_files)}")

    print("\nloading train features ...")
    train_table = _load_features_matrix(train_files, FEATURE_ONLY_COLUMNS)
    train_total = train_table.num_rows
    print(f"  loaded {train_total:,} train rows ({len(FEATURE_ONLY_COLUMNS)} features)")

    sample_idx = _sample_train_indices(train_total, TRAIN_SAMPLE_SIZE, RANDOM_STATE)
    train_sample = train_table.take(pa.array(sample_idx))
    train_X = _arrow_to_float32(train_sample)
    del train_table, train_sample
    print(f"  sampled {train_X.shape[0]:,} rows -> matrix {train_X.shape}")

    print("\nloading test features + identifiers ...")
    test_columns = ("event_time",) + FEATURE_ONLY_COLUMNS
    test_table = _load_features_matrix(test_files, test_columns)
    test_total = test_table.num_rows
    event_time_ns = test_table.column("event_time").to_numpy(zero_copy_only=False).astype("int64")
    test_X = _arrow_to_float32(test_table.select(list(FEATURE_ONLY_COLUMNS)))
    del test_table
    print(f"  loaded {test_total:,} test rows -> matrix {test_X.shape}")

    print("\nlabeling test rows from labels.csv ...")
    intervals = _load_labels(LABELS_CSV)
    print(f"  loaded {len(intervals)} ART intervals")
    is_attack = _label_test_rows(event_time_ns, intervals)
    n_attack = int(is_attack.sum())
    n_benign = int((~is_attack).sum())
    attack_pct = n_attack / test_total
    print(f"  test attack: {n_attack:,}  benign: {n_benign:,}  ({attack_pct:.2%} attack)")

    print(f"\nfitting IsolationForest (n_estimators={n_estimators}, n_jobs={args.n_jobs}) ...")
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=args.n_jobs,
    )
    iso.fit(train_X)
    del train_X

    print("scoring test rows ...")
    # score_samples: higher = more normal. Negate so higher = more anomalous.
    anomaly_score = -iso.score_samples(test_X)
    del test_X

    auroc = float(roc_auc_score(is_attack, anomaly_score))
    ap = float(average_precision_score(is_attack, anomaly_score))
    print(f"  AUROC = {auroc:.4f}   AP = {ap:.4f}")

    print(f"\nbootstrapping AUROC ({args.bootstrap_iters} resamples) ...")
    ci_lo, ci_hi = _bootstrap_auroc_ci(
        is_attack.astype(np.int8), anomaly_score, args.bootstrap_iters, RANDOM_STATE
    )

    wall = time.perf_counter() - t0
    branch = _decision_branch(auroc)

    payload = {
        "v0_2_features_revision": _git_head_sha(),
        "train_sample_size": int(TRAIN_SAMPLE_SIZE),
        "train_total": int(train_total),
        "test_total": int(test_total),
        "test_attack_count": n_attack,
        "test_benign_count": n_benign,
        "test_attack_pct": float(attack_pct),
        "model": {
            "name": "IsolationForest",
            "n_estimators": n_estimators,
            "contamination": "auto",
            "random_state": RANDOM_STATE,
        },
        "metrics": {
            "auroc": auroc,
            "ap": ap,
            "auroc_95ci_lo": ci_lo,
            "auroc_95ci_hi": ci_hi,
        },
        "wall_time_seconds": float(wall),
        "v07_floor_auroc": V07_FLOOR_AUROC,
        "decision_branch": branch,
        "notes": (
            "V.7-equivalent protocol on v0.2 features (33 columns). Train sample "
            "drawn unstratified from train-partition (clean of ART intervals; "
            "splitter cuts at 22:12:38Z, ART starts at 22:45:43Z). Test labels "
            "joined via half-open [start_ts, end_ts) membership. Bootstrap CI "
            "computed by resampling test rows with replacement."
        ),
    }

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(payload, indent=2))
    _print_summary(payload, metrics_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
