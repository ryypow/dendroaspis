"""Shared infrastructure for the v0.2 baseline bake-off.

Everything every baseline needs that is *not* the model itself:

  * label join from ``labels.csv`` via interval-tree on ``event_time``
    (verbatim port from ``scripts/eval_v0_2_mamba.py`` so baseline labels
    match Mamba labels event-for-event).
  * eval-metric helpers — bootstrap AUROC CI, TPR@FPR, per-technique
    AUROC.
  * per-event parquet writer + per-technique CSV writer + eval JSON
    writer matching the Mamba evaluator output schema exactly.
  * tabular feature-matrix loader (XGBoost + IF input) and token-stream
    loader (n-gram input).
  * Aim init / track defensive wrappers — never crash training on aim
    error, never require aim to be installed.
"""

from __future__ import annotations

import csv
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src.core.v0_2_event_encoder import ALL_EMBED_COLUMNS, ALL_FLOAT_COLUMNS
from src.telemetry.tetragon_native_parser import parse_tetragon_ts

try:
    import aim
    _AIM_AVAILABLE = True
except ImportError:
    aim = None  # type: ignore
    _AIM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Feature column inventory — single source of truth for tabular baselines.
# ---------------------------------------------------------------------------

EMBED_FEATURE_COLUMNS: tuple[str, ...] = tuple(s.column for s in ALL_EMBED_COLUMNS)
FLOAT_FEATURE_COLUMNS: tuple[str, ...] = tuple(s.column for s in ALL_FLOAT_COLUMNS)
TABULAR_FEATURE_COLUMNS: tuple[str, ...] = EMBED_FEATURE_COLUMNS + FLOAT_FEATURE_COLUMNS

# High-cardinality hash columns excluded from the IF-flat ablation.
# Dropping these from TABULAR_FEATURE_COLUMNS isolates "low-cardinality
# structural signal" from "encoder hash signal" for the §6 decomposition
# of Mamba's lift over IF-rich.
EXCLUDED_FLAT_HASH_COLUMNS: tuple[str, ...] = (
    "f_proc_name_hash",
    "f_parent_proc_hash",
    "f_proc_cwd_hash",
    "f_lineage_bag_hash",
    "f_parent_child_pair_hash",
    "f_root_ancestor_basename_hash",
    "f_process_tree_id_hash",
)
TABULAR_FEATURE_COLUMNS_FLAT: tuple[str, ...] = tuple(
    c for c in TABULAR_FEATURE_COLUMNS if c not in set(EXCLUDED_FLAT_HASH_COLUMNS)
)

# Token-stream baseline (n-gram) reads these columns. ``token`` is the
# joint-vocab token from the v0.2 behavior builder, ``proc_exec_id`` defines
# per-process sequence boundaries.
TOKEN_STREAM_COLUMNS: tuple[str, ...] = ("event_time", "proc_exec_id", "token")


# ---------------------------------------------------------------------------
# Label join (verbatim port from scripts/eval_v0_2_mamba.py).
# ---------------------------------------------------------------------------


def load_intervals(labels_csv: Path) -> list[tuple[int, int, str]]:
    """Parse labels.csv into a sorted list of (start_ns, end_ns, technique_id)."""
    intervals: list[tuple[int, int, str]] = []
    with Path(labels_csv).open("r", newline="") as fh:
        for row in csv.DictReader(fh):
            start_ns = parse_tetragon_ts(row["start_ts"])
            end_ns = parse_tetragon_ts(row["end_ts"])
            if start_ns is None or end_ns is None:
                raise ValueError(f"unparseable label row: {row}")
            intervals.append((start_ns, end_ns, row["technique_id"]))
    intervals.sort()
    return intervals


def label_event_times(
    event_time_ns: np.ndarray,
    intervals: list[tuple[int, int, str]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (is_attack_bool, technique_id_or_empty_per_row).

    Half-open membership: ``start_ts <= event_time < end_ts``. Intervals
    must be sorted; ``load_intervals`` guarantees that.
    """
    if not intervals:
        return (
            np.zeros(event_time_ns.shape, dtype=bool),
            np.full(event_time_ns.shape, "", dtype=object),
        )
    starts = np.array([s for s, _, _ in intervals], dtype=np.int64)
    ends = np.array([e for _, e, _ in intervals], dtype=np.int64)
    techs = np.array([t for _, _, t in intervals], dtype=object)
    idx = np.searchsorted(starts, event_time_ns, side="right") - 1
    is_attack = np.zeros(event_time_ns.shape, dtype=bool)
    technique = np.full(event_time_ns.shape, "", dtype=object)
    valid = idx >= 0
    valid_idx = idx[valid]
    in_interval = event_time_ns[valid] < ends[valid_idx]
    is_attack[valid] = in_interval
    technique_per_valid = np.where(in_interval, techs[valid_idx], "")
    technique[valid] = technique_per_valid
    return is_attack, technique


# ---------------------------------------------------------------------------
# Eval metric helpers.
# ---------------------------------------------------------------------------


def bootstrap_auroc_ci(
    y: np.ndarray, scores: np.ndarray, iters: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95% percentile bootstrap CI for AUROC over ``iters`` resamples."""
    rng = np.random.default_rng(seed)
    n = y.shape[0]
    aurocs = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        y_b, s_b = y[idx], scores[idx]
        attempts = 0
        while (y_b.sum() == 0 or y_b.sum() == n) and attempts < 5:
            idx = rng.integers(0, n, size=n)
            y_b, s_b = y[idx], scores[idx]
            attempts += 1
        aurocs[i] = roc_auc_score(y_b, s_b)
    return float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))


def tpr_at_fpr(y: np.ndarray, scores: np.ndarray, fpr_target: float = 0.01) -> float:
    """Largest TPR whose FPR <= fpr_target."""
    fpr, tpr, _ = roc_curve(y, scores)
    mask = fpr <= fpr_target
    if mask.any():
        return float(tpr[mask].max())
    return float(tpr[fpr > 0].min() if (fpr > 0).any() else 0.0)


def per_technique_auroc(
    is_attack: np.ndarray,
    technique: np.ndarray,
    scores: np.ndarray,
    *,
    min_attack_events: int = 50,
) -> dict[str, dict]:
    """Per-technique AUROC: keep all benign events + attack events whose
    technique == X; compute AUROC. Skip techniques with fewer than
    ``min_attack_events`` attack rows (estimate too noisy).
    """
    benign_mask = ~is_attack
    out: dict[str, dict] = {}
    for tech in sorted({t for t in technique[is_attack].tolist()}):
        if not tech:
            continue
        tech_mask = (technique == tech) & is_attack
        n_tech = int(tech_mask.sum())
        if n_tech < min_attack_events:
            out[tech] = {"auroc": None, "n_attack_events": n_tech, "skipped": True}
            continue
        sub_mask = tech_mask | benign_mask
        try:
            tech_auroc = float(roc_auc_score(is_attack[sub_mask], scores[sub_mask]))
        except ValueError:
            tech_auroc = None
        out[tech] = {
            "auroc": tech_auroc,
            "n_attack_events": n_tech,
            "skipped": False,
        }
    return out


# ---------------------------------------------------------------------------
# Window-level AUROC — match Mamba's max-aggregation protocol.
# ---------------------------------------------------------------------------


def per_window_auroc(
    is_attack: np.ndarray,
    scores: np.ndarray,
    *,
    window_size: int = 128,
    stride: int = 32,
) -> tuple[float, int, int]:
    """Sliding-window max-pool AUROC over the event-ordered score array.

    Mirrors ``scripts/eval_v0_2_mamba.py`` window construction:
    chronological windows of size ``window_size`` with step ``stride``,
    label = OR over events in the window, score = MAX over events in the
    window.

    Returns ``(per_window_auroc, n_windows, n_attack_windows)``.
    Returns NaN if all windows share a single label.
    """
    n = scores.shape[0]
    if n < window_size:
        return float("nan"), 0, 0
    n_windows = (n - window_size) // stride + 1
    win_label = np.zeros(n_windows, dtype=bool)
    win_score = np.empty(n_windows, dtype=np.float32)
    for w in range(n_windows):
        s = w * stride
        e = s + window_size
        win_label[w] = bool(is_attack[s:e].any())
        win_score[w] = float(scores[s:e].max())
    n_attack = int(win_label.sum())
    if n_attack == 0 or n_attack == n_windows:
        return float("nan"), n_windows, n_attack
    return float(roc_auc_score(win_label, win_score)), n_windows, n_attack


# ---------------------------------------------------------------------------
# Eval payload assembly + writers (match scripts/eval_v0_2_mamba.py output).
# ---------------------------------------------------------------------------


@dataclass
class EvalPayload:
    """Computed eval result for a single baseline run.

    Mirrors the output schema of scripts/eval_v0_2_mamba.py so the
    training analysis notebook can read both Mamba and baseline outputs
    via the same code path.
    """
    model: str
    train_parquet: str
    test_parquet: str
    labels_csv: str
    auroc: float
    ap: float
    ci95_lo: float
    ci95_hi: float
    tpr_at_1pct_fpr: float
    per_window_auroc: float
    n_test_events: int
    n_attack_events: int
    n_windows: int
    n_attack_windows: int
    per_technique: dict
    wall_seconds: float
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {
            "model":              self.model,
            "train_parquet":      self.train_parquet,
            "test_parquet":       self.test_parquet,
            "labels_csv":         self.labels_csv,
            "auroc":              self.auroc,
            "ap":                 self.ap,
            "ci95_lo":            self.ci95_lo,
            "ci95_hi":            self.ci95_hi,
            "tpr_at_1pct_fpr":    self.tpr_at_1pct_fpr,
            "per_window_auroc":   self.per_window_auroc,
            "n_test_events":      self.n_test_events,
            "n_attack_events":    self.n_attack_events,
            "n_windows":          self.n_windows,
            "n_attack_windows":   self.n_attack_windows,
            "per_technique":      self.per_technique,
            "wall_seconds":       self.wall_seconds,
        }
        d.update(self.extra)
        return d


def compute_eval_payload(
    *,
    model_name: str,
    train_parquet: Path,
    test_parquet: Path,
    labels_csv: Path,
    event_time_ns: np.ndarray,
    scores: np.ndarray,
    is_attack: np.ndarray,
    technique: np.ndarray,
    bootstrap_iters: int,
    bootstrap_seed: int,
    window_size: int,
    stride: int,
    wall_seconds: float,
    extra: dict | None = None,
) -> EvalPayload:
    auroc = float(roc_auc_score(is_attack, scores))
    ap = float(average_precision_score(is_attack, scores))
    ci_lo, ci_hi = bootstrap_auroc_ci(is_attack, scores, bootstrap_iters, bootstrap_seed)
    tpr1 = tpr_at_fpr(is_attack, scores, 0.01)
    pw_auroc, n_windows, n_attack_windows = per_window_auroc(
        is_attack, scores, window_size=window_size, stride=stride,
    )
    per_tech = per_technique_auroc(is_attack, technique, scores)
    return EvalPayload(
        model=model_name,
        train_parquet=str(train_parquet),
        test_parquet=str(test_parquet),
        labels_csv=str(labels_csv),
        auroc=auroc,
        ap=ap,
        ci95_lo=ci_lo,
        ci95_hi=ci_hi,
        tpr_at_1pct_fpr=tpr1,
        per_window_auroc=pw_auroc,
        n_test_events=int(len(is_attack)),
        n_attack_events=int(is_attack.sum()),
        n_windows=int(n_windows),
        n_attack_windows=int(n_attack_windows),
        per_technique=per_tech,
        wall_seconds=wall_seconds,
        extra=extra or {},
    )


def write_per_event_parquet(
    output_path: Path,
    *,
    event_time_ns: np.ndarray,
    scores: np.ndarray,
    is_attack: np.ndarray,
    technique: np.ndarray,
) -> None:
    """Write the (event_time, score, is_attack, technique_id) per-event
    parquet matching scripts/eval_v0_2_mamba.py output schema."""
    technique_str = np.where(technique == "", None, technique).astype(object)
    table = pa.table({
        "event_time":   pa.array(event_time_ns.astype(np.int64), type=pa.int64()),
        "score":        pa.array(scores.astype(np.float32)),
        "is_attack":    pa.array(is_attack.astype(bool)),
        "technique_id": pa.array(technique_str.tolist(), type=pa.string()),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="snappy")


def write_per_technique_csv(
    output_path: Path,
    *,
    scores: np.ndarray,
    is_attack: np.ndarray,
    technique: np.ndarray,
) -> None:
    """Per-technique score-distribution CSV, matching the Mamba layout."""
    benign_scores = scores[~is_attack]
    if benign_scores.size:
        b_p50 = float(np.percentile(benign_scores, 50))
        b_p95 = float(np.percentile(benign_scores, 95))
    else:
        b_p50 = b_p95 = float("nan")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        fh.write(
            "technique_id,n_attack,score_mean,score_std,score_p50,"
            "score_p95,score_max,benign_p50,benign_p95\n"
        )
        for tech in sorted({t for t in technique[is_attack].tolist()}):
            if not tech:
                continue
            tm = (technique == tech) & is_attack
            s = scores[tm]
            if s.size == 0:
                continue
            fh.write(
                f"{tech},{int(s.size)},{float(s.mean()):.6f},"
                f"{float(s.std()):.6f},{float(np.percentile(s, 50)):.6f},"
                f"{float(np.percentile(s, 95)):.6f},{float(s.max()):.6f},"
                f"{b_p50:.6f},{b_p95:.6f}\n"
            )


# ---------------------------------------------------------------------------
# Tabular feature-matrix loader (XGBoost + IF).
# ---------------------------------------------------------------------------


def _normalize_event_time_to_ns(table: pa.Table) -> pa.Table:
    """Cast ``event_time`` to ``timestamp[ns]`` if needed.

    DuckDB COPY ... TO PARQUET writes ``timestamp[us]`` regardless of
    input precision. Downstream code (label join, carve thresholds)
    assumes ns-precision int64, so we cast at the boundary. Mirrors
    ``src.core.v0_2_dataloader._read_parquet_sorted``.
    """
    et = table["event_time"]
    if pa.types.is_timestamp(et.type) and et.type.unit != "ns":
        target = pa.timestamp("ns", tz=et.type.tz)
        idx = table.schema.get_field_index("event_time")
        table = table.set_column(idx, "event_time", et.cast(target))
    return table


def load_tabular_features(
    parquet_path: Path,
    *,
    columns: Iterable[str] = TABULAR_FEATURE_COLUMNS,
    extra_columns: Iterable[str] = ("event_time",),
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    """Read parquet, return (X, side_arrays, column_names).

    X: (n_rows, n_features) float32 dense matrix.
    side_arrays: ``{name: np.ndarray}`` for any extra_columns requested
      (e.g., ``event_time`` as int64 ns for label join).

    Embedding columns are read as numeric and cast to float32. Float
    columns are cast to float32. ``event_time`` (if requested) is
    normalized to ns precision and returned as int64 ns.
    """
    cols = list(columns)
    extras = [c for c in extra_columns if c not in cols]
    table = pq.read_table(parquet_path, columns=cols + extras)
    table = _normalize_event_time_to_ns(table)
    table = table.sort_by("event_time")  # match dataloader / eval ordering
    arrays: list[np.ndarray] = []
    for c in cols:
        a = table[c].to_numpy(zero_copy_only=False)
        # Cast everything to float32 for the dense matrix. Bucket ids + binary
        # floats land cleanly; XGBoost's hist tree method handles them as
        # numeric splits, which is fine — splits on bucket IDs work because
        # the IDs are dense in [0, cardinality).
        arrays.append(a.astype(np.float32, copy=False))
    X = np.column_stack(arrays).astype(np.float32, copy=False)
    side: dict[str, np.ndarray] = {}
    for c in extras:
        side[c] = table[c].cast(pa.int64()).to_numpy(zero_copy_only=False).astype(np.int64, copy=True)
    return X, side, cols


def load_token_stream(
    parquet_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read parquet, return (event_time_ns, proc_exec_id, token_strings).

    Sorted by event_time. ``event_time`` is int64 ns precision (cast at
    the boundary; see ``_normalize_event_time_to_ns``). ``proc_exec_id``
    and ``token`` are object arrays (string-like). The n-gram scorer
    factorizes tokens internally so the train→test vocab mapping is
    stable: a token seen in train as code k is the same code k in test,
    and tokens unseen in train map to a fixed OOV bucket.
    """
    table = pq.read_table(parquet_path, columns=list(TOKEN_STREAM_COLUMNS))
    table = _normalize_event_time_to_ns(table)
    table = table.sort_by("event_time")
    event_time = (
        table["event_time"].cast(pa.int64()).to_numpy(zero_copy_only=False).astype(np.int64, copy=True)
    )
    proc_exec_id = np.array(table["proc_exec_id"].to_pylist(), dtype=object)
    token = np.array(table["token"].to_pylist(), dtype=object)
    return event_time, proc_exec_id, token


# ---------------------------------------------------------------------------
# Aim helpers (defensive — never crash training on aim error).
# ---------------------------------------------------------------------------


def maybe_init_aim(
    enabled: bool,
    *,
    repo: str,
    experiment: str,
    run_name: str,
    log_fn=print,
):
    """Open an ``aim.Run`` or return None."""
    if not enabled:
        return None
    if not _AIM_AVAILABLE:
        log_fn("[aim] aim not installed; logging will be file-only.")
        return None
    try:
        run = aim.Run(experiment=experiment, repo=repo)
        run.name = run_name
        return run
    except Exception as e:  # noqa: BLE001
        log_fn(f"[aim] failed to open Run ({e}); logging will be file-only.")
        return None


def aim_set(run, key: str, value) -> None:
    if run is None:
        return
    try:
        run[key] = value
    except Exception:  # noqa: BLE001
        pass


def aim_track(run, name: str, value, *, step=None, epoch=None, context=None) -> None:
    if run is None:
        return
    try:
        kwargs: dict = {}
        if step is not None:
            kwargs["step"] = step
        if epoch is not None:
            kwargs["epoch"] = epoch
        if context is not None:
            kwargs["context"] = context
        run.track(value, name=name, **kwargs)
    except Exception:  # noqa: BLE001
        pass


def aim_close(run) -> None:
    if run is None:
        return
    try:
        run.close()
    except Exception:  # noqa: BLE001
        pass


def system_block(device_str: str = "cpu") -> dict:
    """Standard system-info block for Aim run metadata."""
    block = {
        "device":         device_str,
        "python_version": platform.python_version(),
        "platform":       platform.platform(),
        "host":           platform.node(),
        "cwd":            str(Path.cwd()),
    }
    return block


# ---------------------------------------------------------------------------
# JSON writer (matches Mamba evaluator format).
# ---------------------------------------------------------------------------


def write_eval_json(output_path: Path, payload: EvalPayload) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload.as_dict(), indent=2))


__all__ = [
    "TABULAR_FEATURE_COLUMNS",
    "TABULAR_FEATURE_COLUMNS_FLAT",
    "EXCLUDED_FLAT_HASH_COLUMNS",
    "EMBED_FEATURE_COLUMNS",
    "FLOAT_FEATURE_COLUMNS",
    "TOKEN_STREAM_COLUMNS",
    "load_intervals",
    "label_event_times",
    "bootstrap_auroc_ci",
    "tpr_at_fpr",
    "per_technique_auroc",
    "per_window_auroc",
    "EvalPayload",
    "compute_eval_payload",
    "write_per_event_parquet",
    "write_per_technique_csv",
    "write_eval_json",
    "load_tabular_features",
    "load_token_stream",
    "maybe_init_aim",
    "aim_set",
    "aim_track",
    "aim_close",
    "system_block",
]
