#!/usr/bin/env python3
"""v0.2 XGBoost — leave-one-trial-out supervised baseline.

Replaces the degenerate ``baseline_xgboost`` (AUROC=0.5000, single-class
train fold) with a defensible supervised number. For each labeled trial
T_i in ``data/labels.csv``, fits an XGBoost classifier on (benign train
events) + (test events in all OTHER trials T_j for j != i, labeled
positive), then scores test events. At stitch time:

  * Test events in T_i (positives held out from fold i) -> fold-i score.
  * Test events in T_j (j != i)                          -> fold-j score
                                                            (the fold
                                                            that held T_j
                                                            out).
  * Benign test events                                    -> mean score
                                                            across all 61
                                                            folds (none of
                                                            which saw any
                                                            benign-test
                                                            event in
                                                            training).

This protocol uses test data for training but never scores an event under
a model that saw that event in its train fold; per-event AUROC is
honest. The number is reported as a supervised oracle, clearly distinct
from the unsupervised baselines (n-gram, IF) which see only benign
training.

Outputs to ``mamba-edge/artifacts/v0.2/baseline_xgboost_loocv/``:

  * ``baseline_xgboost_loocv_evaluation.json``           — eval payload
                                                            (matches the
                                                            shared schema).
  * ``xgboost_loocv_evaluation.per_event.parquet``       — per-event
                                                            scores; schema
                                                            (event_time,
                                                            score,
                                                            is_attack,
                                                            technique_id,
                                                            trial_id).
                                                            trial_id is
                                                            ``""`` for
                                                            benign rows.
  * ``baseline_xgboost_loocv_evaluation.per_technique.csv`` — per-technique
                                                            score-distribution
                                                            summary.
  * ``train_summary.json``                                — per-fold fit
                                                            times, fold
                                                            sizes,
                                                            hyperparameters.
  * ``train.log``                                         — stdout
                                                            heartbeat.

The original degenerate ``baseline_xgboost`` artifacts at
``mamba-edge/artifacts/v0.2/baseline_xgboost/`` are NOT touched. Both
results coexist; the report carries both rows.

Fit hyperparameters intentionally diverge from
``XGBoostScorer.DEFAULT_HPARAMS`` in one way: no early stopping, no val
split. Reason: the natural chronological val-tail would be all-benign
(if val carved off benign-train) or all-attack (if val carved off the
test-attack tail), which kills early stopping. Fixed n_estimators=300
matches the budget of the v0.1 supervised baseline within a factor.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

from xgboost import XGBClassifier  # noqa: E402

from src.core.v0_2_baselines.shared import (  # noqa: E402
    compute_eval_payload,
    label_event_times,
    load_intervals,
    load_tabular_features,
    write_eval_json,
    write_per_technique_csv,
)
from src.telemetry.tetragon_native_parser import parse_tetragon_ts  # noqa: E402


# ---------------------------------------------------------------------------
# Trial table (extends labels.csv with a stable per-row LOOCV trial id).
# ---------------------------------------------------------------------------


def load_trials(labels_csv: Path) -> list[dict]:
    """Return per-row trial records sorted chronologically by start_ts."""
    rows: list[dict] = []
    with labels_csv.open("r", newline="") as fh:
        for row in csv.DictReader(fh):
            start_ns = parse_tetragon_ts(row["start_ts"])
            end_ns = parse_tetragon_ts(row["end_ts"])
            if start_ns is None or end_ns is None:
                raise ValueError(f"unparseable label row: {row}")
            trial_id = f"{row['technique_id']}#{row['trial']}"
            rows.append({
                "trial_id":     trial_id,
                "technique_id": row["technique_id"],
                "trial_num":    row["trial"],
                "start_ns":     int(start_ns),
                "end_ns":       int(end_ns),
            })
    rows.sort(key=lambda r: (r["start_ns"], r["end_ns"]))
    # Sanity: trial_id is unique.
    if len({r["trial_id"] for r in rows}) != len(rows):
        raise RuntimeError("non-unique trial_id; check labels.csv")
    return rows


def trial_membership_per_event(
    event_time_ns: np.ndarray,
    trials: list[dict],
) -> np.ndarray:
    """Return an int array of length n_test with the index of the
    containing trial, or -1 for benign events. Half-open membership
    (start_ns <= ts < end_ns) matches ``label_event_times``.

    Trials are non-overlapping (verified at fold-construction time), so
    ``np.searchsorted`` over the sorted start array gives an O(N log T)
    assignment.
    """
    starts = np.array([t["start_ns"] for t in trials], dtype=np.int64)
    ends = np.array([t["end_ns"] for t in trials], dtype=np.int64)
    idx = np.searchsorted(starts, event_time_ns, side="right") - 1
    out = np.full(event_time_ns.shape, -1, dtype=np.int32)
    valid = idx >= 0
    valid_idx = idx[valid]
    in_interval = event_time_ns[valid] < ends[valid_idx]
    out[valid] = np.where(in_interval, valid_idx.astype(np.int32), -1)
    return out


# ---------------------------------------------------------------------------
# Logger.
# ---------------------------------------------------------------------------


def make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = log_path.open("a")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()

    return log, fh


# ---------------------------------------------------------------------------
# Fold runner.
# ---------------------------------------------------------------------------


HPARAMS: dict = {
    "n_estimators":       300,
    "max_depth":          6,
    "learning_rate":      0.05,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "tree_method":        "hist",
    "objective":          "binary:logistic",
    "eval_metric":        "auc",
    "verbosity":          0,
    # Capped at 8 threads. XGBoost hist on 17M-row × 42-feature data
    # saturates memory bandwidth long before it saturates 32 cores; on the
    # prox-class boxes this script targets, n_jobs=-1 spikes load average
    # without proportional speedup. 8 threads is the empirical sweet spot.
    "n_jobs":             8,
}


def fit_one_fold(
    X_benign_train: np.ndarray,
    X_test: np.ndarray,
    test_trial_idx: np.ndarray,
    holdout_i: int,
    *,
    random_state: int = 0,
) -> tuple[XGBClassifier, dict]:
    """Fit XGBoost on (benign_train, y=0) ∪ (test events in trials !=
    holdout_i, y=1). Returns (clf, fit_diag).

    Memory layout: we vstack the benign matrix with the (small) attack
    matrix once. The benign matrix dominates the row count by ~3 orders
    of magnitude.
    """
    attack_mask_other = (test_trial_idx >= 0) & (test_trial_idx != holdout_i)
    X_attack_other = X_test[attack_mask_other]
    n_pos = int(X_attack_other.shape[0])
    n_neg = int(X_benign_train.shape[0])
    if n_pos == 0:
        raise RuntimeError(
            f"fold {holdout_i}: no other-trial attack events; check label table"
        )
    X = np.vstack([X_benign_train, X_attack_other]).astype(np.float32, copy=False)
    y = np.concatenate([
        np.zeros(n_neg, dtype=np.int32),
        np.ones(n_pos, dtype=np.int32),
    ])
    scale_pos_weight = float(n_neg) / float(max(n_pos, 1))
    clf = XGBClassifier(
        **HPARAMS,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
    )
    t_fit = time.time()
    clf.fit(X, y, verbose=False)
    fit_seconds = time.time() - t_fit
    diag = {
        "n_train_total":      int(n_neg + n_pos),
        "n_train_benign":     n_neg,
        "n_train_attack":     n_pos,
        "scale_pos_weight":   scale_pos_weight,
        "fit_seconds":        fit_seconds,
    }
    return clf, diag


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train-parquet",
                   default=Path("data/processed/v0.2/train.parquet"), type=Path)
    p.add_argument("--test-parquet",
                   default=Path("data/processed/v0.2/test.parquet"), type=Path)
    p.add_argument("--labels-csv",
                   default=Path("data/labels.csv"), type=Path)
    p.add_argument("--output-dir",
                   default=Path("artifacts/v0.2/baseline_xgboost_loocv"), type=Path)
    p.add_argument("--bootstrap-iters", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=0)
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument(
        "--max-folds", type=int, default=None,
        help="Optional cap on number of folds for smoke testing.",
    )
    p.add_argument(
        "--max-benign-train", type=int, default=None,
        help="Optional cap on benign-train rows (chronological prefix). "
             "Use only for smoke testing; the headline number must be on "
             "the full benign-train fold.",
    )
    args = p.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log, log_fh = make_logger(output_dir / "train.log")

    log(f"output_dir={output_dir}")
    log(f"train_parquet={args.train_parquet}")
    log(f"test_parquet={args.test_parquet}")
    log(f"labels_csv={args.labels_csv}")

    # --- Load benign-train + test feature matrices ------------------------
    log("loading benign-train tabular features ...")
    t0 = time.time()
    X_train_full, train_side, cols = load_tabular_features(args.train_parquet)
    train_event_time = train_side["event_time"]
    log(f"  train rows={X_train_full.shape[0]:,}  features={X_train_full.shape[1]}")

    intervals = load_intervals(args.labels_csv)
    is_attack_train, _tech_train = label_event_times(train_event_time, intervals)
    n_train_attack = int(is_attack_train.sum())
    log(f"  train attack rows (sanity, expected 0): {n_train_attack:,}")
    if n_train_attack > 0:
        log("  *** train fold contains attack events; dropping them from the "
            "benign-train pool to preserve the unsupervised-train invariant ***")
        X_benign = X_train_full[~is_attack_train].astype(np.float32, copy=False)
    else:
        X_benign = X_train_full
    if args.max_benign_train is not None and X_benign.shape[0] > args.max_benign_train:
        log(f"capping benign-train to first {args.max_benign_train:,} rows (chronological)")
        X_benign = X_benign[: args.max_benign_train]
    log(f"  benign-train rows used: {X_benign.shape[0]:,}")

    log("loading test tabular features ...")
    X_test, test_side, cols_t = load_tabular_features(args.test_parquet)
    test_event_time = test_side["event_time"]
    if cols_t != cols:
        raise RuntimeError("train/test feature columns mismatch")
    log(f"  test rows={X_test.shape[0]:,}  features={X_test.shape[1]}")

    is_attack_test, technique_test = label_event_times(test_event_time, intervals)
    n_test_attack = int(is_attack_test.sum())
    log(f"  test attack rows: {n_test_attack:,} ({n_test_attack/len(is_attack_test):.4%})")

    # --- Trial membership per test event ---------------------------------
    trials = load_trials(args.labels_csv)
    log(f"trials loaded: {len(trials)} (chronologically sorted)")
    test_trial_idx = trial_membership_per_event(test_event_time, trials)
    n_attack_via_trial_idx = int((test_trial_idx >= 0).sum())
    log(f"  test events covered by some trial: {n_attack_via_trial_idx:,} "
        f"(should equal test attack rows)")
    if n_attack_via_trial_idx != n_test_attack:
        log(f"  ! mismatch: {n_attack_via_trial_idx} vs {n_test_attack}")

    n_folds = len(trials) if args.max_folds is None else min(args.max_folds, len(trials))
    log(f"running {n_folds} folds")

    # --- Fold loop --------------------------------------------------------
    all_scores = np.full((n_folds, X_test.shape[0]), np.nan, dtype=np.float32)
    per_fold_diag: list[dict] = []
    t_loop = time.time()
    for i in range(n_folds):
        trial = trials[i]
        t_fold = time.time()
        clf, fdiag = fit_one_fold(
            X_benign,
            X_test,
            test_trial_idx,
            holdout_i=i,
            random_state=args.rng_seed + i,
        )
        proba = clf.predict_proba(X_test)[:, 1].astype(np.float32, copy=False)
        all_scores[i] = proba
        n_held_out = int((test_trial_idx == i).sum())
        wall_fold = time.time() - t_fold
        log(
            f"  fold {i+1:2d}/{n_folds} "
            f"trial={trial['trial_id']:<32s}  "
            f"held_out_events={n_held_out:5d}  "
            f"train_pos={fdiag['n_train_attack']:5d}  "
            f"fit={fdiag['fit_seconds']:5.1f}s  "
            f"wall={wall_fold:5.1f}s"
        )
        per_fold_diag.append({
            "fold_index":         i,
            "trial_id":           trial["trial_id"],
            "technique_id":       trial["technique_id"],
            "trial_num":          trial["trial_num"],
            "start_ns":           trial["start_ns"],
            "end_ns":             trial["end_ns"],
            "n_held_out_events":  n_held_out,
            **fdiag,
            "wall_seconds":       wall_fold,
        })
    loop_seconds = time.time() - t_loop
    log(f"all folds done; loop wall={loop_seconds/60:.1f} min")

    # --- Stitch -----------------------------------------------------------
    # Default per-event score: mean across all folds (good for benigns,
    # since no fold saw any benign-test event in training, every fold's
    # benign score is i.i.d. valid).
    final_scores = np.nanmean(all_scores, axis=0).astype(np.float32, copy=False)
    # Override attack events with the score from the fold that held that
    # trial out (the only honest score).
    for i in range(n_folds):
        mask = test_trial_idx == i
        if mask.any():
            final_scores[mask] = all_scores[i, mask]
    if not np.isfinite(final_scores).all():
        raise RuntimeError("final_scores contains non-finite values; aborting")

    # --- Eval payload (matches shared schema) ----------------------------
    log("computing eval payload ...")
    payload_t0 = time.time()
    payload = compute_eval_payload(
        model_name="xgboost_loocv",
        train_parquet=args.train_parquet,
        test_parquet=args.test_parquet,
        labels_csv=args.labels_csv,
        event_time_ns=test_event_time,
        scores=final_scores,
        is_attack=is_attack_test,
        technique=technique_test,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_seed=args.bootstrap_seed,
        window_size=args.window_size,
        stride=args.stride,
        wall_seconds=loop_seconds,
        extra={
            "design":               "leave-one-trial-out",
            "n_folds":              n_folds,
            "n_benign_train_rows":  int(X_benign.shape[0]),
            "n_test_rows":          int(X_test.shape[0]),
            "hparams":              HPARAMS,
            "supervised_oracle":    True,
        },
    )
    log(
        f"AUROC={payload.auroc:.4f}  AP={payload.ap:.4f}  "
        f"CI95=[{payload.ci95_lo:.4f}, {payload.ci95_hi:.4f}]"
    )
    log(
        f"TPR@1%FPR={payload.tpr_at_1pct_fpr:.4f}  "
        f"per_window_AUROC={payload.per_window_auroc:.4f}"
    )

    # --- Per-window median (offline aggregator parity with notebook §9) ---
    n = final_scores.shape[0]
    win = args.window_size
    stride = args.stride
    n_windows = (n - win) // stride + 1
    win_label = np.zeros(n_windows, dtype=bool)
    win_score_med = np.empty(n_windows, dtype=np.float32)
    for w in range(n_windows):
        s = w * stride
        e = s + win
        win_label[w] = bool(is_attack_test[s:e].any())
        win_score_med[w] = float(np.median(final_scores[s:e]))
    if 0 < int(win_label.sum()) < n_windows:
        from sklearn.metrics import roc_auc_score
        per_window_auroc_median = float(roc_auc_score(win_label, win_score_med))
    else:
        per_window_auroc_median = float("nan")
    log(f"per_window_AUROC (median, offline): {per_window_auroc_median:.4f}")

    # --- Write artifacts --------------------------------------------------
    eval_json_path = output_dir / "baseline_xgboost_loocv_evaluation.json"
    payload_dict = payload.as_dict()
    payload_dict["per_window_auroc_median_offline"] = per_window_auroc_median
    eval_json_path.write_text(json.dumps(payload_dict, indent=2))
    log(f"wrote {eval_json_path}")

    # Per-event parquet at user-specified path with LOOCV trial_id column.
    per_event_path = output_dir / "xgboost_loocv_evaluation.per_event.parquet"
    technique_str = np.where(technique_test == "", None, technique_test).astype(object)
    trial_id_per_event = np.array(
        [trials[i]["trial_id"] if i >= 0 else "" for i in test_trial_idx.tolist()],
        dtype=object,
    )
    trial_id_str = np.where(trial_id_per_event == "", None, trial_id_per_event).astype(object)
    per_event_table = pa.table({
        "event_time":   pa.array(test_event_time.astype(np.int64), type=pa.int64()),
        "score":        pa.array(final_scores.astype(np.float32)),
        "is_attack":    pa.array(is_attack_test.astype(bool)),
        "technique_id": pa.array(technique_str.tolist(), type=pa.string()),
        "trial_id":     pa.array(trial_id_str.tolist(), type=pa.string()),
    })
    pq.write_table(per_event_table, per_event_path, compression="snappy")
    log(f"wrote {per_event_path}")

    per_tech_path = output_dir / "baseline_xgboost_loocv_evaluation.per_technique.csv"
    write_per_technique_csv(
        per_tech_path,
        scores=final_scores,
        is_attack=is_attack_test,
        technique=technique_test,
    )
    log(f"wrote {per_tech_path}")

    # Train summary.
    payload_t1 = time.time()
    summary = {
        "model":                      "xgboost_loocv",
        "design":                     "leave-one-trial-out",
        "train_parquet":              str(args.train_parquet),
        "test_parquet":               str(args.test_parquet),
        "labels_csv":                 str(args.labels_csv),
        "output_dir":                 str(output_dir),
        "n_folds":                    n_folds,
        "n_benign_train_rows":        int(X_benign.shape[0]),
        "n_test_rows":                int(X_test.shape[0]),
        "n_test_attack_rows":         n_test_attack,
        "max_benign_train_arg":       args.max_benign_train,
        "n_jobs_xgboost":             HPARAMS["n_jobs"],
        "loop_seconds":               loop_seconds,
        "eval_seconds":               payload_t1 - payload_t0,
        "rng_seed":                   args.rng_seed,
        "hparams":                    HPARAMS,
        "fold_diag":                  per_fold_diag,
        "median_fit_seconds":         float(np.median([d["fit_seconds"] for d in per_fold_diag])),
        "saved_at":                   datetime.now(timezone.utc).isoformat(),
        "host":                       platform.node(),
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"wrote {output_dir / 'train_summary.json'}")

    wall_total = time.time() - t0
    log(f"done; total wall={wall_total/60:.1f} min")
    log_fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
