#!/usr/bin/env python3
"""v0.2 baseline trainer — unified CLI for n-gram / XGBoost / Isolation
Forest fits.

Usage:
    python scripts/train_v0_2_baseline.py --model ngram \
        --output-dir artifacts/v0.2/baseline_ngram \
        --aim-experiment v0-2_dataset_final \
        --run-name baseline_ngram

    python scripts/train_v0_2_baseline.py --model xgboost \
        --output-dir artifacts/v0.2/baseline_xgboost \
        --aim-experiment v0-2_dataset_final \
        --run-name baseline_xgboost

    python scripts/train_v0_2_baseline.py --model isoforest \
        --output-dir artifacts/v0.2/baseline_isoforest \
        --aim-experiment v0-2_dataset_final \
        --run-name baseline_isoforest

Per-model design lives in
``docs/releases/v0.2-course-milestone/v0.2_baseline_bake_off_design.md``.

Outputs:
  * ``baseline.pkl``            — pickled fitted scorer (load via Scorer.load).
  * ``train.log``               — stdout heartbeat.
  * ``train_summary.json``      — fit diagnostics + wall time + hparams.
  * Aim run                     — hparams / system / dataset blocks +
    final fit-time scalars. Disable with --no-aim.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

import numpy as np

from src.core.v0_2_baselines import (
    IsolationForestFlatScorer,
    IsolationForestScorer,
    NgramScorer,
    XGBoostScorer,
)
from src.core.v0_2_baselines.shared import (
    EXCLUDED_FLAT_HASH_COLUMNS,
    TABULAR_FEATURE_COLUMNS,
    TABULAR_FEATURE_COLUMNS_FLAT,
    aim_close,
    aim_set,
    aim_track,
    label_event_times,
    load_intervals,
    load_tabular_features,
    load_token_stream,
    maybe_init_aim,
    system_block,
)


SUPPORTED_MODELS = ("ngram", "xgboost", "isoforest", "isoforest_flat")


def _make_logger(log_path: Path):
    """Return a `log(msg)` closure that mirrors Mamba trainer formatting."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = log_path.open("a")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()

    return log, fh


# ---------------------------------------------------------------------------
# Helpers used by main().
# ---------------------------------------------------------------------------


def _hparams_block(args) -> dict:
    block: dict = {
        "model":         args.model,
        "rng_seed":      args.rng_seed,
        "val_fraction":  args.val_fraction,
        "max_train_rows": args.max_train_rows,
    }
    if args.model == "ngram":
        block.update({
            "n":             3,
            "alpha":         args.ngram_alpha,
            "backoff":       args.ngram_backoff,
        })
    elif args.model == "xgboost":
        block.update({
            "n_estimators":         args.xgboost_n_estimators,
            "max_depth":            args.xgboost_max_depth,
            "learning_rate":        args.xgboost_learning_rate,
            "early_stopping_rounds": args.xgboost_early_stopping_rounds,
            "device":               args.xgboost_device,
        })
    elif args.model in ("isoforest", "isoforest_flat"):
        block.update({
            "n_estimators":   args.if_n_estimators,
            "max_samples":    args.if_max_samples,
            "contamination":  args.if_contamination,
            "n_jobs":         args.if_n_jobs,
            "max_train_rows": args.if_max_train_rows,
        })
        if args.model == "isoforest_flat":
            block["excluded_columns"] = list(EXCLUDED_FLAT_HASH_COLUMNS)
    return block


def _system_block_full(args) -> dict:
    block = system_block(device_str="cpu")  # baselines don't need GPU device probing
    block["model"] = args.model
    return block


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--train-parquet",
                   default=Path("data/processed/v0.2/train.parquet"), type=Path)
    p.add_argument("--labels-csv", default=Path("data/labels.csv"), type=Path)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.20,
                   help="Fraction of (chronologically last) train held out for "
                        "early stopping. Used by xgboost; ignored by ngram + isoforest.")
    p.add_argument("--max-train-rows", type=int, default=None,
                   help="Cap on training rows (chronological prefix). Smoke / debug.")
    # n-gram
    p.add_argument("--ngram-alpha", type=float, default=1.0)
    p.add_argument("--ngram-backoff", type=float, default=0.4)
    # XGBoost
    p.add_argument("--xgboost-n-estimators", type=int, default=500)
    p.add_argument("--xgboost-max-depth", type=int, default=6)
    p.add_argument("--xgboost-learning-rate", type=float, default=0.05)
    p.add_argument("--xgboost-early-stopping-rounds", type=int, default=25)
    p.add_argument("--xgboost-device", choices=["cpu", "cuda"], default="cpu")
    # Isolation Forest
    p.add_argument("--if-n-estimators", type=int, default=200)
    p.add_argument("--if-max-samples", default="auto",
                   help="IsolationForest max_samples (default 'auto'; integer also accepted).")
    p.add_argument("--if-contamination", default="auto")
    p.add_argument("--if-n-jobs", type=int, default=-1)
    p.add_argument("--if-max-train-rows", type=int, default=None,
                   help="Subsample cap for IF training matrix (V.7-equivalent: 200000).")
    # Aim
    p.add_argument("--no-aim", action="store_true")
    p.add_argument("--aim-repo", default=".aim")
    p.add_argument("--aim-experiment", required=True)
    p.add_argument("--run-name", required=True)
    args = p.parse_args()

    # Coerce numeric strings for max_samples.
    if isinstance(args.if_max_samples, str) and args.if_max_samples not in ("auto",):
        try:
            args.if_max_samples = int(args.if_max_samples)
        except ValueError:
            raise SystemExit(
                f"--if-max-samples must be 'auto' or an integer; got {args.if_max_samples!r}"
            )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log, log_fh = _make_logger(output_dir / "train.log")

    log(f"model={args.model}  output_dir={output_dir}")
    log(f"train_parquet={args.train_parquet}")
    log(f"labels_csv={args.labels_csv}")

    aim_run = maybe_init_aim(
        not args.no_aim,
        repo=args.aim_repo,
        experiment=args.aim_experiment,
        run_name=args.run_name,
        log_fn=log,
    )
    if aim_run is not None:
        log(f"[aim] tracking enabled: experiment={args.aim_experiment} "
            f"run={args.run_name} repo={args.aim_repo}")
    else:
        log("[aim] tracking disabled (or not installed)")

    aim_set(aim_run, "hparams", _hparams_block(args))
    aim_set(aim_run, "system", _system_block_full(args))
    aim_set(aim_run, "tag", "fit")

    np.random.seed(args.rng_seed)
    t0 = time.time()

    if args.model == "ngram":
        log(f"loading token stream from {args.train_parquet}")
        event_time, proc_exec_id, token = load_token_stream(args.train_parquet)
        log(f"  rows={len(event_time):,}  unique_pids={len(np.unique(proc_exec_id)):,}")

        if args.max_train_rows is not None and len(event_time) > args.max_train_rows:
            log(f"capping train to first {args.max_train_rows} rows (chronological)")
            event_time = event_time[: args.max_train_rows]
            proc_exec_id = proc_exec_id[: args.max_train_rows]
            token = token[: args.max_train_rows]

        log(f"joining labels from {args.labels_csv}")
        intervals = load_intervals(args.labels_csv)
        is_attack, _technique = label_event_times(event_time, intervals)
        n_attack = int(is_attack.sum())
        log(f"  attack rows={n_attack:,} ({n_attack / max(len(is_attack), 1):.4%}); benign={len(is_attack) - n_attack:,}")
        aim_set(aim_run, "dataset", {
            "train_parquet":     str(args.train_parquet),
            "labels_csv":        str(args.labels_csv),
            "n_train_events":    int(len(event_time)),
            "n_attack_in_train": n_attack,
            "n_benign_in_train": int(len(event_time) - n_attack),
            "model_input":       "token + proc_exec_id",
        })

        scorer = NgramScorer(n=3, alpha=args.ngram_alpha, backoff=args.ngram_backoff)
        log(f"fitting NgramScorer (alpha={args.ngram_alpha}, backoff={args.ngram_backoff}) on benign-only events ...")
        t_fit = time.time()
        diag = scorer.fit(event_time, proc_exec_id, token, is_attack)
        fit_seconds = time.time() - t_fit
        log(f"  fit done in {fit_seconds:.1f}s")
        log(f"  vocab_size={diag.vocab_size:,}")
        log(f"  unique_trigrams={diag.n_unique_trigrams:,}  unique_bigrams={diag.n_unique_bigrams:,}")
        log(f"  trigram positions={diag.n_trigram_positions:,}  bigram positions={diag.n_bigram_positions:,}")
        log(f"  train_perplexity (sampled)={diag.train_perplexity:.4f}")
        log(f"  backoff rates: trigram={diag.backoff_rates.get('trigram', 0):.3f} "
            f"bigram={diag.backoff_rates.get('bigram', 0):.3f} "
            f"unigram={diag.backoff_rates.get('unigram', 0):.3f}")

        aim_track(aim_run, "fit_seconds", fit_seconds, context={"phase": "fit"})
        aim_track(aim_run, "vocab_size", diag.vocab_size, context={"phase": "fit"})
        aim_track(aim_run, "n_unique_trigrams", diag.n_unique_trigrams, context={"phase": "fit"})
        aim_track(aim_run, "n_unique_bigrams", diag.n_unique_bigrams, context={"phase": "fit"})
        aim_track(aim_run, "train_perplexity", diag.train_perplexity, context={"phase": "fit"})
        for level, rate in diag.backoff_rates.items():
            aim_track(aim_run, "backoff_rate", rate,
                      context={"phase": "fit", "level": level})

        diag_dict = diag.as_dict()

    elif args.model == "xgboost":
        log(f"loading tabular features from {args.train_parquet}")
        X, side, cols = load_tabular_features(args.train_parquet)
        event_time = side["event_time"]
        log(f"  rows={X.shape[0]:,}  features={X.shape[1]}")

        if args.max_train_rows is not None and X.shape[0] > args.max_train_rows:
            log(f"capping train to first {args.max_train_rows} rows (chronological)")
            X = X[: args.max_train_rows]
            event_time = event_time[: args.max_train_rows]

        log(f"joining labels from {args.labels_csv}")
        intervals = load_intervals(args.labels_csv)
        is_attack, _technique = label_event_times(event_time, intervals)
        n_attack = int(is_attack.sum())
        log(f"  attack rows={n_attack:,} ({n_attack / max(len(is_attack), 1):.4%})")
        aim_set(aim_run, "dataset", {
            "train_parquet":     str(args.train_parquet),
            "labels_csv":        str(args.labels_csv),
            "n_train_events":    int(X.shape[0]),
            "n_attack_in_train": n_attack,
            "n_benign_in_train": int(X.shape[0] - n_attack),
            "n_features":        int(X.shape[1]),
            "feature_columns":   list(cols),
            "model_input":       "f_* tabular",
        })

        device = "cuda" if args.xgboost_device == "cuda" else "cpu"
        scorer = XGBoostScorer(
            n_estimators=args.xgboost_n_estimators,
            max_depth=args.xgboost_max_depth,
            learning_rate=args.xgboost_learning_rate,
            early_stopping_rounds=args.xgboost_early_stopping_rounds,
            device=device,
            random_state=args.rng_seed,
        )
        log(f"fitting XGBoostScorer (n_estimators={args.xgboost_n_estimators}, "
            f"max_depth={args.xgboost_max_depth}, "
            f"early_stopping={args.xgboost_early_stopping_rounds}, device={device}) ...")
        t_fit = time.time()
        diag = scorer.fit(X, is_attack, val_fraction=args.val_fraction, feature_columns=cols)
        fit_seconds = time.time() - t_fit
        log(f"  fit done in {fit_seconds:.1f}s")
        if diag.degenerate:
            log(f"  *** DEGENERATE MODE: {diag.degenerate_reason} ***")
            log(f"  *** XGBoost will return 0.5 for every event at score time ***")
        else:
            log(f"  scale_pos_weight={diag.scale_pos_weight:.4f}")
            log(f"  best_iteration={diag.best_iteration}  best_val_AUC={diag.best_val_metric:.4f}")

        aim_track(aim_run, "fit_seconds", fit_seconds, context={"phase": "fit"})
        aim_track(aim_run, "scale_pos_weight", diag.scale_pos_weight, context={"phase": "fit"})
        aim_track(aim_run, "best_iteration", diag.best_iteration, context={"phase": "fit"})
        aim_track(aim_run, "best_val_auc", diag.best_val_metric, context={"phase": "fit"})
        aim_track(aim_run, "n_train_positives", diag.n_train_positives, context={"phase": "fit"})
        aim_track(aim_run, "n_val_positives", diag.n_val_positives, context={"phase": "fit"})
        aim_track(aim_run, "degenerate", float(diag.degenerate), context={"phase": "fit"})
        for it, (t_auc, v_auc) in enumerate(
            zip(diag.train_metric_history, diag.val_metric_history)
        ):
            aim_track(aim_run, "train_auc", t_auc, step=it, context={"subset": "train"})
            aim_track(aim_run, "val_auc",   v_auc, step=it, context={"subset": "val"})

        diag_dict = diag.as_dict()

    elif args.model in ("isoforest", "isoforest_flat"):
        is_flat = args.model == "isoforest_flat"
        feature_columns = (
            TABULAR_FEATURE_COLUMNS_FLAT if is_flat else TABULAR_FEATURE_COLUMNS
        )
        log(f"loading tabular features from {args.train_parquet}")
        if is_flat:
            log(
                f"  flat variant: dropping {len(EXCLUDED_FLAT_HASH_COLUMNS)} hash "
                f"columns -> {len(feature_columns)} features"
            )
            log(f"  excluded: {list(EXCLUDED_FLAT_HASH_COLUMNS)}")
        X, side, cols = load_tabular_features(args.train_parquet, columns=feature_columns)
        event_time = side["event_time"]
        log(f"  rows={X.shape[0]:,}  features={X.shape[1]}")

        if args.max_train_rows is not None and X.shape[0] > args.max_train_rows:
            log(f"capping train to first {args.max_train_rows} rows (chronological)")
            X = X[: args.max_train_rows]
            event_time = event_time[: args.max_train_rows]

        log(f"joining labels from {args.labels_csv}")
        intervals = load_intervals(args.labels_csv)
        is_attack, _technique = label_event_times(event_time, intervals)
        n_attack = int(is_attack.sum())
        log(f"  attack rows={n_attack:,} ({n_attack / max(len(is_attack), 1):.4%}); will drop before fit")
        aim_set(aim_run, "dataset", {
            "train_parquet":     str(args.train_parquet),
            "labels_csv":        str(args.labels_csv),
            "n_train_events":    int(X.shape[0]),
            "n_attack_in_train": n_attack,
            "n_benign_in_train": int(X.shape[0] - n_attack),
            "n_features":        int(X.shape[1]),
            "feature_columns":   list(cols),
            "excluded_columns":  list(EXCLUDED_FLAT_HASH_COLUMNS) if is_flat else [],
            "model_input":       "f_* tabular (benign-only, flat)" if is_flat else "f_* tabular (benign-only)",
        })

        scorer_cls = IsolationForestFlatScorer if is_flat else IsolationForestScorer
        scorer = scorer_cls(
            n_estimators=args.if_n_estimators,
            max_samples=args.if_max_samples,
            contamination=args.if_contamination,
            random_state=args.rng_seed,
            n_jobs=args.if_n_jobs,
            max_train_rows=args.if_max_train_rows,
        )
        log(f"fitting {scorer_cls.__name__} (n_estimators={args.if_n_estimators}, "
            f"max_samples={args.if_max_samples}, contamination={args.if_contamination}) ...")
        t_fit = time.time()
        diag = scorer.fit(X, is_attack, feature_columns=cols)
        fit_seconds = time.time() - t_fit
        log(f"  fit done in {fit_seconds:.1f}s")
        log(f"  n_train_used={diag.n_train_used:,}  n_features={diag.n_features}")
        log(f"  attack events dropped from train: {diag.n_train_events_attack_dropped:,}")

        aim_track(aim_run, "fit_seconds", fit_seconds, context={"phase": "fit"})
        aim_track(aim_run, "n_train_used", diag.n_train_used, context={"phase": "fit"})
        aim_track(aim_run, "n_features", diag.n_features, context={"phase": "fit"})
        aim_track(aim_run, "n_attack_dropped", diag.n_train_events_attack_dropped, context={"phase": "fit"})

        diag_dict = diag.as_dict()

    else:
        raise SystemExit(f"unknown model: {args.model!r}")

    save_path = output_dir / "baseline.pkl"
    log(f"saving scorer to {save_path}")
    scorer.save(save_path)

    wall = time.time() - t0
    summary = {
        "model":         args.model,
        "train_parquet": str(args.train_parquet),
        "labels_csv":    str(args.labels_csv),
        "output_dir":    str(output_dir),
        "wall_seconds":  wall,
        "rng_seed":      args.rng_seed,
        "diag":          diag_dict,
        "hparams":       _hparams_block(args),
        "saved_at":      datetime.now(timezone.utc).isoformat(),
        "host":          platform.node(),
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"wrote {output_dir / 'train_summary.json'}")
    log(f"done; wall={wall:.1f}s")

    if aim_run is not None:
        try:
            aim_run["final"] = {
                "model":        args.model,
                "wall_seconds": wall,
                "saved_path":   str(save_path),
            }
            aim_close(aim_run)
        except Exception as e:  # noqa: BLE001
            log(f"[aim] error closing run: {e}")

    log_fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
