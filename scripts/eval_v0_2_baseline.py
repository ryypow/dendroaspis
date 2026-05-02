#!/usr/bin/env python3
"""v0.2 baseline evaluator — unified CLI for n-gram / XGBoost / Isolation
Forest checkpoints on test.parquet + labels.csv.

Usage:
    python scripts/eval_v0_2_baseline.py --model ngram \
        --checkpoint artifacts/v0.2/baseline_ngram/baseline.pkl \
        --output artifacts/v0.2/baseline_ngram/baseline_ngram_evaluation.json \
        --aim-experiment v0-2_dataset_final \
        --run-name baseline_ngram_eval

Mirrors ``scripts/eval_v0_2_mamba.py`` outputs so the training analysis
notebook can plot Mamba and baseline curves on the same axes:

  * ``<output>``                              — eval JSON (AUROC, AP, CI95,
                                                TPR@1%FPR, per_window_AUROC,
                                                per_technique).
  * ``<output>.per_event.parquet``            — (event_time, score,
                                                is_attack, technique_id).
  * ``<output>.per_technique.csv``            — score-distribution summary
                                                per technique.
  * Aim run                                    — hparams / system blocks +
                                                headline metrics + per-
                                                technique AUROC tracks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
    compute_eval_payload,
    label_event_times,
    load_intervals,
    load_tabular_features,
    load_token_stream,
    maybe_init_aim,
    system_block,
    write_eval_json,
    write_per_event_parquet,
    write_per_technique_csv,
)


SUPPORTED_MODELS = ("ngram", "xgboost", "isoforest", "isoforest_flat")


def _make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = log_path.open("a")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()

    return log, fh


def _load_scorer(model: str, checkpoint: Path):
    if model == "ngram":
        return NgramScorer.load(checkpoint)
    if model == "xgboost":
        return XGBoostScorer.load(checkpoint)
    if model == "isoforest":
        return IsolationForestScorer.load(checkpoint)
    if model == "isoforest_flat":
        return IsolationForestFlatScorer.load(checkpoint)
    raise ValueError(f"unknown model: {model!r}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--test-parquet",
                   default=Path("data/processed/v0.2/test.parquet"), type=Path)
    p.add_argument("--labels-csv", default=Path("data/labels.csv"), type=Path)
    p.add_argument("--train-parquet",
                   default=Path("data/processed/v0.2/train.parquet"), type=Path,
                   help="Path recorded in the eval JSON for traceability; not "
                        "loaded at eval time.")
    p.add_argument("--output", required=True, type=Path,
                   help="Eval JSON path; per-event parquet + per-technique "
                        "CSV are written alongside with .per_event.parquet "
                        "and .per_technique.csv suffixes.")
    p.add_argument("--bootstrap-iters", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=0)
    p.add_argument("--window-size", type=int, default=128,
                   help="Window size for per-window AUROC aggregation. "
                        "Matches Mamba evaluator default.")
    p.add_argument("--stride", type=int, default=32,
                   help="Stride for per-window AUROC. Matches Mamba evaluator default.")
    # Aim
    p.add_argument("--no-aim", action="store_true")
    p.add_argument("--aim-repo", default=".aim")
    p.add_argument("--aim-experiment", required=True)
    p.add_argument("--run-name", required=True)
    args = p.parse_args()

    output_dir: Path = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    log, log_fh = _make_logger(output_dir / "eval.log")

    log(f"model={args.model}  checkpoint={args.checkpoint}")
    log(f"test_parquet={args.test_parquet}")
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

    aim_set(aim_run, "hparams", {
        "model":           args.model,
        "checkpoint":      str(args.checkpoint),
        "test_parquet":    str(args.test_parquet),
        "labels_csv":      str(args.labels_csv),
        "bootstrap_iters": args.bootstrap_iters,
        "window_size":     args.window_size,
        "stride":          args.stride,
    })
    aim_set(aim_run, "system", {**system_block(), "model": args.model})
    aim_set(aim_run, "tag", "eval")

    log(f"loading scorer from {args.checkpoint}")
    scorer = _load_scorer(args.model, args.checkpoint)
    log(f"  loaded {type(scorer).__name__}")

    t0 = time.time()
    if args.model == "ngram":
        log(f"loading test token stream from {args.test_parquet}")
        event_time, proc_exec_id, token = load_token_stream(args.test_parquet)
        log(f"  test rows={len(event_time):,}")
        log("scoring ...")
        scores = scorer.score_events(event_time, proc_exec_id, token)
    elif args.model in ("xgboost", "isoforest", "isoforest_flat"):
        log(f"loading test tabular features from {args.test_parquet}")
        if args.model == "isoforest_flat":
            feature_columns = TABULAR_FEATURE_COLUMNS_FLAT
            log(
                f"  flat variant: dropping {len(EXCLUDED_FLAT_HASH_COLUMNS)} hash "
                f"columns -> {len(feature_columns)} features"
            )
        else:
            feature_columns = TABULAR_FEATURE_COLUMNS
        X, side, cols = load_tabular_features(args.test_parquet, columns=feature_columns)
        event_time = side["event_time"]
        log(f"  test rows={X.shape[0]:,}  features={X.shape[1]}")
        log("scoring ...")
        scores = scorer.score_events(X)
    else:
        raise SystemExit(f"unknown model: {args.model!r}")

    log(f"  scored {len(scores):,} events in {time.time() - t0:.1f}s")

    log(f"joining labels from {args.labels_csv}")
    intervals = load_intervals(args.labels_csv)
    is_attack, technique = label_event_times(event_time, intervals)
    n_attack = int(is_attack.sum())
    if n_attack == 0:
        raise RuntimeError(
            "no attack events landed in test.parquet — labeling broken or "
            "labels.csv intervals do not overlap test event_time range"
        )
    log(f"  attack rows={n_attack:,} ({n_attack / len(is_attack):.4%}); "
        f"benign={len(is_attack) - n_attack:,}")

    wall = time.time() - t0
    log("computing eval metrics ...")
    payload = compute_eval_payload(
        model_name=args.model,
        train_parquet=args.train_parquet,
        test_parquet=args.test_parquet,
        labels_csv=args.labels_csv,
        event_time_ns=event_time,
        scores=scores,
        is_attack=is_attack,
        technique=technique,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_seed=args.bootstrap_seed,
        window_size=args.window_size,
        stride=args.stride,
        wall_seconds=wall,
    )

    log(f"AUROC={payload.auroc:.4f}  AP={payload.ap:.4f}  "
        f"CI95=[{payload.ci95_lo:.4f}, {payload.ci95_hi:.4f}]")
    log(f"TPR@1%FPR={payload.tpr_at_1pct_fpr:.4f}  "
        f"per_window_AUROC={payload.per_window_auroc:.4f}")
    log(f"per_technique entries: {len(payload.per_technique)}")

    # Write outputs.
    write_eval_json(args.output, payload)
    log(f"wrote {args.output}")

    per_event_path = args.output.with_suffix(".per_event.parquet")
    write_per_event_parquet(
        per_event_path,
        event_time_ns=event_time,
        scores=scores,
        is_attack=is_attack,
        technique=technique,
    )
    log(f"wrote {per_event_path}")

    per_tech_path = args.output.with_suffix(".per_technique.csv")
    write_per_technique_csv(
        per_tech_path,
        scores=scores,
        is_attack=is_attack,
        technique=technique,
    )
    log(f"wrote {per_tech_path}")

    # Aim — headline scalars + per-technique tracks.
    aim_track(aim_run, "auroc", payload.auroc, context={"phase": "eval"})
    aim_track(aim_run, "ap", payload.ap, context={"phase": "eval"})
    aim_track(aim_run, "ci95_lo", payload.ci95_lo, context={"phase": "eval"})
    aim_track(aim_run, "ci95_hi", payload.ci95_hi, context={"phase": "eval"})
    aim_track(aim_run, "tpr_at_1pct_fpr", payload.tpr_at_1pct_fpr, context={"phase": "eval"})
    aim_track(aim_run, "per_window_auroc", payload.per_window_auroc, context={"phase": "eval"})
    aim_track(aim_run, "n_test_events", float(payload.n_test_events), context={"phase": "eval"})
    aim_track(aim_run, "n_attack_events", float(payload.n_attack_events), context={"phase": "eval"})
    aim_track(aim_run, "n_attack_windows", float(payload.n_attack_windows), context={"phase": "eval"})

    for tech, tech_data in payload.per_technique.items():
        tech_auroc = tech_data.get("auroc")
        if tech_auroc is None:
            continue
        aim_track(aim_run, "per_technique_auroc", float(tech_auroc),
                  context={"phase": "eval", "technique": tech})
        aim_track(aim_run, "per_technique_n_attack_events",
                  float(tech_data.get("n_attack_events", 0)),
                  context={"phase": "eval", "technique": tech})

    if aim_run is not None:
        try:
            aim_run["final"] = {
                "auroc":          payload.auroc,
                "ap":             payload.ap,
                "wall_seconds":   wall,
            }
            aim_close(aim_run)
        except Exception as e:  # noqa: BLE001
            log(f"[aim] error closing run: {e}")

    log_fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
