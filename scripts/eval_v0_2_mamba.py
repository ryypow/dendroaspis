#!/usr/bin/env python3
"""Plan E v2 — eval a v0.2 Mamba checkpoint against test.parquet + labels.csv.

Usage:
    python scripts/eval_v0_2_mamba.py \\
        --checkpoint <path>/best.pt \\
        --objective {nll,mem} \\
        --output artifacts/v0.2/diagnostics/<name>_metrics.json

Reads ``data/processed/v0.2/test.parquet``, scores every event via the
trained scorer, joins the timestamp to ``data/labels.csv``, computes:

  * Per-event AUROC + AP
  * 95% bootstrap CI for AUROC (1000 resamples; matches V.7 / IF protocol)
  * Per-window AUROC (max-aggregated)
  * TPR @ 1% FPR
  * Per-technique AUROC (one row per ATT&CK technique)

Per-event score aggregation: each event appears in up to (window_size //
stride) = 4 windows; we take the MAX score across windows that include
the event (matches v0.1's window_score protocol).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src.core.v0_2_dataloader import V02SequenceDataset, load_parquet_dataset
from src.core.v0_2_mamba_scorer import build_scorer
from src.telemetry.tetragon_native_parser import parse_tetragon_ts

try:
    import aim
    _AIM_AVAILABLE = True
except ImportError:
    aim = None  # type: ignore
    _AIM_AVAILABLE = False


def _load_intervals(labels_csv: Path) -> list[tuple[int, int, str]]:
    """Parse labels.csv into sorted (start_ns, end_ns, technique_id)."""
    intervals: list[tuple[int, int, str]] = []
    with labels_csv.open("r", newline="") as fh:
        for row in csv.DictReader(fh):
            start_ns = parse_tetragon_ts(row["start_ts"])
            end_ns = parse_tetragon_ts(row["end_ts"])
            if start_ns is None or end_ns is None:
                raise ValueError(f"unparseable label row: {row}")
            intervals.append((start_ns, end_ns, row["technique_id"]))
    intervals.sort()
    return intervals


def _label_event_times(
    event_time_ns: np.ndarray,
    intervals: list[tuple[int, int, str]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (is_attack_bool, technique_id_or_empty_per_row)."""
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


def _bootstrap_auroc_ci(
    y: np.ndarray, scores: np.ndarray, iters: int, seed: int
) -> tuple[float, float]:
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


def _tpr_at_fpr(y: np.ndarray, scores: np.ndarray, fpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y, scores)
    # Largest TPR whose FPR <= target; if none, return TPR at smallest FPR > 0.
    mask = fpr <= fpr_target
    if mask.any():
        return float(tpr[mask].max())
    return float(tpr[fpr > 0].min() if (fpr > 0).any() else 0.0)


def _collate(batch: list[dict]) -> dict:
    out: dict = {}
    feature_keys = batch[0]["features"].keys()
    out["features"] = {k: torch.stack([s["features"][k] for s in batch], dim=0) for k in feature_keys}
    out["target_id"] = torch.stack([s["target_id"] for s in batch], dim=0)
    out["event_time"] = torch.stack([s["event_time"] for s in batch], dim=0)
    if "mask" in batch[0]:
        out["mask"] = torch.stack([s["mask"] for s in batch], dim=0)
    return out


def _move(batch: dict, device: torch.device) -> dict:
    moved = {
        "features": {k: v.to(device, non_blocking=True) for k, v in batch["features"].items()},
        "target_id": batch["target_id"].to(device, non_blocking=True),
        "event_time": batch["event_time"].to(device, non_blocking=True),
    }
    if "mask" in batch:
        moved["mask"] = batch["mask"].to(device, non_blocking=True)
    return moved


def _score_batch_mem_repeated_random(
    model,
    features: dict[str, torch.Tensor],
    n_samples: int,
    mask_fraction: float,
) -> torch.Tensor:
    """Fallback MEM scoring: average per-event SE over N independent
    random masks. Each position is masked w.p. ~mask_fraction per sample,
    so after N samples a position has been masked at least once with
    probability 1 - (1 - mask_fraction)^N. With N=10, fraction=0.15 ->
    coverage ~80%; positions that are never masked fall back to 0."""
    sample = next(iter(features.values()))
    B, L = sample.shape[0], sample.shape[1]
    device = sample.device
    out = torch.zeros(B, L, device=device, dtype=torch.float32)
    coverage = torch.zeros(B, L, device=device, dtype=torch.int32)
    n_mask = max(1, int(round(mask_fraction * L)))
    for _ in range(n_samples):
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            perm = torch.randperm(L, device=device)
            mask[b, perm[:n_mask]] = True
        per_event_se = model.forward(features, mask)
        out = out + per_event_se.to(out.dtype)
        coverage = coverage + mask.to(coverage.dtype)
    coverage_safe = coverage.clamp_min(1).to(out.dtype)
    return out / coverage_safe


_RAW_SCORE_SCHEMA = pa.schema([
    pa.field("window_idx", pa.int32()),
    pa.field("window_start_event_idx", pa.int64()),
    pa.field("position_idx", pa.int8()),
    pa.field("score", pa.float32()),
])


@torch.no_grad()
def score_dataset(
    model,
    dataset: V02SequenceDataset,
    device: torch.device,
    objective: str,
    batch_size: int = 8,
    num_workers: int = 0,
    *,
    mem_score_mode: str = "all_positions",
    mem_score_chunk_size: int = 16,
    mem_repeated_random_n: int = 10,
    raw_scores_out: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (per_event_scores_aggregated, per_window_scores).

    per_event_scores_aggregated: shape (n_rows,). Max per-event score across
      all windows containing each event; events not covered by any window
      get the dataset minimum.
    per_window_scores: shape (n_windows,). Max per-event score within each
      window — used for per-window AUROC.

    For NLL: forward returns per-event NLL directly.

    For MEM: random-mask training scoring is wrong at eval time (anomalous
    events that don't fall in the 15% sample get zero score). Two eval
    modes available:
      * ``all_positions`` (default; recommended): walks contiguous chunks
        of size ``mem_score_chunk_size``, masking each chunk in turn so
        every position is reconstructed exactly once.
      * ``repeated_random``: averages SE across ``mem_repeated_random_n``
        random 15% masks. Cheaper but coverage gaps remain.
    """
    if mem_score_mode not in ("all_positions", "repeated_random"):
        raise ValueError(
            f"mem_score_mode must be 'all_positions' or 'repeated_random', "
            f"got {mem_score_mode!r}"
        )
    n_rows = dataset.n_rows
    per_event = np.full(n_rows, -np.inf, dtype=np.float32)
    per_window = np.empty(len(dataset), dtype=np.float32)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=_collate,
        pin_memory=(device.type == "cuda"),
    )

    model.eval()
    starts = dataset.window_starts
    win = dataset.window_size

    autocast_dtype = torch.bfloat16 if device.type == "cuda" else None
    autocast_enabled = device.type == "cuda"

    def _score_batch(batch):
        if objective == "nll":
            return model(batch["features"], batch["target_id"])
        # MEM eval path.
        if mem_score_mode == "all_positions":
            return model.score_all_positions(batch["features"], chunk_size=mem_score_chunk_size)
        return _score_batch_mem_repeated_random(
            model, batch["features"], n_samples=mem_repeated_random_n, mask_fraction=0.15
        )

    raw_writer: Optional[pq.ParquetWriter] = None
    raw_position_idx = np.arange(win, dtype=np.int8)
    if raw_scores_out is not None:
        raw_scores_out.parent.mkdir(parents=True, exist_ok=True)
        raw_writer = pq.ParquetWriter(
            str(raw_scores_out), _RAW_SCORE_SCHEMA, compression="snappy"
        )

    try:
        window_idx = 0
        for batch in loader:
            batch = _move(batch, device)
            if autocast_enabled:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    per_event_b = _score_batch(batch)
            else:
                per_event_b = _score_batch(batch)
            per_event_np = per_event_b.float().cpu().numpy()  # (B, L)
            B = per_event_np.shape[0]
            if raw_writer is not None:
                # Build long-format batch: B*L rows. window_idx values are
                # the true global indices into dataset.window_starts.
                batch_window_ids = np.arange(window_idx, window_idx + B, dtype=np.int32)
                batch_starts = np.asarray(starts[window_idx:window_idx + B], dtype=np.int64)
                col_window_idx = np.repeat(batch_window_ids, win)
                col_window_start = np.repeat(batch_starts, win)
                col_position_idx = np.tile(raw_position_idx, B)
                col_score = per_event_np.reshape(-1).astype(np.float32, copy=False)
                rb = pa.RecordBatch.from_arrays(
                    [
                        pa.array(col_window_idx, type=pa.int32()),
                        pa.array(col_window_start, type=pa.int64()),
                        pa.array(col_position_idx, type=pa.int8()),
                        pa.array(col_score, type=pa.float32()),
                    ],
                    schema=_RAW_SCORE_SCHEMA,
                )
                raw_writer.write_batch(rb)
            for j in range(B):
                ws = starts[window_idx]
                scores_window = per_event_np[j]
                per_window[window_idx] = float(scores_window.max())
                target_slice = per_event[ws:ws + win]
                np.maximum(target_slice, scores_window[: len(target_slice)], out=target_slice)
                window_idx += 1
    finally:
        if raw_writer is not None:
            raw_writer.close()

    # Events outside any window: replace -inf with dataset min so they
    # don't dominate AUROC.
    covered = np.isfinite(per_event)
    if covered.any():
        floor = float(per_event[covered].min())
    else:
        floor = 0.0
    per_event[~covered] = floor
    return per_event, per_window


def _maybe_init_aim(enabled: bool, repo: str, experiment: str, run_name: str):
    if not enabled or not _AIM_AVAILABLE:
        return None
    try:
        run = aim.Run(experiment=experiment, repo=repo)
        run.name = run_name
        return run
    except Exception as e:  # noqa: BLE001
        print(f"[aim] failed to open Run ({e}); eval will write JSON only.")
        return None


def _aim_set(run, key: str, value) -> None:
    if run is None:
        return
    try:
        run[key] = value
    except Exception:  # noqa: BLE001
        pass


def _aim_track(run, name: str, value, *, step=None, context=None) -> None:
    if run is None:
        return
    try:
        kwargs: dict = {}
        if step is not None:
            kwargs["step"] = step
        if context is not None:
            kwargs["context"] = context
        run.track(value, name=name, **kwargs)
    except Exception:  # noqa: BLE001
        pass


def evaluate(
    checkpoint_path: Path,
    objective: str,
    test_parquet: Path,
    labels_csv: Path,
    output_path: Path,
    aim_experiment: str,
    run_name: str,
    *,
    batch_size: int = 8,
    bootstrap_iters: int = 1000,
    bootstrap_seed: int = 0,
    num_workers: int = 0,
    mem_score_mode: str = "all_positions",
    mem_score_chunk_size: int = 16,
    mem_repeated_random_n: int = 10,
    aim_enabled: bool = True,
    aim_repo: str = ".aim",
    raw_scores_out: Optional[Path] = None,
) -> dict:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    print(f"loading dataset: {test_parquet}", flush=True)
    dataset = load_parquet_dataset(test_parquet, objective=objective)
    print(f"  rows={dataset.n_rows} windows={len(dataset)}", flush=True)

    # ``feature_set`` defaults to "rich" so existing checkpoints written
    # before the §6 ablation (mamba_mem_run2/, mamba_nll_run2/, ...) load
    # unchanged. Flat checkpoints persist the field at training time.
    print(f"loading checkpoint: {checkpoint_path}", flush=True)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    feature_set = ckpt.get("feature_set", "rich")
    print(f"  feature_set={feature_set}", flush=True)
    model = build_scorer(objective, feature_set=feature_set).to(device)
    model.load_state_dict(ckpt["model_state"])

    if objective in ("mem", "mem-fa"):
        print(f"scoring ... mem_score_mode={mem_score_mode}", flush=True)
        if mem_score_mode == "all_positions":
            print(f"  chunk_size={mem_score_chunk_size}", flush=True)
        else:
            print(f"  n_samples={mem_repeated_random_n}", flush=True)
    else:
        print("scoring ...", flush=True)
    if raw_scores_out is not None:
        print(f"streaming raw per-position scores -> {raw_scores_out}", flush=True)
    per_event_scores, per_window_scores = score_dataset(
        model, dataset, device, objective,
        batch_size=batch_size, num_workers=num_workers,
        mem_score_mode=mem_score_mode,
        mem_score_chunk_size=mem_score_chunk_size,
        mem_repeated_random_n=mem_repeated_random_n,
        raw_scores_out=raw_scores_out,
    )
    event_time_ns = dataset.arrays["event_time"]

    print("labeling rows from labels.csv ...", flush=True)
    intervals = _load_intervals(labels_csv)
    is_attack, technique = _label_event_times(event_time_ns, intervals)
    n_attack = int(is_attack.sum())
    if n_attack == 0:
        raise RuntimeError("no attack events landed in test.parquet — labeling broken")
    print(f"  attack rows={n_attack} ({n_attack / len(is_attack):.4f})", flush=True)

    auroc = float(roc_auc_score(is_attack, per_event_scores))
    ap = float(average_precision_score(is_attack, per_event_scores))
    ci_lo, ci_hi = _bootstrap_auroc_ci(is_attack, per_event_scores, bootstrap_iters, bootstrap_seed)
    tpr_1pct = _tpr_at_fpr(is_attack, per_event_scores, 0.01)

    # Per-window AUROC: any event_time in a window that lies in an attack
    # interval marks the window attack.
    win_starts = np.array(dataset.window_starts, dtype=np.int64)
    win_ends = win_starts + dataset.window_size
    win_is_attack = np.zeros(len(win_starts), dtype=bool)
    for w_idx, (s, e) in enumerate(zip(win_starts, win_ends)):
        if is_attack[s:e].any():
            win_is_attack[w_idx] = True
    if win_is_attack.sum() > 0 and win_is_attack.sum() < len(win_is_attack):
        per_window_auroc = float(roc_auc_score(win_is_attack, per_window_scores))
    else:
        per_window_auroc = float("nan")

    # Per-technique AUROC: keep all benign events + attack events whose
    # technique == X; compute AUROC. Skip techniques with too few events.
    benign_mask = ~is_attack
    per_tech: dict[str, dict] = {}
    for tech in sorted(set(t for t in technique[is_attack].tolist())):
        if not tech:
            continue
        tech_mask = (technique == tech) & is_attack
        n_tech = int(tech_mask.sum())
        if n_tech < 50:  # skip noisy estimates
            per_tech[tech] = {"auroc": None, "n_attack_events": n_tech, "skipped": True}
            continue
        sub_mask = tech_mask | benign_mask
        try:
            tech_auroc = float(roc_auc_score(is_attack[sub_mask], per_event_scores[sub_mask]))
        except ValueError:
            tech_auroc = None
        per_tech[tech] = {
            "auroc": tech_auroc,
            "n_attack_events": n_tech,
            "skipped": False,
        }

    payload = {
        "objective": objective,
        "feature_set": feature_set,
        "checkpoint": str(checkpoint_path),
        "test_parquet": str(test_parquet),
        "labels_csv": str(labels_csv),
        "mem_score_mode": mem_score_mode if objective in ("mem", "mem-fa") else None,
        "mem_score_chunk_size": mem_score_chunk_size if (
            objective in ("mem", "mem-fa") and mem_score_mode == "all_positions"
        ) else None,
        "mem_repeated_random_n": mem_repeated_random_n if (
            objective in ("mem", "mem-fa") and mem_score_mode == "repeated_random"
        ) else None,
        "auroc": auroc,
        "ap": ap,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "tpr_at_1pct_fpr": tpr_1pct,
        "per_window_auroc": per_window_auroc,
        "n_test_events": int(len(is_attack)),
        "n_attack_events": n_attack,
        "n_windows": int(len(per_window_scores)),
        "n_attack_windows": int(win_is_attack.sum()),
        "per_technique": per_tech,
        "wall_seconds": time.time() - t0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"\nAUROC={auroc:.4f}  AP={ap:.4f}  CI95=[{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"TPR@1%FPR={tpr_1pct:.4f}  per_window_AUROC={per_window_auroc:.4f}")
    print(f"saved: {output_path}")

    scores_path = output_path.with_suffix(".per_event.parquet")
    technique_str = np.where(technique == "", None, technique).astype(object)
    scores_table = pa.table({
        "event_time": pa.array(event_time_ns, type=pa.int64()),
        "score": pa.array(per_event_scores.astype(np.float32)),
        "is_attack": pa.array(is_attack),
        "technique_id": pa.array(technique_str.tolist(), type=pa.string()),
    })
    pq.write_table(scores_table, scores_path, compression="snappy")
    print(f"saved per-event scores: {scores_path}")

    summary_path = output_path.with_suffix(".per_technique.csv")
    with summary_path.open("w") as fh:
        fh.write("technique_id,n_attack,score_mean,score_std,score_p50,score_p95,score_max,benign_p50,benign_p95\n")
        benign_scores = per_event_scores[benign_mask]
        b_p50 = float(np.percentile(benign_scores, 50)) if benign_scores.size else float("nan")
        b_p95 = float(np.percentile(benign_scores, 95)) if benign_scores.size else float("nan")
        for tech in sorted(set(t for t in technique[is_attack].tolist()) if n_attack else []):
            if not tech:
                continue
            tm = (technique == tech) & is_attack
            s = per_event_scores[tm]
            if s.size == 0:
                continue
            fh.write(f"{tech},{int(s.size)},{float(s.mean()):.6f},{float(s.std()):.6f},"
                     f"{float(np.percentile(s, 50)):.6f},{float(np.percentile(s, 95)):.6f},"
                     f"{float(s.max()):.6f},{b_p50:.6f},{b_p95:.6f}\n")
    print(f"saved per-technique score summary: {summary_path}")

    # ---- Aim tracking ----
    aim_run = _maybe_init_aim(aim_enabled, aim_repo, aim_experiment, run_name)
    if aim_run is not None:
        # Hyperparams / context for filtering in the UI.
        _aim_set(aim_run, "hparams", {
            "objective": objective,
            "feature_set": feature_set,
            "checkpoint": str(checkpoint_path),
            "test_parquet": str(test_parquet),
            "labels_csv": str(labels_csv),
            "batch_size": batch_size,
            "bootstrap_iters": bootstrap_iters,
            "mem_score_mode": mem_score_mode if objective in ("mem", "mem-fa") else None,
            "mem_score_chunk_size": (
                mem_score_chunk_size if (objective in ("mem", "mem-fa") and mem_score_mode == "all_positions") else None
            ),
            "mem_repeated_random_n": (
                mem_repeated_random_n if (objective in ("mem", "mem-fa") and mem_score_mode == "repeated_random") else None
            ),
        })
        # System block.
        sysblock = {
            "device":         str(device),
            "torch_version":  str(torch.__version__),
            "python_version": platform.python_version(),
            "platform":       platform.platform(),
            "host":           platform.node(),
        }
        if device.type == "cuda" and torch.cuda.is_available():
            sysblock["gpu_name"] = torch.cuda.get_device_name(0)
            sysblock["cuda_version"] = torch.version.cuda
        _aim_set(aim_run, "system", sysblock)
        _aim_set(aim_run, "tag", "eval")

        # Headline metrics — track without step so they show up as scalar
        # cards in the run summary.
        _aim_track(aim_run, "auroc", auroc, context={"phase": "eval"})
        _aim_track(aim_run, "ap", ap, context={"phase": "eval"})
        _aim_track(aim_run, "ci95_lo", ci_lo, context={"phase": "eval"})
        _aim_track(aim_run, "ci95_hi", ci_hi, context={"phase": "eval"})
        _aim_track(aim_run, "tpr_at_1pct_fpr", tpr_1pct, context={"phase": "eval"})
        _aim_track(aim_run, "per_window_auroc", per_window_auroc, context={"phase": "eval"})
        _aim_track(aim_run, "n_test_events", float(len(is_attack)), context={"phase": "eval"})
        _aim_track(aim_run, "n_attack_events", float(n_attack), context={"phase": "eval"})
        _aim_track(aim_run, "n_attack_windows", float(win_is_attack.sum()), context={"phase": "eval"})

        # Per-technique AUROC: one track per technique, indexed by technique
        # so the UI groups them together.
        for tech, tech_data in per_tech.items():
            tech_auroc = tech_data.get("auroc")
            if tech_auroc is None:
                continue
            _aim_track(
                aim_run, "per_technique_auroc", float(tech_auroc),
                context={"phase": "eval", "technique": tech},
            )
            _aim_track(
                aim_run, "per_technique_n_attack_events",
                float(tech_data.get("n_attack_events", 0)),
                context={"phase": "eval", "technique": tech},
            )

        try:
            aim_run.close()
        except Exception:  # noqa: BLE001
            pass

    return payload


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--objective", required=True, choices=["nll", "mem", "mem-fa"],
                   help="Objective the checkpoint was trained on. mem-fa is the "
                        "3a.1 field-aware categorical MEM.")
    p.add_argument("--test-parquet", default=Path("data/processed/v0.2/test.parquet"), type=Path)
    p.add_argument("--labels-csv", default=Path("data/labels.csv"), type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--bootstrap-iters", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--mem-score-mode",
        choices=["all_positions", "repeated_random"],
        default="all_positions",
        help=(
            "MEM eval scoring strategy. 'all_positions' (default) walks the "
            "sequence in contiguous chunks and reconstructs each position "
            "exactly once — required for reliable anomaly scoring. "
            "'repeated_random' averages SE over N random masks (faster but "
            "leaves coverage gaps)."
        ),
    )
    p.add_argument(
        "--mem-score-chunk-size", type=int, default=16,
        help="Contiguous positions masked per forward pass under all_positions mode.",
    )
    p.add_argument(
        "--mem-repeated-random-n", type=int, default=10,
        help="Number of random masks under repeated_random mode.",
    )
    p.add_argument(
        "--save-raw-window-scores", type=Path, default=None,
        help=(
            "Optional path. When set, stream raw per-position scores "
            "(one row per (window, position) pair) to this parquet via "
            "ParquetWriter. Used by the aggregator-recompute notebook to "
            "study aggregator choice independently of the per-event "
            "max-smoothing applied in score_dataset."
        ),
    )
    # Aim
    p.add_argument("--no-aim", action="store_true",
                   help="Disable Aim experiment tracking even if aim is installed.")
    p.add_argument("--aim-repo", type=str, default=".aim",
                   help="Aim repo path (default: ./.aim relative to mamba-edge).")
    p.add_argument("--aim-experiment", type=str, required=True,
                   help="Aim experiment name (groups runs in the UI). Required.")
    p.add_argument("--run-name", type=str, required=True,
                   help="Aim run name. Required — uniquely identifies the eval in the UI.")
    args = p.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        objective=args.objective,
        test_parquet=args.test_parquet,
        labels_csv=args.labels_csv,
        output_path=args.output,
        batch_size=args.batch_size,
        bootstrap_iters=args.bootstrap_iters,
        num_workers=args.num_workers,
        mem_score_mode=args.mem_score_mode,
        mem_score_chunk_size=args.mem_score_chunk_size,
        mem_repeated_random_n=args.mem_repeated_random_n,
        aim_enabled=not args.no_aim,
        aim_repo=args.aim_repo,
        aim_experiment=args.aim_experiment,
        run_name=args.run_name,
        raw_scores_out=args.save_raw_window_scores,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
