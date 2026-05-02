#!/usr/bin/env python3
"""Plan E v2 — train v0.2 Mamba on train.parquet.

Usage:
    python scripts/train_v0_2_mamba.py --objective {nll,mem} --output-dir <path> [--max-train-steps N]

Recipe (mirrors v0.1's trainer; locked in Plan E v2 §6):
  AdamW lr=3e-4 wd=0.01
  Cosine schedule with 10% linear warmup
  Gradient clip 1.0
  bf16 autocast on CUDA, fp32 on CPU
  batch_size=8, 15 epochs
  Early stopping on val loss (patience=5, min_delta=1e-4)
  Val carved from train.parquet via carve_train_val(val_fraction=0.20,
    boundary_gap_seconds=900) — 15-minute hold-out band; no window spans
    the boundary.

Logging:
  * train.log   — text heartbeat every --log-every steps + per-epoch summary
  * train_summary.json + split_diagnostics.json — final + carve metadata
  * Aim tracker (if installed): hparams / model / system / dataset blocks +
    per-step (train_loss, lr) and per-epoch (val_loss, epoch_train_loss,
    epoch_wall_seconds) tracks. Disable with --no-aim. Browse via `aim up`
    in the directory containing the .aim repo (default ./.aim).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

# Aim is optional — if not installed, training proceeds with file-only
# logging and the Aim wrapper is a no-op.
try:
    import aim
    _AIM_AVAILABLE = True
except ImportError:
    aim = None  # type: ignore
    _AIM_AVAILABLE = False

import numpy as np

from src.core.v0_2_dataloader import V02ValCarveConfig, carve_train_val
from src.core.v0_2_mamba_scorer import build_scorer


def _collate(batch: list[dict]) -> dict:
    """Stack a list of per-window samples into a batch dict."""
    out: dict = {}
    feature_keys = batch[0]["features"].keys()
    out["features"] = {
        k: torch.stack([s["features"][k] for s in batch], dim=0) for k in feature_keys
    }
    out["target_id"] = torch.stack([s["target_id"] for s in batch], dim=0)
    out["event_time"] = torch.stack([s["event_time"] for s in batch], dim=0)
    if "mask" in batch[0]:
        out["mask"] = torch.stack([s["mask"] for s in batch], dim=0)
    return out


def _move(batch: dict, device: torch.device) -> dict:
    """Move all tensors in a nested batch dict to device."""
    moved = {
        "features": {k: v.to(device, non_blocking=True) for k, v in batch["features"].items()},
        "target_id": batch["target_id"].to(device, non_blocking=True),
        "event_time": batch["event_time"].to(device, non_blocking=True),
    }
    if "mask" in batch:
        moved["mask"] = batch["mask"].to(device, non_blocking=True)
    return moved


@contextmanager
def _autocast(device: torch.device):
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


def _cosine_warmup_lr(step: int, total_steps: int, base_lr: float, warmup_frac: float = 0.10) -> float:
    warmup_steps = max(1, int(total_steps * warmup_frac))
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _forward_with_metrics(model, batch: dict, objective: str) -> dict:
    """Run one forward pass and return loss + per-event tensors needed by
    the metrics accumulators (accuracy / perplexity for NLL; per-event MSE
    for MEM). All tensors are detached from the autograd graph so the
    accumulator can accept them without holding gradients."""
    if objective == "nll":
        nll_padded, shifted_logits, shifted_targets = model(
            batch["features"], batch["target_id"], return_logits=True
        )
        # Scorer front-pads: position 0 is the artificial zero (event 0 has
        # no predecessor), positions 1..L-1 are the real per-event NLLs.
        # Matches MambaNLLScorer.loss().
        nll_real = nll_padded[:, 1:]
        loss = nll_real.mean()
        preds = shifted_logits.argmax(dim=-1)
        correct = (preds == shifted_targets).float()
        return {
            "loss": loss,
            "per_event_nll": nll_real.detach(),
            "correct": correct.detach(),
        }
    # MEM
    per_event_se = model(batch["features"], batch["mask"])  # (B, L), zero outside mask
    n_masked = batch["mask"].sum().clamp_min(1).to(per_event_se.dtype)
    loss = per_event_se.sum() / n_masked
    return {
        "loss": loss,
        "per_event_se": per_event_se.detach(),
        "mask": batch["mask"].detach(),
        "target_id": batch["target_id"].detach(),
    }


class _NLLAcc:
    """Streaming accumulator for NLL eval-on-val metrics."""

    def __init__(self) -> None:
        self.n_events = 0
        self.sum_nll = 0.0
        self.sum_sq_nll = 0.0
        self.n_correct = 0

    def update(self, fwd: dict) -> None:
        nll = fwd["per_event_nll"].float()
        correct = fwd["correct"]
        self.n_events += nll.numel()
        self.sum_nll += float(nll.sum().item())
        self.sum_sq_nll += float((nll ** 2).sum().item())
        self.n_correct += int(correct.sum().item())

    def summary(self) -> dict:
        if self.n_events == 0:
            return {}
        mean = self.sum_nll / self.n_events
        var = max((self.sum_sq_nll / self.n_events) - mean ** 2, 0.0)
        return {
            "val_nll_mean": mean,
            "val_nll_var": var,
            "val_perplexity": math.exp(min(mean, 50)),  # cap to avoid overflow
            "val_token_accuracy": self.n_correct / self.n_events,
        }


class _MEMAcc:
    """Streaming accumulator for MEM eval-on-val metrics.

    Naming note: the log keys (`val_recon_error_mean`, `val_recon_error_var`,
    `val_recon_error_by_action_family`) are inherited from vanilla MEM
    (continuous-embedding MSE reconstruction). For `--objective mem-fa`
    the same keys carry the **field-aware masked prediction loss**
    (sum of CE+BCE heads), not MSE. Names retained for Aim cross-run
    continuity; the report must describe MEM-FA values as field-aware
    loss, not reconstruction error.
    """

    def __init__(self, n_action_families: int) -> None:
        self.n_action_families = n_action_families
        self.n_events = 0
        self.sum_se = 0.0
        self.sum_sq_se = 0.0
        self.af_sum = [0.0] * n_action_families
        self.af_count = [0] * n_action_families

    def update(self, fwd: dict) -> None:
        per_event_se = fwd["per_event_se"].float()
        mask = fwd["mask"]
        target_id = fwd["target_id"]
        # Only masked positions have meaningful SE; non-masked are zeroed.
        sel = per_event_se[mask]
        af_sel = target_id[mask]
        n = sel.numel()
        if n == 0:
            return
        self.n_events += n
        self.sum_se += float(sel.sum().item())
        self.sum_sq_se += float((sel ** 2).sum().item())
        # Per-action-family bucketing.
        for i in range(self.n_action_families):
            af_mask = (af_sel == i)
            n_i = int(af_mask.sum().item())
            if n_i:
                self.af_sum[i] += float(sel[af_mask].sum().item())
                self.af_count[i] += n_i

    def summary(self, vocab: tuple[str, ...]) -> dict:
        if self.n_events == 0:
            return {}
        mean = self.sum_se / self.n_events
        var = max((self.sum_sq_se / self.n_events) - mean ** 2, 0.0)
        per_af: dict[str, float] = {}
        for i, name in enumerate(vocab):
            if self.af_count[i]:
                per_af[name] = self.af_sum[i] / self.af_count[i]
        # OOV bucket if cardinality > vocab.
        if len(self.af_sum) > len(vocab):
            i = len(vocab)
            if self.af_count[i]:
                per_af["OOV"] = self.af_sum[i] / self.af_count[i]
        return {
            "val_recon_error_mean": mean,
            "val_recon_error_var": var,
            "val_recon_error_by_action_family": per_af,
        }


def _maybe_init_aim(
    enabled: bool,
    repo: str,
    experiment: str,
    run_name: str,
):
    """Open an aim.Run, or return None if Aim is disabled / not installed."""
    if not enabled:
        return None
    if not _AIM_AVAILABLE:
        print("[aim] aim not installed; logging will be file-only.", flush=True)
        return None
    try:
        run = aim.Run(experiment=experiment, repo=repo)
        run.name = run_name
        return run
    except Exception as e:  # noqa: BLE001 — defensive: never let aim crash training
        print(f"[aim] failed to open Run ({e}); logging will be file-only.", flush=True)
        return None


def _log_run_metadata(
    aim_run,
    *,
    model: torch.nn.Module,
    objective: str,
    train_parquet: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    base_lr: float,
    weight_decay: float,
    grad_clip: float,
    patience: int,
    min_delta: float,
    val_fraction: float,
    boundary_gap_seconds: int,
    window_size: int,
    stride: int,
    num_workers: int,
    torch_num_threads: int | None,
    max_train_steps: int | None,
    rng_seed: int,
    device: torch.device,
    diag,
    feature_set: str = "rich",
) -> None:
    """Populate the Aim run's hparams / model / system / dataset blocks."""
    if aim_run is None:
        return

    aim_run["hparams"] = {
        # Objective + recipe
        "objective":            objective,
        "feature_set":          feature_set,
        "epochs":               epochs,
        "batch_size":           batch_size,
        "lr":                   base_lr,
        "weight_decay":         weight_decay,
        "grad_clip":            grad_clip,
        "patience":             patience,
        "min_delta":            min_delta,
        # Val carve
        "val_fraction":         val_fraction,
        "boundary_gap_seconds": boundary_gap_seconds,
        # Windowing
        "window_size":          window_size,
        "stride":               stride,
        # Architecture (read from the constructed model)
        "d_model":              getattr(getattr(model, "body", None), "d_model", None),
        "n_layers":             len(getattr(getattr(model, "body", None), "blocks", []) or []),
        "dropout":              0.1,  # locked in MambaBody default
        # MEM-specific
        "mask_fraction":        0.15 if objective in ("mem", "mem-fa") else None,
        # Runtime
        "num_workers":          num_workers,
        "torch_num_threads":    torch_num_threads,
        "max_train_steps":      max_train_steps,
        "rng_seed":             rng_seed,
    }

    # Model summary.
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_encoder = sum(p.numel() for p in model.encoder.parameters()) if hasattr(model, "encoder") else 0
    n_body = sum(p.numel() for p in model.body.parameters()) if hasattr(model, "body") else 0
    n_head = sum(p.numel() for p in model.head.parameters()) if hasattr(model, "head") else 0
    aim_run["model"] = {
        "model_class":      type(model).__name__,
        "total_params":     int(n_total),
        "trainable_params": int(n_trainable),
        "encoder_params":   int(n_encoder),
        "body_params":      int(n_body),
        "head_params":      int(n_head),
        "vocab_size":       getattr(model, "vocab_size", None),
        "d_model":          getattr(getattr(model, "body", None), "d_model", None),
    }

    # System.
    sysblock = {
        "device":            str(device),
        "torch_version":     str(torch.__version__),
        "python_version":    platform.python_version(),
        "platform":          platform.platform(),
        "host":              platform.node(),
        "cwd":               str(Path.cwd()),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        sysblock["gpu_name"] = torch.cuda.get_device_name(0)
        sysblock["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1
        )
        sysblock["cuda_version"] = torch.version.cuda
    aim_run["system"] = sysblock

    # Dataset / split diagnostics.
    aim_run["dataset"] = {
        "train_parquet": str(train_parquet),
        "output_dir":    str(output_dir),
        **diag.as_dict(),
    }


def _aim_track(aim_run, name: str, value, *, step=None, epoch=None, context=None) -> None:
    """Defensive tracker — never crash training on aim error."""
    if aim_run is None:
        return
    try:
        kwargs: dict = {}
        if step is not None:
            kwargs["step"] = step
        if epoch is not None:
            kwargs["epoch"] = epoch
        if context is not None:
            kwargs["context"] = context
        aim_run.track(value, name=name, **kwargs)
    except Exception:  # noqa: BLE001
        pass


def _seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker so MEM mask sampling (and any other
    per-getitem RNG use) is reproducible across runs.

    Pattern from https://pytorch.org/docs/stable/notes/randomness.html
    — derive each worker's seed from torch's initial seed so that running
    the same training command with the same ``--rng-seed`` produces the
    same masks even with multi-process loading.
    """
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    import random as _random
    _random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)


def train(
    objective: str,
    train_parquet: Path,
    output_dir: Path,
    *,
    epochs: int = 15,
    batch_size: int = 8,
    base_lr: float = 3e-4,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    patience: int = 5,
    min_delta: float = 1e-4,
    val_fraction: float = 0.20,
    boundary_gap_seconds: int = 900,
    window_size: int = 128,
    stride: int = 32,
    num_workers: int = 0,
    torch_num_threads: int | None = None,
    max_train_steps: int | None = None,
    rng_seed: int = 0,
    aim_enabled: bool = True,
    aim_repo: str = ".aim",
    aim_experiment: str = "v0.2-mamba-run-1",
    run_name: str | None = None,
    log_every: int = 200,
    feature_set: str = "rich",
) -> dict:
    # Cap intra-op torch threading. With num_workers > 0 each worker also
    # spawns its own torch thread pool; without this cap on a 32-core box
    # the workers and the main-process model thrash each other.
    if torch_num_threads is not None and torch_num_threads > 0:
        torch.set_num_threads(torch_num_threads)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    log_fh = log_path.open("a")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    torch.manual_seed(rng_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} torch={torch.__version__}")

    log(f"loading + carving {train_parquet}")
    cfg = V02ValCarveConfig(
        val_fraction=val_fraction,
        boundary_gap_seconds=boundary_gap_seconds,
        shuffle_train_windows=True,
        shuffle_val_windows=False,
    )
    train_ds, val_ds, diag = carve_train_val(
        train_parquet,
        cfg,
        window_size=window_size,
        stride=stride,
        objective=objective,
    )
    log("split diagnostics:")
    for k, v in diag.as_dict().items():
        log(f"  {k}: {v}")

    # Persist diagnostics alongside the training summary so reviewers can
    # verify the carve without rerunning anything.
    (output_dir / "split_diagnostics.json").write_text(json.dumps(diag.as_dict(), indent=2))

    # ---- Aim tracker init (after diag so we can log it as run metadata) ----
    if run_name is None:
        run_name = f"{objective}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    aim_run = _maybe_init_aim(
        enabled=aim_enabled,
        repo=aim_repo,
        experiment=aim_experiment,
        run_name=run_name,
    )
    if aim_run is not None:
        log(f"[aim] tracking enabled: experiment={aim_experiment} run={run_name} repo={aim_repo}")
    else:
        log("[aim] tracking disabled (or not installed)")

    g = torch.Generator()
    g.manual_seed(rng_seed)

    # persistent_workers + prefetch_factor are only valid when num_workers>0.
    loader_kwargs: dict = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=_seed_worker,
        generator=g,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_ds,
        shuffle=cfg.shuffle_train_windows,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=cfg.shuffle_val_windows,  # False per Plan E v2 §6
        drop_last=False,
        **loader_kwargs,
    )

    model = build_scorer(objective, feature_set=feature_set).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model={type(model).__name__} params={n_params/1e6:.2f}M feature_set={feature_set}")
    raw_input_dim = getattr(model.encoder, "raw_input_dim", None)
    n_embed_cols = len(getattr(model.encoder, "embed_specs", []))
    n_float_cols = len(getattr(model.encoder, "float_specs", []))
    log(f"  encoder: raw_input_dim={raw_input_dim} embed_cols={n_embed_cols} float_cols={n_float_cols}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr, weight_decay=weight_decay, betas=(0.9, 0.95)
    )

    total_steps = epochs * len(train_loader)
    if max_train_steps is not None:
        total_steps = min(total_steps, max_train_steps)
    log(f"total_steps={total_steps}")

    # Populate Aim run metadata now that the model + diag are built.
    _log_run_metadata(
        aim_run,
        model=model,
        objective=objective,
        train_parquet=train_parquet,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        base_lr=base_lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        patience=patience,
        min_delta=min_delta,
        val_fraction=val_fraction,
        boundary_gap_seconds=boundary_gap_seconds,
        window_size=window_size,
        stride=stride,
        num_workers=num_workers,
        torch_num_threads=torch_num_threads,
        max_train_steps=max_train_steps,
        rng_seed=rng_seed,
        device=device,
        diag=diag,
        feature_set=feature_set,
    )

    best_val = float("inf")
    no_improve_epochs = 0
    history: list[dict] = []
    global_step = 0
    t_start = time.time()

    from src.processing.v0_2_behavior_builder import ACTION_FAMILY_VOCAB
    n_action_families = len(ACTION_FAMILY_VOCAB) + 1  # incl. OOV

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_loss_n = 0
        train_correct = 0  # NLL only
        train_n_events = 0  # NLL only
        recent_losses: list[float] = []  # rolling window for the heartbeat line
        epoch_start = time.time()
        n_train_batches = len(train_loader)
        for batch in train_loader:
            if max_train_steps is not None and global_step >= max_train_steps:
                break
            batch = _move(batch, device)
            lr = _cosine_warmup_lr(global_step, total_steps, base_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                fwd = _forward_with_metrics(model, batch, objective)
            loss = fwd["loss"]
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            loss_val = float(loss.item())
            grad_norm_val = float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm)
            train_loss_sum += loss_val
            train_loss_n += 1
            recent_losses.append(loss_val)
            if len(recent_losses) > 50:
                recent_losses.pop(0)
            global_step += 1

            # Per-step Aim tracks.
            _aim_track(aim_run, "train_loss", loss_val,
                       step=global_step, context={"subset": "train"})
            _aim_track(aim_run, "lr", lr, step=global_step)
            _aim_track(aim_run, "grad_norm", grad_norm_val, step=global_step)

            # Objective-specific per-step tracks.
            if objective == "nll":
                step_acc = float(fwd["correct"].mean().item())
                step_ppl = math.exp(min(loss_val, 50))
                train_correct += int(fwd["correct"].sum().item())
                train_n_events += int(fwd["correct"].numel())
                _aim_track(aim_run, "train_token_accuracy", step_acc,
                           step=global_step, context={"subset": "train"})
                _aim_track(aim_run, "train_perplexity", step_ppl,
                           step=global_step, context={"subset": "train"})

            # Per-step heartbeat in the text log every ``log_every`` steps.
            if log_every and global_step % log_every == 0:
                elapsed = time.time() - epoch_start
                in_epoch_step = global_step - epoch * n_train_batches
                steps_per_sec = in_epoch_step / max(elapsed, 1e-9)
                running = sum(recent_losses) / max(len(recent_losses), 1)
                eta_in_epoch = (n_train_batches - in_epoch_step) / max(steps_per_sec, 1e-9)
                log(
                    f"  epoch {epoch:02d} step {in_epoch_step:>5}/{n_train_batches} | "
                    f"loss(50)={running:.4f} grad_norm={grad_norm_val:.3f} lr={lr:.2e} "
                    f"{steps_per_sec:.2f}step/s eta={eta_in_epoch:.0f}s"
                )
                _aim_track(aim_run, "throughput_steps_per_sec", steps_per_sec, step=global_step)

        # ---- validation ----
        model.eval()
        val_loss_sum = 0.0
        val_loss_n = 0
        val_acc: _NLLAcc | _MEMAcc
        if objective == "nll":
            val_acc = _NLLAcc()
        else:
            val_acc = _MEMAcc(n_action_families=n_action_families)
        with torch.no_grad():
            for batch in val_loader:
                batch = _move(batch, device)
                with _autocast(device):
                    fwd = _forward_with_metrics(model, batch, objective)
                val_loss_sum += float(fwd["loss"].item())
                val_loss_n += 1
                val_acc.update(fwd)

        train_loss = train_loss_sum / max(1, train_loss_n)
        val_loss = val_loss_sum / max(1, val_loss_n)
        wall = time.time() - epoch_start
        train_val_gap = train_loss - val_loss

        if objective == "nll":
            val_summary = val_acc.summary()  # type: ignore[arg-type]
            train_token_acc_epoch = (train_correct / train_n_events) if train_n_events else 0.0
        else:
            val_summary = val_acc.summary(ACTION_FAMILY_VOCAB)  # type: ignore[arg-type]
            train_token_acc_epoch = None

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_val_gap": train_val_gap,
            "wall_seconds": wall,
            "lr": lr,
            **val_summary,
            **({"train_token_accuracy": train_token_acc_epoch} if train_token_acc_epoch is not None else {}),
        })
        log(
            f"epoch {epoch:02d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"gap={train_val_gap:+.4f} lr={lr:.2e} wall={wall:.0f}s"
        )
        if objective == "nll":
            log(
                f"  val_token_acc={val_summary.get('val_token_accuracy', 0.0):.4f} "
                f"val_perplexity={val_summary.get('val_perplexity', 0.0):.4f} "
                f"val_nll_var={val_summary.get('val_nll_var', 0.0):.4f}"
            )
        else:
            per_af = val_summary.get("val_recon_error_by_action_family", {})
            log(
                f"  val_recon_mean={val_summary.get('val_recon_error_mean', 0.0):.4f} "
                f"val_recon_var={val_summary.get('val_recon_error_var', 0.0):.4f}"
            )
            if per_af:
                top = sorted(per_af.items(), key=lambda kv: -kv[1])[:5]
                log("  recon by action_family (top 5): "
                    + ", ".join(f"{k}={v:.3f}" for k, v in top))

        # ---- Per-epoch Aim tracks ----
        _aim_track(aim_run, "epoch_train_loss", train_loss,
                   epoch=epoch, context={"subset": "train"})
        _aim_track(aim_run, "val_loss", val_loss,
                   epoch=epoch, context={"subset": "val"})
        _aim_track(aim_run, "epoch_wall_seconds", wall, epoch=epoch)
        _aim_track(aim_run, "train_val_gap", train_val_gap, epoch=epoch)
        # Best-val-so-far includes the current epoch.
        running_best = min(best_val, val_loss)
        _aim_track(aim_run, "best_val_loss", running_best, epoch=epoch)

        if objective == "nll":
            _aim_track(aim_run, "val_token_accuracy",
                       val_summary.get("val_token_accuracy", 0.0),
                       epoch=epoch, context={"subset": "val"})
            _aim_track(aim_run, "val_perplexity",
                       val_summary.get("val_perplexity", 0.0),
                       epoch=epoch, context={"subset": "val"})
            _aim_track(aim_run, "val_nll_mean",
                       val_summary.get("val_nll_mean", 0.0),
                       epoch=epoch, context={"subset": "val"})
            _aim_track(aim_run, "val_nll_var",
                       val_summary.get("val_nll_var", 0.0),
                       epoch=epoch, context={"subset": "val"})
            if train_token_acc_epoch is not None:
                _aim_track(aim_run, "epoch_train_token_accuracy",
                           train_token_acc_epoch,
                           epoch=epoch, context={"subset": "train"})
        else:
            _aim_track(aim_run, "val_recon_error_mean",
                       val_summary.get("val_recon_error_mean", 0.0),
                       epoch=epoch, context={"subset": "val"})
            _aim_track(aim_run, "val_recon_error_var",
                       val_summary.get("val_recon_error_var", 0.0),
                       epoch=epoch, context={"subset": "val"})
            for af_name, af_mean in val_summary.get(
                "val_recon_error_by_action_family", {}
            ).items():
                _aim_track(aim_run,
                           "val_recon_error",
                           af_mean,
                           epoch=epoch,
                           context={"subset": "val", "action_family": af_name})

        # Per-epoch checkpoint.
        torch.save(
            {"epoch": epoch, "model_state": model.state_dict(), "objective": objective,
             "feature_set": feature_set,
             "val_loss": val_loss, "train_loss": train_loss},
            output_dir / f"epoch_{epoch:02d}.pt",
        )

        if val_loss < best_val - min_delta:
            best_val = val_loss
            no_improve_epochs = 0
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "objective": objective,
                 "feature_set": feature_set,
                 "val_loss": val_loss, "train_loss": train_loss},
                output_dir / "best.pt",
            )
            log(f"  best.pt updated (val_loss={val_loss:.4f})")
        else:
            no_improve_epochs += 1
            log(f"  no improvement ({no_improve_epochs}/{patience})")
            if no_improve_epochs >= patience:
                log(f"early stopping at epoch {epoch}")
                break
        if max_train_steps is not None and global_step >= max_train_steps:
            log(f"reached max_train_steps={max_train_steps}; stopping")
            break

    summary = {
        "objective": objective,
        "feature_set": feature_set,
        "encoder_raw_input_dim": raw_input_dim,
        "encoder_n_embed_cols": n_embed_cols,
        "encoder_n_float_cols": n_float_cols,
        "best_val_loss": best_val,
        "epochs_run": len(history),
        "wall_seconds": time.time() - t_start,
        "history": history,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"done; best_val_loss={best_val:.4f} wall={summary['wall_seconds']:.0f}s")

    if aim_run is not None:
        try:
            aim_run["final"] = {
                "best_val_loss": best_val,
                "epochs_run": len(history),
                "wall_seconds": summary["wall_seconds"],
                "global_step": global_step,
            }
            aim_run.close()
        except Exception as e:  # noqa: BLE001
            log(f"[aim] error closing run: {e}")

    log_fh.close()
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--objective", required=True, choices=["nll", "mem", "mem-fa"],
                   help="nll = next-event prediction; mem = vanilla masked event modeling "
                        "(continuous-MSE; run-#1 negative result); mem-fa = field-aware "
                        "categorical MEM (3a.1, recommended over mem).")
    p.add_argument("--feature-set", choices=["rich", "flat"], default="rich",
                   help="Encoder feature inventory. 'rich' (default) = all 42 columns "
                        "(33 embed + 9 float). 'flat' = drops the 7 high-cardinality hash "
                        "columns to match the IF-flat baseline; used for the §6 encoder "
                        "ablation. Existing checkpoints (no field) load as 'rich'.")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--train-parquet", default=Path("data/processed/v0.2/train.parquet"), type=Path)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument(
        "--torch-num-threads", type=int, default=4,
        help=(
            "Cap intra-op torch threading. Default 4 pairs with the default "
            "num_workers=6 so dataloader workers and the main-process model "
            "don't oversubscribe on a typical CPU box."
        ),
    )
    p.add_argument("--max-train-steps", type=int, default=None,
                   help="Cap total optimizer steps for smoke / debug runs.")
    p.add_argument("--rng-seed", type=int, default=0)
    # Aim tracker
    p.add_argument("--no-aim", action="store_true",
                   help="Disable Aim experiment tracking even if aim is installed.")
    p.add_argument("--aim-repo", type=str, default=".aim",
                   help="Aim repo path (default: ./.aim relative to mamba-edge).")
    p.add_argument("--aim-experiment", type=str, required=True,
                   help="Aim experiment name (groups runs in the UI). Required.")
    p.add_argument("--run-name", type=str, required=True,
                   help="Aim run name. Required — uniquely identifies the run in the UI.")
    p.add_argument("--log-every", type=int, default=100,
                   help="Per-step heartbeat interval (steps). 0 disables in-epoch logging.")
    args = p.parse_args()

    train(
        objective=args.objective,
        train_parquet=args.train_parquet,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        base_lr=args.lr,
        num_workers=args.num_workers,
        torch_num_threads=args.torch_num_threads,
        max_train_steps=args.max_train_steps,
        rng_seed=args.rng_seed,
        aim_enabled=not args.no_aim,
        aim_repo=args.aim_repo,
        aim_experiment=args.aim_experiment,
        run_name=args.run_name,
        log_every=args.log_every,
        feature_set=args.feature_set,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
