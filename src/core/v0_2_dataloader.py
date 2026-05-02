"""v0.2 sequence dataloader.

Reads ``train.parquet`` / ``test.parquet`` (Plan G outputs), forms sliding
windows over the event_time-sorted rows, and returns per-window dicts
suitable for the ``V02EventEncoder``.

For training time, the ``carve_train_val`` factory splits a single
``train.parquet`` into chronologically-disjoint train and val datasets
with a 15-minute boundary gap (no window can span the train/val boundary).
It also returns a diagnostics dict (row counts, window counts, effective
val fraction, split timestamp) for the trainer to log.

Plan E v2 §6 lock:

  * ``val_fraction = 0.20``
  * Boundary gap = 15 minutes (split_ts - 7.5min .. split_ts + 7.5min)
  * Train: DataLoader(shuffle=True); val: DataLoader(shuffle=False).
    The dataset itself does NOT shuffle window_starts — that lets the
    trainer get fresh per-epoch ordering via the standard DataLoader API.
  * Process-tree exclusion is OFF for run #1.

By default the dataset only materializes the model-input columns + the
``event_time`` column. Set ``include_aux_columns=True`` to additionally
load string side columns like ``process_tree_root_exec_id`` (eval / audit).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from src.core.v0_2_event_encoder import (
    ALL_EMBED_COLUMNS,
    ALL_FLOAT_COLUMNS,
    model_input_columns,
)


@dataclass(frozen=True)
class V02ValCarveConfig:
    """Configuration for ``carve_train_val``. See Plan E v2 §6.

    The two ``shuffle_*`` flags here are advisory — the dataset itself no
    longer shuffles window_starts; the trainer is expected to read these
    flags and pass them to ``torch.utils.data.DataLoader(shuffle=...)``.
    Kept on the config for documentation and to feed the diagnostics dict.
    """
    val_fraction: float = 0.20
    boundary_gap_seconds: int = 900  # 15-minute gap (split_ts ± 7.5min)
    shuffle_train_windows: bool = True
    shuffle_val_windows: bool = False


def _load_columns_to_arrays(
    table: pa.Table,
    *,
    include_aux_columns: bool,
) -> dict[str, np.ndarray]:
    """Convert a pa.Table into a dict of in-memory numpy arrays.

    Always loaded:
      * every column in ``ALL_EMBED_COLUMNS`` (cast to int64)
      * every column in ``ALL_FLOAT_COLUMNS`` (cast to float32)
      * ``event_time`` (cast to int64 ns; needed for eval-time labeling
        and val-carve boundary checks)

    Loaded only when ``include_aux_columns=True``:
      * ``process_tree_root_exec_id`` (string side col) — used by audit
        notebooks / future ablations, not by the encoder.

    Keeping aux columns out of the training-time load saves ~250 MB on
    a 17M-row train parquet.
    """
    out: dict[str, np.ndarray] = {}
    # ``copy=True`` materializes a writable buffer so downstream
    # ``torch.from_numpy`` on a slice produces a writable tensor without a
    # per-call copy in __getitem__. Pyarrow's zero-copy view is read-only;
    # one copy here saves N copies per training step.
    for spec in ALL_EMBED_COLUMNS:
        out[spec.column] = table[spec.column].to_numpy(zero_copy_only=False).astype(np.int64, copy=True)
    for spec in ALL_FLOAT_COLUMNS:
        out[spec.column] = table[spec.column].to_numpy(zero_copy_only=False).astype(np.float32, copy=True)
    out["event_time"] = table["event_time"].cast(pa.int64()).to_numpy(zero_copy_only=False).astype(np.int64, copy=True)
    if include_aux_columns and "process_tree_root_exec_id" in table.column_names:
        out["process_tree_root_exec_id"] = np.array(
            table["process_tree_root_exec_id"].to_pylist(), dtype=object
        )
    return out


class V02SequenceDataset(Dataset):
    """Sliding-window dataset over a single in-memory pa.Table.

    The table must already be sorted by ``event_time`` and must have the
    encoder columns plus ``event_time``. Windows are formed only within
    this table, so a caller that wants train/val partitions should call
    ``carve_train_val`` first and pass each side here.

    Per-epoch shuffling is the caller's responsibility via
    ``torch.utils.data.DataLoader(shuffle=True)`` — the dataset stores
    ``window_starts`` in chronological order and never permutes it.

    For MEM, the per-sample 15% mask is sampled via ``torch.randperm`` in
    ``__getitem__``; reproducibility across runs requires the trainer to
    seed each DataLoader worker (see ``train_v0_2_mamba.py:_seed_worker``).
    """

    def __init__(
        self,
        table: pa.Table,
        window_size: int = 128,
        stride: int = 32,
        objective: str = "nll",
        mask_fraction: float = 0.15,
        include_aux_columns: bool = False,
    ) -> None:
        if objective not in ("nll", "mem", "mem-fa"):
            raise ValueError(
                f"objective must be 'nll', 'mem', or 'mem-fa', got {objective!r}"
            )
        self.window_size = window_size
        self.stride = stride
        self.objective = objective
        self.mask_fraction = mask_fraction
        self.n_rows = table.num_rows
        self.include_aux_columns = include_aux_columns

        # Row-count safety: a dataset with fewer than window_size rows
        # cannot produce a single full window. Caller-side carve mistakes
        # (val_fraction too small, gap too large) surface here cleanly.
        if self.n_rows < window_size:
            raise ValueError(
                f"V02SequenceDataset got n_rows={self.n_rows} < window_size={window_size}; "
                f"check val_fraction / boundary_gap_seconds settings"
            )

        self.arrays = _load_columns_to_arrays(table, include_aux_columns=include_aux_columns)

        # Chronological window starts. DataLoader(shuffle=True) handles
        # per-epoch randomization; we never permute the list ourselves.
        n_windows = (self.n_rows - window_size) // stride + 1
        self.window_starts: list[int] = [i * stride for i in range(n_windows)]

    def __len__(self) -> int:
        return len(self.window_starts)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        start = self.window_starts[i]
        end = start + self.window_size

        # _load_columns_to_arrays already produces writable, correctly-typed
        # numpy arrays at construction time. A slice of a writable numpy
        # array is itself writable, so torch.from_numpy on a slice does
        # NOT need to copy.
        features: dict[str, torch.Tensor] = {}
        for spec in ALL_EMBED_COLUMNS:
            features[spec.column] = torch.from_numpy(self.arrays[spec.column][start:end])
        for spec in ALL_FLOAT_COLUMNS:
            features[spec.column] = torch.from_numpy(self.arrays[spec.column][start:end])

        target = torch.from_numpy(self.arrays["f_action_family"][start:end])
        event_time = torch.from_numpy(self.arrays["event_time"][start:end])

        sample: dict[str, torch.Tensor] = {
            "features": features,
            "target_id": target,
            "event_time": event_time,
        }

        if self.objective in ("mem", "mem-fa"):
            # Per-sample 15% mask. Uses torch's RNG state, which the
            # trainer seeds per-worker for reproducibility. Both vanilla
            # MEM and field-aware MEM use the same mask sampling.
            n_mask = max(1, int(round(self.mask_fraction * self.window_size)))
            perm = torch.randperm(self.window_size)
            mask = torch.zeros(self.window_size, dtype=torch.bool)
            mask[perm[:n_mask]] = True
            sample["mask"] = mask

        return sample


def _read_parquet_sorted(parquet_path: Path) -> pa.Table:
    """Read a parquet, sort by event_time, normalize event_time precision
    to ``timestamp[ns]``, and strip any pyarrow.dataset augmented fields
    that ``sort_by`` may have injected.

    The precision normalization is load-bearing: DuckDB's COPY ... TO
    PARQUET (used by ``scripts/run_v0.2_parser.py`` for the chronological
    sort+concat) writes ``timestamp[us]`` regardless of the input
    precision, because DuckDB's internal TIMESTAMP type is microsecond.
    Downstream code (carve thresholds, eval joins) assumes ns-precision
    int64 arithmetic and would silently mis-scale by 1000 if given
    ``timestamp[us]``. Casting at the boundary keeps the rest of the
    module unit-agnostic.
    """
    table = pq.read_table(parquet_path)
    table = table.sort_by("event_time")
    if "event_time" in table.column_names:
        et = table["event_time"]
        if pa.types.is_timestamp(et.type) and et.type.unit != "ns":
            target = pa.timestamp("ns", tz=et.type.tz)
            idx = table.schema.get_field_index("event_time")
            table = table.set_column(idx, "event_time", et.cast(target))
    real_cols = [c for c in table.column_names if not c.startswith("__")]
    if len(real_cols) != len(table.column_names):
        table = table.select(real_cols)
    return table


@dataclass(frozen=True)
class V02ValCarveDiagnostics:
    """Returned alongside the (train, val) datasets so the trainer can log
    the carve outcome and so unit tests can assert mass-conservation."""
    n_total: int
    n_train: int
    n_val: int
    n_gap: int
    train_windows: int
    val_windows: int
    effective_val_fraction: float
    split_ts_ns: int
    boundary_gap_seconds: int

    def as_dict(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_gap": self.n_gap,
            "train_windows": self.train_windows,
            "val_windows": self.val_windows,
            "effective_val_fraction": self.effective_val_fraction,
            "split_ts_ns": self.split_ts_ns,
            "boundary_gap_seconds": self.boundary_gap_seconds,
        }


def carve_train_val(
    parquet_path: Path,
    cfg: V02ValCarveConfig | None = None,
    *,
    window_size: int = 128,
    stride: int = 32,
    objective: str = "nll",
    include_aux_columns: bool = False,
) -> tuple[V02SequenceDataset, V02SequenceDataset, V02ValCarveDiagnostics]:
    """Split ``parquet_path`` into chronologically-disjoint train + val
    datasets with a 15-minute boundary gap.

    Boundary semantics (Plan E v2 §6):

      ``split_ts`` = event_time of the row at index
        floor((1 - val_fraction) * n_rows) after sorting by event_time.
      train: event_time <  split_ts - 7.5min
      gap  : split_ts - 7.5min <= event_time < split_ts + 7.5min  (dropped)
      val  : event_time >= split_ts + 7.5min

    Windows are formed independently within train and val, so no window
    spans the boundary.

    Returns ``(train_ds, val_ds, diagnostics)``. Both datasets must contain
    at least ``window_size`` rows after the gap drop; if either is too
    small, ``V02SequenceDataset.__init__`` raises ``ValueError``.
    """
    cfg = cfg or V02ValCarveConfig()
    parquet_path = Path(parquet_path)

    table = _read_parquet_sorted(parquet_path)
    n_rows = table.num_rows
    if n_rows == 0:
        raise ValueError(f"empty parquet: {parquet_path}")

    split_idx = int((1.0 - cfg.val_fraction) * n_rows)
    split_idx = max(1, min(split_idx, n_rows - 1))
    split_ts_ns = int(table["event_time"].cast(pa.int64())[split_idx].as_py())
    half_gap_ns = (cfg.boundary_gap_seconds * 1_000_000_000) // 2

    ts = table["event_time"].cast(pa.int64())
    train_threshold = pa.scalar(split_ts_ns - half_gap_ns, type=pa.int64())
    val_threshold = pa.scalar(split_ts_ns + half_gap_ns, type=pa.int64())

    train_table = table.filter(pc.less(ts, train_threshold))
    val_table = table.filter(pc.greater_equal(ts, val_threshold))
    n_train = train_table.num_rows
    n_val = val_table.num_rows
    n_gap = n_rows - n_train - n_val
    assert n_gap >= 0, f"gap row count is negative ({n_gap}); carve logic bug"

    if n_train < window_size:
        raise ValueError(
            f"train partition has {n_train} rows < window_size={window_size}; "
            f"loosen val_fraction or boundary_gap_seconds"
        )
    if n_val < window_size:
        raise ValueError(
            f"val partition has {n_val} rows < window_size={window_size}; "
            f"loosen val_fraction or boundary_gap_seconds"
        )

    train_ds = V02SequenceDataset(
        train_table, window_size=window_size, stride=stride,
        objective=objective, include_aux_columns=include_aux_columns,
    )
    val_ds = V02SequenceDataset(
        val_table, window_size=window_size, stride=stride,
        objective=objective, include_aux_columns=include_aux_columns,
    )
    diag = V02ValCarveDiagnostics(
        n_total=n_rows,
        n_train=n_train,
        n_val=n_val,
        n_gap=n_gap,
        train_windows=len(train_ds),
        val_windows=len(val_ds),
        effective_val_fraction=(n_val / (n_train + n_val)) if (n_train + n_val) else 0.0,
        split_ts_ns=split_ts_ns,
        boundary_gap_seconds=cfg.boundary_gap_seconds,
    )
    return train_ds, val_ds, diag


def load_parquet_dataset(
    parquet_path: Path,
    *,
    window_size: int = 128,
    stride: int = 32,
    objective: str = "nll",
    include_aux_columns: bool = False,
) -> V02SequenceDataset:
    """Convenience for eval / inference: read a single parquet (e.g.,
    test.parquet) and return one chronological dataset. No carve.
    """
    table = _read_parquet_sorted(Path(parquet_path))
    return V02SequenceDataset(
        table, window_size=window_size, stride=stride,
        objective=objective, include_aux_columns=include_aux_columns,
    )


__all__ = [
    "V02ValCarveConfig",
    "V02ValCarveDiagnostics",
    "V02SequenceDataset",
    "carve_train_val",
    "load_parquet_dataset",
    "model_input_columns",
]
