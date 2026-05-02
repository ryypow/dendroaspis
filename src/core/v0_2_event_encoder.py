"""v0.2 per-event encoder.

Reads the v0.2 model columns produced by v0_2_behavior_builder +
v0_2_features: existing f_* features (minus dropped constants), the
seven Plan G lineage / action columns, and the four 3a' promoted-
to-model categorical columns (f_path_category, f_dst_ip_category,
f_dst_port_category, f_object_category). Each column gets its own
embedding table or float pass-through; concatenated and projected to
``d_model``. Authoritative current-state listing is in
``docs/releases/v0.2-course-milestone/v0.2_encoder_design_final.md``.

Cardinalities are pulled from the constants in
``src.features.v0_2_features`` and ``src.processing.v0_2_behavior_builder``
so the encoder can never drift from the on-disk schema.

Pre-Mamba per-event vector flow:

    parquet column tensors (B, L) per field
        -> per-field embedding tables / float pass-through
        -> concat along last dim (B, L, sum_of_dims)
        -> Linear(sum_of_dims, d_model)
        -> (B, L, d_model)

The MEM scorer detaches this output before computing its target, so
inserting a [MASK] vector at masked positions does not gradient-update
the embedding tables for those positions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.features.v0_2_features import (
    CMDLINE_COMPRESSION_INCOMPRESSIBLE,
    CMDLINE_ENTROPY_HIGH,
    EVENT_TYPE_OOV_INDEX,
    KPROBE_ACTION_OOV_INDEX,
    KPROBE_FUNCTION_OOV_INDEX,
    KPROBE_POLICY_OOV_INDEX,
    LINEAGE_BAG_HASH_V1_BUCKETS,
    PARENT_PROC_HASH_BUCKETS,
    PATH_SENSITIVITY_NONE,
    PROC_CWD_HASH_BUCKETS,
    PROC_NAME_HASH_BUCKETS,
    PROC_UID_BUCKET_OTHER,
    SOCK_FAMILY_OOV_INDEX,
    TIME_SINCE_PARENT_HR,
)
from src.processing.v0_2_behavior_builder import (
    ACTION_FAMILY_CARDINALITY,
    DST_IP_CATEGORY_CARDINALITY,
    DST_PORT_CATEGORY_CARDINALITY,
    OBJECT_CATEGORY_CARDINALITY,
    PATH_CATEGORY_CARDINALITY,
)


# Each cardinality is len(VOCAB) + 1 OOV (or +1 NONE + 1 OOV) so the
# resolved indices fit. These mirror v0_2_features.py constants — single
# source of truth so the encoder can never be sized wrong.

EVENT_TYPE_CARD: int = EVENT_TYPE_OOV_INDEX + 1                              # 4
KPROBE_FUNCTION_CARD: int = KPROBE_FUNCTION_OOV_INDEX + 1                    # 14
KPROBE_POLICY_CARD: int = KPROBE_POLICY_OOV_INDEX + 1                        # 8
KPROBE_ACTION_CARD: int = KPROBE_ACTION_OOV_INDEX + 1                        # 3
PROC_UID_BUCKET_CARD: int = PROC_UID_BUCKET_OTHER + 1                        # 5
DST_PORT_BUCKET_CARD: int = 7   # NONE..HIGH (constants in v0_2_features)
ARGS_LENGTH_BUCKET_CARD: int = 6
CAP_COUNT_BUCKET_CARD: int = 5
PATH_SENSITIVITY_CARD: int = PATH_SENSITIVITY_NONE + 1                       # 8
CMDLINE_ENTROPY_CARD: int = CMDLINE_ENTROPY_HIGH + 1                         # 6
CMDLINE_COMPRESSION_CARD: int = CMDLINE_COMPRESSION_INCOMPRESSIBLE + 1       # 6
TIME_SINCE_PARENT_CARD: int = TIME_SINCE_PARENT_HR + 1                       # 6
SOCK_FAMILY_CARD: int = SOCK_FAMILY_OOV_INDEX + 1                            # 4

LINEAGE_DEPTH_CARD: int = 16
DELTA_T_LOG_BUCKET_CARD: int = 10
PROCESS_AGE_LOG_BUCKET_CARD: int = 10
PARENT_CHILD_PAIR_HASH_CARD: int = 1024
ROOT_ANCESTOR_BASENAME_HASH_CARD: int = 1024
PROCESS_TREE_ID_HASH_CARD: int = 4096


@dataclass(frozen=True)
class EmbedSpec:
    """Embedding-backed column specification."""
    column: str
    cardinality: int
    dim: int


@dataclass(frozen=True)
class FloatSpec:
    """Float pass-through column specification.

    ``scale`` is applied as ``x / scale`` after casting to float32. Use 1.0
    for already-binary booleans; larger scales for raw integer columns
    (mmap prot bits, fd numbers) so values stay in roughly [0, 1].
    """
    column: str
    scale: float = 1.0


# --- Existing 33 features (drop f_args_truncated; constant in current corpus) ---

EXISTING_EMBED_COLUMNS: tuple[EmbedSpec, ...] = (
    EmbedSpec("f_event_type",                          EVENT_TYPE_CARD, 8),
    EmbedSpec("f_kprobe_function",                     KPROBE_FUNCTION_CARD, 16),
    EmbedSpec("f_kprobe_policy",                       KPROBE_POLICY_CARD, 8),
    EmbedSpec("f_kprobe_action",                       KPROBE_ACTION_CARD, 4),
    EmbedSpec("f_proc_uid_bucket",                     PROC_UID_BUCKET_CARD, 4),
    EmbedSpec("f_dst_port_bucket",                     DST_PORT_BUCKET_CARD, 4),
    EmbedSpec("f_args_length_bucket",                  ARGS_LENGTH_BUCKET_CARD, 4),
    EmbedSpec("f_cap_count_bucket",                    CAP_COUNT_BUCKET_CARD, 4),
    EmbedSpec("f_path_sens_cwd",                       PATH_SENSITIVITY_CARD, 8),
    EmbedSpec("f_path_sens_binary",                    PATH_SENSITIVITY_CARD, 8),
    EmbedSpec("f_path_sens_kp",                        PATH_SENSITIVITY_CARD, 8),
    EmbedSpec("f_proc_name_hash",                      PROC_NAME_HASH_BUCKETS, 32),
    EmbedSpec("f_parent_proc_hash",                    PARENT_PROC_HASH_BUCKETS, 32),
    EmbedSpec("f_proc_cwd_hash",                       PROC_CWD_HASH_BUCKETS, 16),
    EmbedSpec("f_lineage_bag_hash",                    LINEAGE_BAG_HASH_V1_BUCKETS, 16),
    EmbedSpec("f_cmdline_entropy",                     CMDLINE_ENTROPY_CARD, 4),
    EmbedSpec("f_cmdline_compress",                    CMDLINE_COMPRESSION_CARD, 4),
    EmbedSpec("f_time_since_parent_exec",              TIME_SINCE_PARENT_CARD, 4),
    EmbedSpec("f_kp_fd_install_path_sensitivity",      PATH_SENSITIVITY_CARD, 4),
    EmbedSpec("f_kp_mmap_path_sensitivity",            PATH_SENSITIVITY_CARD, 4),
    EmbedSpec("f_kp_tcp_connect_dst_port_bucket",      DST_PORT_BUCKET_CARD, 4),
    EmbedSpec("f_kp_tcp_connect_sock_family",          SOCK_FAMILY_CARD, 4),
)

EXISTING_FLOAT_COLUMNS: tuple[FloatSpec, ...] = (
    # Booleans (already 0/1). f_in_init_tree dropped 3a' — confirmed
    # constant zero in both train and test (audit cell 31 + cell 33).
    # f_is_procfs_walk RETAINED — became a real model feature after the
    # parser tag-not-drop fix (1,250 train + 96 test events have it = 1).
    FloatSpec("f_is_procfs_walk"),
    FloatSpec("f_uid_eq_parent"),
    FloatSpec("f_is_setuid_exec"),
    FloatSpec("f_kp_commit_creds_uid_change"),
    FloatSpec("f_kp_commit_creds_cap_change"),
    FloatSpec("f_kp_udp_sendmsg_dport_eq_53"),
    # Numerics (normalize to roughly [0, 1]; clipped after cast).
    FloatSpec("f_kp_fd_install_fd_int32", scale=1024.0),
    FloatSpec("f_kp_mmap_prot_uint",       scale=256.0),
    FloatSpec("f_kp_mprotect_prot_uint",   scale=256.0),
)

# --- 7 new Plan G model columns ---

PLAN_G_EMBED_COLUMNS: tuple[EmbedSpec, ...] = (
    EmbedSpec("f_action_family",                ACTION_FAMILY_CARDINALITY, 8),     # 15
    EmbedSpec("f_lineage_depth",                LINEAGE_DEPTH_CARD, 4),
    EmbedSpec("f_parent_child_pair_hash",       PARENT_CHILD_PAIR_HASH_CARD, 16),
    EmbedSpec("f_root_ancestor_basename_hash",  ROOT_ANCESTOR_BASENAME_HASH_CARD, 8),
    EmbedSpec("f_process_tree_id_hash",         PROCESS_TREE_ID_HASH_CARD, 8),
    EmbedSpec("f_delta_t_log_bucket",           DELTA_T_LOG_BUCKET_CARD, 4),
    EmbedSpec("f_process_age_log_bucket",       PROCESS_AGE_LOG_BUCKET_CARD, 4),
    # 3a' — promoted from side-only categories to model features.
    # Sized to design vocab (not observed-in-train counts) for OOV safety.
    EmbedSpec("f_path_category",                PATH_CATEGORY_CARDINALITY, 4),     # 11
    EmbedSpec("f_dst_ip_category",              DST_IP_CATEGORY_CARDINALITY, 4),   # 6
    EmbedSpec("f_dst_port_category",            DST_PORT_CATEGORY_CARDINALITY, 4), # 7
    EmbedSpec("f_object_category",              OBJECT_CATEGORY_CARDINALITY, 4),   # 8
)


ALL_EMBED_COLUMNS: tuple[EmbedSpec, ...] = EXISTING_EMBED_COLUMNS + PLAN_G_EMBED_COLUMNS
ALL_FLOAT_COLUMNS: tuple[FloatSpec, ...] = EXISTING_FLOAT_COLUMNS

# Total pre-projection dim: sum of embed dims + count of float columns.
RAW_INPUT_DIM: int = sum(s.dim for s in ALL_EMBED_COLUMNS) + len(ALL_FLOAT_COLUMNS)


# ---------------------------------------------------------------------------
# Flat feature variant — used by the §6 Mamba-flat encoder ablation. Drops
# the seven high-cardinality hash columns (matching the IF-flat baseline at
# `src/core/v0_2_baselines/`) so the §6 decomposition of Mamba's lift over
# IF-rich is apples-to-apples.
#
# Behavior is purely additive: callers that don't pass ``feature_set`` get
# the rich (42-feature) tuples and existing checkpoints remain valid.
# ---------------------------------------------------------------------------

EXCLUDED_FLAT_HASH_COLUMNS: tuple[str, ...] = (
    "f_proc_name_hash",
    "f_parent_proc_hash",
    "f_proc_cwd_hash",
    "f_lineage_bag_hash",
    "f_parent_child_pair_hash",
    "f_root_ancestor_basename_hash",
    "f_process_tree_id_hash",
)


def _drop_embed_columns(
    specs: tuple[EmbedSpec, ...],
    drop: tuple[str, ...],
) -> tuple[EmbedSpec, ...]:
    drop_set = set(drop)
    return tuple(s for s in specs if s.column not in drop_set)


EXISTING_EMBED_COLUMNS_FLAT: tuple[EmbedSpec, ...] = _drop_embed_columns(
    EXISTING_EMBED_COLUMNS, EXCLUDED_FLAT_HASH_COLUMNS
)
PLAN_G_EMBED_COLUMNS_FLAT: tuple[EmbedSpec, ...] = _drop_embed_columns(
    PLAN_G_EMBED_COLUMNS, EXCLUDED_FLAT_HASH_COLUMNS
)
ALL_EMBED_COLUMNS_FLAT: tuple[EmbedSpec, ...] = (
    EXISTING_EMBED_COLUMNS_FLAT + PLAN_G_EMBED_COLUMNS_FLAT
)
ALL_FLOAT_COLUMNS_FLAT: tuple[FloatSpec, ...] = ALL_FLOAT_COLUMNS  # no float cols dropped

RAW_INPUT_DIM_FLAT: int = (
    sum(s.dim for s in ALL_EMBED_COLUMNS_FLAT) + len(ALL_FLOAT_COLUMNS_FLAT)
)


FEATURE_SETS: tuple[str, ...] = ("rich", "flat")


def embed_specs_for(feature_set: str = "rich") -> tuple[EmbedSpec, ...]:
    """Return the embed-column spec tuple for ``feature_set``."""
    if feature_set == "rich":
        return ALL_EMBED_COLUMNS
    if feature_set == "flat":
        return ALL_EMBED_COLUMNS_FLAT
    raise ValueError(f"feature_set must be one of {FEATURE_SETS}, got {feature_set!r}")


def float_specs_for(feature_set: str = "rich") -> tuple[FloatSpec, ...]:
    """Return the float-column spec tuple for ``feature_set``."""
    if feature_set == "rich":
        return ALL_FLOAT_COLUMNS
    if feature_set == "flat":
        return ALL_FLOAT_COLUMNS_FLAT
    raise ValueError(f"feature_set must be one of {FEATURE_SETS}, got {feature_set!r}")


def raw_input_dim_for(feature_set: str = "rich") -> int:
    """Pre-projection concat dim for ``feature_set``."""
    embeds = embed_specs_for(feature_set)
    floats = float_specs_for(feature_set)
    return sum(s.dim for s in embeds) + len(floats)


def model_input_columns(feature_set: str = "rich") -> list[str]:
    """All parquet column names the encoder reads under ``feature_set``.

    The ``feature_set`` parameter is optional and defaults to ``"rich"``
    so existing callers (dataloader, audit notebooks) keep their previous
    behavior unchanged.
    """
    embeds = embed_specs_for(feature_set)
    floats = float_specs_for(feature_set)
    return [s.column for s in embeds] + [s.column for s in floats]


class V02EventEncoder(nn.Module):
    """Per-event embedding-table encoder for v0.2 features.

    Produces a (B, L, d_model) tensor from a dict of per-column tensors.
    Embedding columns expect int64 indices in ``[0, cardinality)``; float
    columns expect any numeric dtype (cast and divided by ``scale``).

    The optional ``feature_set`` argument selects the column inventory:
      * ``"rich"`` (default) — all 33 embed + 9 float columns (RAW_INPUT_DIM = 277)
      * ``"flat"``           — drops the 7 high-cardinality hash columns
                               (RAW_INPUT_DIM_FLAT = 149); used for the §6
                               Mamba-flat ablation paired with IF-flat.
    """

    def __init__(
        self,
        d_model: int = 128,
        *,
        feature_set: str = "rich",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.feature_set = feature_set
        self.embed_specs = embed_specs_for(feature_set)
        self.float_specs = float_specs_for(feature_set)
        raw_input_dim = raw_input_dim_for(feature_set)
        self.raw_input_dim = raw_input_dim

        self.embeddings = nn.ModuleDict({
            spec.column: nn.Embedding(spec.cardinality, spec.dim)
            for spec in self.embed_specs
        })
        self.proj = nn.Linear(raw_input_dim, d_model)

    def forward(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Encode (B, L) per-column tensors into a (B, L, d_model) sequence.

        Embedding columns must have dtype torch.long; float columns may be
        any numeric dtype.
        """
        parts: list[torch.Tensor] = []
        for spec in self.embed_specs:
            x = features[spec.column]
            if x.dtype != torch.long:
                x = x.long()
            parts.append(self.embeddings[spec.column](x))
        for spec in self.float_specs:
            x = features[spec.column]
            x = x.to(torch.float32)
            if spec.scale != 1.0:
                x = x / spec.scale
            parts.append(x.unsqueeze(-1))
        cat = torch.cat(parts, dim=-1)  # (B, L, RAW_INPUT_DIM)
        return self.proj(cat)
