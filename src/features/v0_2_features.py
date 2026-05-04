"""v0.2 feature encoders

Turns the human-readable raw parquet into fixed-width, all-int matrix
Function roles:
    - Encoder(encode_*): converts raw columns into feature column
        -outputs: small integers in defined ranges (per-tier)
    -internal helpers: shared math by all encoders )entropy/compression/hashing)
    -orchestration:
        - _build_feature_table: runs every encoder
        - build_features: walks the parquet tree

feature coverage:
* Tier 1 — categorical embeddings (event_type, kprobe_*, proc_uid_bucket).
        - vocab(string) lookup -> int (buckets)
* Tier 2 — bucketed scalars (dst_port, args_length, cap_count, path_sensitivity).
        - range-bucketed (numeric) -> int(thresholds)
* Tier 3 — hash buckets (proc_name, parent_proc, proc_cwd, lineage_bag_v1).
        - variable-string hashes -> fixed N buckets
* Tier 4 — boolean flags (in_init_tree, procfs_walk, uid_eq_parent_uid, ...).
        - the inherited true/false fields
* Tier 5 — derived continuous (cmdline entropy, gzip ratio, time_since_parent_exec).
        - entropy/compression/time ->> bucketed
* Tier 6 — kprobe-specific feature dicts (one per hooked function we feature).
        - specific to kprobe characteristics

The ``build_features`` orchestrator at the bottom wires the encoders together:
it reads partitioned parquet from ``input_path``, applies all 27 Tier 1-6
encoders per file, and writes a feature parquet under ``output_path`` whose
filename mirrors the input. Output schema is FIXED — see ``FEATURE_COLUMNS``.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Union

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.telemetry.tetragon_native_parser import KPROBE_FUNCTIONS

ArrowTabular = Union[pa.Table, pa.RecordBatch]


# ---------- vocab tables ----------

EVENT_TYPE_VOCAB: tuple[str, ...] = (
    "process_exec",
    "process_exit",
    "process_kprobe",
)
EVENT_TYPE_OOV_INDEX: int = len(EVENT_TYPE_VOCAB)  # 3

# KPROBE_FUNCTIONS is the parser's locked tuple of 12 hooked functions; reuse
# it directly so the encoder's vocab can never drift from the parser.
KPROBE_FUNCTION_VOCAB: tuple[str, ...] = KPROBE_FUNCTIONS
KPROBE_FUNCTION_NOT_KPROBE_INDEX: int = len(KPROBE_FUNCTION_VOCAB)  # 12
KPROBE_FUNCTION_OOV_INDEX: int = KPROBE_FUNCTION_NOT_KPROBE_INDEX + 1  # 13

#the 6 tetragon trace policies
KPROBE_POLICY_VOCAB: tuple[str, ...] = (
    "file-monitor",
    "memory-monitor",
    "privilege-monitor",
    "network-monitor",
    "injection-monitor",
    "attribute-monitor",
)
KPROBE_POLICY_NONE_INDEX: int = len(KPROBE_POLICY_VOCAB)  # 6
KPROBE_POLICY_OOV_INDEX: int = KPROBE_POLICY_NONE_INDEX + 1  # 7

# Tetragon kprobe action enum: only KPROBE_ACTION_POST appears in the
# v0.2 corpus (verified by full-corpus scan over data/processed/v0.2-split/).
# adds 1 to length for "OTHER/NONE"
KPROBE_ACTION_VOCAB: tuple[str, ...] = ("KPROBE_ACTION_POST",)
KPROBE_ACTION_NONE_INDEX: int = len(KPROBE_ACTION_VOCAB)  # 1
KPROBE_ACTION_OOV_INDEX: int = KPROBE_ACTION_NONE_INDEX + 1  # 2

# proc_uid bucket assignments. Indices form a closed set 0..4.
PROC_UID_BUCKET_ROOT: int = 0
PROC_UID_BUCKET_RYAN: int = 1
PROC_UID_BUCKET_SYSTEM_LOW: int = 2
PROC_UID_BUCKET_DAEMON_HIGH: int = 3
PROC_UID_BUCKET_OTHER: int = 4


# ---------- Tier 2 vocab / sentinel constants ----------

# dst_port bucket assignments — closed set 0..6.
# `loopback` slot is reserved for dport == 0 (Tetragon emits 0 for loopback /
# unset dport in some kprobe paths)
DST_PORT_BUCKET_NONE: int = 0
DST_PORT_BUCKET_LOOPBACK: int = 1
DST_PORT_BUCKET_SSH: int = 2
DST_PORT_BUCKET_DNS: int = 3
DST_PORT_BUCKET_WEB: int = 4
DST_PORT_BUCKET_PRIV: int = 5
DST_PORT_BUCKET_HIGH: int = 6
DST_PORT_WEB_SET: frozenset[int] = frozenset({80, 443, 8080, 8443})

# args_length bucket assignments — closed set 0..5.
ARGS_LENGTH_BUCKET_ZERO: int = 0
ARGS_LENGTH_BUCKET_LT50: int = 1
ARGS_LENGTH_BUCKET_50_200: int = 2
ARGS_LENGTH_BUCKET_200_1K: int = 3
ARGS_LENGTH_BUCKET_1K_5K: int = 4
ARGS_LENGTH_BUCKET_GT5K: int = 5

# cap_count bucket assignments — closed set 0..4.
CAP_COUNT_BUCKET_ZERO: int = 0
CAP_COUNT_BUCKET_1_2: int = 1
CAP_COUNT_BUCKET_3_5: int = 2
CAP_COUNT_BUCKET_6_15: int = 3
CAP_COUNT_BUCKET_16PLUS: int = 4

# path_sensitivity bucket assignments — closed set 0..7. Order MUST match
# the path_sensitivity table 
# # first-match wins: more-specific buckets come first so they win over generic ones
PATH_SENSITIVITY_SENSITIVE_SYS: int = 0
PATH_SENSITIVITY_STAGING: int = 1
PATH_SENSITIVITY_RUNTIME: int = 2
PATH_SENSITIVITY_USER_HOME: int = 3
PATH_SENSITIVITY_PROC_SELF: int = 4
PATH_SENSITIVITY_VSCODE_DEV: int = 5
PATH_SENSITIVITY_OTHER: int = 6
PATH_SENSITIVITY_NONE: int = 7

_PATH_SENSITIVE_SYS_PREFIXES: tuple[str, ...] = (
    "/etc/", "/root/", "/boot/", "/usr/sbin/", "/sbin/",
)
_PATH_STAGING_PREFIXES: tuple[str, ...] = (
    "/tmp/", "/var/tmp/", "/dev/shm/",
)
_PATH_RUNTIME_PREFIXES: tuple[str, ...] = (
    "/var/run/", "/run/",
)
_PATH_USER_HOME_PREFIXES: tuple[str, ...] = (
    "/home/", "/Users/",
)
# proc_self vs runtime: 
# proc_Self: the current process looking at its own runtime info
# runtime: one process is poking at a different process's runtime info
#   - cross memory inspection without ptrace
_PATH_PROC_SELF_PREFIX: str = "/proc/self/"
_PATH_PROC_NUMERIC_REGEX: str = r"^/proc/[0-9]+/"
_PATH_VSCODE_DEV_SUBSTRINGS: tuple[str, ...] = (
    ".vscode-server/", "node_modules/", ".git/",
)


# ---------- Tier 3 hash bucket sizes ----------

PROC_NAME_HASH_BUCKETS: int = 2048   # was 1024 (27.7% collision per audit cell 43)
PARENT_PROC_HASH_BUCKETS: int = 1024
PROC_CWD_HASH_BUCKETS: int = 4096    # was 256 (90.2% collision per audit cell 43)
LINEAGE_BAG_HASH_V1_BUCKETS: int = 256


# ---------- Tier 5 vocab / sentinel constants ----------

"""
shannon entropy of the argv characters
    - measures how spread out the character distribution is
    - "" (empty) -- entropy = 0.0 (no args)
    - "-c 'echo hello world'" -- entropy = 3.0-4.0
    - random/encrypted blob: entropy=6.0+

Compression: same pipeline as entropy but measures the gzip size of an argv string
    - divides compressed size by original size in _compression_ratio()

low ratio: <<1.0 --> string compressed a lot (lots of repetition/structure)
high ratio: 1.0+ ---> string barely compressed (random/dense)

example:
    - "-la" -- compression: ~3.0: tiny string, gzip header dominates
    - structured json: data='{\"k\":1,\"k\":2,\"k\":3,...}'" -- repeated keys, very compressable

Compression + entropy together:
    - 
Argv pattern	                    Entropy	        Compression	            What it tells you
"hello hello hello hello hello"	    LOW (~2.5)	    VERY (~0.4)	            repetitive low-info text
Long base64 payload	                HIGH (~5.8)	    WEAK (~0.75)	        encoded payload with structure
AES-encrypted blob (base64'd)	    HIGH (~5.9)	    INCOMPRESSIBLE (~0.95)	true random-looking data
"ls -la"	                           LOW (~2.3)	INCOMPRESSIBLE (~3.0)	normal short command (artifact of gzip overhead)
"""
# cmdline_entropy bucket assignments — closed set 0..5 (plus null sentinel 0).
# Indices: null=0, low<2.0, mid_low<3.5, mid<4.5, mid_high<5.5, high>=5.5.
CMDLINE_ENTROPY_NULL: int = 0 #null if a process has no arguments at all -- like ls
CMDLINE_ENTROPY_LOW: int = 1
CMDLINE_ENTROPY_MID_LOW: int = 2
CMDLINE_ENTROPY_MID: int = 3
CMDLINE_ENTROPY_MID_HIGH: int = 4
CMDLINE_ENTROPY_HIGH: int = 5

# cmdline_compression_ratio bucket assignments — closed set 0..5.
# Indices: null=0, very_compressible<0.3, compressible<0.5, moderate<0.7,
# weakly_compressible<0.9, incompressible>=0.9.
CMDLINE_COMPRESSION_NULL: int = 0
CMDLINE_COMPRESSION_VERY: int = 1
CMDLINE_COMPRESSION_COMPRESSIBLE: int = 2
CMDLINE_COMPRESSION_MODERATE: int = 3
CMDLINE_COMPRESSION_WEAK: int = 4
CMDLINE_COMPRESSION_INCOMPRESSIBLE: int = 5

# time_since_parent_exec bucket assignments — closed set 0..5.
TIME_SINCE_PARENT_NULL: int = 0
TIME_SINCE_PARENT_SUB_MS: int = 1  # < 1e6 ns
TIME_SINCE_PARENT_MS: int = 2  # 1e6 .. 1e9
TIME_SINCE_PARENT_SEC: int = 3  # 1e9 .. 60e9
TIME_SINCE_PARENT_MIN: int = 4  # 60e9 .. 3600e9
TIME_SINCE_PARENT_HR: int = 5  # >= 3600e9


# ---------- Tier 6 socket family vocab ----------

# kp_sock_family is emitted by Tetragon as the AF_* string. Empirically only
# AF_INET and AF_INET6 appear in the v0.2 corpus.
SOCK_FAMILY_VOCAB: tuple[str, ...] = ("AF_INET", "AF_INET6")
SOCK_FAMILY_NONE_INDEX: int = len(SOCK_FAMILY_VOCAB)  # 2
SOCK_FAMILY_OOV_INDEX: int = SOCK_FAMILY_NONE_INDEX + 1  # 3


# ---------- internal helpers ----------
def _column(table: ArrowTabular, name: str) -> pa.Array | pa.ChunkedArray:
    """Fetch a column from either a Table or a RecordBatch."""
    if isinstance(table, pa.RecordBatch):
        return table.column(name)
    return table.column(name)


def _index_with_sentinels(
    column: pa.Array | pa.ChunkedArray,
    vocab: tuple[str, ...],
    null_index: int,
    oov_index: int,
) -> pa.Array:
    """Cleaner two-pass vocab lookup that disambiguates null vs OOV.

    1. ``pc.index_in`` -> indices; null both for null-input and OOV-input.
    2. Replace the OOV-input case first (where input is non-null) with
       ``oov_index``.
    3. Replace the null-input case with ``null_index``.
    """
    value_set = pa.array(vocab, type=pa.string())
    raw = pc.index_in(column, value_set=value_set)  # int32, nullable
    is_null_input = pc.is_null(column)
    # OOV mask: input is non-null AND index is null.
    is_oov = pc.and_(pc.invert(is_null_input), pc.is_null(raw))
    null_scalar = pa.scalar(null_index, type=raw.type)
    oov_scalar = pa.scalar(oov_index, type=raw.type)
    # Apply OOV first, then null.
    after_oov = pc.if_else(is_oov, oov_scalar, raw)
    after_null = pc.if_else(is_null_input, null_scalar, after_oov)
    arr = pc.cast(after_null, pa.uint8())
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    return arr


# ---------- public encoders ----------


def encode_event_type(table: ArrowTabular) -> pa.Array:
    """Map ``event_type`` column to ``uint8`` indices.

    Vocabulary: process_exec=0, process_exit=1, process_kprobe=2.
    Null or unknown values both collapse to ``EVENT_TYPE_OOV_INDEX`` (3) —
    the parser guarantees non-null event_type for surviving rows, so a non-zero
    OOV count on real data signals a parser/encoder vocab drift.
    """
    column = _column(table, "event_type")
    return _index_with_sentinels(
        column,
        EVENT_TYPE_VOCAB,
        null_index=EVENT_TYPE_OOV_INDEX,
        oov_index=EVENT_TYPE_OOV_INDEX,
    )


def encode_kprobe_function_name(table: ArrowTabular) -> pa.Array:
    """Map ``kprobe_function_name`` column to ``uint8`` indices.

    Vocabulary: the 12 hooked functions in
    ``src.telemetry.tetragon_native_parser.KPROBE_FUNCTIONS`` get indices
    0..11 in their declared order. Null (i.e., the row is not a kprobe
    event) maps to ``KPROBE_FUNCTION_NOT_KPROBE_INDEX`` (12). Unknown
    function names map to ``KPROBE_FUNCTION_OOV_INDEX`` (13).
    """
    column = _column(table, "kprobe_function_name")
    return _index_with_sentinels(
        column,
        KPROBE_FUNCTION_VOCAB,
        null_index=KPROBE_FUNCTION_NOT_KPROBE_INDEX,
        oov_index=KPROBE_FUNCTION_OOV_INDEX,
    )


def encode_kprobe_policy_name(table: ArrowTabular) -> pa.Array:
    """Map ``kprobe_policy_name`` column to ``uint8`` indices.

    Vocabulary: the six monitor policies observed in V.1 get indices 0..5
    in declaration order. Null maps to ``KPROBE_POLICY_NONE_INDEX`` (6),
    unknown policy names to ``KPROBE_POLICY_OOV_INDEX`` (7).
    """
    column = _column(table, "kprobe_policy_name")
    return _index_with_sentinels(
        column,
        KPROBE_POLICY_VOCAB,
        null_index=KPROBE_POLICY_NONE_INDEX,
        oov_index=KPROBE_POLICY_OOV_INDEX,
    )


def encode_kprobe_action(table: ArrowTabular) -> pa.Array:
    """Map ``kprobe_action`` column to ``uint8`` indices.

    Vocabulary: only ``KPROBE_ACTION_POST`` (index 0) appears in the v0.2
    corpus. Null maps to ``KPROBE_ACTION_NONE_INDEX`` (1), any other
    Tetragon action enum (FOLLOWFD, SIGKILL, OVERRIDE, ...) maps to
    ``KPROBE_ACTION_OOV_INDEX`` (2).
    """
    column = _column(table, "kprobe_action")
    return _index_with_sentinels(
        column,
        KPROBE_ACTION_VOCAB,
        null_index=KPROBE_ACTION_NONE_INDEX,
        oov_index=KPROBE_ACTION_OOV_INDEX,
    )


def encode_proc_uid_bucket(table: ArrowTabular) -> pa.Array:
    """Bucket ``proc_uid`` into 5 identity classes.

    Buckets:
        * ``root`` (uid == 0) -> 0
        * ``ryan_admin`` (uid == 1000) -> 1
        * ``system_low`` (1 <= uid < 100) -> 2
        * ``daemon_high`` (100 <= uid < 1000) -> 3
        * ``other`` (uid >= 1001 OR null) -> 4
    """
    column = _column(table, "proc_uid")
    # Cast to a signed type wide enough to hold all uint32 values without
    # losing the null mask. Use int64 throughout the comparisons.
    uid64 = pc.cast(column, pa.int64())

    is_null = pc.is_null(uid64)
    is_root = pc.equal(uid64, pa.scalar(0, type=pa.int64()))
    is_ryan = pc.equal(uid64, pa.scalar(1000, type=pa.int64()))
    is_system_low = pc.and_(
        pc.greater_equal(uid64, pa.scalar(1, type=pa.int64())),
        pc.less(uid64, pa.scalar(100, type=pa.int64())),
    )
    is_daemon_high = pc.and_(
        pc.greater_equal(uid64, pa.scalar(100, type=pa.int64())),
        pc.less(uid64, pa.scalar(1000, type=pa.int64())),
    )

    # Layered if_else: outermost null check, then root/ryan equality,
    # then range buckets, default to "other". int8 throughout to ensure
    # if_else preserves type, then cast to uint8 at the end.
    other = pa.scalar(PROC_UID_BUCKET_OTHER, type=pa.int8())
    base = pc.if_else(
        is_daemon_high,
        pa.scalar(PROC_UID_BUCKET_DAEMON_HIGH, type=pa.int8()),
        other,
    )
    base = pc.if_else(
        is_system_low,
        pa.scalar(PROC_UID_BUCKET_SYSTEM_LOW, type=pa.int8()),
        base,
    )
    base = pc.if_else(
        is_ryan,
        pa.scalar(PROC_UID_BUCKET_RYAN, type=pa.int8()),
        base,
    )
    base = pc.if_else(
        is_root,
        pa.scalar(PROC_UID_BUCKET_ROOT, type=pa.int8()),
        base,
    )
    base = pc.if_else(
        is_null,
        pa.scalar(PROC_UID_BUCKET_OTHER, type=pa.int8()),
        base,
    )
    arr = pc.cast(base, pa.uint8())
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    return arr


# ---------- internal helpers (Tier 2+) ----------


def _to_array(maybe_chunked: pa.Array | pa.ChunkedArray) -> pa.Array:
    """Force a column to a contiguous Array (collapsing ChunkedArray)."""
    if isinstance(maybe_chunked, pa.ChunkedArray):
        return maybe_chunked.combine_chunks()
    return maybe_chunked


def _u8(arr: pa.Array | pa.ChunkedArray) -> pa.Array:
    """Cast to ``uint8`` and collapse any ChunkedArray."""
    out = pc.cast(arr, pa.uint8())
    return _to_array(out)


# ---------- Tier 2 encoders ----------


def encode_dst_port_bucket(table: ArrowTabular) -> pa.Array:
    """Bucket ``kp_sock_dport`` into 7 connection-class buckets.

    Buckets (per spec §5 Tier 2):

    * ``none`` (null) -> 0  — non-network event or kprobe with no dport
    * ``loopback`` (0) -> 1 — Tetragon emits 0 for loopback / unset
    * ``ssh`` (22) -> 2
    * ``dns`` (53) -> 3
    * ``web`` ({80, 443, 8080, 8443}) -> 4
    * ``priv`` (1..1023, excluding above) -> 5
    * ``high`` (>=1024) -> 6
    """
    column = _column(table, "kp_sock_dport")
    # Cast to int64 so that null mask is preserved through arithmetic.
    dport = pc.cast(column, pa.int64())

    is_null = pc.is_null(dport)
    is_loop = pc.equal(dport, pa.scalar(0, type=pa.int64()))
    is_ssh = pc.equal(dport, pa.scalar(22, type=pa.int64()))
    is_dns = pc.equal(dport, pa.scalar(53, type=pa.int64()))
    is_web = pc.is_in(
        dport,
        value_set=pa.array(sorted(DST_PORT_WEB_SET), type=pa.int64()),
    )
    is_priv = pc.and_(
        pc.greater(dport, pa.scalar(0, type=pa.int64())),
        pc.less(dport, pa.scalar(1024, type=pa.int64())),
    )

    # Layered if_else: order matters — first match wins.
    high = pa.scalar(DST_PORT_BUCKET_HIGH, type=pa.int8())
    base = pc.if_else(is_priv, pa.scalar(DST_PORT_BUCKET_PRIV, type=pa.int8()), high)
    base = pc.if_else(is_web, pa.scalar(DST_PORT_BUCKET_WEB, type=pa.int8()), base)
    base = pc.if_else(is_dns, pa.scalar(DST_PORT_BUCKET_DNS, type=pa.int8()), base)
    base = pc.if_else(is_ssh, pa.scalar(DST_PORT_BUCKET_SSH, type=pa.int8()), base)
    base = pc.if_else(
        is_loop, pa.scalar(DST_PORT_BUCKET_LOOPBACK, type=pa.int8()), base
    )
    base = pc.if_else(is_null, pa.scalar(DST_PORT_BUCKET_NONE, type=pa.int8()), base)
    return _u8(base)


def encode_args_length_bucket(table: ArrowTabular) -> pa.Array:
    """Bucket ``proc_arguments`` byte length into 6 buckets.

    Buckets (per spec §5 Tier 2):

    * 0 (null or empty) -> 0
    * < 50 chars -> 1
    * 50-200 -> 2
    * 200-1k -> 3
    * 1k-5k -> 4
    * > 5k -> 5
    """
    column = _column(table, "proc_arguments")
    length = pc.cast(pc.binary_length(column), pa.int64())

    is_null = pc.is_null(length)
    is_zero = pc.equal(length, pa.scalar(0, type=pa.int64()))
    is_lt50 = pc.less(length, pa.scalar(50, type=pa.int64()))
    is_lt200 = pc.less(length, pa.scalar(200, type=pa.int64()))
    is_lt1k = pc.less(length, pa.scalar(1_000, type=pa.int64()))
    is_lt5k = pc.less(length, pa.scalar(5_000, type=pa.int64()))

    gt5k = pa.scalar(ARGS_LENGTH_BUCKET_GT5K, type=pa.int8())
    base = pc.if_else(
        is_lt5k, pa.scalar(ARGS_LENGTH_BUCKET_1K_5K, type=pa.int8()), gt5k
    )
    base = pc.if_else(
        is_lt1k, pa.scalar(ARGS_LENGTH_BUCKET_200_1K, type=pa.int8()), base
    )
    base = pc.if_else(
        is_lt200, pa.scalar(ARGS_LENGTH_BUCKET_50_200, type=pa.int8()), base
    )
    base = pc.if_else(
        is_lt50, pa.scalar(ARGS_LENGTH_BUCKET_LT50, type=pa.int8()), base
    )
    base = pc.if_else(
        is_zero, pa.scalar(ARGS_LENGTH_BUCKET_ZERO, type=pa.int8()), base
    )
    base = pc.if_else(
        is_null, pa.scalar(ARGS_LENGTH_BUCKET_ZERO, type=pa.int8()), base
    )
    return _u8(base)


def encode_cap_count_bucket(table: ArrowTabular) -> pa.Array:
    """Bucket ``proc_cap_effective`` list length into 5 capability buckets.

    Buckets (per spec §5 Tier 2):

    * 0 (null or empty list) -> 0
    * 1-2 -> 1
    * 3-5 -> 2
    * 6-15 -> 3
    * 16+ -> 4
    """
    column = _column(table, "proc_cap_effective")
    # list_value_length returns null for null inputs.
    length = pc.cast(pc.list_value_length(column), pa.int64())

    is_null = pc.is_null(length)
    is_zero = pc.equal(length, pa.scalar(0, type=pa.int64()))
    is_le2 = pc.less_equal(length, pa.scalar(2, type=pa.int64()))
    is_le5 = pc.less_equal(length, pa.scalar(5, type=pa.int64()))
    is_le15 = pc.less_equal(length, pa.scalar(15, type=pa.int64()))

    gt15 = pa.scalar(CAP_COUNT_BUCKET_16PLUS, type=pa.int8())
    base = pc.if_else(is_le15, pa.scalar(CAP_COUNT_BUCKET_6_15, type=pa.int8()), gt15)
    base = pc.if_else(is_le5, pa.scalar(CAP_COUNT_BUCKET_3_5, type=pa.int8()), base)
    base = pc.if_else(is_le2, pa.scalar(CAP_COUNT_BUCKET_1_2, type=pa.int8()), base)
    base = pc.if_else(is_zero, pa.scalar(CAP_COUNT_BUCKET_ZERO, type=pa.int8()), base)
    base = pc.if_else(is_null, pa.scalar(CAP_COUNT_BUCKET_ZERO, type=pa.int8()), base)
    return _u8(base)


def encode_path_sensitivity(table: ArrowTabular, column_name: str) -> pa.Array:
    """Map a path string column to 8 sensitivity buckets.

    Bucket order (first match wins, per spec §5 Tier 2 path_sensitivity table):

    1. ``sensitive_sys`` — /etc/, /root/, /boot/, /usr/sbin/, /sbin/
    2. ``staging`` — /tmp/, /var/tmp/, /dev/shm/
    3. ``runtime`` — /var/run/, /run/, /proc/<numeric_pid>/ (non-self)
    4. ``user_home`` — /home/<user>/, /Users/<user>/
    5. ``proc_self`` — /proc/self/
    6. ``vscode_dev`` — substring: .vscode-server/, node_modules/, .git/
    7. ``other`` — everything else
    8. ``none`` — null/empty

    The proc_self vs runtime distinction is approximated by prefix only:
    /proc/self/ -> proc_self; /proc/<digits>/ -> runtime. We do not compare
    PIDs (no per-row context). For v0.2 this is sufficient.

    The encoder accepts ``column_name`` as a parameter so the same logic can
    apply to ``proc_cwd``, ``proc_binary``, ``kp_fd_install_path``, etc.
    """
    column = _column(table, column_name)
    arr = _to_array(column)
    # Defensive: if the column came from an untyped Python list of None and
    # ended up as null-type, cast to string so pc.starts_with kernels exist.
    if arr.type == pa.null():
        arr = pc.cast(arr, pa.string())

    is_null = pc.is_null(arr)
    # Empty string also collapses to "none" per spec.
    is_empty = pc.equal(arr, pa.scalar("", type=pa.string()))

    # Build masks in spec order. Each mask is null where input is null, so
    # we bypass that with `fill_null(False)` at the very end via if_else
    # using is_null as the gate.
    def _starts_with_any(prefixes: tuple[str, ...]) -> pa.Array:
        mask = pc.starts_with(arr, prefixes[0])
        for p in prefixes[1:]:
            mask = pc.or_(mask, pc.starts_with(arr, p))
        return mask

    def _contains_any(substrs: tuple[str, ...]) -> pa.Array:
        mask = pc.match_substring(arr, substrs[0])
        for s in substrs[1:]:
            mask = pc.or_(mask, pc.match_substring(arr, s))
        return mask

    is_sensitive_sys = _starts_with_any(_PATH_SENSITIVE_SYS_PREFIXES)
    is_staging = _starts_with_any(_PATH_STAGING_PREFIXES)
    # runtime: /var/run/, /run/, OR /proc/<numeric>/ (non-self).
    is_runtime_prefix = _starts_with_any(_PATH_RUNTIME_PREFIXES)
    is_proc_numeric = pc.match_substring_regex(arr, _PATH_PROC_NUMERIC_REGEX)
    is_proc_self = pc.starts_with(arr, _PATH_PROC_SELF_PREFIX)
    is_runtime_proc = pc.and_(is_proc_numeric, pc.invert(is_proc_self))
    is_runtime = pc.or_(is_runtime_prefix, is_runtime_proc)
    is_user_home = _starts_with_any(_PATH_USER_HOME_PREFIXES)
    is_vscode = _contains_any(_PATH_VSCODE_DEV_SUBSTRINGS)

    # Apply in REVERSE order so highest-priority overrides last (first-match wins).
    other = pa.scalar(PATH_SENSITIVITY_OTHER, type=pa.int8())
    base = pc.if_else(
        is_vscode, pa.scalar(PATH_SENSITIVITY_VSCODE_DEV, type=pa.int8()), other
    )
    base = pc.if_else(
        is_proc_self, pa.scalar(PATH_SENSITIVITY_PROC_SELF, type=pa.int8()), base
    )
    base = pc.if_else(
        is_user_home, pa.scalar(PATH_SENSITIVITY_USER_HOME, type=pa.int8()), base
    )
    base = pc.if_else(
        is_runtime, pa.scalar(PATH_SENSITIVITY_RUNTIME, type=pa.int8()), base
    )
    base = pc.if_else(
        is_staging, pa.scalar(PATH_SENSITIVITY_STAGING, type=pa.int8()), base
    )
    base = pc.if_else(
        is_sensitive_sys,
        pa.scalar(PATH_SENSITIVITY_SENSITIVE_SYS, type=pa.int8()),
        base,
    )
    # null + empty -> none (must fire after all others; gate is unconditional).
    none_scalar = pa.scalar(PATH_SENSITIVITY_NONE, type=pa.int8())
    base = pc.if_else(is_empty, none_scalar, base)
    base = pc.if_else(is_null, none_scalar, base)
    return _u8(base)


# ---------- Tier 3 hash encoders ----------


def _hash_to_bucket(value: str | None, buckets: int) -> int:
    """Stable hash of ``value`` modulo ``buckets``.

    Uses ``hashlib.blake2b(digest_size=8)`` so we don't pull in xxhash. The
    8-byte digest is interpreted as an unsigned big-endian integer; collisions
    over 1024 buckets are statistically uniform.
    """
    if not value:
        return 0
    h = hashlib.blake2b(value.encode("utf-8", errors="replace"), digest_size=8)
    return int.from_bytes(h.digest(), "big") % buckets


def _basename(path: str | None) -> str | None:
    if path is None:
        return None
    # Last '/'-segment; if no '/' the whole string is the basename.
    idx = path.rfind("/")
    if idx == -1:
        return path
    return path[idx + 1 :]


def _first_n_path_components(path: str | None, n: int) -> str | None:
    """Return the first ``n`` path components of ``path``.

    e.g., ``/home/ryan/foo/bar`` with n=2 returns ``/home/ryan``.
    A null or empty input returns None.
    """
    if not path:
        return None
    if not path.startswith("/"):
        # Relative path — return up to n components joined.
        parts = path.split("/", n)
        head = parts[:n]
        return "/".join(head)
    parts = path[1:].split("/", n)
    head = parts[:n]
    return "/" + "/".join(head)


def _vector_hash(values: list[str | None], buckets: int, dtype: pa.DataType) -> pa.Array:
    """Apply ``_hash_to_bucket`` over a Python list and wrap as ``pa.Array``."""
    out = [_hash_to_bucket(v, buckets) for v in values]
    return pa.array(out, type=dtype)


def encode_proc_name_hash(table: ArrowTabular) -> pa.Array:
    """Hash of ``proc_binary`` basename modulo PROC_NAME_HASH_BUCKETS.

    Returns a ``uint16`` array (capacity exceeds uint8). Null/empty
    binaries hash to bucket 0 (a normal hash collision; documented).
    Capacity is 2048 as of 3a' (was 1024; bumped per audit cell 43's
    27.7% collision rate).
    """
    column = _column(table, "proc_binary")
    values = [_basename(p) for p in _to_array(column).to_pylist()]
    return _vector_hash(values, PROC_NAME_HASH_BUCKETS, pa.uint16())


def encode_parent_proc_hash(table: ArrowTabular) -> pa.Array:
    """Hash of ``parent_binary`` basename modulo 1024.

    Returns a ``uint16`` array (1024 buckets exceeds uint8). Null parents
    hash to bucket 0.
    """
    column = _column(table, "parent_binary")
    values = [_basename(p) for p in _to_array(column).to_pylist()]
    return _vector_hash(values, PARENT_PROC_HASH_BUCKETS, pa.uint16())


def encode_proc_cwd_hash(table: ArrowTabular) -> pa.Array:
    """Hash of first 4 path components of ``proc_cwd`` modulo PROC_CWD_HASH_BUCKETS.

    e.g., ``/home/ryan/projects/foo/bar`` -> hash(``/home/ryan/projects/foo``).
    Depth=4 instead of depth=2 (3a' fix): depth-2 saturated at ~17 distinct
    buckets in this corpus (almost every cwd collapsed to ``/home/rypow``),
    making the 4096 capacity dead weight. Depth-4 captures per-project
    granularity (`/home/rypow/projects/dendroaspis` distinct from
    `/home/rypow/projects/other`), exercising more of the embedding table.
    Null/empty cwd hashes to bucket 0. Returns ``uint16``.
    """
    column = _column(table, "proc_cwd")
    values = [_first_n_path_components(p, 4) for p in _to_array(column).to_pylist()]
    return _vector_hash(values, PROC_CWD_HASH_BUCKETS, pa.uint16())


def encode_lineage_bag_hash_v1(table: ArrowTabular) -> pa.Array:
    """Parent-only lineage hash: hash of ``parent_binary`` basename mod 256.

    Per spec §5 Tier 3 v0.2.3 patch: v1 = parent only (available immediately,
    no lineage utility needed). v2 (full ancestor chain) is deferred.
    """
    column = _column(table, "parent_binary")
    values = [_basename(p) for p in _to_array(column).to_pylist()]
    return _vector_hash(values, LINEAGE_BAG_HASH_V1_BUCKETS, pa.uint8())


# ---------- Tier 4 boolean encoders ----------


def _bool_passthrough(column: pa.Array | pa.ChunkedArray) -> pa.Array:
    """Cast a bool column to uint8, mapping nulls to False (0)."""
    filled = pc.fill_null(column, False)
    return _u8(filled)


def encode_proc_in_init_tree(table: ArrowTabular) -> pa.Array:
    """Pass through ``proc_in_init_tree`` as ``uint8`` (null -> 0)."""
    return _bool_passthrough(_column(table, "proc_in_init_tree"))


def encode_proc_is_procfs_walk(table: ArrowTabular) -> pa.Array:
    """Pass through ``proc_is_procfs_walk`` as ``uint8`` (null -> 0)."""
    return _bool_passthrough(_column(table, "proc_is_procfs_walk"))


def encode_proc_uid_eq_parent_uid(table: ArrowTabular) -> pa.Array:
    """True iff ``proc_uid == parent_uid``; null on either side -> False."""
    proc_uid = pc.cast(_column(table, "proc_uid"), pa.int64())
    parent_uid = pc.cast(_column(table, "parent_uid"), pa.int64())
    eq = pc.equal(proc_uid, parent_uid)
    filled = pc.fill_null(eq, False)
    return _u8(filled)


def encode_is_setuid_exec(table: ArrowTabular) -> pa.Array:
    """Approximate setuid-exec detection.

    True iff ALL of:

    * ``proc_cap_effective`` is non-empty
    * ``parent_cap_effective`` is empty/null
    * ``event_type == "process_exec"``

    This is an approximation — true setuid detection requires comparing
    capability *sets*, not just presence. Documented per the task brief.
    """
    proc_caps = _column(table, "proc_cap_effective")
    parent_caps = _column(table, "parent_cap_effective")
    event_type = _column(table, "event_type")

    proc_len = pc.fill_null(pc.list_value_length(proc_caps), 0)
    parent_len = pc.fill_null(pc.list_value_length(parent_caps), 0)

    proc_has = pc.greater(proc_len, pa.scalar(0, type=pa.int32()))
    parent_empty = pc.equal(parent_len, pa.scalar(0, type=pa.int32()))
    is_exec = pc.equal(event_type, pa.scalar("process_exec", type=pa.string()))
    is_exec = pc.fill_null(is_exec, False)

    combined = pc.and_(proc_has, pc.and_(parent_empty, is_exec))
    return _u8(combined)


def encode_args_truncated(table: ArrowTabular) -> pa.Array:
    """Pass through ``exec_truncated_args`` if present, else all-False.

    NOTE: as of v0.2 schema (commit 2e3af7a), ``exec_truncated_args`` is NOT
    present in the parquet — the parser/writer never emits it. This encoder
    therefore returns an all-zero ``uint8`` array of length ``num_rows`` with
    a documented warning. If a future parser revision adds the column, the
    encoder will pass it through.
    """
    schema = table.schema
    n = table.num_rows
    if "exec_truncated_args" in schema.names:
        return _bool_passthrough(_column(table, "exec_truncated_args"))
    return pa.array([0] * n, type=pa.uint8())


# ---------- Tier 5 derived continuous encoders ----------


def _shannon_entropy(s: str) -> float:
    """Shannon entropy over characters of ``s`` (base 2). Empty -> 0.0."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _entropy_bucket(h: float | None) -> int:
    if h is None:
        return CMDLINE_ENTROPY_NULL
    if h < 2.0:
        return CMDLINE_ENTROPY_LOW
    if h < 3.5:
        return CMDLINE_ENTROPY_MID_LOW
    if h < 4.5:
        return CMDLINE_ENTROPY_MID
    if h < 5.5:
        return CMDLINE_ENTROPY_MID_HIGH
    return CMDLINE_ENTROPY_HIGH


def encode_cmdline_entropy(table: ArrowTabular) -> pa.Array:
    """Shannon entropy of ``proc_arguments`` characters, bucketed into 6.

    Per-row Python compute: pyarrow has no vectorized Shannon entropy. Empty
    or null arguments map to ``CMDLINE_ENTROPY_NULL`` (0).
    """
    column = _column(table, "proc_arguments")
    values = _to_array(column).to_pylist()
    out: list[int] = []
    for v in values:
        if not v:
            out.append(CMDLINE_ENTROPY_NULL)
            continue
        out.append(_entropy_bucket(_shannon_entropy(v)))
    return pa.array(out, type=pa.uint8())


def _compression_ratio(s: str) -> float:
    """gzip-compressed length divided by uncompressed length. Empty -> 1.0."""
    if not s:
        return 1.0
    raw = s.encode("utf-8", errors="replace")
    if not raw:
        return 1.0
    compressed = gzip.compress(raw)
    return len(compressed) / len(raw)


def _ratio_bucket(r: float | None) -> int:
    if r is None:
        return CMDLINE_COMPRESSION_NULL
    if r < 0.3:
        return CMDLINE_COMPRESSION_VERY
    if r < 0.5:
        return CMDLINE_COMPRESSION_COMPRESSIBLE
    if r < 0.7:
        return CMDLINE_COMPRESSION_MODERATE
    if r < 0.9:
        return CMDLINE_COMPRESSION_WEAK
    return CMDLINE_COMPRESSION_INCOMPRESSIBLE


def encode_cmdline_compression_ratio(table: ArrowTabular) -> pa.Array:
    """``len(gzip(args)) / len(args)`` bucketed into 6.

    Per-row Python compute: stdlib ``gzip.compress`` is not vectorizable.
    Empty / null arguments map to ``CMDLINE_COMPRESSION_NULL`` (0).
    """
    column = _column(table, "proc_arguments")
    values = _to_array(column).to_pylist()
    out: list[int] = []
    for v in values:
        if not v:
            out.append(CMDLINE_COMPRESSION_NULL)
            continue
        out.append(_ratio_bucket(_compression_ratio(v)))
    return pa.array(out, type=pa.uint8())


def encode_time_since_parent_exec(table: ArrowTabular) -> pa.Array:
    """Bucket ``event_time - parent_start_time`` (in ns) into 6 buckets.

    Buckets (per spec §5 Tier 5):

    * null sentinel (either timestamp missing) -> 0
    * sub-ms (< 1e6 ns) -> 1
    * ms (1e6 .. 1e9) -> 2
    * sec (1e9 .. 60e9) -> 3
    * min (60e9 .. 3600e9) -> 4
    * hr (>= 3600e9) -> 5

    The schema column is ``parent_start_time`` (not ``parent_proc_start_time``
    as the spec text mentions).
    """
    event_ns = pc.cast(_column(table, "event_time"), pa.int64())
    parent_ns = pc.cast(_column(table, "parent_start_time"), pa.int64())
    delta = pc.subtract(event_ns, parent_ns)

    is_null = pc.is_null(delta)
    is_sub_ms = pc.less(delta, pa.scalar(1_000_000, type=pa.int64()))
    is_ms = pc.less(delta, pa.scalar(1_000_000_000, type=pa.int64()))
    is_sec = pc.less(delta, pa.scalar(60_000_000_000, type=pa.int64()))
    is_min = pc.less(delta, pa.scalar(3_600_000_000_000, type=pa.int64()))

    hr = pa.scalar(TIME_SINCE_PARENT_HR, type=pa.int8())
    base = pc.if_else(is_min, pa.scalar(TIME_SINCE_PARENT_MIN, type=pa.int8()), hr)
    base = pc.if_else(is_sec, pa.scalar(TIME_SINCE_PARENT_SEC, type=pa.int8()), base)
    base = pc.if_else(is_ms, pa.scalar(TIME_SINCE_PARENT_MS, type=pa.int8()), base)
    base = pc.if_else(
        is_sub_ms, pa.scalar(TIME_SINCE_PARENT_SUB_MS, type=pa.int8()), base
    )
    base = pc.if_else(
        is_null, pa.scalar(TIME_SINCE_PARENT_NULL, type=pa.int8()), base
    )
    return _u8(base)


# ---------- Tier 6 kprobe-specific encoders ----------


def _encode_sock_family(column: pa.Array | pa.ChunkedArray) -> pa.Array:
    """Encode kp_sock_family AF_INET/AF_INET6 with sentinels."""
    return _index_with_sentinels(
        column,
        SOCK_FAMILY_VOCAB,
        null_index=SOCK_FAMILY_NONE_INDEX,
        oov_index=SOCK_FAMILY_OOV_INDEX,
    )


def encode_kp_fd_install_features(table: ArrowTabular) -> dict[str, pa.Array]:
    """Extract per-event features for the ``fd_install`` kprobe.

    Returns a dict with two arrays:

    * ``kp_fd_install_fd_int32`` — ``kp_fd_install_fd`` (int32, null -> -1)
    * ``kp_fd_install_path_sensitivity`` — path_sensitivity of the path arg

    On non-fd_install events the source columns are null, so the FD slot
    becomes -1 and the path slot becomes ``PATH_SENSITIVITY_NONE``.
    """
    fd_col = _column(table, "kp_fd_install_fd")
    fd_filled = pc.fill_null(fd_col, pa.scalar(-1, type=pa.int32()))
    fd_arr = pc.cast(fd_filled, pa.int32())
    return {
        "kp_fd_install_fd_int32": _to_array(fd_arr),
        "kp_fd_install_path_sensitivity": encode_path_sensitivity(
            table, "kp_fd_install_path"
        ),
    }


def encode_kp_security_mmap_file_features(
    table: ArrowTabular,
) -> dict[str, pa.Array]:
    """Extract per-event features for the ``security_mmap_file`` kprobe.

    Returns a dict with two arrays:

    * ``kp_mmap_path_sensitivity`` — path_sensitivity of the file path
    * ``kp_mmap_prot_uint`` — ``kp_mmap_prot`` (uint32, null -> 0)
    """
    prot_col = _column(table, "kp_mmap_prot")
    prot_filled = pc.fill_null(prot_col, pa.scalar(0, type=pa.uint32()))
    return {
        "kp_mmap_path_sensitivity": encode_path_sensitivity(table, "kp_mmap_path"),
        "kp_mmap_prot_uint": _to_array(pc.cast(prot_filled, pa.uint32())),
    }


def encode_kp_security_file_mprotect_features(
    table: ArrowTabular,
) -> dict[str, pa.Array]:
    """Extract per-event features for the ``security_file_mprotect`` kprobe.

    Returns a dict with one array:

    * ``kp_mprotect_prot_uint`` — ``kp_mprotect_prot`` (uint64, null -> 0).
      This is the T1055 RWX shellcode signal.

    Spec mentions ``kp_security_file_mprotect_size_arg_0`` but the parser's
    actual column name is ``kp_mprotect_prot`` (per parser §_kp_security_file_mprotect
    which writes the second positional arg as the prot value); we use the
    parser's name.
    """
    prot_col = _column(table, "kp_mprotect_prot")
    prot_filled = pc.fill_null(prot_col, pa.scalar(0, type=pa.uint64()))
    return {
        "kp_mprotect_prot_uint": _to_array(pc.cast(prot_filled, pa.uint64())),
    }


def encode_kp_commit_creds_features(table: ArrowTabular) -> dict[str, pa.Array]:
    """Extract per-event features for the ``commit_creds`` kprobe.

    Returns a dict with two booleans (cast to ``uint8``):

    * ``kp_commit_creds_uid_change`` — true iff ``kp_creds_uid != proc_uid``
      (when both are non-null). The spec hints at ``kp_creds_uid !=
      proc_creds_uid`` but those columns may not coexist on the same row;
      ``proc_uid`` is always populated for kprobe events on the source
      process so it's a more reliable signal.
    * ``kp_commit_creds_cap_change`` — true iff ``kp_creds_caps`` differs
      from ``proc_cap_effective`` (set inequality). Falls back to "non-null
      kp_creds_caps with different length" when full set comparison would be
      brittle (lists may differ only in order from the same effective set).
    """
    kp_creds_uid = pc.cast(_column(table, "kp_creds_uid"), pa.int64())
    proc_uid = pc.cast(_column(table, "proc_uid"), pa.int64())
    uid_change = pc.not_equal(kp_creds_uid, proc_uid)
    uid_change = pc.fill_null(uid_change, False)

    kp_caps_len = pc.fill_null(
        pc.list_value_length(_column(table, "kp_creds_caps")),
        pa.scalar(-1, type=pa.int32()),
    )
    proc_caps_len = pc.fill_null(
        pc.list_value_length(_column(table, "proc_cap_effective")),
        pa.scalar(-1, type=pa.int32()),
    )
    # Approximation: cap_change only meaningful when kp_creds_caps is non-null
    # (i.e., on a commit_creds event). When kp_creds_len == -1 (null), no
    # change. When both populated, length difference is the proxy for set
    # inequality.
    is_kp_present = pc.greater_equal(kp_caps_len, pa.scalar(0, type=pa.int32()))
    cap_change_raw = pc.not_equal(kp_caps_len, proc_caps_len)
    cap_change = pc.and_(is_kp_present, cap_change_raw)

    return {
        "kp_commit_creds_uid_change": _u8(uid_change),
        "kp_commit_creds_cap_change": _u8(cap_change),
    }


def encode_kp_tcp_connect_features(table: ArrowTabular) -> dict[str, pa.Array]:
    """Extract per-event features for the ``tcp_connect`` kprobe.

    Returns a dict with two arrays:

    * ``kp_tcp_connect_dst_port_bucket`` — dst_port_bucket of ``kp_sock_dport``
    * ``kp_tcp_connect_sock_family`` — sock_family of ``kp_sock_family``
      (AF_INET=0 / AF_INET6=1 / none=2 / oov=3)
    """
    return {
        "kp_tcp_connect_dst_port_bucket": encode_dst_port_bucket(table),
        "kp_tcp_connect_sock_family": _encode_sock_family(
            _column(table, "kp_sock_family")
        ),
    }


def encode_kp_udp_sendmsg_features(table: ArrowTabular) -> dict[str, pa.Array]:
    """Extract per-event features for the ``udp_sendmsg`` kprobe.

    Returns a dict with one boolean (cast to ``uint8``):

    * ``kp_udp_sendmsg_dport_eq_53`` — true iff ``kp_sock_dport == 53``
      (DNS query signal). Null dport -> False.
    """
    dport = pc.cast(_column(table, "kp_sock_dport"), pa.int64())
    is_53 = pc.equal(dport, pa.scalar(53, type=pa.int64()))
    is_53 = pc.fill_null(is_53, False)
    return {
        "kp_udp_sendmsg_dport_eq_53": _u8(is_53),
    }


# ---------- build_features orchestrator (Phase 1) ----------


# kp_*_path columns are coalesced (first non-null wins) into a single virtual
# path column before path_sensitivity is applied. Order matches the spec's
# kprobe-priority list in §7.1: file-write (fd_install) takes precedence over
# memory-mapping (mmap), then unlink, then chmod. Non-kprobe rows have all
# four null and the coalesced result is null -> PATH_SENSITIVITY_NONE.
_KP_PATH_COLUMNS: tuple[str, ...] = (
    "kp_fd_install_path",
    "kp_mmap_path",
    "kp_unlink_path",
    "kp_chmod_path",
)


# The fixed output schema for the feature parquet — column order is locked so
# downstream consumers (C4 IF baseline, C5 sequence dataset, ...) can assume it.
# 3 identifier columns + 33 feature columns = 36 columns.
FEATURE_COLUMNS: tuple[str, ...] = (
    # Identifiers
    "event_time",
    "proc_exec_id",
    "proc_pid",
    # Tier 1 (5)
    "f_event_type",
    "f_kprobe_function",
    "f_kprobe_policy",
    "f_kprobe_action",
    "f_proc_uid_bucket",
    # Tier 2 (6)
    "f_dst_port_bucket",
    "f_args_length_bucket",
    "f_cap_count_bucket",
    "f_path_sens_cwd",
    "f_path_sens_binary",
    "f_path_sens_kp",
    # Tier 3 (4)
    "f_proc_name_hash",
    "f_parent_proc_hash",
    "f_proc_cwd_hash",
    "f_lineage_bag_hash",
    # Tier 4 (5)
    "f_in_init_tree",
    "f_is_procfs_walk",
    "f_uid_eq_parent",
    "f_is_setuid_exec",
    "f_args_truncated",
    # Tier 5 (3)
    "f_cmdline_entropy",
    "f_cmdline_compress",
    "f_time_since_parent_exec",
    # Tier 6 (10): two from fd_install, two from mmap, one from mprotect,
    # two from commit_creds, two from tcp_connect, one from udp_sendmsg.
    "f_kp_fd_install_fd_int32",
    "f_kp_fd_install_path_sensitivity",
    "f_kp_mmap_path_sensitivity",
    "f_kp_mmap_prot_uint",
    "f_kp_mprotect_prot_uint",
    "f_kp_commit_creds_uid_change",
    "f_kp_commit_creds_cap_change",
    "f_kp_tcp_connect_dst_port_bucket",
    "f_kp_tcp_connect_sock_family",
    "f_kp_udp_sendmsg_dport_eq_53",
)


def _dirname(path: str | None) -> str | None:
    """Return the parent directory of ``path`` (with trailing slash).

    Mirrors POSIX ``dirname`` behavior closely enough for path_sensitivity:
    ``/usr/sbin/cron`` -> ``/usr/sbin/``. The trailing slash is intentional so
    the resulting string still matches the prefix kernels (``/usr/sbin/``).
    """
    if not path:
        return None
    idx = path.rfind("/")
    if idx == -1:
        return path
    if idx == 0:
        # Path is "/foo" -> dirname "/".
        return "/"
    return path[: idx + 1]


def _coalesce_kp_path_column(table: ArrowTabular) -> pa.Array:
    """Coalesce the four kp_*_path columns (first non-null wins).

    Returns a string ``pa.Array`` whose value at row i is the first non-null
    among ``kp_fd_install_path``, ``kp_mmap_path``, ``kp_unlink_path``,
    ``kp_chmod_path`` — or null if all four are null. Vectorized via
    ``pc.coalesce``.
    """
    cols = [_to_array(_column(table, name)) for name in _KP_PATH_COLUMNS]
    coalesced = pc.coalesce(*cols)
    return _to_array(coalesced)


def _build_feature_table(table: pa.Table) -> pa.Table:
    """Apply all Tier 1-6 encoders + identifier passthrough to a single table.

    Returns a new ``pa.Table`` with exactly the columns in ``FEATURE_COLUMNS``
    and the same row count + row order as the input.
    """
    n = table.num_rows

    # Identifier columns — passthrough exact dtypes from the v0.2 split schema:
    # event_time: timestamp[ns, tz=UTC]; proc_exec_id: string; proc_pid: uint32.
    event_time = _to_array(_column(table, "event_time"))
    proc_exec_id = _to_array(_column(table, "proc_exec_id"))
    proc_pid = _to_array(_column(table, "proc_pid"))

    # Tier 1
    f_event_type = encode_event_type(table)
    f_kprobe_function = encode_kprobe_function_name(table)
    f_kprobe_policy = encode_kprobe_policy_name(table)
    f_kprobe_action = encode_kprobe_action(table)
    f_proc_uid_bucket = encode_proc_uid_bucket(table)

    # Tier 2
    f_dst_port_bucket = encode_dst_port_bucket(table)
    f_args_length_bucket = encode_args_length_bucket(table)
    f_cap_count_bucket = encode_cap_count_bucket(table)
    f_path_sens_cwd = encode_path_sensitivity(table, "proc_cwd")

    # f_path_sens_binary: dirname of proc_binary, then path_sensitivity.
    # Per-row Python (no vectorized dirname kernel); ~17M rows still fits the
    # 30-min budget per the smoke estimate (0.013 ms/row encoder cost).
    bin_dirnames = [_dirname(p) for p in _to_array(_column(table, "proc_binary")).to_pylist()]
    bin_dirname_table = pa.table({"proc_binary_dirname": pa.array(bin_dirnames, type=pa.string())})
    f_path_sens_binary = encode_path_sensitivity(bin_dirname_table, "proc_binary_dirname")

    # f_path_sens_kp: coalesce 4 kp_*_path columns then apply path_sensitivity.
    coalesced_kp_path = _coalesce_kp_path_column(table)
    coalesced_kp_table = pa.table({"_kp_path_coalesced": coalesced_kp_path})
    f_path_sens_kp = encode_path_sensitivity(coalesced_kp_table, "_kp_path_coalesced")

    # Tier 3
    f_proc_name_hash = encode_proc_name_hash(table)
    f_parent_proc_hash = encode_parent_proc_hash(table)
    f_proc_cwd_hash = encode_proc_cwd_hash(table)
    f_lineage_bag_hash = encode_lineage_bag_hash_v1(table)

    # Tier 4
    f_in_init_tree = encode_proc_in_init_tree(table)
    f_is_procfs_walk = encode_proc_is_procfs_walk(table)
    f_uid_eq_parent = encode_proc_uid_eq_parent_uid(table)
    f_is_setuid_exec = encode_is_setuid_exec(table)
    f_args_truncated = encode_args_truncated(table)

    # Tier 5
    f_cmdline_entropy = encode_cmdline_entropy(table)
    f_cmdline_compress = encode_cmdline_compression_ratio(table)
    f_time_since_parent_exec = encode_time_since_parent_exec(table)

    # Tier 6 (each returns a dict; flatten with the f_kp_* prefix).
    fd_dict = encode_kp_fd_install_features(table)
    mmap_dict = encode_kp_security_mmap_file_features(table)
    mprot_dict = encode_kp_security_file_mprotect_features(table)
    creds_dict = encode_kp_commit_creds_features(table)
    tcp_dict = encode_kp_tcp_connect_features(table)
    udp_dict = encode_kp_udp_sendmsg_features(table)

    columns: dict[str, pa.Array] = {
        "event_time": event_time,
        "proc_exec_id": proc_exec_id,
        "proc_pid": proc_pid,
        "f_event_type": f_event_type,
        "f_kprobe_function": f_kprobe_function,
        "f_kprobe_policy": f_kprobe_policy,
        "f_kprobe_action": f_kprobe_action,
        "f_proc_uid_bucket": f_proc_uid_bucket,
        "f_dst_port_bucket": f_dst_port_bucket,
        "f_args_length_bucket": f_args_length_bucket,
        "f_cap_count_bucket": f_cap_count_bucket,
        "f_path_sens_cwd": f_path_sens_cwd,
        "f_path_sens_binary": f_path_sens_binary,
        "f_path_sens_kp": f_path_sens_kp,
        "f_proc_name_hash": f_proc_name_hash,
        "f_parent_proc_hash": f_parent_proc_hash,
        "f_proc_cwd_hash": f_proc_cwd_hash,
        "f_lineage_bag_hash": f_lineage_bag_hash,
        "f_in_init_tree": f_in_init_tree,
        "f_is_procfs_walk": f_is_procfs_walk,
        "f_uid_eq_parent": f_uid_eq_parent,
        "f_is_setuid_exec": f_is_setuid_exec,
        "f_args_truncated": f_args_truncated,
        "f_cmdline_entropy": f_cmdline_entropy,
        "f_cmdline_compress": f_cmdline_compress,
        "f_time_since_parent_exec": f_time_since_parent_exec,
        "f_kp_fd_install_fd_int32": fd_dict["kp_fd_install_fd_int32"],
        "f_kp_fd_install_path_sensitivity": fd_dict["kp_fd_install_path_sensitivity"],
        "f_kp_mmap_path_sensitivity": mmap_dict["kp_mmap_path_sensitivity"],
        "f_kp_mmap_prot_uint": mmap_dict["kp_mmap_prot_uint"],
        "f_kp_mprotect_prot_uint": mprot_dict["kp_mprotect_prot_uint"],
        "f_kp_commit_creds_uid_change": creds_dict["kp_commit_creds_uid_change"],
        "f_kp_commit_creds_cap_change": creds_dict["kp_commit_creds_cap_change"],
        "f_kp_tcp_connect_dst_port_bucket": tcp_dict["kp_tcp_connect_dst_port_bucket"],
        "f_kp_tcp_connect_sock_family": tcp_dict["kp_tcp_connect_sock_family"],
        "f_kp_udp_sendmsg_dport_eq_53": udp_dict["kp_udp_sendmsg_dport_eq_53"],
    }

    # Defensive: every produced array must be exactly n rows; mismatched
    # encoders are a serious bug.
    for name, arr in columns.items():
        if len(arr) != n:
            raise RuntimeError(
                f"feature column {name!r} has {len(arr)} rows, expected {n}"
            )

    # Build table in FEATURE_COLUMNS order.
    return pa.table({name: columns[name] for name in FEATURE_COLUMNS})


def build_features(
    input_path: Path,
    output_path: Path,
) -> dict[str, int]:
    """Apply all Tier 1-6 encoders to every parquet under ``input_path``.

    Mirrors the input partition structure under ``output_path``::

        input_path/<src>.parquet -> output_path/<src>.parquet

    Each output row corresponds 1:1 to an input row (same row order). The
    output schema is fixed by ``FEATURE_COLUMNS``: 3 identifier columns
    (event_time, proc_exec_id, proc_pid) + 33 feature columns. Raw parser
    fields like ``proc_arguments`` and ``kp_*_path`` are NOT passed through;
    they are inputs to the encoders only.

    Per-event ``is_attack`` labels are NOT computed here — the IF baseline
    script joins to ``labels.csv`` separately on (event_time, proc_exec_id).

    Args:
        input_path: directory containing one or more ``*.parquet`` files
            produced by the v0.2 splitter.
        output_path: directory to write feature parquet to. Created if it
            does not exist. Filenames mirror the input file basenames.

    Returns:
        Per-file row counts plus a ``'total'`` key. Example::

            {"events-2026-04-14T03-53-05.161.parquet": 12345, ..., "total": 9876543}
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"input_path does not exist: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"input_path is not a directory: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    total = 0
    # Recurse to handle the date-partitioned tree the splitter writes
    # (partition=train/year=2026/month=04/day=NN/events-X.parquet). Preserve
    # the relative path in output to avoid basename-collision overwrites
    # (same root cause as splitter fix d76cd54).
    files = sorted(p for p in input_path.rglob("*.parquet"))
    for src in files:
        rel = src.relative_to(input_path)
        # ParquetFile.read() bypasses pyarrow's ParquetDataset auto-detection
        # which would otherwise see the `partition=train/year=YYYY/...` path
        # and try to merge schemas of nearby files (failing on cross-file
        # `year` type variance).
        table = pq.ParquetFile(str(src)).read()
        feature_table = _build_feature_table(table)
        dst = output_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(feature_table, dst, compression="snappy")
        counts[str(rel)] = feature_table.num_rows
        total += feature_table.num_rows
    counts["total"] = total
    return counts


# ---------- CLI ----------


def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.features.v0_2_features",
        description="v0.2 feature builder (Phase 1): apply Tier 1-6 encoders to "
        "v0.2-split parquet and write feature parquet.",
    )
    p.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Directory of v0.2-split parquet files.",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory to write feature parquet to. Mirrors input filenames.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    t0 = time.perf_counter()
    counts = build_features(args.input, args.output)
    elapsed = time.perf_counter() - t0
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "wall_time_seconds": round(elapsed, 2),
        "total_rows": counts.get("total", 0),
        "files": {k: v for k, v in counts.items() if k != "total"},
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
