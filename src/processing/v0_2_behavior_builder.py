"""v0.2 behavior builder.

Reads a raw normalized parquet (output of TetragonNativeWriter) and
augments it with model + analysis features per Plan G:

  - 7 model columns: f_action_family, f_lineage_depth,
    f_parent_child_pair_hash, f_root_ancestor_basename_hash,
    f_process_tree_id_hash, f_delta_t_log_bucket, f_process_age_log_bucket
  - 14 side columns (strings / bools / raw int64) for analysis + debug:
    proc_binary_basename, parent_binary_basename, proc_cwd_sanitized,
    root_ancestor_basename, process_tree_root_exec_id, parent_child_pair,
    path_category, dst_ip_category, dst_port_category, object_category,
    lineage_parent_fallback_used, lineage_missing_parent,
    lineage_cycle_detected, token, delta_t_prev_ns, process_age_ns
  - The existing 33 f_* features via _build_feature_table

Lineage walks the (proc_exec_id -> proc_parent_exec_id) graph globally,
which is why this is a separate stage from the parser (the parser only
sees one file at a time and cannot resolve parents that lived in earlier
files).

Tokens use categories only — never raw IPs or paths. See
~/.claude/plans/v0.2-final-feature-design.md for the full design.
"""

from __future__ import annotations

import math
import os
import re
import time
from ipaddress import ip_address
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.features.v0_2_features import _build_feature_table, _hash_to_bucket


# ---------- enums ----------

ACTION_FAMILY_VOCAB: tuple[str, ...] = (
    "PROC_EXEC",
    "PROC_EXIT",
    "NET_CONNECT",
    "NET_CLOSE",
    "NET_ACCEPT",
    "NET_SEND",
    "FILE_OPEN",
    "FILE_DELETE",
    "FILE_CHMOD",
    "PRIV_CHANGE",
    "PROC_INJECT_PTRACE",
    "PROC_INJECT_VMWRITE",
    "MEM_MAP",
    "MEM_PROTECT",
)
ACTION_FAMILY_OOV: str = "OTHER"
ACTION_FAMILY_OOV_INDEX: int = len(ACTION_FAMILY_VOCAB)  # 14

# Total integer cardinality of f_action_family is len(VOCAB) + 1 to leave
# room for the OOV row at index ACTION_FAMILY_OOV_INDEX (= 14). The Plan E
# encoder MUST size its embedding as Embedding(ACTION_FAMILY_CARDINALITY, ...)
# = Embedding(15, ...). Sizing it to 14 will produce IndexError on any
# unrecognized event_type / kprobe_function combination.
ACTION_FAMILY_CARDINALITY: int = len(ACTION_FAMILY_VOCAB) + 1  # 15

_ACTION_FAMILY_INDEX: dict[str, int] = {name: i for i, name in enumerate(ACTION_FAMILY_VOCAB)}

_KPROBE_TO_ACTION: dict[str, str] = {
    "tcp_connect": "NET_CONNECT",
    "tcp_close": "NET_CLOSE",
    "inet_csk_accept": "NET_ACCEPT",
    "udp_sendmsg": "NET_SEND",
    "fd_install": "FILE_OPEN",
    "do_unlinkat": "FILE_DELETE",
    "chmod_common": "FILE_CHMOD",
    "commit_creds": "PRIV_CHANGE",
    "sys_ptrace": "PROC_INJECT_PTRACE",
    "sys_process_vm_writev": "PROC_INJECT_VMWRITE",
    "security_mmap_file": "MEM_MAP",
    "security_file_mprotect": "MEM_PROTECT",
}

_KPROBE_TO_OBJECT: dict[str, str] = {
    "tcp_connect": "NETWORK",
    "tcp_close": "NETWORK",
    "inet_csk_accept": "NETWORK",
    "udp_sendmsg": "NETWORK",
    "fd_install": "FILE",
    "do_unlinkat": "FILE",
    "chmod_common": "FILE",
    "commit_creds": "CREDENTIAL",
    "sys_ptrace": "INJECT",
    "sys_process_vm_writev": "INJECT",
    "security_mmap_file": "MEMORY",
    "security_file_mprotect": "MEMORY",
}

# 3a' — promoted-to-model-feature categorical vocabularies. Ordered enums
# include every value the corresponding ``derive_*`` function can produce,
# regardless of whether the value was observed in any specific corpus.
# Sized to the design vocab, not the observed-in-train counts, so the
# encoder embedding table does not crash on a value the test (or future)
# corpus exercises that train didn't.

PATH_CATEGORY_VOCAB: tuple[str, ...] = (
    "VSCODE_DEV", "SENSITIVE_SYS", "TEMP_STAGING", "RUNTIME",
    "LIBRARY", "USR_BIN", "OPT",
    "USER_HOME", "PROC_SELF", "OTHER", "NONE",
)
DST_IP_CATEGORY_VOCAB: tuple[str, ...] = (
    "LOOPBACK", "LINK_LOCAL", "PRIVATE", "MULTICAST", "PUBLIC", "NONE",
)
DST_PORT_CATEGORY_VOCAB: tuple[str, ...] = (
    "DNS", "WEB", "SSH", "WELL_KNOWN", "REGISTERED", "HIGH", "NONE",
)
OBJECT_CATEGORY_VOCAB: tuple[str, ...] = (
    "PROCESS", "FILE", "NETWORK", "MEMORY", "CREDENTIAL", "INJECT",
    "OTHER", "NONE",
)

PATH_CATEGORY_CARDINALITY: int = len(PATH_CATEGORY_VOCAB)        # 11
DST_IP_CATEGORY_CARDINALITY: int = len(DST_IP_CATEGORY_VOCAB)    # 6
DST_PORT_CATEGORY_CARDINALITY: int = len(DST_PORT_CATEGORY_VOCAB)  # 7
OBJECT_CATEGORY_CARDINALITY: int = len(OBJECT_CATEGORY_VOCAB)    # 8

_PATH_CATEGORY_INDEX: dict[str, int] = {v: i for i, v in enumerate(PATH_CATEGORY_VOCAB)}
_DST_IP_CATEGORY_INDEX: dict[str, int] = {v: i for i, v in enumerate(DST_IP_CATEGORY_VOCAB)}
_DST_PORT_CATEGORY_INDEX: dict[str, int] = {v: i for i, v in enumerate(DST_PORT_CATEGORY_VOCAB)}
_OBJECT_CATEGORY_INDEX: dict[str, int] = {v: i for i, v in enumerate(OBJECT_CATEGORY_VOCAB)}

# OOV indices for any value the deriver might emit that's not in the vocab
# (defensive; should not happen given the deriver functions are closed).
_PATH_CATEGORY_OOV: int = _PATH_CATEGORY_INDEX["OTHER"]
_DST_IP_CATEGORY_OOV: int = _DST_IP_CATEGORY_INDEX["NONE"]
_DST_PORT_CATEGORY_OOV: int = _DST_PORT_CATEGORY_INDEX["NONE"]
_OBJECT_CATEGORY_OOV: int = _OBJECT_CATEGORY_INDEX["OTHER"]

# Path category prefix table. Order matters: more-specific buckets come
# first. VSCODE_DEV must precede USER_HOME because vscode-server paths live
# under /home/. PROC_SELF is special-cased before the loop because
# /proc/self/ is a substring of the RUNTIME prefix /proc/. SENSITIVE_SYS
# precedes LIBRARY/USR_BIN because /usr/sbin/ is a more sensitive bucket
# than the generic /usr/lib or /usr/bin and we want it called out
# distinctly. LIBRARY / USR_BIN / OPT split out the previously-OTHER
# bucket which audit cell 29 showed to be 30.7% of train events.
_PATH_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("VSCODE_DEV", (".vscode-server/", "node_modules/", ".git/")),
    ("SENSITIVE_SYS", ("/etc/", "/root/", "/boot/", "/usr/sbin/", "/sbin/")),
    ("TEMP_STAGING", ("/tmp/", "/var/tmp/", "/dev/shm/")),
    ("RUNTIME", ("/var/run/", "/run/", "/proc/")),
    ("LIBRARY", ("/usr/lib/", "/usr/lib32/", "/usr/lib64/", "/usr/local/lib/", "/lib/", "/lib32/", "/lib64/")),
    ("USR_BIN", ("/usr/bin/", "/usr/local/bin/", "/usr/local/sbin/")),
    ("OPT", ("/opt/",)),
    ("USER_HOME", ("/home/", "/Users/")),
)


# ---------- category derivers ----------

def derive_action_family(event_type: str | None, kprobe_function: str | None) -> str:
    """Map (event_type, kprobe_function_name) to an action family token."""
    if event_type == "process_exec":
        return "PROC_EXEC"
    if event_type == "process_exit":
        return "PROC_EXIT"
    if event_type == "process_kprobe":
        return _KPROBE_TO_ACTION.get(kprobe_function or "", ACTION_FAMILY_OOV)
    return ACTION_FAMILY_OOV


def derive_object_category(event_type: str | None, kprobe_function: str | None) -> str:
    """Map an event to its high-level object category."""
    if event_type in ("process_exec", "process_exit"):
        return "PROCESS"
    if event_type == "process_kprobe":
        return _KPROBE_TO_OBJECT.get(kprobe_function or "", "OTHER")
    if event_type is None:
        return "NONE"
    return "OTHER"


def derive_path_category(path: str | None) -> str:
    """Bucket a filesystem path into a coarse category. NONE for null/empty."""
    if not path:
        return "NONE"
    if "/proc/self/" in path:
        return "PROC_SELF"
    for label, prefixes in _PATH_BUCKETS:
        if any(p in path for p in prefixes):
            return label
    return "OTHER"


def derive_dst_ip_category(addr: str | None) -> str:
    """Bucket an IP address (v4 or v6) into a coarse category."""
    if not addr:
        return "NONE"
    try:
        ip = ip_address(addr.strip())
    except ValueError:
        return "NONE"
    if ip.is_loopback:
        return "LOOPBACK"
    # is_link_local must precede is_private because some link-local
    # ranges overlap private ranges in older Python versions.
    if ip.is_link_local:
        return "LINK_LOCAL"
    if ip.is_private:
        return "PRIVATE"
    if ip.is_multicast:
        return "MULTICAST"
    return "PUBLIC"


def derive_dst_port_category(port: int | None) -> str:
    """Bucket a destination port number into a coarse category."""
    if port is None:
        return "NONE"
    if port == 53:
        return "DNS"
    if port in (80, 443):
        return "WEB"
    if port == 22:
        return "SSH"
    if port < 1024:
        return "WELL_KNOWN"
    if port < 49152:
        return "REGISTERED"
    return "HIGH"


# ---------- token formatter ----------

_RAW_IP_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
_RAW_PATH_HINT_RE = re.compile(r"/(?:etc|tmp|var|root|home|proc|usr|run|dev|boot)\b")


def format_token(
    action_family: str,
    proc_basename: str,
    parent_basename: str,
    path_category: str,
    dst_ip_category: str,
    dst_port_category: str,
) -> str:
    """Format an event token using categories only — never raw IPs or paths.

    Format rules:
        PROC_EXEC -> PROC_EXEC:{parent_basename}->{proc_basename}
        PROC_EXIT -> PROC_EXIT:{proc_basename}
        NET_*     -> {action}:{proc_basename}:{dst_ip_category}:{dst_port_category}
        FILE_*    -> {action}:{proc_basename}:{path_category}
        otherwise -> {action}:{proc_basename}
    """
    proc = proc_basename or "unknown"
    parent = parent_basename or "unknown"
    if action_family == "PROC_EXEC":
        token = f"PROC_EXEC:{parent}->{proc}"
    elif action_family == "PROC_EXIT":
        token = f"PROC_EXIT:{proc}"
    elif action_family.startswith("NET_"):
        token = (
            f"{action_family}:{proc}:"
            f"{dst_ip_category or 'NONE'}:{dst_port_category or 'NONE'}"
        )
    elif action_family.startswith("FILE_"):
        token = f"{action_family}:{proc}:{path_category or 'NONE'}"
    else:
        token = f"{action_family}:{proc}"
    return token


def token_contains_raw_artifact(token: str) -> bool:
    """Return True iff the token leaks a raw IPv4 or a raw filesystem path."""
    if not token:
        return False
    return bool(_RAW_IP_RE.search(token) or _RAW_PATH_HINT_RE.search(token))


# ---------- lineage walker ----------

class LineageWalker:
    """Resolves per-event lineage features via a global (exec_id -> parent)
    map.

    Parent precedence at registration time: proc_parent_exec_id first,
    then parent_exec_id as fallback. Two distinct debug bools come out of
    this:

      lineage_parent_fallback_used: proc_parent_exec_id was null and the
        parent_exec_id fallback was used (regardless of whether the
        resolved parent ended up in the process map).
      lineage_missing_parent: no usable parent at all (both null) OR the
        resolved parent exec_id is not present in the process map
        (orphaned link).

    Cycle detection uses a per-walk visited set; tripping it sets
    lineage_cycle_detected on the affected event.
    """

    MAX_DEPTH: int = 64  # safety cap; real lineage rarely exceeds 15

    def __init__(self) -> None:
        # exec_id -> (resolved_parent_exec_id_or_None, basename)
        self.process_map: dict[str, tuple[str | None, str]] = {}
        # exec_id -> (depth, root_exec_id, root_basename, cycle_detected)
        self._cache: dict[str, tuple[int, str | None, str, bool]] = {}
        # 3a' diagnostic: counts events where the same exec_id was seen
        # twice. proc_exec_id is supposed to be globally unique (Tetragon
        # hashes pid + ns + time), so a non-zero value either means a hash
        # collision or that a process emitted multiple process_exec rows
        # (e.g., a procfs-walk re-emission across an LXC restart). Either
        # way, first-write-wins semantics apply.
        self.duplicate_exec_ids: int = 0

    def add_process(
        self,
        exec_id: str | None,
        proc_parent_exec_id: str | None,
        parent_exec_id: str | None,
        proc_binary: str | None,
    ) -> None:
        if not exec_id:
            return
        resolved_parent = proc_parent_exec_id or parent_exec_id
        basename = os.path.basename(proc_binary) if proc_binary else ""
        if exec_id in self.process_map:
            # add_process is called for every event row (process_exec,
            # process_kprobe, process_exit), so most calls hit an existing
            # entry; that is normal repetition of the same process across
            # its lifetime. Only count *content-mismatched* duplicates —
            # same exec_id mapped to a different (parent, basename) pair.
            # Those would indicate a genuine collision or upstream
            # corruption.
            existing_parent, existing_basename = self.process_map[exec_id]
            if existing_parent != resolved_parent or existing_basename != basename:
                self.duplicate_exec_ids += 1
            return
        self.process_map[exec_id] = (resolved_parent, basename)

    def lookup(self, exec_id: str | None) -> tuple[int, str | None, str, bool]:
        """Walk parents from exec_id; return (depth, root_exec_id, root_basename, cycle)."""
        if not exec_id or exec_id not in self.process_map:
            return (0, None, "", False)
        if exec_id in self._cache:
            return self._cache[exec_id]

        visited: set[str] = set()
        cur: str | None = exec_id
        depth = 0
        cycle = False
        while cur is not None and cur in self.process_map:
            if cur in visited:
                cycle = True
                break
            visited.add(cur)
            parent, _basename = self.process_map[cur]
            if parent is None or parent not in self.process_map:
                break  # walked off the map
            cur = parent
            depth += 1
            if depth >= self.MAX_DEPTH:
                cycle = True  # treat depth blowout as a cycle for safety
                break

        # `cur` is now the deepest exec_id resolvable in the map.
        if cur is not None and cur in self.process_map:
            root_exec_id = cur
            root_basename = self.process_map[cur][1]
        else:
            root_exec_id = None
            root_basename = ""
        result = (min(depth, 15), root_exec_id, root_basename, cycle)
        self._cache[exec_id] = result
        return result

    def lineage_for(
        self,
        exec_id: str | None,
        proc_parent_exec_id: str | None,
        parent_exec_id: str | None,
    ) -> dict:
        """Return per-event lineage fields for one row."""
        fallback_used = (proc_parent_exec_id is None) and (parent_exec_id is not None)
        resolved_parent = proc_parent_exec_id or parent_exec_id
        if proc_parent_exec_id is None and parent_exec_id is None:
            missing = True
        elif resolved_parent is not None and resolved_parent not in self.process_map:
            missing = True
        else:
            missing = False
        depth, root_exec_id, root_basename, cycle = self.lookup(exec_id)
        return {
            "lineage_depth": depth,
            "root_ancestor_exec_id": root_exec_id,
            "root_ancestor_basename": root_basename,
            "lineage_parent_fallback_used": fallback_used,
            "lineage_missing_parent": missing,
            "lineage_cycle_detected": cycle,
        }


# ---------- bucketing helpers ----------

def log_bucket_ns(value_ns: int | None, num_buckets: int = 10) -> int:
    """Log-bucket a non-negative ns value into 0..num_buckets-1.

    Bucket 0 = None / non-positive. Higher buckets correspond to
    log10(ns)+1, capped at num_buckets-1.
    """
    if value_ns is None or value_ns <= 0:
        return 0
    b = int(math.log10(value_ns)) + 1
    return min(max(b, 0), num_buckets - 1)


def _normalize_cwd(cwd: str | None) -> str:
    """Normalize a cwd string: lowercase + ensure trailing slash. NOT a
    redaction step — usernames and other identifiers are preserved. The
    column name reflects this; a real sanitization pass (PII redaction) is
    a separate concern for any dataset-release work.
    """
    if not cwd:
        return ""
    s = cwd.lower()
    if not s.endswith("/"):
        s = s + "/"
    return s


def _path_for_category(
    kp_path: str | None,
    event_type: str | None,
    proc_binary: str | None,
    proc_cwd: str | None,
) -> str | None:
    """Pick the right path to bucket per row.

    Priority:
      1. kp_*_path (set on FILE_* / fd_install / mmap / unlink / chmod kprobes).
      2. proc_binary for process_exec — captures /tmp/payload-style staging
         that proc_cwd alone misses.
      3. proc_cwd as a final fallback for everything else.
    """
    if kp_path:
        return kp_path
    if event_type == "process_exec" and proc_binary:
        return proc_binary
    return proc_cwd


# ---------- main builder (streaming) ----------

_ID_COLS_FOR_LINEAGE: tuple[str, ...] = (
    "proc_exec_id",
    "proc_parent_exec_id",
    "parent_exec_id",
    "proc_binary",
)


def _populate_lineage_map(
    walker: LineageWalker,
    pf: pq.ParquetFile,
    batch_size: int,
) -> None:
    """Add every (exec_id -> parent, basename) tuple in pf to walker.

    Mutates walker in place. Uses LineageWalker.add_process's
    first-write-wins semantic, so callers seeding from multiple
    parquets keep the earliest registration (typically the seed
    parquet's). Reading only the id + binary columns avoids
    materializing the full ~17M-row table in memory.
    """
    for batch in pf.iter_batches(batch_size=batch_size, columns=list(_ID_COLS_FOR_LINEAGE)):
        eids = batch.column("proc_exec_id").to_pylist()
        ppids = batch.column("proc_parent_exec_id").to_pylist()
        pids = batch.column("parent_exec_id").to_pylist()
        bins = batch.column("proc_binary").to_pylist()
        for eid, ppid, pid, pb in zip(eids, ppids, pids, bins):
            walker.add_process(eid, ppid, pid, pb)


def _build_lineage_map(
    pf: pq.ParquetFile,
    batch_size: int,
) -> LineageWalker:
    """Pass 1: scan id + binary columns, build a fresh global walker.

    Thin wrapper around _populate_lineage_map for callers that want a
    standalone walker; ~50–100k unique processes per parquet, <100 MB.
    """
    walker = LineageWalker()
    _populate_lineage_map(walker, pf, batch_size)
    return walker


def _resolve_parent_basename(
    walker: LineageWalker,
    proc_parent_exec_id: str | None,
    parent_exec_id: str | None,
    fallback_basename: str,
) -> str:
    """Prefer the parent basename recorded in the global process_map (first
    occurrence wins) over the row's own parent_binary. Falls back to the
    row's own parent_binary basename if the parent isn't in the map."""
    parent_id = proc_parent_exec_id or parent_exec_id
    if parent_id and parent_id in walker.process_map:
        return walker.process_map[parent_id][1]
    return fallback_basename


def _derive_batch_columns(
    batch_table: pa.Table,
    walker: LineageWalker,
    prev_ts_by_pid: dict[str, int],
) -> tuple[dict, dict[str, int], dict]:
    """Pass 2 inner loop: derive Plan G columns for one batch.

    ``prev_ts_by_pid`` maps ``proc_exec_id`` to the last seen
    ``event_time`` for that process; it is mutated in place across
    batches so per-process ``delta_t_prev_ns`` spans batch boundaries
    correctly. The dict is also returned for explicit-state clarity at
    the call site.

    Returns (new_columns_dict, prev_ts_by_pid, batch_stats_dict). Stats
    include ``negative_delta_t_prev`` (count of within-pid negatives,
    clamped to 0; a residual noise floor from cross-CPU reordering of
    a single process's events).
    """
    n = batch_table.num_rows

    proc_exec_id = batch_table["proc_exec_id"].to_pylist()
    proc_parent_exec_id = batch_table["proc_parent_exec_id"].to_pylist()
    parent_exec_id = batch_table["parent_exec_id"].to_pylist()
    proc_binary = batch_table["proc_binary"].to_pylist()
    parent_binary = batch_table["parent_binary"].to_pylist()
    proc_cwd = batch_table["proc_cwd"].to_pylist()
    event_type = batch_table["event_type"].to_pylist()

    kprobe_fn = (
        batch_table["kprobe_function_name"].to_pylist()
        if "kprobe_function_name" in batch_table.column_names else [None] * n
    )
    kp_sock_daddr = (
        batch_table["kp_sock_daddr"].to_pylist()
        if "kp_sock_daddr" in batch_table.column_names else [None] * n
    )
    kp_sock_dport = (
        batch_table["kp_sock_dport"].to_pylist()
        if "kp_sock_dport" in batch_table.column_names else [None] * n
    )
    kp_paths: list[str | None] = [None] * n
    for col in ("kp_fd_install_path", "kp_mmap_path", "kp_unlink_path", "kp_chmod_path"):
        if col in batch_table.column_names:
            arr = batch_table[col].to_pylist()
            for i, v in enumerate(arr):
                if kp_paths[i] is None and v is not None:
                    kp_paths[i] = v

    event_time = batch_table["event_time"].cast(pa.int64()).to_pylist()
    proc_start_time = batch_table["proc_start_time"].cast(pa.int64()).to_pylist()

    proc_basenames = [os.path.basename(p) if p else "" for p in proc_binary]
    parent_basenames_row = [os.path.basename(p) if p else "" for p in parent_binary]
    proc_cwd_normalized = [_normalize_cwd(c) for c in proc_cwd]

    action_families: list[str] = []
    object_categories: list[str] = []
    path_categories: list[str] = []
    dst_ip_categories: list[str] = []
    dst_port_categories: list[str] = []
    tokens: list[str] = []
    parent_child_pair: list[str] = []
    parent_basenames_resolved: list[str] = []
    delta_t_prev: list[int | None] = []
    process_age: list[int | None] = []

    f_action_family: list[int] = []
    f_lineage_depth: list[int] = []
    f_pcp_hash: list[int] = []
    f_root_basename_hash: list[int] = []
    f_tree_id_hash: list[int] = []
    f_delta_t_log: list[int] = []
    f_age_log: list[int] = []
    # 3a' — promoted-to-model categorical features.
    f_path_category: list[int] = []
    f_dst_ip_category: list[int] = []
    f_dst_port_category: list[int] = []
    f_object_category: list[int] = []

    root_ancestor_basenames: list[str] = []
    process_tree_root_exec_ids: list[str] = []
    lineage_parent_fallback_used: list[bool] = []
    lineage_missing_parent: list[bool] = []
    lineage_cycle_detected: list[bool] = []

    neg_within_pid = 0
    for i in range(n):
        proc_b = proc_basenames[i]
        parent_b_resolved = _resolve_parent_basename(
            walker, proc_parent_exec_id[i], parent_exec_id[i], parent_basenames_row[i]
        )
        parent_basenames_resolved.append(parent_b_resolved)

        af = derive_action_family(event_type[i], kprobe_fn[i])
        oc = derive_object_category(event_type[i], kprobe_fn[i])
        path_for_cat = _path_for_category(kp_paths[i], event_type[i], proc_binary[i], proc_cwd[i])
        pc_cat = derive_path_category(path_for_cat)
        dst_ip = derive_dst_ip_category(kp_sock_daddr[i])
        dst_port = derive_dst_port_category(kp_sock_dport[i])
        tok = format_token(af, proc_b, parent_b_resolved, pc_cat, dst_ip, dst_port)

        action_families.append(af)
        object_categories.append(oc)
        path_categories.append(pc_cat)
        dst_ip_categories.append(dst_ip)
        dst_port_categories.append(dst_port)
        tokens.append(tok)

        pcp = f"{parent_b_resolved}->{proc_b}" if (parent_b_resolved or proc_b) else ""
        parent_child_pair.append(pcp)

        # Per-process delta_t: gap to this process's previous event, not
        # the previous row in the global stream. Tetragon emits events
        # via per-CPU perf ring buffers, so global row order is not a
        # reliable chronological signal across processes; per-pid order
        # is reliable up to rare within-process cross-CPU reordering.
        # Residual negatives are clamped to 0 and counted.
        ts = event_time[i]
        pid = proc_exec_id[i]
        prev_for_pid = prev_ts_by_pid.get(pid) if pid is not None else None
        if ts is None or prev_for_pid is None:
            dt = None
        else:
            dt = ts - prev_for_pid
            if dt < 0:
                neg_within_pid += 1
                dt = 0
        delta_t_prev.append(dt)
        if ts is not None and pid is not None:
            prev_ts_by_pid[pid] = ts

        st = proc_start_time[i]
        age = (ts - st) if (ts is not None and st is not None) else None
        process_age.append(age)

        lin = walker.lineage_for(proc_exec_id[i], proc_parent_exec_id[i], parent_exec_id[i])
        root_ancestor_basenames.append(lin["root_ancestor_basename"])
        process_tree_root_exec_ids.append(lin["root_ancestor_exec_id"] or "")
        lineage_parent_fallback_used.append(lin["lineage_parent_fallback_used"])
        lineage_missing_parent.append(lin["lineage_missing_parent"])
        lineage_cycle_detected.append(lin["lineage_cycle_detected"])

        f_action_family.append(_ACTION_FAMILY_INDEX.get(af, ACTION_FAMILY_OOV_INDEX))
        f_lineage_depth.append(lin["lineage_depth"])
        f_pcp_hash.append(_hash_to_bucket(pcp, 1024))
        f_root_basename_hash.append(_hash_to_bucket(lin["root_ancestor_basename"], 1024))
        f_tree_id_hash.append(_hash_to_bucket(lin["root_ancestor_exec_id"], 4096))
        f_delta_t_log.append(log_bucket_ns(dt))
        f_age_log.append(log_bucket_ns(age))
        f_path_category.append(_PATH_CATEGORY_INDEX.get(pc_cat, _PATH_CATEGORY_OOV))
        f_dst_ip_category.append(_DST_IP_CATEGORY_INDEX.get(dst_ip, _DST_IP_CATEGORY_OOV))
        f_dst_port_category.append(_DST_PORT_CATEGORY_INDEX.get(dst_port, _DST_PORT_CATEGORY_OOV))
        f_object_category.append(_OBJECT_CATEGORY_INDEX.get(oc, _OBJECT_CATEGORY_OOV))

    new_columns = {
        "proc_binary_basename": pa.array(proc_basenames, type=pa.string()),
        "parent_binary_basename": pa.array(parent_basenames_resolved, type=pa.string()),
        "proc_cwd_normalized": pa.array(proc_cwd_normalized, type=pa.string()),
        "root_ancestor_basename": pa.array(root_ancestor_basenames, type=pa.string()),
        "process_tree_root_exec_id": pa.array(process_tree_root_exec_ids, type=pa.string()),
        "parent_child_pair": pa.array(parent_child_pair, type=pa.string()),
        "path_category": pa.array(path_categories, type=pa.string()),
        "dst_ip_category": pa.array(dst_ip_categories, type=pa.string()),
        "dst_port_category": pa.array(dst_port_categories, type=pa.string()),
        "object_category": pa.array(object_categories, type=pa.string()),
        "lineage_parent_fallback_used": pa.array(lineage_parent_fallback_used, type=pa.bool_()),
        "lineage_missing_parent": pa.array(lineage_missing_parent, type=pa.bool_()),
        "lineage_cycle_detected": pa.array(lineage_cycle_detected, type=pa.bool_()),
        "token": pa.array(tokens, type=pa.string()),
        "delta_t_prev_ns": pa.array(delta_t_prev, type=pa.int64()),
        "process_age_ns": pa.array(process_age, type=pa.int64()),
        "f_action_family": pa.array(f_action_family, type=pa.uint8()),
        "f_lineage_depth": pa.array(f_lineage_depth, type=pa.uint8()),
        "f_parent_child_pair_hash": pa.array(f_pcp_hash, type=pa.uint16()),
        "f_root_ancestor_basename_hash": pa.array(f_root_basename_hash, type=pa.uint16()),
        "f_process_tree_id_hash": pa.array(f_tree_id_hash, type=pa.uint16()),
        "f_delta_t_log_bucket": pa.array(f_delta_t_log, type=pa.uint8()),
        "f_process_age_log_bucket": pa.array(f_age_log, type=pa.uint8()),
        "f_path_category": pa.array(f_path_category, type=pa.uint8()),
        "f_dst_ip_category": pa.array(f_dst_ip_category, type=pa.uint8()),
        "f_dst_port_category": pa.array(f_dst_port_category, type=pa.uint8()),
        "f_object_category": pa.array(f_object_category, type=pa.uint8()),
    }
    batch_stats = {
        "lineage_parent_fallback_used": int(sum(lineage_parent_fallback_used)),
        "lineage_missing_parent": int(sum(lineage_missing_parent)),
        "lineage_cycle_detected": int(sum(lineage_cycle_detected)),
        "negative_delta_t_prev": neg_within_pid,
    }
    return new_columns, prev_ts_by_pid, batch_stats


def build_behavior_features(
    raw_parquet_path: Path,
    output_path: Path,
    batch_size: int = 100_000,
    seed_from_parquets: list[Path] | None = None,
) -> dict:
    """Read a raw parquet, derive lineage / categories / tokens / features,
    write the augmented parquet to ``output_path``.

    Two-pass streaming design:
      Pass 1 (lineage): iterate over only the id + binary columns to build
        the global LineageWalker. ~50–100k unique processes -> <100 MB RAM.
      Pass 2 (features): iterate over all columns in ``batch_size`` chunks,
        derive Plan G columns + run ``_build_feature_table`` per batch, write
        each batch to a single output ParquetWriter. Per-batch memory peaks
        at ~one batch worth of Python lists rather than the full corpus.

    The builder DOES NOT sort the input. ``delta_t_prev_ns`` is computed
    per-process (keyed on ``proc_exec_id``), not against the previous row
    in the global stream, because Tetragon's per-CPU perf ring buffers
    can produce cross-process row-order interleaving in the parquet.
    Within a single process, residual cross-CPU reordering is rare;
    those negative deltas are clamped to 0 and counted in
    ``negative_delta_t_prev``. A non-zero value is the noise floor, not
    a correctness failure.

    ``seed_from_parquets`` (optional) is a list of additional raw
    parquets to walk in pass 1 BEFORE the primary raw parquet, so the
    LineageWalker resolves parents whose ``process_exec`` records live
    in a sibling parquet. Used to seed the test walker from the train
    parquet so test events whose parents lived only during the train
    window are not flagged as ``lineage_missing_parent``. First-write-
    wins, so seeded processes win on collision (collisions are
    expected to be rare since ``proc_exec_id`` is a hash of pid + ns +
    time).

    Returns a stats dict with wall-time, row count, unique process count,
    and lineage / ordering counters.
    """
    t0 = time.time()
    raw_parquet_path = Path(raw_parquet_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(raw_parquet_path)
    n_rows = pf.metadata.num_rows

    # ---- Pass 1: build lineage map (with optional seeding). ----
    walker = LineageWalker()
    seed_processes = 0
    if seed_from_parquets:
        for sp in seed_from_parquets:
            _populate_lineage_map(walker, pq.ParquetFile(sp), batch_size)
        seed_processes = len(walker.process_map)
    _populate_lineage_map(walker, pf, batch_size)

    # ---- Pass 2: stream batches, derive features, write. ----
    writer: pq.ParquetWriter | None = None
    prev_ts_by_pid: dict[str, int] = {}
    cumulative_stats = {
        "lineage_parent_fallback_used": 0,
        "lineage_missing_parent": 0,
        "lineage_cycle_detected": 0,
        "negative_delta_t_prev": 0,
    }
    try:
        for batch in pf.iter_batches(batch_size=batch_size):
            batch_table = pa.Table.from_batches([batch])
            new_cols, prev_ts_by_pid, batch_stats = _derive_batch_columns(
                batch_table, walker, prev_ts_by_pid
            )
            for k, v in batch_stats.items():
                cumulative_stats[k] += v
            feature_table = _build_feature_table(batch_table)
            out_columns = {name: feature_table[name] for name in feature_table.column_names}
            out_columns.update(new_cols)
            out_table = pa.table(out_columns)
            if writer is None:
                writer = pq.ParquetWriter(output_path, out_table.schema, compression="snappy")
            writer.write_table(out_table)
    finally:
        if writer is not None:
            writer.close()

    return {
        "wall_seconds": time.time() - t0,
        "rows": n_rows,
        "unique_processes": len(walker.process_map),
        "seed_processes": seed_processes,
        "duplicate_exec_ids": walker.duplicate_exec_ids,
        **cumulative_stats,
        "output": str(output_path),
    }
