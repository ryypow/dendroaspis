"""Tetragon-native parser.

Implements §4 of docs/releases/v0.2-course-milestone/v0.2_design_plan.md.

Stateless: each input file processed independently. Stats accumulate across
parse_file() calls within a single parser instance and reset only when the
parser is recreated.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import yaml

from .tetragon_native_writer import PARSER_VERSION, SCHEMA

log = logging.getLogger(__name__)

DEFAULT_TARGET_PID_NS_INUMS: frozenset[int] = frozenset({4026533329, 4026533387})

# Sentinel intervals to exclude from the output parquet.
# Events whose top-level `time` falls inside any (start, end) tuple are dropped.
# Per docs/releases/v0.2-course-milestone/v0.2_design_plan.md §1.9 / §4.11.
DEFAULT_SENTINEL_INTERVALS: tuple[tuple[str, str, str], ...] = (
    ("2026-04-19T07:54:00Z", "2026-04-19T09:40:00Z", "setup_listener_shakedown"),
    ("2026-04-19T09:40:00Z", "2026-04-19T10:15:00Z", "T1195.002_dry_run_99"),
    ("2026-04-19T11:16:00Z", "2026-04-19T11:18:00Z", "T1057_smoke_99"),
    ("2026-04-19T18:41:00Z", "2026-04-19T18:46:00Z", "T1195.002_anomaly_99"),
)

KPROBE_FUNCTIONS = (
    "fd_install",
    "security_mmap_file",
    "commit_creds",
    "tcp_connect",
    "tcp_close",
    "udp_sendmsg",
    "inet_csk_accept",
    "do_unlinkat",
    "chmod_common",
    "security_file_mprotect",
    "sys_ptrace",
    "sys_process_vm_writev",
)


# ---------- timestamp parsing ----------

def parse_tetragon_ts(ts: str | None) -> int | None:
    """Parse Tetragon RFC3339-with-nanoseconds-and-Z to int64 ns since epoch."""
    if not ts or not isinstance(ts, str) or not ts.endswith("Z"):
        return None
    body = ts[:-1]
    if "." in body:
        date_part, frac = body.split(".", 1)
        frac = frac[:9].ljust(9, "0")
        try:
            ns_frac = int(frac)
        except ValueError:
            return None
    else:
        date_part, ns_frac = body, 0
    try:
        dt = datetime.fromisoformat(date_part).replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return int(dt.timestamp()) * 1_000_000_000 + ns_frac


# ---------- empty-row template ----------

_EMPTY_ROW: dict = {f.name: None for f in SCHEMA}


def _empty_row() -> dict:
    return dict(_EMPTY_ROW)


# ---------- config ----------

class TetragonNativeParserConfig:
    def __init__(
        self,
        target_pid_ns_inums: frozenset[int] = DEFAULT_TARGET_PID_NS_INUMS,
        node_name: str = "prox1",
        parser_version: str = PARSER_VERSION,
        sentinel_intervals: tuple[tuple[str, str, str], ...] = DEFAULT_SENTINEL_INTERVALS,
    ) -> None:
        self.target_pid_ns_inums = frozenset(target_pid_ns_inums)
        self.node_name = node_name
        self.parser_version = parser_version
        self.sentinel_intervals = tuple(sentinel_intervals)
        bounds: list[tuple[int, int, str]] = []
        for s_start, s_end, label in self.sentinel_intervals:
            start_ns = parse_tetragon_ts(s_start)
            end_ns = parse_tetragon_ts(s_end)
            if start_ns is None or end_ns is None:
                raise ValueError(
                    f"sentinel_intervals entry {label!r} has unparseable bound: "
                    f"start={s_start!r} end={s_end!r}"
                )
            bounds.append((start_ns, end_ns, label))
        self._sentinel_bounds_ns: tuple[tuple[int, int, str], ...] = tuple(bounds)

    @classmethod
    def from_yaml(cls, path: Path) -> "TetragonNativeParserConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        inums = data.get("target_pid_ns_inums")
        if inums is None:
            single = data.get("target_pid_ns_inum")
            inums = [single] if single is not None else DEFAULT_TARGET_PID_NS_INUMS
        return cls(
            target_pid_ns_inums=frozenset(int(x) for x in inums),
            node_name=str(data.get("host_id", "prox1")),
        )


# ---------- process-block extraction (Groups B, C, D / E) ----------

def _flag_tokens(flags_str: str | None) -> set[str]:
    if not flags_str:
        return set()
    return set(flags_str.split())


def _extract_process_block(proc: dict, prefix: str, event_type: str) -> dict:
    """Extract Group B + C + D fields from a process block dict."""
    out: dict = {}
    if not proc:
        return out

    flags_str = proc.get("flags") or ""
    tokens = _flag_tokens(flags_str)

    # Group B
    out[f"{prefix}exec_id"] = proc.get("exec_id")
    out[f"{prefix}parent_exec_id"] = proc.get("parent_exec_id")
    out[f"{prefix}pid"] = proc.get("pid")
    out[f"{prefix}tid"] = proc.get("tid")
    out[f"{prefix}uid"] = proc.get("uid")
    out[f"{prefix}auid"] = proc.get("auid")
    out[f"{prefix}binary"] = proc.get("binary")
    out[f"{prefix}arguments"] = proc.get("arguments")
    out[f"{prefix}cwd"] = proc.get("cwd")
    out[f"{prefix}flags"] = flags_str
    out[f"{prefix}start_time"] = parse_tetragon_ts(proc.get("start_time"))
    out[f"{prefix}in_init_tree"] = bool(proc.get("in_init_tree", False))
    out[f"{prefix}refcnt"] = proc.get("refcnt")

    # Group C
    out[f"{prefix}is_procfs_walk"] = ("procFS" in tokens) and (event_type == "process_exec")
    out[f"{prefix}flag_execve"] = "execve" in tokens
    out[f"{prefix}flag_clone"] = "clone" in tokens
    out[f"{prefix}flag_rootcwd"] = "rootcwd" in tokens

    # Group D — caps
    cap = proc.get("cap") or {}
    out[f"{prefix}cap_permitted"] = cap.get("permitted") or []
    out[f"{prefix}cap_effective"] = cap.get("effective") or []
    out[f"{prefix}cap_inheritable"] = cap.get("inheritable") or []

    # Group D — namespaces
    ns = proc.get("ns") or {}
    for ns_name in ("pid", "mnt", "net", "user", "uts", "ipc", "cgroup"):
        block = ns.get(ns_name) or {}
        out[f"{prefix}ns_{ns_name}_inum"] = block.get("inum")
    for ns_name in ("pid", "user", "time"):
        block = ns.get(ns_name) or {}
        out[f"{prefix}ns_{ns_name}_is_host"] = bool(block.get("is_host", False))

    # Group D — process_credentials
    creds = proc.get("process_credentials") or {}
    for k in ("uid", "gid", "euid", "egid", "suid", "sgid", "fsuid", "fsgid"):
        out[f"{prefix}creds_{k}"] = creds.get(k)

    return out


# ---------- kprobe arg extractors ----------

def _kp_fd_install(args: list) -> dict:
    out: dict = {}
    if len(args) >= 1 and isinstance(args[0], dict) and "int_arg" in args[0]:
        try:
            out["kp_fd_install_fd"] = int(args[0]["int_arg"])
        except (TypeError, ValueError):
            pass
    if len(args) >= 2 and isinstance(args[1], dict) and "file_arg" in args[1]:
        f = args[1]["file_arg"] or {}
        out["kp_fd_install_path"] = f.get("path")
        out["kp_fd_install_permission"] = f.get("permission")
    return out


def _kp_security_mmap_file(args: list) -> dict:
    out: dict = {}
    if len(args) >= 1 and isinstance(args[0], dict) and "file_arg" in args[0]:
        f = args[0]["file_arg"] or {}
        out["kp_mmap_path"] = f.get("path")
        out["kp_mmap_permission"] = f.get("permission")
    if len(args) >= 2 and isinstance(args[1], dict):
        for k in ("uint_arg", "int_arg", "size_arg"):
            if k in args[1]:
                try:
                    out["kp_mmap_prot"] = int(args[1][k])
                except (TypeError, ValueError):
                    pass
                break
    return out


def _kp_commit_creds(args: list) -> dict:
    out: dict = {}
    if not args or not isinstance(args[0], dict):
        return out
    pca = args[0].get("process_credentials_arg") or {}
    for k in ("uid", "gid", "euid", "egid", "suid", "sgid", "fsuid", "fsgid"):
        if k in pca:
            out[f"kp_creds_{k}"] = pca[k]
    caps = pca.get("caps") or {}
    out["kp_creds_caps"] = caps.get("effective") or []
    user_ns = (pca.get("user_ns") or {}).get("ns") or {}
    if "inum" in user_ns:
        out["kp_creds_user_ns_inum"] = user_ns["inum"]
    out["kp_creds_user_ns_is_host"] = bool(user_ns.get("is_host", False))
    return out


def _extract_sock(args: list) -> dict:
    out: dict = {}
    for a in args:
        if isinstance(a, dict) and "sock_arg" in a:
            s = a["sock_arg"] or {}
            out["kp_sock_family"] = s.get("family")
            out["kp_sock_type"] = s.get("type")
            out["kp_sock_protocol"] = s.get("protocol")
            out["kp_sock_saddr"] = s.get("saddr")
            out["kp_sock_daddr"] = s.get("daddr")
            for k_in, k_out in (("sport", "kp_sock_sport"), ("dport", "kp_sock_dport")):
                if k_in in s:
                    try:
                        out[k_out] = int(s[k_in])
                    except (TypeError, ValueError):
                        pass
            out["kp_sock_state"] = s.get("state")
            cookie = s.get("cookie")
            out["kp_sock_cookie"] = str(cookie) if cookie is not None else None
            break
    return out


def _kp_tcp_connect(args: list) -> dict:
    return _extract_sock(args)


def _kp_tcp_close(args: list) -> dict:
    return _extract_sock(args)


def _kp_inet_csk_accept(args: list) -> dict:
    return _extract_sock(args)


def _kp_udp_sendmsg(args: list) -> dict:
    out = _extract_sock(args)
    for a in args:
        if isinstance(a, dict) and "int_arg" in a:
            try:
                out["kp_msg_length"] = int(a["int_arg"])
            except (TypeError, ValueError):
                pass
            break
    return out


def _kp_do_unlinkat(args: list) -> dict:
    out: dict = {}
    if len(args) >= 1 and isinstance(args[0], dict) and "int_arg" in args[0]:
        try:
            out["kp_unlink_dirfd"] = int(args[0]["int_arg"])
        except (TypeError, ValueError):
            pass
    if len(args) >= 2 and isinstance(args[1], dict):
        for k in ("string_arg", "file_arg"):
            if k in args[1]:
                v = args[1][k]
                if k == "file_arg" and isinstance(v, dict):
                    out["kp_unlink_path"] = v.get("path")
                else:
                    out["kp_unlink_path"] = v if isinstance(v, str) else None
                break
    return out


def _kp_chmod_common(args: list) -> dict:
    out: dict = {}
    if len(args) >= 1 and isinstance(args[0], dict) and "file_arg" in args[0]:
        f = args[0]["file_arg"] or {}
        out["kp_chmod_path"] = f.get("path")
        out["kp_chmod_permission_before"] = f.get("permission")
    if len(args) >= 2 and isinstance(args[1], dict):
        for k in ("uint_arg", "int_arg"):
            if k in args[1]:
                try:
                    out["kp_chmod_mode"] = int(args[1][k])
                except (TypeError, ValueError):
                    pass
                break
    return out


def _kp_security_file_mprotect(args: list) -> dict:
    out: dict = {}
    vals = []
    for a in args:
        if not isinstance(a, dict):
            continue
        for k in ("size_arg", "uint_arg", "int_arg"):
            if k in a:
                try:
                    vals.append(int(a[k]))
                except (TypeError, ValueError):
                    pass
                break
    if len(vals) >= 1:
        out["kp_mprotect_reqprot"] = vals[0]
    if len(vals) >= 2:
        out["kp_mprotect_prot"] = vals[1]
    return out


def _kp_sys_ptrace(args: list) -> dict:
    out: dict = {}
    if len(args) >= 1 and isinstance(args[0], dict):
        for k in ("long_arg", "int_arg", "size_arg"):
            if k in args[0]:
                try:
                    out["kp_ptrace_request"] = int(args[0][k])
                except (TypeError, ValueError):
                    pass
                break
    if len(args) >= 2 and isinstance(args[1], dict):
        for k in ("int_arg", "long_arg"):
            if k in args[1]:
                try:
                    out["kp_ptrace_target_pid"] = int(args[1][k])
                except (TypeError, ValueError):
                    pass
                break
    return out


def _kp_sys_process_vm_writev(args: list) -> dict:
    out: dict = {}
    if len(args) >= 1 and isinstance(args[0], dict):
        for k in ("int_arg", "long_arg"):
            if k in args[0]:
                try:
                    out["kp_pvw_target_pid"] = int(args[0][k])
                except (TypeError, ValueError):
                    pass
                break
    return out


KPROBE_EXTRACTORS = {
    "fd_install": _kp_fd_install,
    "security_mmap_file": _kp_security_mmap_file,
    "commit_creds": _kp_commit_creds,
    "tcp_connect": _kp_tcp_connect,
    "tcp_close": _kp_tcp_close,
    "udp_sendmsg": _kp_udp_sendmsg,
    "inet_csk_accept": _kp_inet_csk_accept,
    "do_unlinkat": _kp_do_unlinkat,
    "chmod_common": _kp_chmod_common,
    "security_file_mprotect": _kp_security_file_mprotect,
    "sys_ptrace": _kp_sys_ptrace,
    "sys_process_vm_writev": _kp_sys_process_vm_writev,
}


# ---------- parser ----------

class TetragonNativeParser:
    """Stateless Tetragon JSONL → row-dict parser.

    Stats accumulate across parse_file() calls within a single instance.
    """

    def __init__(self, config: TetragonNativeParserConfig) -> None:
        self.config = config
        self.stats: Counter[str] = Counter()

    # --- container filter ---

    def _passes_container_filter(self, proc: dict) -> bool:
        ns = (proc.get("ns") or {}).get("pid") or {}
        if ns.get("is_host"):
            self.stats["drop_container_filter_is_host"] += 1
            return False
        inum = ns.get("inum")
        if inum is None:
            self.stats["drop_container_filter_no_inum"] += 1
            return False
        if self.config.target_pid_ns_inums and inum not in self.config.target_pid_ns_inums:
            self.stats["drop_container_filter_wrong_inum"] += 1
            return False
        return True

    # --- sentinel filter ---

    def _is_in_sentinel_interval(self, event_time_ns: int | None) -> bool:
        if event_time_ns is None:
            self.stats["unparseable_event_time"] += 1
            return False
        for s_start, s_end, _ in self.config._sentinel_bounds_ns:
            if s_start <= event_time_ns <= s_end:
                return True
        return False

    # --- per-event extraction ---

    def _build_row(self, raw: dict, source_file: str, source_line: int) -> dict | None:
        if not isinstance(raw, dict):
            self.stats["drop_wrong_event_type"] += 1
            return None

        if "process_exec" in raw:
            event_type = "process_exec"
            inner = raw["process_exec"]
        elif "process_exit" in raw:
            event_type = "process_exit"
            inner = raw["process_exit"]
        elif "process_kprobe" in raw:
            event_type = "process_kprobe"
            inner = raw["process_kprobe"]
        else:
            self.stats["drop_wrong_event_type"] += 1
            return None

        proc = inner.get("process") or {}
        if not proc:
            self.stats["drop_no_process"] += 1
            return None

        if not self._passes_container_filter(proc):
            return None

        event_time_ns = parse_tetragon_ts(raw.get("time"))
        if self._is_in_sentinel_interval(event_time_ns):
            self.stats["drop_sentinel_interval"] += 1
            return None

        row = _empty_row()

        # Group A
        row["event_type"] = event_type
        row["event_time"] = event_time_ns
        row["node_name"] = raw.get("node_name") or self.config.node_name

        # Groups B + C + D for source process
        row.update(_extract_process_block(proc, "proc_", event_type))

        # Tag-not-drop procfs-walk execs per design plan
        # (plan_archive/tetragon_native_parquet_rebuild.md:393, :436) and
        # V.5-redo evidence in tetragon-dataset-validation.ipynb cells 17-23
        # (~28K procFS-flagged events fall inside ART intervals; they are not
        # startup-only). Dropping breaks lineage for pre-Tetragon processes
        # and deletes real attack-window signal. The proc_is_procfs_walk flag
        # survives on the row; downstream consumers (encoder f_is_procfs_walk)
        # use it to discount backdated event_time on these specific rows.
        # Only process_exec carries a meaningful procfs-walk tag; exit/kprobe
        # paths clear the flag below per §1.11. The kept-tag counter fires
        # at row-finalize time (see tagged_procfs_walk_execs below).

        # Groups B + C + D for parent process (mirrors B-D per §2.2 line 303)
        parent = inner.get("parent") or {}
        row.update(_extract_process_block(parent, "parent_", event_type))

        # Group F + G — event-type-specific
        if event_type == "process_exit":
            row["exit_status"] = inner.get("status")
            row["exit_signal"] = inner.get("signal")
            if row["proc_is_procfs_walk"]:
                row["proc_is_procfs_walk"] = False  # only execs get tagged per §1.11
            if row["parent_is_procfs_walk"]:
                row["parent_is_procfs_walk"] = False
        elif event_type == "process_kprobe":
            fn = inner.get("function_name")
            row["kprobe_function_name"] = fn
            row["kprobe_policy_name"] = inner.get("policy_name")
            row["kprobe_action"] = inner.get("action")
            row["kprobe_return_action"] = inner.get("return_action")
            args = inner.get("args") or []
            extractor = KPROBE_EXTRACTORS.get(fn)
            if extractor is not None:
                try:
                    row.update(extractor(args))
                except Exception as e:
                    self.stats["errors_extraction"] += 1
                    log.warning(
                        "kprobe extraction failure file=%s line=%d fn=%s: %s",
                        source_file, source_line, fn, e,
                    )
            try:
                row["kprobe_args_json"] = json.dumps(args, separators=(",", ":"))
            except (TypeError, ValueError):
                self.stats["errors_extraction"] += 1
                row["kprobe_args_json"] = None
            if row["proc_is_procfs_walk"]:
                row["proc_is_procfs_walk"] = False
            if row["parent_is_procfs_walk"]:
                row["parent_is_procfs_walk"] = False

        # Group H
        row["_source_file"] = source_file
        row["_source_line"] = source_line
        row["_parser_version"] = self.config.parser_version

        self.stats["lines_kept"] += 1
        if row.get("proc_is_procfs_walk"):
            self.stats["tagged_procfs_walk_execs"] += 1
        self.stats[f"kept_{event_type}"] += 1

        return row

    # --- public API ---

    def parse_file(self, path: Path) -> Iterator[dict]:
        path = Path(path)
        try:
            opener = gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, "r", encoding="utf-8")
        except OSError as e:
            self.stats["files_failed"] += 1
            log.error("could not open file %s: %s", path, e)
            return
        self.stats["files_read"] += 1
        try:
            with opener as fh:
                for lineno, line in enumerate(fh, start=1):
                    self.stats["lines_read"] += 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError as e:
                        self.stats["errors_json_parse"] += 1
                        log.warning("json parse error %s:%d: %s", path.name, lineno, e)
                        continue
                    try:
                        row = self._build_row(raw, path.name, lineno)
                    except Exception as e:
                        self.stats["errors_extraction"] += 1
                        log.warning("extraction error %s:%d: %s", path.name, lineno, e)
                        continue
                    if row is not None:
                        yield row
        except OSError as e:
            self.stats["files_failed"] += 1
            log.error("read error on file %s: %s", path, e)

    def parse_files(self, paths: list[Path]) -> Iterator[dict]:
        for p in paths:
            yield from self.parse_file(p)
