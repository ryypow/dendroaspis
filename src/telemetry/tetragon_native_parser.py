"""Tetragon-native parser.

It reads the ~140 tetragon events.json.gz files collected for this experiment. Each line in each events.json file is one kernel event
-The tetragon trace policies (config/tetragon) controls the event types to collect.
    - the trace policies were designed to collect the events that can provide the model enough information to determine if there is an attack taking place

-Filtering:
    -the container used in this experiment by its INUM (specific to my homelab proxmox)
        - Since there were additional containers deployed on the proxmox host, this parser filters for
        - by default, tetragon traces EVERY process on the host by default (including all containers and the proxmox host itself), which is why the container filter is necessary.

    - "sentinel intervals":
        - time windows during which the lab was being set up, smoke-tested, and testing the atomic red team setup/configuration
        - none of these attacks are included in the training or evaluation set

    - any events missing an actual process binary:
        - rows (each row being a kernel event) missing a process binary are useless for the model training so there is no need to include them

After the events are filtered, the parser pulls out the fields used in the encoder. This is where the attack signal resides.
Custom trace policies: the tetragon eBPF probes that monitor for the defined kernel functions
- kernel probes (kprobe): the "tripwire" that triggers event collection. 
    - everytime a program/process wants to do something, it has to go through the kernel
    - the kprobe is like the "hidden camera" that collects the events for the functions defined in the tetragon trace policies
    - no malicious process or package can avoid the kernel - it will see everything, regardless of how an attacker tries to obfuscate their attack

- for this experiment, we are monitoring the following kernel functions:
        - fd_install: catches file opens, dropped payloads, and freshly created sockets in one place
        - commit_creds: ground-truth privilege-escalation tripwire: setuid binaries, sudo, exploits that flip a process to UID 0
        - tcp_close: Fires when a TCP socket is torn down. Pairs with tcp_connect / inet_csk_accept
        - udp_sendmsg: Fires on every outbound UDP datagram. Captures addresses, ports, and message length. Catches DNS queries (so DNS-tunneling C2 shows up here), NTP, and any custom UDP-based exfil
        - inet_csk_accept: Fires when a process accepts an inbound TCP connection
            - Key signal for reverse shells dialing back, bind shells, and unauthorized listeners
        - do_unlinkat: fires when a file is deleted. Primary signal for log wiping, evidence destruction, and ransomware cleanup of originals after encryption.
        - chmod_common: Fires whenever a file's permission bits change, with the path and the new mode. Catches "drop a payload then make it executable"
        - security_file_mprotect: LSM hook that gates mprotect — the syscall that changes the protection bits on an already-mapped memory region
        - sys_ptrace: fires on ptrace syscalls:
            - Used legitimately by debuggers (gdb, strace), and used maliciously for code injection, credential theft (reading another process's memory), and anti-debug evasion
        - sys_process_vm_writev:  Fires when one process writes directly into another process's memory via process_vm_writev
            - if triggered, it's a strong signal for cross-process code/data injection

Default event types tetragon collects regardless of trace policy config:
- process_exec/process_exit: when a process starts/ends
    - this is used to trace process lineage
    - this is how we understand where events originated from
    - shows attack signal for the chained attacks like: file-download -> chmod +x -> process execute -> data exfiltration
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

#the target containers namespace inums: ignoring the other containers present on the proxmox host
DEFAULT_TARGET_PID_NS_INUMS: frozenset[int] = frozenset({4026533329, 4026533387})

# Sentinel intervals to exclude from the output parquet.
# Events whose top-level `time` falls inside any (start, end) tuple are dropped.
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


#timestamp parsing: converts the the tetragon event timestamp into nanoseconds since the unix epoch (January 1, 1970)
#timestamps (in nanoseconds) are stored as int64
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


#empty-row template

_EMPTY_ROW: dict = {f.name: None for f in SCHEMA}


def _empty_row() -> dict:
    return dict(_EMPTY_ROW)


#config:
#NOTE: scoped to the target containers namespace inums specific to the proxmox host (two of them, due to container respawn)

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


#process-block extraction
""" 
Builds the per-event parquet schema.
- each tetragon event comes wrapped in a process dictionary describing the process that triggered the event
- this function takes the dictionary and flattens it into a bunch of columns suitable for a parquet row
- called twice per event: once for the process itself (prefix "proc_") and once for its parent (prefix "parent_"),
  so a single row carries both sides of the lineage

Each event contains:
 - process identity: proc_pid, proc_binary, proc_uid, proc_arguments, proc_cwd
     - exec_id / parent_exec_id are the stable lineage keys (PIDs get reused, exec_ids do not)
     - auid is the original login uid preserved across su/sudo — divergence from uid is itself a priv-esc signal

 - flags: proc_flag_execve, proc_flag_clone, proc_flag_rootcwd, proc_is_procfs_walk
     - flag_execve / flag_clone tell you whether this process arrived via execve or fork/clone
     - is_procfs_walk marks execs that tetragon backfilled by walking /proc at startup
       (real exec happened earlier; the encoder backdates event_time to start_time for these)

 - timing: proc_start_time (int64 ns since epoch)
     - lets the encoder compute process age at event time

 - capabilities: proc_cap_permitted, proc_cap_effective, proc_cap_inheritable
     - linux's fine-grained replacement for binary root/non-root
     - effective = active right now; permitted = ceiling; inheritable = passed across execve

 - namespaces: proc_ns_{pid,mnt,net,user,uts,ipc,cgroup}_inum (+ is_host for pid/user/time)
     - the inum fields are how the container filter scopes events to the target LXC
     - is_host distinguishes host-namespace events from container-namespace events

 - credentials: proc_creds_{uid,gid,euid,egid,suid,sgid,fsuid,fsgid}
     - linux tracks four parallel uid/gid sets (real, effective, saved-set, filesystem)
     - matched values = normal; divergence = setuid binary mid-exec or active priv manipulation

All of the above is mirrored under the parent_ prefix for the parent process block.
"""

def _flag_tokens(flags_str: str | None) -> set[str]:
    if not flags_str:
        return set()
    return set(flags_str.split())


def _extract_process_block(proc: dict, prefix: str, event_type: str) -> dict:
    out: dict = {}
    if not proc:
        return out

    flags_str = proc.get("flags") or ""
    tokens = _flag_tokens(flags_str)

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

    out[f"{prefix}is_procfs_walk"] = ("procFS" in tokens) and (event_type == "process_exec")
    out[f"{prefix}flag_execve"] = "execve" in tokens
    out[f"{prefix}flag_clone"] = "clone" in tokens
    out[f"{prefix}flag_rootcwd"] = "rootcwd" in tokens

    #capabilities
    cap = proc.get("cap") or {}
    out[f"{prefix}cap_permitted"] = cap.get("permitted") or []
    out[f"{prefix}cap_effective"] = cap.get("effective") or []
    out[f"{prefix}cap_inheritable"] = cap.get("inheritable") or []

    #namespaces
    ns = proc.get("ns") or {}
    for ns_name in ("pid", "mnt", "net", "user", "uts", "ipc", "cgroup"):
        block = ns.get(ns_name) or {}
        out[f"{prefix}ns_{ns_name}_inum"] = block.get("inum")
    for ns_name in ("pid", "user", "time"):
        block = ns.get(ns_name) or {}
        out[f"{prefix}ns_{ns_name}_is_host"] = bool(block.get("is_host", False))

    #process_credentials
    creds = proc.get("process_credentials") or {}
    for k in ("uid", "gid", "euid", "egid", "suid", "sgid", "fsuid", "fsgid"):
        out[f"{prefix}creds_{k}"] = creds.get(k)

    return out


#kprobe arg extractor
"""
Twelve functions, one per kprobe defined in trace policies
- each arg field is a positional list of argument wrappers
- each element is a one-key dict, like {"int_arg": 5}, {"file_arg": {...}}, {"sock_arg": {...}}
- drops any args that are not expected (truncated, )
"""

# -----------file descriptor installation--------
# mirrors kernel signature: fd_install(unsigned int fd, struct file *file)
# if a process opens /etc/shadow or drops a payload at /tmp/x, this is where you see the path and the mode bits at install time
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

# ------------ file-backed memory mapping-----------
# Signature: security_mmap_file(file, prot, flags)
# captures the file being mapped and the protection bits requested at mmap time
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

# --------- credential transition -------------
#takes one struct of new credentials
#Tetragon unpacks the entire thing — all four UID/GID sets, the new effective capability list, and the user namespace the creds belong to
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

# ------------ helper for the four network probes------------
# iterates across tcp_connect, tcp_close, inet_csk_accept, udp_sendmsg
# enables the reconstruction of a full network 4-tuple: (saddr, sport, daddr, dport) plus protocol family and TCP state
# The cookie cast to str is because socket cookies are 64-bit unsigned integers that overflow Parquet's int64 column
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

# udp_sendmsg: standard sock fields plus the size of the datagram payload
# Message length is the dominant signal for DNS-tunneling C2
#abnormally large UDP/53 payloads stand out hard against normal DNS traffic
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

# -----------file deletion---------
# Signature: do_unlinkat(dfd, name)
# dirfd is the directory the path is relative to (AT_FDCWD = -100 means "relative to cwd")
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

# -------------permission-bit changes---------
# captures the path, the old mode (_before), and the new mode being requested
# lets the model see the delta betwen permission changes
# like 0644 → 0755 (added executable)
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

#---------- memory protection changes--------
# accumulates all numeric args into a list and takes the first two
# reqprot is what the process asked for; prot is what the LSM ultimately allowed
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

#------------debugger / injection syscall---------
#captures the ptrace operation code and the target PID
#the request code is the benign/attack discriminator
#PTRACE_ATTACH (attach to victim) and PTRACE_POKEDATA (write into victim's memory) provide attack signal
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

#----------direct cross-process memory write-----
#captures writes to the target containers processes
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


#parser
class TetragonNativeParser:
    """Stateless Tetragon JSONL → row-dict parser.

    Stats accumulate across parse_file() calls within a single instance.
    """

    def __init__(self, config: TetragonNativeParserConfig) -> None:
        self.config = config
        self.stats: Counter[str] = Counter()

    #container filter: filters for the monitored LXC container by cgroup/inum

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

        # event type/time and node name extraction
        row["event_type"] = event_type
        row["event_time"] = event_time_ns
        row["node_name"] = raw.get("node_name") or self.config.node_name

        # source process fields
        row.update(_extract_process_block(proc, "proc_", event_type))

        """
        NOTE: I tagged procfs-walk execs due to dropping breaking the lineage/ancestor trace
        for tetragon processes.

        IN v1, dropping these broke the lineage between processes and dropped events within attack intervals

        The encoder uses it for backdated event_times on the rows with proc_is_procfs_walk= true
        
        """
        parent = inner.get("parent") or {}
        row.update(_extract_process_block(parent, "parent_", event_type))

        #event-type-specific
        if event_type == "process_exit":
            row["exit_status"] = inner.get("status")
            row["exit_signal"] = inner.get("signal")
            if row["proc_is_procfs_walk"]:
                row["proc_is_procfs_walk"] = False
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
                #captures full args blob for field recovery, incase the extractor functions missed a field
                row["kprobe_args_json"] = json.dumps(args, separators=(",", ":"))
            except (TypeError, ValueError):
                self.stats["errors_extraction"] += 1
                row["kprobe_args_json"] = None
            if row["proc_is_procfs_walk"]:
                row["proc_is_procfs_walk"] = False
            if row["parent_is_procfs_walk"]:
                row["parent_is_procfs_walk"] = False

        # provenance/metadata
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
