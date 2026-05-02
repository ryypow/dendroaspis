"""Tetragon-native parquet writer.

Holds the locked raw-event SCHEMA literal (§2.2) and a single-output
writer that accumulates raw parser rows across multiple parse_file()
calls and writes one parquet at the path the caller specifies. The
on-disk schema is the raw event SCHEMA below (no feature computation).

Feature computation is a downstream stage in v0_2_behavior_builder, which
reads the raw parquet, joins lineage / categories / tokens, and runs
_build_feature_table for the 33 f_* features. Keeping raw and feature
parquets separate preserves a re-encodable raw artifact and lets lineage
features see global state across all files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

PARSER_VERSION = "0.1.0"

_TS = pa.timestamp("ns", tz="UTC")
_LIST_STR = pa.list_(pa.string())


def _proc_fields(prefix: str) -> list[pa.Field]:
    """Group B + C + D fields with the given prefix ("proc_" or "parent_")."""
    return [
        # Group B
        pa.field(f"{prefix}exec_id", pa.string()),
        pa.field(f"{prefix}parent_exec_id", pa.string()),
        pa.field(f"{prefix}pid", pa.uint32()),
        pa.field(f"{prefix}tid", pa.uint32()),
        pa.field(f"{prefix}uid", pa.uint32()),
        pa.field(f"{prefix}auid", pa.uint32()),
        pa.field(f"{prefix}binary", pa.string()),
        pa.field(f"{prefix}arguments", pa.string()),
        pa.field(f"{prefix}cwd", pa.string()),
        pa.field(f"{prefix}flags", pa.string()),
        pa.field(f"{prefix}start_time", _TS),
        pa.field(f"{prefix}in_init_tree", pa.bool_()),
        pa.field(f"{prefix}refcnt", pa.uint32()),
        # Group C
        pa.field(f"{prefix}is_procfs_walk", pa.bool_()),
        pa.field(f"{prefix}flag_execve", pa.bool_()),
        pa.field(f"{prefix}flag_clone", pa.bool_()),
        pa.field(f"{prefix}flag_rootcwd", pa.bool_()),
        # Group D
        pa.field(f"{prefix}cap_permitted", _LIST_STR),
        pa.field(f"{prefix}cap_effective", _LIST_STR),
        pa.field(f"{prefix}cap_inheritable", _LIST_STR),
        pa.field(f"{prefix}ns_pid_inum", pa.uint64()),
        pa.field(f"{prefix}ns_mnt_inum", pa.uint64()),
        pa.field(f"{prefix}ns_net_inum", pa.uint64()),
        pa.field(f"{prefix}ns_user_inum", pa.uint64()),
        pa.field(f"{prefix}ns_uts_inum", pa.uint64()),
        pa.field(f"{prefix}ns_ipc_inum", pa.uint64()),
        pa.field(f"{prefix}ns_cgroup_inum", pa.uint64()),
        pa.field(f"{prefix}ns_pid_is_host", pa.bool_()),
        pa.field(f"{prefix}ns_user_is_host", pa.bool_()),
        pa.field(f"{prefix}ns_time_is_host", pa.bool_()),
        pa.field(f"{prefix}creds_uid", pa.uint32()),
        pa.field(f"{prefix}creds_gid", pa.uint32()),
        pa.field(f"{prefix}creds_euid", pa.uint32()),
        pa.field(f"{prefix}creds_egid", pa.uint32()),
        pa.field(f"{prefix}creds_suid", pa.uint32()),
        pa.field(f"{prefix}creds_sgid", pa.uint32()),
        pa.field(f"{prefix}creds_fsuid", pa.uint32()),
        pa.field(f"{prefix}creds_fsgid", pa.uint32()),
    ]


_GROUP_A = [
    pa.field("event_type", pa.string()),
    pa.field("event_time", _TS),
    pa.field("node_name", pa.string()),
]

_GROUP_F = [
    pa.field("exit_status", pa.int32()),
    pa.field("exit_signal", pa.string()),
    pa.field("kprobe_function_name", pa.string()),
    pa.field("kprobe_policy_name", pa.string()),
    pa.field("kprobe_action", pa.string()),
    pa.field("kprobe_return_action", pa.string()),
]

_GROUP_G = [
    # fd_install
    pa.field("kp_fd_install_fd", pa.int32()),
    pa.field("kp_fd_install_path", pa.string()),
    pa.field("kp_fd_install_permission", pa.string()),
    # security_mmap_file
    pa.field("kp_mmap_path", pa.string()),
    pa.field("kp_mmap_permission", pa.string()),
    pa.field("kp_mmap_prot", pa.uint32()),
    # commit_creds
    pa.field("kp_creds_uid", pa.uint32()),
    pa.field("kp_creds_gid", pa.uint32()),
    pa.field("kp_creds_euid", pa.uint32()),
    pa.field("kp_creds_egid", pa.uint32()),
    pa.field("kp_creds_suid", pa.uint32()),
    pa.field("kp_creds_sgid", pa.uint32()),
    pa.field("kp_creds_fsuid", pa.uint32()),
    pa.field("kp_creds_fsgid", pa.uint32()),
    pa.field("kp_creds_caps", _LIST_STR),
    pa.field("kp_creds_user_ns_inum", pa.uint64()),
    pa.field("kp_creds_user_ns_is_host", pa.bool_()),
    # tcp_connect / tcp_close / inet_csk_accept / udp_sendmsg shared sock columns
    pa.field("kp_sock_family", pa.string()),
    pa.field("kp_sock_type", pa.string()),
    pa.field("kp_sock_protocol", pa.string()),
    pa.field("kp_sock_saddr", pa.string()),
    pa.field("kp_sock_daddr", pa.string()),
    pa.field("kp_sock_sport", pa.uint32()),
    pa.field("kp_sock_dport", pa.uint32()),
    pa.field("kp_sock_state", pa.string()),
    pa.field("kp_sock_cookie", pa.string()),
    # udp_sendmsg extra
    pa.field("kp_msg_length", pa.int32()),
    # do_unlinkat
    pa.field("kp_unlink_dirfd", pa.int32()),
    pa.field("kp_unlink_path", pa.string()),
    # chmod_common
    pa.field("kp_chmod_path", pa.string()),
    pa.field("kp_chmod_permission_before", pa.string()),
    pa.field("kp_chmod_mode", pa.uint32()),
    # security_file_mprotect
    pa.field("kp_mprotect_reqprot", pa.uint64()),
    pa.field("kp_mprotect_prot", pa.uint64()),
    # sys_ptrace
    pa.field("kp_ptrace_request", pa.int64()),
    pa.field("kp_ptrace_target_pid", pa.int32()),
    # sys_process_vm_writev
    pa.field("kp_pvw_target_pid", pa.int32()),
    # JSON catchall (always populated for kprobe events)
    pa.field("kprobe_args_json", pa.string()),
]

_GROUP_H = [
    pa.field("_source_file", pa.string()),
    pa.field("_source_line", pa.uint32()),
    pa.field("_parser_version", pa.string()),
]

SCHEMA: pa.Schema = pa.schema(
    _GROUP_A
    + _proc_fields("proc_")
    + _proc_fields("parent_")
    + _GROUP_F
    + _GROUP_G
    + _GROUP_H
)


def _rows_to_record_batch(rows: list[dict]) -> pa.RecordBatch:
    """Build a RecordBatch from row dicts. Timestamps are int64 ns (or None)."""
    arrays = []
    for field in SCHEMA:
        col = [r.get(field.name) for r in rows]
        if pa.types.is_timestamp(field.type):
            arrays.append(pa.array(col, type=pa.int64()).cast(field.type, safe=False))
        else:
            arrays.append(pa.array(col, type=field.type))
    return pa.RecordBatch.from_arrays(arrays, schema=SCHEMA)


class TetragonNativeWriter:
    """Single-output raw-event parquet writer.

    Accumulates raw parser rows across multiple parse_file() calls and
    writes one parquet at ``output_path`` using the raw-event ``SCHEMA``
    above. Feature computation lives downstream in v0_2_behavior_builder.
    """

    def __init__(
        self,
        output_path: Path,
        batch_size: int = 50_000,
        compression: str = "snappy",
    ) -> None:
        self.output_path = Path(output_path)
        self.batch_size = batch_size
        self.compression = compression
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict] = []
        self._writer: pq.ParquetWriter | None = None

    def _flush(self) -> None:
        if not self._buffer:
            return
        batch = _rows_to_record_batch(self._buffer)
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self.output_path,
                SCHEMA,
                compression=self.compression,
            )
        self._writer.write_batch(batch)
        self._buffer.clear()

    def write(self, rows: Iterable[dict]) -> None:
        """Accumulate rows; flush in ``batch_size`` chunks."""
        for row in rows:
            self._buffer.append(row)
            if len(self._buffer) >= self.batch_size:
                self._flush()

    def close(self) -> None:
        """Flush the final partial batch and close the underlying ParquetWriter."""
        self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
