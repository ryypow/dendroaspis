"""v0.2 feature engineering.

Tier-1 categorical encoders for the v0.2 Tetragon-native feature pipeline.
Spec: docs/releases/v0.2-course-milestone/v0.2_tetragon_feature_engineering.md §5.
"""
from __future__ import annotations

from src.features.v0_2_features import (
    EVENT_TYPE_OOV_INDEX,
    EVENT_TYPE_VOCAB,
    KPROBE_ACTION_NONE_INDEX,
    KPROBE_ACTION_OOV_INDEX,
    KPROBE_ACTION_VOCAB,
    KPROBE_FUNCTION_NOT_KPROBE_INDEX,
    KPROBE_FUNCTION_OOV_INDEX,
    KPROBE_FUNCTION_VOCAB,
    KPROBE_POLICY_NONE_INDEX,
    KPROBE_POLICY_OOV_INDEX,
    KPROBE_POLICY_VOCAB,
    PROC_UID_BUCKET_DAEMON_HIGH,
    PROC_UID_BUCKET_OTHER,
    PROC_UID_BUCKET_ROOT,
    PROC_UID_BUCKET_RYAN,
    PROC_UID_BUCKET_SYSTEM_LOW,
    encode_event_type,
    encode_kprobe_action,
    encode_kprobe_function_name,
    encode_kprobe_policy_name,
    encode_proc_uid_bucket,
)

__all__ = [
    "EVENT_TYPE_OOV_INDEX",
    "EVENT_TYPE_VOCAB",
    "KPROBE_ACTION_NONE_INDEX",
    "KPROBE_ACTION_OOV_INDEX",
    "KPROBE_ACTION_VOCAB",
    "KPROBE_FUNCTION_NOT_KPROBE_INDEX",
    "KPROBE_FUNCTION_OOV_INDEX",
    "KPROBE_FUNCTION_VOCAB",
    "KPROBE_POLICY_NONE_INDEX",
    "KPROBE_POLICY_OOV_INDEX",
    "KPROBE_POLICY_VOCAB",
    "PROC_UID_BUCKET_DAEMON_HIGH",
    "PROC_UID_BUCKET_OTHER",
    "PROC_UID_BUCKET_ROOT",
    "PROC_UID_BUCKET_RYAN",
    "PROC_UID_BUCKET_SYSTEM_LOW",
    "encode_event_type",
    "encode_kprobe_action",
    "encode_kprobe_function_name",
    "encode_kprobe_policy_name",
    "encode_proc_uid_bucket",
]
