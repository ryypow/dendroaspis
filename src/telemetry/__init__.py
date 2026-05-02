"""Telemetry parsers for the Dendroaspis edge agent.

v0.2: Tetragon-native parser. See
docs/releases/v0.2-course-milestone/v0.2.1_design_plan.md §4.
"""

from .tetragon_native_parser import (
    TetragonNativeParser,
    TetragonNativeParserConfig,
)
from .tetragon_native_writer import SCHEMA, TetragonNativeWriter

__all__ = [
    "TetragonNativeParser",
    "TetragonNativeParserConfig",
    "TetragonNativeWriter",
    "SCHEMA",
]
