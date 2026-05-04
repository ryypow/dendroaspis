"""v0.2 Isolation Forest baseline — *flat* feature variant.

Identical training procedure to ``IsolationForestScorer`` but consumes
the reduced ``TABULAR_FEATURE_COLUMNS_FLAT`` set (35 cols = 26 embed +
9 float) instead of the full 42-column matrix. The seven dropped
columns are the high-cardinality hashes: see
``EXCLUDED_FLAT_HASH_COLUMNS`` in ``shared.py``.

Purpose: decomposition of Mamba MEM-FA's 0.085 AUROC lift over
IF-rich-v0.2 into "encoder hash richness" vs "sequence body +
field-aware reconstruction". A separate scorer class (rather than a
column flag on the rich one) keeps the saved ``baseline.pkl`` and CLI
dispatch unambiguous.
"""

from __future__ import annotations

from .isoforest_scorer import IsolationForestScorer


class IsolationForestFlatScorer(IsolationForestScorer):
    """IsolationForest fitted on the v0.2 tabular matrix minus the seven
    high-cardinality hash columns. Behaviorally identical to the
    parent — feature selection happens at load time in the CLI."""
