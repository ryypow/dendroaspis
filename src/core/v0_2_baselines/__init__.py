"""v0.2 classical baselines: n-gram, XGBoost, Isolation Forest.

Three pre-registered baselines for the §3b architecture bake-off, each
probing an orthogonal question about where signal in the v0.2 corpus
lives:

  * NgramScorer        — does long-range sequence modeling matter?
  * XGBoostScorer      — is this a supervised problem in disguise?
  * IsolationForestScorer — does per-event density estimation alone work?

Authoritative spec: ``docs/releases/v0.2-course-milestone/
v0.2_baseline_bake_off_design.md``. All three scorers expose the same
interface (``fit``, ``score_events``, ``save``, ``load``) and emit the
same per-event score schema as the Mamba evaluator so the existing
training-analysis notebook can graph all four lines without
per-baseline special casing.
"""

from .ngram_scorer import NgramScorer
from .xgboost_scorer import XGBoostScorer
from .isoforest_scorer import IsolationForestScorer
from .isoforest_flat_scorer import IsolationForestFlatScorer

__all__ = [
    "NgramScorer",
    "XGBoostScorer",
    "IsolationForestScorer",
    "IsolationForestFlatScorer",
]
