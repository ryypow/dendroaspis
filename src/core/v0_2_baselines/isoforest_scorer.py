"""v0.2 Isolation Forest baseline — unsupervised tabular outlier
detection on the f_* feature columns.

Trains on **benign-only** train events (events whose timestamp falls
outside any ART label interval). Scores every test event by negated
path-length anomaly score so higher = more anomalous, matching the
sign convention used by Mamba / n-gram.

This is the natural baseline for the **MEM-FA** Mamba run, since both
are unsupervised reconstruction-style anomaly scorers — IF asks
"is this event a per-event outlier in feature space?" and MEM-FA asks
"is this event reconstructable from its sequence context?" If IF beats
MEM-FA at headline AUROC, the field-aware reconstruction objective is
not adding contextual value over plain density estimation.

Supersedes ``scripts/run_v0_2_if_baseline.py`` for the bake-off
context. The legacy script reads ``data/processed/v0.2-features/``
(pre-Plan-G, pre-rebuilt-corpus) and emits a metrics-only JSON without
per-event scores, per-technique attribution, or Aim logging. Kept in
the tree for v0.1-feature reproducibility; do not use for v0.2 work.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class IsolationForestFitDiagnostics:
    """Reportable diagnostics from a fit. Logged to train.log + Aim."""
    n_train_events_total: int = 0
    n_train_events_benign: int = 0
    n_train_events_attack_dropped: int = 0
    n_train_used: int = 0
    n_features: int = 0
    n_estimators: int = 0
    contamination: str = "auto"
    max_samples: int | str = "auto"

    def as_dict(self) -> dict:
        return {
            "n_train_events_total":          self.n_train_events_total,
            "n_train_events_benign":         self.n_train_events_benign,
            "n_train_events_attack_dropped": self.n_train_events_attack_dropped,
            "n_train_used":                  self.n_train_used,
            "n_features":                    self.n_features,
            "n_estimators":                  self.n_estimators,
            "contamination":                 self.contamination,
            "max_samples":                   self.max_samples,
        }


class IsolationForestScorer:
    """Unsupervised IF on per-event tabular features."""

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_samples: int | str = "auto",
        contamination: str = "auto",
        random_state: int = 0,
        n_jobs: int = -1,
        max_train_rows: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = n_jobs
        # ``max_train_rows`` lets the caller cap the training matrix to
        # mirror the v0.1 V.7-equivalent protocol (200k stratified
        # subsample) when running cheap diagnostic comparisons. Set None
        # for the full benign train set.
        self.max_train_rows = max_train_rows

        self.model: IsolationForest | None = None
        self.feature_columns: list[str] | None = None
        self.diag: IsolationForestFitDiagnostics = IsolationForestFitDiagnostics()

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        feature_columns: list[str] | None = None,
    ) -> IsolationForestFitDiagnostics:
        """Train on benign-only events from (X, y).

        ``y`` is the per-event ``is_attack`` boolean array. Attack events
        are dropped before fitting so the unsupervised model learns the
        benign distribution only — matching the bake-off design's clean
        unsupervised parity with the n-gram baseline.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2-d, got shape {X.shape}")
        if y.shape != (X.shape[0],):
            raise ValueError(f"y shape {y.shape} != X.shape[0]={X.shape[0]}")

        n_total = X.shape[0]
        benign = ~y.astype(bool)
        X_benign = X[benign]
        n_attack_dropped = int(n_total - X_benign.shape[0])

        # Optional subsample cap (V.7-equivalent protocol).
        rng = np.random.default_rng(self.random_state)
        if self.max_train_rows is not None and X_benign.shape[0] > self.max_train_rows:
            sample_idx = rng.choice(
                X_benign.shape[0], size=self.max_train_rows, replace=False
            )
            sample_idx.sort()  # preserve chronological order
            X_train = X_benign[sample_idx]
        else:
            X_train = X_benign

        if X_train.shape[0] == 0:
            raise RuntimeError(
                "no benign training events — IsolationForest cannot fit. "
                "Check that labels.csv / is_attack array are not all True."
            )

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model.fit(X_train)
        self.feature_columns = list(feature_columns or [])

        self.diag = IsolationForestFitDiagnostics(
            n_train_events_total=n_total,
            n_train_events_benign=int(X_benign.shape[0]),
            n_train_events_attack_dropped=n_attack_dropped,
            n_train_used=int(X_train.shape[0]),
            n_features=int(X.shape[1]),
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
        )
        return self.diag

    # ----------------------------------------------------------------- score

    def score_events(self, X: np.ndarray) -> np.ndarray:
        """Per-event anomaly score; higher = more anomalous.

        ``score_samples`` returns the negated path-length where higher =
        more normal; we negate again so the sign matches Mamba / n-gram.
        """
        if self.model is None:
            raise RuntimeError(
                "IsolationForestScorer.fit() must be called before score_events()"
            )
        # IF.score_samples: higher = more normal. Negate for anomaly.
        scores = -self.model.score_samples(X)
        return scores.astype(np.float32, copy=False)

    # ------------------------------------------------------------------ I/O

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({
                "n_estimators":     self.n_estimators,
                "max_samples":      self.max_samples,
                "contamination":    self.contamination,
                "random_state":     self.random_state,
                "n_jobs":           self.n_jobs,
                "max_train_rows":   self.max_train_rows,
                "model":            self.model,
                "feature_columns":  self.feature_columns,
                "diag":             self.diag.as_dict(),
            }, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> IsolationForestScorer:
        with Path(path).open("rb") as fh:
            d = pickle.load(fh)
        s = cls(
            n_estimators=d["n_estimators"],
            max_samples=d["max_samples"],
            contamination=d["contamination"],
            random_state=d["random_state"],
            n_jobs=d["n_jobs"],
            max_train_rows=d.get("max_train_rows"),
        )
        s.model = d["model"]
        s.feature_columns = d["feature_columns"]
        s.diag = IsolationForestFitDiagnostics(**{
            k: v for k, v in d.get("diag", {}).items()
            if k in IsolationForestFitDiagnostics.__dataclass_fields__
        })
        return s
