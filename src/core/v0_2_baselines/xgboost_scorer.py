"""v0.2 XGBoost baseline — per-event supervised gradient boosting on
the f_* feature columns.

This is the supervised half of the bake-off: a classifier fit directly
to ``is_attack`` labels derived from labels.csv interval-join. Classical
imbalanced-classification setup with class-weighted loss.

Differs from ``src/core/v01_baselines/xgboost_model.py`` in two ways
documented in
``docs/releases/v0.2-course-milestone/v0.2_baseline_bake_off_design.md``
§3.2:

  1. Per-event, not window-level. v0.1 fed 29-d window aggregates and
     broadcast scores to events; v0.2 produces a native per-event score
     via ``predict_proba`` so the bake-off comparison is apples-to-apples
     with Mamba and IF.
  2. v0.2 ``f_*`` features (42 columns: 33 embed + 9 float, the full
     ``TABULAR_FEATURE_COLUMNS`` from ``shared.py``) instead of v0.1
     windowed aggregates.

**Degenerate mode.** If labeled train has zero positive (or zero
negative) events, XGBClassifier.fit raises on a single-class target.
The scorer detects this, logs a warning, and switches to a constant-0.5
score for every event. The eval payload's ``extra`` field records the
degenerate flag so the report can explain why XGBoost failed if the
v0.2 train→test time split puts all ART intervals on the test side.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None  # type: ignore
    _XGBOOST_AVAILABLE = False


@dataclass
class XGBoostFitDiagnostics:
    """Reportable diagnostics from a fit. Logged to train.log + Aim."""
    n_train_events: int = 0
    n_train_positives: int = 0
    n_train_negatives: int = 0
    n_val_events: int = 0
    n_val_positives: int = 0
    scale_pos_weight: float = 1.0
    best_iteration: int = -1
    best_val_metric: float = float("nan")
    train_metric_history: list[float] = field(default_factory=list)
    val_metric_history: list[float] = field(default_factory=list)
    degenerate: bool = False
    degenerate_reason: str = ""

    def as_dict(self) -> dict:
        return {
            "n_train_events":         self.n_train_events,
            "n_train_positives":      self.n_train_positives,
            "n_train_negatives":      self.n_train_negatives,
            "n_val_events":           self.n_val_events,
            "n_val_positives":        self.n_val_positives,
            "scale_pos_weight":       self.scale_pos_weight,
            "best_iteration":         self.best_iteration,
            "best_val_metric":        self.best_val_metric,
            "n_iterations_used":      len(self.train_metric_history),
            "degenerate":             self.degenerate,
            "degenerate_reason":      self.degenerate_reason,
        }


class XGBoostScorer:
    """Supervised binary classifier on per-event tabular features."""

    DEFAULT_HPARAMS: dict = {
        "n_estimators":       500,
        "max_depth":          6,
        "learning_rate":      0.05,
        "subsample":          0.8,
        "colsample_bytree":   0.8,
        "tree_method":        "hist",
        "objective":          "binary:logistic",
        "eval_metric":        "auc",
        "verbosity":          0,
    }

    def __init__(
        self,
        *,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 25,
        device: str = "cpu",  # "cpu" or "cuda"
        random_state: int = 0,
    ) -> None:
        if not _XGBOOST_AVAILABLE:
            raise ImportError(
                "xgboost is required for XGBoostScorer; install via "
                "`pip install xgboost`"
            )
        self.hparams = dict(self.DEFAULT_HPARAMS)
        self.hparams.update({
            "n_estimators":     n_estimators,
            "max_depth":        max_depth,
            "learning_rate":    learning_rate,
            "subsample":        subsample,
            "colsample_bytree": colsample_bytree,
            "random_state":     random_state,
        })
        # Device handling: pass "device" to XGBClassifier for GPU; keep
        # "cpu" off to avoid a no-op kwarg in older xgboost versions.
        if device == "cuda":
            self.hparams["device"] = "cuda"
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.model: XGBClassifier | None = None
        self.feature_columns: list[str] | None = None
        self._degenerate: bool = False
        self.diag: XGBoostFitDiagnostics = XGBoostFitDiagnostics()

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        val_fraction: float = 0.20,
        feature_columns: list[str] | None = None,
    ) -> XGBoostFitDiagnostics:
        """Train on (X, y), with the last ``val_fraction`` of the
        time-sorted matrix held out for early stopping.

        Caller is responsible for sorting X + y by event_time before
        passing them in (``shared.load_tabular_features`` does so).
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2-d, got shape {X.shape}")
        if y.shape != (X.shape[0],):
            raise ValueError(f"y shape {y.shape} != X.shape[0]={X.shape[0]}")

        n = X.shape[0]
        n_val = max(1, int(n * val_fraction))
        n_train = n - n_val
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]

        n_pos_train = int(y_train.sum())
        n_neg_train = int((~y_train.astype(bool)).sum())
        n_pos_val = int(y_val.sum())

        # Degenerate-mode guard: XGBClassifier requires both classes in y.
        if n_pos_train == 0 or n_neg_train == 0:
            reason = (
                "train has zero positive events"
                if n_pos_train == 0
                else "train has zero negative events"
            )
            self._degenerate = True
            self.feature_columns = list(feature_columns or [])
            self.diag = XGBoostFitDiagnostics(
                n_train_events=n_train,
                n_train_positives=n_pos_train,
                n_train_negatives=n_neg_train,
                n_val_events=n_val,
                n_val_positives=n_pos_val,
                scale_pos_weight=1.0,
                degenerate=True,
                degenerate_reason=reason,
            )
            return self.diag

        # Class imbalance handling (negative-to-positive ratio).
        scale_pos_weight = float(n_neg_train) / float(max(n_pos_train, 1))

        clf = XGBClassifier(
            **self.hparams,
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        # eval_set must be a list of (X, y) tuples; first entry tracked as
        # training metric, last as validation.
        clf.fit(
            X_train, y_train.astype(np.int32),
            eval_set=[(X_train, y_train.astype(np.int32)),
                      (X_val,   y_val.astype(np.int32))],
            verbose=False,
        )

        self.model = clf
        self.feature_columns = list(feature_columns or [])
        self._degenerate = False

        # Persist training history for the report. evals_result_ is
        # ``{"validation_0": {"auc": [...]}, "validation_1": {"auc": [...]}}``.
        evals = clf.evals_result_
        keys = list(evals.keys())
        train_hist = evals[keys[0]].get("auc", []) if len(keys) >= 1 else []
        val_hist = evals[keys[1]].get("auc", []) if len(keys) >= 2 else []
        best_iter = int(getattr(clf, "best_iteration", len(val_hist) - 1))
        best_val = float(val_hist[best_iter]) if val_hist else float("nan")

        self.diag = XGBoostFitDiagnostics(
            n_train_events=n_train,
            n_train_positives=n_pos_train,
            n_train_negatives=n_neg_train,
            n_val_events=n_val,
            n_val_positives=n_pos_val,
            scale_pos_weight=scale_pos_weight,
            best_iteration=best_iter,
            best_val_metric=best_val,
            train_metric_history=[float(v) for v in train_hist],
            val_metric_history=[float(v) for v in val_hist],
            degenerate=False,
        )
        return self.diag

    # ----------------------------------------------------------------- score

    def score_events(self, X: np.ndarray) -> np.ndarray:
        """Return per-event P(attack) probabilities."""
        if self._degenerate:
            return np.full(X.shape[0], 0.5, dtype=np.float32)
        if self.model is None:
            raise RuntimeError("XGBoostScorer.fit() must be called before score_events()")
        proba = self.model.predict_proba(X)[:, 1]
        return proba.astype(np.float32, copy=False)

    # ------------------------------------------------------------------ I/O

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({
                "hparams":               self.hparams,
                "early_stopping_rounds": self.early_stopping_rounds,
                "random_state":          self.random_state,
                "model":                 self.model,
                "feature_columns":       self.feature_columns,
                "degenerate":            self._degenerate,
                "diag":                  self.diag.as_dict(),
            }, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> XGBoostScorer:
        with Path(path).open("rb") as fh:
            d = pickle.load(fh)
        s = cls(
            n_estimators=d["hparams"].get("n_estimators", 500),
            max_depth=d["hparams"].get("max_depth", 6),
            learning_rate=d["hparams"].get("learning_rate", 0.05),
            subsample=d["hparams"].get("subsample", 0.8),
            colsample_bytree=d["hparams"].get("colsample_bytree", 0.8),
            early_stopping_rounds=d["early_stopping_rounds"],
            device="cuda" if d["hparams"].get("device") == "cuda" else "cpu",
            random_state=d["random_state"],
        )
        s.hparams = d["hparams"]
        s.model = d["model"]
        s.feature_columns = d["feature_columns"]
        s._degenerate = d["degenerate"]
        diag_dict = d.get("diag", {})
        # Drop derived keys not in the dataclass definition.
        diag_dict.pop("n_iterations_used", None)
        s.diag = XGBoostFitDiagnostics(**{
            k: v for k, v in diag_dict.items()
            if k in XGBoostFitDiagnostics.__dataclass_fields__
        })
        return s
