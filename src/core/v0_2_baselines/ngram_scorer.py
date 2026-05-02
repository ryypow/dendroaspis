"""v0.2 n-gram baseline — trigram language model on the joint ``token``
column with Laplace smoothing + stupid backoff.

The scorer is trained on **benign-only** events (events whose timestamp
falls outside any ART label interval). At score time, every event in
``test.parquet`` gets a per-event ``-log P(token | prev2, prev1)``
estimate, with stupid backoff: trigram if the (prev2, prev1) prefix was
ever seen in training, else bigram if (prev1) was seen, else unigram.
Higher score = lower likelihood under benign = more anomalous.

Sequence boundary = ``proc_exec_id``. The first event of each process
has no prior context (unigram fallback); the second event has only
``prev1`` (bigram fallback). This mirrors how Mamba's encoder + Mamba
body see a per-process lineage-walked sequence — n-gram is genuinely
"same context, less capacity," not "different context."

Adapted from ``src/core/v01_baselines/ngram_model.py`` but rewritten to
read the v0.2 ``token`` column directly (the v0.1 implementation
recomputed a (action × object_type) joint id from event objects).

Hyperparameters committed in
``docs/releases/v0.2-course-milestone/v0.2_baseline_bake_off_design.md``
§3.1: n=3, Laplace alpha=1.0, stupid-backoff factor 0.4.
"""

from __future__ import annotations

import math
import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class NgramFitDiagnostics:
    """Reportable diagnostics from a fit. Logged to train.log + Aim."""
    n_train_events: int = 0
    n_processes: int = 0
    vocab_size: int = 0
    n_unique_trigrams: int = 0
    n_unique_bigrams: int = 0
    n_trigram_positions: int = 0  # events that contributed a trigram count
    n_bigram_positions: int = 0
    train_perplexity: float = float("nan")
    backoff_rates: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_train_events":      self.n_train_events,
            "n_processes":         self.n_processes,
            "vocab_size":          self.vocab_size,
            "n_unique_trigrams":   self.n_unique_trigrams,
            "n_unique_bigrams":    self.n_unique_bigrams,
            "n_trigram_positions": self.n_trigram_positions,
            "n_bigram_positions":  self.n_bigram_positions,
            "train_perplexity":    self.train_perplexity,
            "backoff_rates":       self.backoff_rates,
        }


def _factorize_object_array(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map a 1-d object array of strings to integer codes.

    Returns ``(codes, uniques)`` like pandas.factorize but without the
    pandas dep. Stable: codes[i] = index of arr[i] in uniques.
    """
    uniques, inv = np.unique(arr, return_inverse=True)
    return inv.astype(np.int64), uniques


def _group_starts_from_codes(codes: np.ndarray) -> np.ndarray:
    """Indices of group boundaries in a code array sorted by code.

    Returns the starting index of every contiguous run.
    """
    if codes.size == 0:
        return np.zeros(0, dtype=np.int64)
    diff = np.diff(codes)
    starts = np.r_[0, np.flatnonzero(diff != 0) + 1]
    return starts.astype(np.int64)


def _per_position_group_offset(
    n: int, group_starts: np.ndarray
) -> np.ndarray:
    """Position-within-group offset (0 for first event of process)."""
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    group_id = np.zeros(n, dtype=np.int64)
    group_id[group_starts] = 1
    group_id = np.cumsum(group_id) - 1
    return np.arange(n, dtype=np.int64) - group_starts[group_id]


class NgramScorer:
    """Trigram language model with Laplace + stupid backoff.

    Train: ``fit(event_time, proc_exec_id, token, is_attack)``. Only
      benign events (where ``is_attack`` is False) are counted.
    Score: ``score_events(event_time, proc_exec_id, token)``. Returns a
      per-event ``-log P`` array (higher = more anomalous).
    """

    # Sentinel int code returned by ``_encode_tokens`` for token strings not
    # seen in the training vocabulary. Stored under a single bucket so the
    # n-gram backoff falls through to the unigram smoothed prior for all
    # OOV strings (count = 0 in unigram → P = alpha / (n_total + alpha * V)
    # by Laplace).
    OOV_CODE: int = -1

    def __init__(
        self,
        *,
        n: int = 3,
        alpha: float = 1.0,
        backoff: float = 0.4,
    ) -> None:
        if n != 3:
            raise ValueError(
                f"only n=3 is supported (spec §3.1); got n={n}"
            )
        self.n = n
        self.alpha = alpha
        self.backoff = backoff

        # Fitted state.
        self._unigram: dict[int, int] = {}
        self._n_total: int = 0
        self._bigram_count: dict[tuple[int, int], int] = {}
        self._bigram_total: dict[int, int] = {}
        self._trigram_count: dict[tuple[int, int, int], int] = {}
        self._trigram_total: dict[tuple[int, int], int] = {}
        self._vocab_size: int = 0
        # token string -> int code, stable across fit/score so train + test
        # factorizations share codes. OOV strings at score time map to OOV_CODE.
        self._vocab: dict = {}
        self._fitted: bool = False
        self.diag: NgramFitDiagnostics = NgramFitDiagnostics()

    def _build_vocab(self, token: np.ndarray) -> None:
        """Build the string->int vocab from training tokens. Sorted-string
        order so the codes are deterministic across runs."""
        uniq = sorted({str(t) for t in token.tolist()})
        self._vocab = {s: i for i, s in enumerate(uniq)}

    def _encode_tokens(self, token: np.ndarray) -> np.ndarray:
        """Map an object array of token strings to int64 codes, using
        ``OOV_CODE`` for unseen strings."""
        if self._vocab:
            v = self._vocab
            oov = self.OOV_CODE
            return np.fromiter(
                (v.get(str(t), oov) for t in token.tolist()),
                dtype=np.int64,
                count=token.shape[0],
            )
        # No vocab yet (during fit); build directly.
        if token.dtype != object:
            return token.astype(np.int64, copy=False)
        # token is object dtype, vocab not yet built.
        raise RuntimeError("_encode_tokens called before fit / _build_vocab")

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        event_time: np.ndarray,
        proc_exec_id: np.ndarray,
        token: np.ndarray,
        is_attack: np.ndarray,
    ) -> NgramFitDiagnostics:
        """Build n-gram count tables on benign-only events."""
        if event_time.shape != proc_exec_id.shape or event_time.shape != token.shape:
            raise ValueError(
                f"shape mismatch: event_time={event_time.shape} "
                f"proc_exec_id={proc_exec_id.shape} token={token.shape}"
            )
        if is_attack.shape != event_time.shape:
            raise ValueError(
                f"is_attack shape {is_attack.shape} != event_time shape {event_time.shape}"
            )

        # Benign-only filter.
        benign = ~is_attack.astype(bool)
        et = event_time[benign]
        pid = proc_exec_id[benign]
        tk_raw = token[benign]

        # Build a stable string->code vocab on benign training tokens.
        # Token may already be int (test fixtures) or object/string (real
        # parquet); the vocab is bypassed for int input.
        if tk_raw.dtype == object:
            self._build_vocab(tk_raw)
            tk = self._encode_tokens(tk_raw)
        else:
            tk = tk_raw.astype(np.int64, copy=False)
            # No vocab needed; self._vocab stays empty and _encode_tokens
            # treats incoming arrays as already-encoded ints.
            self._vocab = {}

        # Sort by (proc_exec_id, event_time). proc_exec_id is object dtype;
        # factorize to int code so np.lexsort works without object overhead.
        pid_codes, _uniques = _factorize_object_array(pid)
        order = np.lexsort((et, pid_codes))
        pid_codes = pid_codes[order]
        tk = tk[order]
        n = tk.shape[0]

        # Group boundaries + per-position offset within group.
        group_starts = _group_starts_from_codes(pid_codes)
        offset = _per_position_group_offset(n, group_starts)

        # Vectorized counting via np.unique on tuples.
        unigram_keys, unigram_counts = np.unique(tk, return_counts=True)
        unigram = {int(k): int(c) for k, c in zip(unigram_keys, unigram_counts)}

        # Bigrams: positions with offset >= 1.
        bi_pos = np.flatnonzero(offset >= 1)
        bi_total_positions = int(bi_pos.size)
        bigram_count: dict[tuple[int, int], int] = {}
        bigram_total: dict[int, int] = {}
        if bi_pos.size:
            bi_pairs = np.column_stack((tk[bi_pos - 1], tk[bi_pos]))
            uniq_bi, cnt_bi = np.unique(bi_pairs, axis=0, return_counts=True)
            for (p1, w), c in zip(uniq_bi, cnt_bi):
                bigram_count[(int(p1), int(w))] = int(c)
                bigram_total[int(p1)] = bigram_total.get(int(p1), 0) + int(c)

        # Trigrams: positions with offset >= 2.
        tri_pos = np.flatnonzero(offset >= 2)
        tri_total_positions = int(tri_pos.size)
        trigram_count: dict[tuple[int, int, int], int] = {}
        trigram_total: dict[tuple[int, int], int] = {}
        if tri_pos.size:
            tri_triples = np.column_stack(
                (tk[tri_pos - 2], tk[tri_pos - 1], tk[tri_pos])
            )
            uniq_tri, cnt_tri = np.unique(tri_triples, axis=0, return_counts=True)
            for (p2, p1, w), c in zip(uniq_tri, cnt_tri):
                trigram_count[(int(p2), int(p1), int(w))] = int(c)
                key = (int(p2), int(p1))
                trigram_total[key] = trigram_total.get(key, 0) + int(c)

        # Persist.
        self._unigram = unigram
        self._n_total = int(tk.shape[0])
        self._bigram_count = bigram_count
        self._bigram_total = bigram_total
        self._trigram_count = trigram_count
        self._trigram_total = trigram_total
        self._vocab_size = len(unigram)
        self._fitted = True

        # Train-perplexity sanity + backoff-rate diagnostics.
        if n > 0:
            train_logp_sum = 0.0
            backoff_counts = {"trigram": 0, "bigram": 0, "unigram": 0}
            # Sample at most 100k events to keep the diag pass cheap.
            sample_n = min(100_000, n)
            sample_idx = np.linspace(0, n - 1, sample_n, dtype=np.int64)
            for i in sample_idx:
                w = int(tk[i])
                off = int(offset[i])
                p1 = int(tk[i - 1]) if off >= 1 else None
                p2 = int(tk[i - 2]) if off >= 2 else None
                neg_logp, level = self._score_one(p2, p1, w, return_level=True)
                train_logp_sum += neg_logp
                backoff_counts[level] += 1
            mean_logp = train_logp_sum / sample_n
            train_ppl = math.exp(min(mean_logp, 50))
            backoff_rates = {k: v / sample_n for k, v in backoff_counts.items()}
        else:
            train_ppl = float("nan")
            backoff_rates = {}

        # Report unique-process count from sorted code array.
        n_processes = int(group_starts.size)

        self.diag = NgramFitDiagnostics(
            n_train_events=n,
            n_processes=n_processes,
            vocab_size=self._vocab_size,
            n_unique_trigrams=len(trigram_count),
            n_unique_bigrams=len(bigram_count),
            n_trigram_positions=tri_total_positions,
            n_bigram_positions=bi_total_positions,
            train_perplexity=train_ppl,
            backoff_rates=backoff_rates,
        )
        return self.diag

    # ----------------------------------------------------------------- score

    def _score_one(
        self,
        prev2: int | None,
        prev1: int | None,
        w: int,
        *,
        return_level: bool = False,
    ) -> float | tuple[float, str]:
        """Compute -log P(w | prev2, prev1) under stupid backoff."""
        V = self._vocab_size
        alpha = self.alpha
        beta = self.backoff
        if prev2 is not None and prev1 is not None:
            t_total = self._trigram_total.get((prev2, prev1), 0)
            if t_total > 0:
                t_count = self._trigram_count.get((prev2, prev1, w), 0)
                P = (t_count + alpha) / (t_total + alpha * V)
                neg_logp = -math.log(P)
                return (neg_logp, "trigram") if return_level else neg_logp
        if prev1 is not None:
            b_total = self._bigram_total.get(prev1, 0)
            if b_total > 0:
                b_count = self._bigram_count.get((prev1, w), 0)
                P = beta * (b_count + alpha) / (b_total + alpha * V)
                neg_logp = -math.log(P)
                return (neg_logp, "bigram") if return_level else neg_logp
        # Unigram fallback. ``prev1 is None`` means start-of-process; use one
        # backoff factor. ``prev1 is not None`` (came here because trigram +
        # bigram both empty) means two backoffs; multiply by beta^2.
        if prev1 is None and prev2 is None:
            backoff_factor = 1.0
        elif prev1 is None:
            backoff_factor = beta
        else:
            backoff_factor = beta * beta
        u_count = self._unigram.get(w, 0)
        denom = self._n_total + alpha * max(V, 1)
        P = backoff_factor * (u_count + alpha) / denom
        neg_logp = -math.log(P)
        return (neg_logp, "unigram") if return_level else neg_logp

    def score_events(
        self,
        event_time: np.ndarray,
        proc_exec_id: np.ndarray,
        token: np.ndarray,
    ) -> np.ndarray:
        """Per-event -log P scores in original event order.

        Score array is parallel to the input arrays — caller does not need
        to know the internal sort order.
        """
        if not self._fitted:
            raise RuntimeError("NgramScorer.fit() must be called before score_events()")
        if event_time.shape != proc_exec_id.shape or event_time.shape != token.shape:
            raise ValueError(
                f"shape mismatch: event_time={event_time.shape} "
                f"proc_exec_id={proc_exec_id.shape} token={token.shape}"
            )

        n = event_time.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.float32)

        # Encode tokens through the fitted vocab (string -> int code, with
        # OOV_CODE for unseen strings). For int-typed input (unit tests),
        # the vocab is empty and tokens pass through unchanged.
        if token.dtype == object and self._vocab:
            token_int = self._encode_tokens(token)
        else:
            token_int = token.astype(np.int64, copy=False)

        pid_codes, _ = _factorize_object_array(proc_exec_id)
        order = np.lexsort((event_time, pid_codes))
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(n, dtype=np.int64)

        pid_sorted = pid_codes[order]
        tk_sorted = token_int[order]

        group_starts = _group_starts_from_codes(pid_sorted)
        offset = _per_position_group_offset(n, group_starts)

        sorted_scores = np.empty(n, dtype=np.float32)
        for i in range(n):
            w = int(tk_sorted[i])
            off = int(offset[i])
            p1 = int(tk_sorted[i - 1]) if off >= 1 else None
            p2 = int(tk_sorted[i - 2]) if off >= 2 else None
            sorted_scores[i] = self._score_one(p2, p1, w)
        return sorted_scores[inv_order]

    # ------------------------------------------------------------------ I/O

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({
                "n":            self.n,
                "alpha":        self.alpha,
                "backoff":      self.backoff,
                "unigram":      self._unigram,
                "n_total":      self._n_total,
                "bigram_count": self._bigram_count,
                "bigram_total": self._bigram_total,
                "trigram_count": self._trigram_count,
                "trigram_total": self._trigram_total,
                "vocab_size":   self._vocab_size,
                "vocab":        self._vocab,
                "diag":         self.diag.as_dict(),
            }, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> NgramScorer:
        with Path(path).open("rb") as fh:
            d = pickle.load(fh)
        s = cls(n=d["n"], alpha=d["alpha"], backoff=d["backoff"])
        s._unigram = d["unigram"]
        s._n_total = d["n_total"]
        s._bigram_count = d["bigram_count"]
        s._bigram_total = d["bigram_total"]
        s._trigram_count = d["trigram_count"]
        s._trigram_total = d["trigram_total"]
        s._vocab_size = d["vocab_size"]
        s._vocab = d.get("vocab", {})
        s._fitted = True
        s.diag = NgramFitDiagnostics(**{
            k: v for k, v in d.get("diag", {}).items() if k in NgramFitDiagnostics.__dataclass_fields__
        })
        return s
