"""v0.2 Mamba scorer.

Two scorer variants share the v0.1 Mamba body and the v0.2 event encoder;
they differ only in head + loss + scoring rule:

  * NLL: head = ``Linear(d_model, ACTION_FAMILY_CARDINALITY)``; loss is
    cross-entropy at each position predicting the next position's
    ``f_action_family`` index. Anomaly score = per-event NLL; window
    score = max NLL across positions.

  * MEM: head = ``Linear(d_model, d_model)`` predicting the per-event
    encoder embedding at masked positions. Target is the encoder output
    **detached** before loss compute so the encoder is not gradient-
    updated to make masked positions trivially predictable. Anomaly
    score = per-event reconstruction MSE; window score = max across
    positions.

Both variants stack ``n_layers`` MambaBlocks with pre-norm residual:

    x = x + block(norm(x))

The v0.1 body was tuned at L=128 / d_model=64; Plan E v2 keeps L=128 and
bumps d_model to 128 to accommodate the larger encoder concat dim.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.mamba_block import MambaBlock
from src.core.v0_2_event_encoder import V02EventEncoder
from src.features.v0_2_features import KPROBE_FUNCTION_OOV_INDEX
from src.processing.v0_2_behavior_builder import (
    ACTION_FAMILY_CARDINALITY,
    DST_PORT_CATEGORY_CARDINALITY,
    OBJECT_CATEGORY_CARDINALITY,
    PATH_CATEGORY_CARDINALITY,
)


# 3a.1 — field-aware categorical MEM heads. CE on each named categorical
# field, BCE on each named binary flag. The encoder embedding for each
# masked position is replaced with the [MASK] token before the body, so
# the body cannot peek at the field values; each head must reconstruct
# its field from context. Per-event loss = uniform mean across heads.
_MEM_FA_CE_FIELDS: dict[str, int] = {
    "f_action_family":     ACTION_FAMILY_CARDINALITY,        # 15
    "f_kprobe_function":   KPROBE_FUNCTION_OOV_INDEX + 1,    # 14
    "f_path_category":     PATH_CATEGORY_CARDINALITY,        # 11
    "f_dst_port_category": DST_PORT_CATEGORY_CARDINALITY,    # 7
    "f_object_category":   OBJECT_CATEGORY_CARDINALITY,      # 8
}
_MEM_FA_BCE_FIELDS: tuple[str, ...] = (
    "f_uid_eq_parent",
    "f_kp_commit_creds_uid_change",
    "f_kp_commit_creds_cap_change",
    "f_is_setuid_exec",
)
MEM_FA_NUM_HEADS: int = len(_MEM_FA_CE_FIELDS) + len(_MEM_FA_BCE_FIELDS)  # 9


class MambaBody(nn.Module):
    """Stacked MambaBlocks with pre-norm residual + final LayerNorm.

    Same shape as v0.1's MambaScorer body; just renamed so it can be
    reused for both heads without duplicating the SSM.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model=d_model, d_state=d_state) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        for norm, block in zip(self.norms, self.blocks):
            x = x + block(norm(x))
        return self.dropout(self.final_norm(x))


class MambaNLLScorer(nn.Module):
    """Next-event NLL over ``f_action_family`` (15 classes incl. OOV).

    forward(features, target_id) -> (B, L) per-event NLL with the last
    position padded with zero so the output length matches the input.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        n_layers: int = 2,
        dropout: float = 0.1,
        *,
        feature_set: str = "rich",
    ) -> None:
        super().__init__()
        self.feature_set = feature_set
        self.encoder = V02EventEncoder(d_model=d_model, feature_set=feature_set)
        self.body = MambaBody(d_model=d_model, d_state=d_state, n_layers=n_layers, dropout=dropout)
        self.head = nn.Linear(d_model, ACTION_FAMILY_CARDINALITY)
        self.vocab_size = ACTION_FAMILY_CARDINALITY

    def forward(
        self,
        features: dict[str, torch.Tensor],
        target_id: torch.Tensor,
        *,
        return_logits: bool = False,
    ):
        # encoder: (B, L, d_model)
        x = self.encoder(features)
        h = self.body(x)
        logits = self.head(h)                        # (B, L, V)
        # Predict t+1 from t: logits at position t -> targets at position t+1.
        shifted_logits = logits[:, :-1, :]            # (B, L-1, V)
        shifted_targets = target_id[:, 1:]            # (B, L-1)
        nll = F.cross_entropy(
            shifted_logits.reshape(-1, self.vocab_size),
            shifted_targets.reshape(-1),
            reduction="none",
        ).reshape(shifted_targets.shape)              # (B, L-1)
        # Per-event alignment: nll[i] is the loss for predicting target[i+1]
        # from logits[i], so it is event (i+1)'s surprise score. Pad at
        # the FRONT so nll_padded[i] is event i's surprise score (with
        # event 0 having no predecessor and therefore zero). This makes
        # nll_padded join correctly to the row-aligned event_time when
        # downstream eval pulls per-event scores out of a window.
        pad = torch.zeros(nll.shape[0], 1, device=nll.device, dtype=nll.dtype)
        nll_padded = torch.cat([pad, nll], dim=1)     # (B, L)
        if return_logits:
            return nll_padded, shifted_logits, shifted_targets
        return nll_padded

    def loss(self, features: dict[str, torch.Tensor], target_id: torch.Tensor) -> torch.Tensor:
        """Mean per-event NLL across all (non-padded) positions."""
        per_event = self.forward(features, target_id)
        # Position 0 carries the front-pad zero (no predecessor for the
        # first event); only L-1 real positions contribute to the mean.
        return per_event[:, 1:].mean()

    def window_score(
        self,
        features: dict[str, torch.Tensor],
        target_id: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(features, target_id).max(dim=1).values


class MambaMEMScorer(nn.Module):
    """Masked event modeling over per-event encoder embeddings.

    Mask injection: at masked positions, the encoder's output is replaced
    with a learned ``[MASK]`` embedding before being fed to the body.
    The target is the encoder output BEFORE masking, ``.detach()``'d.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        n_layers: int = 2,
        dropout: float = 0.1,
        *,
        feature_set: str = "rich",
    ) -> None:
        super().__init__()
        self.feature_set = feature_set
        self.encoder = V02EventEncoder(d_model=d_model, feature_set=feature_set)
        self.body = MambaBody(d_model=d_model, d_state=d_state, n_layers=n_layers, dropout=dropout)
        self.head = nn.Linear(d_model, d_model)
        # Learned [MASK] token; broadcast to (B, L, d_model) at masked positions.
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        self.d_model = d_model

    def _encode_with_mask(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (target, masked_input).

        target: encoder output detached (no gradient flows into the encoder
                via the target path — prevents collapse).
        masked_input: same as target but with mask_token at masked positions.
        """
        clean = self.encoder(features)                # (B, L, d_model)
        target = clean.detach()
        # Broadcast mask_token over batch + length where mask is True.
        mask_expanded = mask.unsqueeze(-1).to(clean.dtype)         # (B, L, 1)
        mask_token = self.mask_token.view(1, 1, -1).expand_as(clean)
        masked_input = clean * (1 - mask_expanded) + mask_token * mask_expanded
        return target, masked_input

    def forward(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns per-event reconstruction MSE (B, L). Non-masked positions
        get zero loss. Used for both training (mean over masked) and
        scoring (max over positions)."""
        target, masked_input = self._encode_with_mask(features, mask)
        h = self.body(masked_input)
        pred = self.head(h)                            # (B, L, d_model)
        per_event_se = ((pred - target) ** 2).mean(dim=-1)  # (B, L)
        # Zero out non-masked positions so window-scoring sees only masked
        # reconstruction errors.
        return per_event_se * mask.to(per_event_se.dtype)

    def loss(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean MSE over masked positions only."""
        per_event = self.forward(features, mask)
        n_masked = mask.sum().clamp_min(1).to(per_event.dtype)
        return per_event.sum() / n_masked

    def window_score(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(features, mask).max(dim=1).values

    def score_all_positions(
        self,
        features: dict[str, torch.Tensor],
        chunk_size: int = 16,
    ) -> torch.Tensor:
        """Deterministic eval-time per-event scoring with INTERLEAVED masks.

        Random 15% mask training is correct for *training* but wrong for
        *evaluation*: at eval, an anomalous event that happened not to be
        sampled into the 15% mask would receive zero reconstruction error
        and be silently missed.

        This method reconstructs every position EXACTLY ONCE across N
        passes, where N = ceil(L / chunk_size) (default 8 at L=128). Each
        pass c masks every Nth position starting from offset c — i.e.,
        positions c, c+N, c+2N, .... So every position is masked in
        exactly one pass (the pass with c == position mod N).

        Why interleaved instead of contiguous chunks (the original v0.1-style
        eval): because the body is causal Mamba, contiguous-chunk masking
        means a position late in a chunk has its entire local prior context
        masked out. E.g., if positions 32–47 are masked together, position
        47 can only attend to positions 0–31 plus 16 [MASK] tokens; the
        SSM's hidden state at position 47 is dominated by the [MASK]
        propagation, which is out-of-distribution relative to training's
        random 15% mask. Interleaved masks preserve ~7/8 of the local
        context unmasked, keeping eval inputs in-distribution.

        Mask density per pass: L/N positions masked = ~12.5% at L=128,
        N=8. Same total per-pass mask count as the old contiguous scheme
        (16 positions); only the layout changes.

        NOT for training. Use ``loss(features, random_mask)`` for that.

        Args:
            features: per-column tensors of shape (B, L). Same dict the
                training forward consumes.
            chunk_size: positions masked per forward pass (interleaved).
                Default 16. Clamped to [1, L]. Number of passes is
                ``ceil(L / chunk_size)``.

        Returns:
            (B, L) per-event reconstruction MSE. Every position is the
            output of one forward pass that masked it.
        """
        sample = next(iter(features.values()))
        B, L = sample.shape[0], sample.shape[1]
        device = sample.device

        chunk_size = max(1, min(chunk_size, L))
        n_chunks = (L + chunk_size - 1) // chunk_size
        output = torch.zeros(B, L, device=device, dtype=torch.float32)
        for c in range(n_chunks):
            mask = torch.zeros(B, L, dtype=torch.bool, device=device)
            mask[:, c::n_chunks] = True   # interleaved: c, c+N, c+2N, ...
            per_event_se = self.forward(features, mask)  # zero outside mask
            # Each position is True in exactly one pass's mask, so the sum
            # equals "the SE from the pass that masked this position".
            output = output + per_event_se.to(output.dtype)
        return output


class MambaFieldAwareMEMScorer(nn.Module):
    """3a.1 — field-aware categorical MEM.

    Replaces vanilla MEM's monolithic ``Linear(d_model, d_model)`` MSE
    head with **per-field** heads:

      * 5 cross-entropy heads on categorical fields:
          ``f_action_family``       (15 classes)
          ``f_kprobe_function``     (14 classes)
          ``f_path_category``       (11 classes)
          ``f_dst_port_category``   (7 classes)
          ``f_object_category``     (8 classes)

      * 4 binary cross-entropy heads on flag fields:
          ``f_uid_eq_parent``
          ``f_kp_commit_creds_uid_change``
          ``f_kp_commit_creds_cap_change``
          ``f_is_setuid_exec``

    Per-event loss = ``(1/N_heads) * sum(field_losses)`` (uniform mean
    across all 9 heads). Anomaly score at masked positions = same
    weighted sum.

    Targets §11.3 Cause 3 of vanilla MEM (post-projection target lets
    the bottleneck route around content). With per-field categorical
    heads, sparse security markers like priv-change get explicit
    cross-entropy penalty rather than being averaged into a 128-d MSE
    where they are dominated by high-norm hash dimensions.

    Mask injection mirrors vanilla MEM exactly: at masked positions,
    the encoder output is replaced with the learned ``[MASK]`` token
    before the body. The body never sees the masked field values; each
    head reconstructs its field's value from context.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        n_layers: int = 2,
        dropout: float = 0.1,
        *,
        feature_set: str = "rich",
    ) -> None:
        super().__init__()
        self.feature_set = feature_set
        self.encoder = V02EventEncoder(d_model=d_model, feature_set=feature_set)
        self.body = MambaBody(d_model=d_model, d_state=d_state, n_layers=n_layers, dropout=dropout)
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        self.d_model = d_model
        # Per-field heads. ModuleDict is iteration-stable so loss is
        # deterministic across runs. None of the 5 CE fields and 4 BCE
        # fields are among the 7 hash columns dropped by feature_set="flat",
        # so the heads, targets, and loss math are invariant to feature_set.
        self.ce_heads = nn.ModuleDict(
            {field: nn.Linear(d_model, ncls) for field, ncls in _MEM_FA_CE_FIELDS.items()}
        )
        self.bce_heads = nn.ModuleDict(
            {field: nn.Linear(d_model, 1) for field in _MEM_FA_BCE_FIELDS}
        )
        self.n_heads = MEM_FA_NUM_HEADS

    def _encode_with_mask(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run encoder, replace masked positions with [MASK], return
        ``masked_input`` (B, L, d_model). No detached target needed —
        targets are the raw field values, not the encoder embedding."""
        clean = self.encoder(features)
        mask_expanded = mask.unsqueeze(-1).to(clean.dtype)
        mask_token = self.mask_token.view(1, 1, -1).expand_as(clean)
        return clean * (1 - mask_expanded) + mask_token * mask_expanded

    def forward(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns per-event total loss (B, L). Non-masked positions get
        zero. Loss = uniform mean across 9 per-field heads."""
        masked_input = self._encode_with_mask(features, mask)
        h = self.body(masked_input)                     # (B, L, d_model)
        B, L = h.shape[0], h.shape[1]

        total = torch.zeros(B, L, device=h.device, dtype=h.dtype)

        for field, ncls in _MEM_FA_CE_FIELDS.items():
            logits = self.ce_heads[field](h)            # (B, L, ncls)
            target = features[field].long()             # (B, L)
            ce = F.cross_entropy(
                logits.reshape(B * L, ncls),
                target.reshape(B * L),
                reduction="none",
            ).reshape(B, L)
            total = total + ce

        for field in _MEM_FA_BCE_FIELDS:
            logits = self.bce_heads[field](h).squeeze(-1)   # (B, L)
            target = features[field].to(h.dtype)            # 0/1 floats
            bce = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none",
            )
            total = total + bce

        per_event = total / float(self.n_heads)
        return per_event * mask.to(per_event.dtype)

    def loss(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean field-aware loss over masked positions only."""
        per_event = self.forward(features, mask)
        n_masked = mask.sum().clamp_min(1).to(per_event.dtype)
        return per_event.sum() / n_masked

    def window_score(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(features, mask).max(dim=1).values

    def score_all_positions(
        self,
        features: dict[str, torch.Tensor],
        chunk_size: int = 16,
    ) -> torch.Tensor:
        """Deterministic per-event scoring with INTERLEAVED masks. Same
        scheme as vanilla MEM's score_all_positions — see that docstring
        for the causal-Mamba rationale. Each position is reconstructed
        in exactly one of N=ceil(L/chunk_size) passes; pass c masks
        positions c, c+N, c+2N, ...; ~7/8 of context stays unmasked."""
        sample = next(iter(features.values()))
        B, L = sample.shape[0], sample.shape[1]
        device = sample.device
        chunk_size = max(1, min(chunk_size, L))
        n_chunks = (L + chunk_size - 1) // chunk_size
        output = torch.zeros(B, L, device=device, dtype=torch.float32)
        for c in range(n_chunks):
            mask = torch.zeros(B, L, dtype=torch.bool, device=device)
            mask[:, c::n_chunks] = True   # interleaved
            per_event = self.forward(features, mask)
            output = output + per_event.to(output.dtype)
        return output


def build_scorer(
    objective: str,
    *,
    d_model: int = 128,
    d_state: int = 16,
    n_layers: int = 2,
    dropout: float = 0.1,
    feature_set: str = "rich",
) -> nn.Module:
    """Factory: ``objective`` selects one of three scorer variants.

    Choices:
      ``"nll"``    — autoregressive next-event prediction (run #1 + #2 baseline)
      ``"mem"``    — vanilla masked event modeling with continuous-MSE
                     reconstruction (run #1 negative result; kept for paper §11
                     reference, not recommended for new training)
      ``"mem-fa"`` — field-aware categorical MEM (3a.1; per-field CE + BCE
                     heads). Targets vanilla MEM's collapse mode.

    ``feature_set`` (``"rich"`` default; ``"flat"`` drops 7 hash columns)
    is forwarded to the underlying ``V02EventEncoder``. Existing callers
    that omit it get the rich encoder unchanged.
    """
    if objective == "nll":
        return MambaNLLScorer(
            d_model=d_model, d_state=d_state, n_layers=n_layers, dropout=dropout,
            feature_set=feature_set,
        )
    if objective == "mem":
        return MambaMEMScorer(
            d_model=d_model, d_state=d_state, n_layers=n_layers, dropout=dropout,
            feature_set=feature_set,
        )
    if objective == "mem-fa":
        return MambaFieldAwareMEMScorer(
            d_model=d_model, d_state=d_state, n_layers=n_layers, dropout=dropout,
            feature_set=feature_set,
        )
    raise ValueError(f"objective must be 'nll', 'mem', or 'mem-fa', got {objective!r}")


__all__ = [
    "MambaBody",
    "MambaNLLScorer",
    "MambaMEMScorer",
    "MambaFieldAwareMEMScorer",
    "build_scorer",
]
