"""Core model, dataset, collator, and evaluation utilities for the Belief World Model.

Architecture (supports two belief backends)
============================================

**GRU mode** (belief_type="gru", original):
    slots → sequential GRU rollout → beliefs + priors

**JEPA mode** (belief_type="jepa", Plan C):
    slots → Conformer+DAAM online encoder  → beliefs  (bidirectional)
    slots → CausalConformer prior encoder  → priors   (unidirectional)
    masked slots → online encoder + predictor → z_pred (for JEPA loss)
    full slots   → EMA target encoder        → z_target (stop-gradient)

Read-out heads are shared across both modes:
    - frame-level phone CTC  (evidence + belief context → realized phones)
    - slot-level canonical CTC (upsampled beliefs → canonical phones)
    - slot-level evidence CTC (upsampled slots → realized phones)
    - future prediction / JEPA prediction
    - reconstruction
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from wm_common import Vocabulary, levenshtein_distance, load_jsonl, read_json
from wm_jepa import (
    JEPAConfig,
    ConformerDAAMEncoder,
    CausalConformerEncoder,
    JEPAPredictor,
    block_mask_slots,
    compute_jepa_loss,
    update_ema,
    check_collapse,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class BeliefWMConfig:
    evidence_dim: int = 46
    hidden_dim: int = 256
    phone_vocab_size: int = 90
    belief_type: str = "gru"  # gru | jepa
    pooling_type: str = "mean"  # mean | attention
    upsample_factor: int = 4
    dropout: float = 0.1
    # --- Identity-aware modulation (proposal §5.5) ---
    use_identity: bool = False
    identity_dim: int = 128
    # --- Prosody-aware modulation (proposal §5.7) ---
    use_prosody: bool = False
    prosody_dim: int = 64
    # --- Uncertainty estimation (proposal §5.4) ---
    use_uncertainty: bool = False
    uncertainty_dim: int = 32
    # --- Mismatch / conflict signal (proposal §5.9) ---
    use_mismatch: bool = False
    mismatch_dim: int = 64
    # --- Frame phone head gradient control ---
    detach_belief_for_frame_phone: bool = False
    belief_grad_scale: float = 0.1
    frame_phone_dropout: float = 0.1
    # --- Canonical head regularisation ---
    canonical_head_dropout: float = 0.0
    # --- JEPA configuration (Plan C) ---
    jepa_encoder_layers: int = 6
    jepa_encoder_heads: int = 8
    jepa_encoder_ff_dim: int = 1024
    jepa_encoder_conv_kernel: int = 15
    jepa_daam_num_gaussians: int = 4
    jepa_daam_alpha_init: float = 0.05
    jepa_predictor_layers: int = 2
    jepa_predictor_heads: int = 8
    jepa_prior_layers: int = 2
    jepa_prior_heads: int = 8
    jepa_ema_tau: float = 0.996
    jepa_mask_ratio: float = 0.5
    jepa_mask_min_span: int = 1
    jepa_mask_max_span: int | None = None


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


def pool_evidence_mean(
    projected: torch.Tensor,
    boundaries: torch.Tensor,
    slot_mask: torch.Tensor,
) -> torch.Tensor:
    """Vectorised mean-pool of frame features into syllable slots."""
    B, T, H = projected.shape
    K = slot_mask.shape[1]
    tidx = torch.arange(T, device=projected.device).view(1, T, 1)
    starts = boundaries[:, :, 0].unsqueeze(1)  # (B,1,K)
    ends = boundaries[:, :, 1].unsqueeze(1)  # (B,1,K)
    in_slot = (tidx >= starts) & (tidx < ends) & (slot_mask.unsqueeze(1) > 0)  # (B,T,K)
    w = in_slot.float()
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1.0)
    return torch.einsum("btk,bth->bkh", w, projected)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score_fn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(
        self,
        projected: torch.Tensor,
        boundaries: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T, H = projected.shape
        K = slot_mask.shape[1]
        scores = self.score_fn(projected).squeeze(-1)  # (B,T)
        tidx = torch.arange(T, device=projected.device).view(1, T, 1)
        starts = boundaries[:, :, 0].unsqueeze(1)
        ends = boundaries[:, :, 1].unsqueeze(1)
        in_slot = (tidx >= starts) & (tidx < ends) & (slot_mask.unsqueeze(1) > 0)
        masked = scores.unsqueeze(-1).expand(-1, -1, K)
        masked = masked.masked_fill(~in_slot, float("-inf"))
        w = F.softmax(masked, dim=1).masked_fill(~in_slot, 0.0)
        return torch.einsum("btk,bth->bkh", w, projected)


# ---------------------------------------------------------------------------
# Broadcast slots → frames  (for frame-level read-outs)
# ---------------------------------------------------------------------------


def broadcast_to_frames(
    slot_values: torch.Tensor,
    boundaries: torch.Tensor,
    slot_mask: torch.Tensor,
    total_frames: int,
) -> torch.Tensor:
    B, K, H = slot_values.shape
    tidx = torch.arange(total_frames, device=slot_values.device).view(1, -1, 1)
    starts = boundaries[:, :, 0].unsqueeze(1)
    ends = boundaries[:, :, 1].unsqueeze(1)
    in_slot = (tidx >= starts) & (tidx < ends) & (slot_mask.unsqueeze(1) > 0)
    w = in_slot.float()
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return torch.einsum("btk,bkh->bth", w, slot_values)


# ---------------------------------------------------------------------------
# Slot up-sampler  (K slots → K*F sub-frames for CTC)
# ---------------------------------------------------------------------------


class SlotUpsampler(nn.Module):
    def __init__(self, hidden_dim: int, factor: int = 4) -> None:
        super().__init__()
        self.factor = factor
        self.pos_embed = nn.Parameter(torch.randn(factor, hidden_dim) * 0.02)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())

    def forward(
        self, slots: torch.Tensor, slot_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, K, H = slots.shape
        F_ = self.factor
        repeated = slots.unsqueeze(2).expand(-1, -1, F_, -1)  # (B,K,F,H)
        pos = self.pos_embed.view(1, 1, F_, H)
        up = self.proj((repeated + pos).reshape(B, K * F_, H))
        up_mask = slot_mask.unsqueeze(-1).expand(-1, -1, F_).reshape(B, K * F_)
        return up, up_mask


# ---------------------------------------------------------------------------
# Uncertainty Estimator  (proposal §5.4)
# ---------------------------------------------------------------------------


class UncertaintyEstimator(nn.Module):
    """Derive per-frame reliability signals from the recogniser's raw emissions.

    For logit evidence the module first computes analytical features (entropy,
    confidence margin) and then passes them through a learnable projection.
    For hidden-state evidence a small MLP learns to map the representation to
    an uncertainty embedding.
    """

    def __init__(self, evidence_dim: int, uncertainty_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.evidence_dim = evidence_dim
        self.uncertainty_dim = uncertainty_dim
        analytical_feat_dim = 2  # entropy + margin
        self.analytical_proj = nn.Sequential(
            nn.Linear(analytical_feat_dim, uncertainty_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.learned_proj = nn.Sequential(
            nn.Linear(evidence_dim, evidence_dim // 2),
            nn.GELU(),
            nn.Linear(evidence_dim // 2, uncertainty_dim),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(uncertainty_dim * 2, uncertainty_dim),
            nn.Sigmoid(),
        )

    @staticmethod
    def _analytical_features(evidence: torch.Tensor) -> torch.Tensor:
        """Entropy and top-1 vs top-2 confidence margin from logit-like evidence."""
        p = F.softmax(evidence, dim=-1).clamp(min=1e-8)
        entropy = -(p * p.log()).sum(dim=-1, keepdim=True)
        top2 = p.topk(min(2, p.shape[-1]), dim=-1).values
        margin = top2[..., 0:1] - top2[..., 1:2] if top2.shape[-1] > 1 else top2[..., 0:1]
        return torch.cat([entropy, margin], dim=-1)

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        """Returns (B, T, uncertainty_dim) reliability embedding."""
        ana = self.analytical_proj(self._analytical_features(evidence))
        lrn = self.learned_proj(evidence)
        g = self.gate(torch.cat([ana, lrn], dim=-1))
        return g * ana + (1 - g) * lrn


# ---------------------------------------------------------------------------
# Identity Encoder  (proposal §5.5)
# ---------------------------------------------------------------------------


class IdentityEncoder(nn.Module):
    """Utterance-level speaker identity from raw (pre-projection) evidence.

    Attention-weighted mean pooling over frames followed by an MLP produces a
    fixed-size identity vector **I** that acts as a slow calibration state
    within the belief transition.
    """

    def __init__(self, evidence_dim: int, identity_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn_score = nn.Sequential(
            nn.Linear(evidence_dim, evidence_dim // 4),
            nn.Tanh(),
            nn.Linear(evidence_dim // 4, 1),
        )
        self.proj = nn.Sequential(
            nn.Linear(evidence_dim, identity_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(identity_dim * 2, identity_dim),
            nn.LayerNorm(identity_dim),
        )

    def forward(
        self,
        evidence: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns (B, identity_dim)."""
        scores = self.attn_score(evidence).squeeze(-1)  # (B, T)
        if frame_mask is not None:
            scores = scores.masked_fill(frame_mask < 0.5, float("-inf"))
        w = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, T, 1)
        pooled = (evidence * w).sum(dim=1)  # (B, E)
        return self.proj(pooled)


# ---------------------------------------------------------------------------
# Prosody Extractor  (proposal §5.7)
# ---------------------------------------------------------------------------


class ProsodyExtractor(nn.Module):
    """Syllable-aligned prosodic features from raw evidence and boundaries.

    Within each syllable slot the module computes statistics that approximate
    relative pitch contour, duration / tempo, prominence, and boundary
    tendency, then projects them into a compact prosody embedding **P_k**.
    """

    def __init__(self, evidence_dim: int, prosody_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        stat_dim = evidence_dim * 2 + 3  # mean, std, duration, energy, energy_delta
        self.proj = nn.Sequential(
            nn.Linear(stat_dim, prosody_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(prosody_dim * 2, prosody_dim),
            nn.LayerNorm(prosody_dim),
        )

    def forward(
        self,
        evidence: torch.Tensor,
        boundaries: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns (B, K, prosody_dim)."""
        B, T, E = evidence.shape
        K = slot_mask.shape[1]
        device = evidence.device

        tidx = torch.arange(T, device=device).view(1, T, 1)
        starts = boundaries[:, :, 0].unsqueeze(1)
        ends = boundaries[:, :, 1].unsqueeze(1)
        in_slot = (tidx >= starts) & (tidx < ends) & (slot_mask.unsqueeze(1) > 0)  # (B,T,K)
        w = in_slot.float()
        counts = w.sum(dim=1).clamp_min(1.0)  # (B, K)

        slot_mean = torch.einsum("btk,bte->bke", w, evidence) / counts.unsqueeze(-1)
        # Var(X) = E[X²] - E[X]²  — avoids materialising (B,T,K,E) tensor
        slot_mean_of_sq = torch.einsum("btk,bte->bke", w, evidence ** 2) / counts.unsqueeze(-1)
        slot_std = (slot_mean_of_sq - slot_mean ** 2).clamp(min=0).sqrt()

        energy = evidence.norm(dim=-1, keepdim=True)  # (B, T, 1)
        slot_energy = torch.einsum("btk,bte->bke", w, energy) / counts.unsqueeze(-1)
        duration = counts.unsqueeze(-1) / 50.0  # normalise to seconds at ~50Hz

        energy_shifted = torch.cat([energy[:, :1], energy[:, :-1]], dim=1)
        energy_delta = (energy - energy_shifted).abs()
        slot_delta = torch.einsum("btk,bte->bke", w, energy_delta) / counts.unsqueeze(-1)

        features = torch.cat(
            [slot_mean, slot_std, duration, slot_energy, slot_delta], dim=-1
        )
        return self.proj(features)


# ---------------------------------------------------------------------------
# Enriched Slotizer  (proposal §5.8)
# ---------------------------------------------------------------------------


class EnrichedSlotizer(nn.Module):
    """Uncertainty-aware pooling that fuses evidence, uncertainty, and prosody.

    Frames with higher reliability (lower uncertainty) receive more weight.
    Prosodic features are concatenated after pooling.
    """

    def __init__(
        self,
        hidden_dim: int,
        uncertainty_dim: int,
        prosody_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.reliability_gate = nn.Sequential(
            nn.Linear(uncertainty_dim, 1),
            nn.Sigmoid(),
        )
        in_dim = hidden_dim + prosody_dim
        self.merge = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        projected: torch.Tensor,
        uncertainty: torch.Tensor,
        prosody: torch.Tensor,
        boundaries: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns (B, K, hidden_dim)."""
        B, T, H = projected.shape
        K = slot_mask.shape[1]

        rel = self.reliability_gate(uncertainty)  # (B, T, 1)
        weighted = projected * rel  # reliability-gated evidence

        tidx = torch.arange(T, device=projected.device).view(1, T, 1)
        starts = boundaries[:, :, 0].unsqueeze(1)
        ends = boundaries[:, :, 1].unsqueeze(1)
        in_slot = (tidx >= starts) & (tidx < ends) & (slot_mask.unsqueeze(1) > 0)
        w = in_slot.float()
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = torch.einsum("btk,bth->bkh", w, weighted)

        return self.merge(torch.cat([pooled, prosody], dim=-1))


# ---------------------------------------------------------------------------
# Belief World Model  (supports both GRU and JEPA backends)
# ---------------------------------------------------------------------------


def _build_jepa_config(cfg: BeliefWMConfig) -> JEPAConfig:
    return JEPAConfig(
        hidden_dim=cfg.hidden_dim,
        encoder_layers=cfg.jepa_encoder_layers,
        encoder_heads=cfg.jepa_encoder_heads,
        encoder_ff_dim=cfg.jepa_encoder_ff_dim,
        encoder_conv_kernel=cfg.jepa_encoder_conv_kernel,
        daam_num_gaussians=cfg.jepa_daam_num_gaussians,
        daam_alpha_init=cfg.jepa_daam_alpha_init,
        predictor_layers=cfg.jepa_predictor_layers,
        predictor_heads=cfg.jepa_predictor_heads,
        prior_layers=cfg.jepa_prior_layers,
        prior_heads=cfg.jepa_prior_heads,
        dropout=cfg.dropout,
        ema_tau=cfg.jepa_ema_tau,
        mask_ratio=cfg.jepa_mask_ratio,
        mask_min_span=cfg.jepa_mask_min_span,
        mask_max_span=cfg.jepa_mask_max_span,
    )


class BeliefWorldModel(nn.Module):
    def __init__(self, config: BeliefWMConfig) -> None:
        super().__init__()
        self.config = config
        H = config.hidden_dim
        E = config.evidence_dim

        # ---- evidence projection ----
        mid = (E + H) // 2 if E > H else H
        self.evidence_proj = nn.Sequential(
            nn.Linear(E, mid),
            nn.LayerNorm(mid),
            nn.GELU(),
            nn.Linear(mid, H),
            nn.LayerNorm(H),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # ---- optional front-end modules ----
        self.uncertainty_estimator: UncertaintyEstimator | None = None
        if config.use_uncertainty:
            self.uncertainty_estimator = UncertaintyEstimator(
                E, config.uncertainty_dim, config.dropout
            )

        self.identity_encoder: IdentityEncoder | None = None
        self._identity_proj: nn.Module | None = None
        if config.use_identity:
            self.identity_encoder = IdentityEncoder(
                E, config.identity_dim, config.dropout
            )
            self._identity_proj = nn.Linear(config.identity_dim, H)

        self.prosody_extractor: ProsodyExtractor | None = None
        self._prosody_proj: nn.Module | None = None
        if config.use_prosody:
            self.prosody_extractor = ProsodyExtractor(
                E, config.prosody_dim, config.dropout
            )
            self._prosody_proj = nn.Linear(config.prosody_dim, H)

        # ---- pooling / slotizer ----
        self.enriched_slotizer: EnrichedSlotizer | None = None
        self.pooling: nn.Module | None = None
        if config.use_uncertainty and config.use_prosody:
            self.enriched_slotizer = EnrichedSlotizer(
                H, config.uncertainty_dim, config.prosody_dim, config.dropout
            )
        elif config.pooling_type == "attention":
            self.pooling = AttentionPooling(H)

        # ==================================================================
        # Belief backend: GRU or JEPA
        # ==================================================================
        self._use_jepa = config.belief_type == "jepa"

        if self._use_jepa:
            import copy as _copy

            jcfg = _build_jepa_config(config)
            self._jepa_cfg = jcfg

            self.online_encoder = ConformerDAAMEncoder(jcfg)
            self.target_encoder = _copy.deepcopy(self.online_encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad_(False)

            self.jepa_predictor = JEPAPredictor(jcfg)
            self.prior_encoder = CausalConformerEncoder(jcfg)
            self.mask_token = nn.Parameter(torch.randn(H) * 0.02)
        else:
            # Legacy GRU path
            self._mismatch_mlp: nn.Module | None = None
            mismatch_in = H * 2
            if config.use_identity:
                mismatch_in += H
            if config.use_mismatch:
                self._mismatch_mlp = nn.Sequential(
                    nn.Linear(mismatch_in, config.mismatch_dim),
                    nn.GELU(),
                    nn.Linear(config.mismatch_dim, config.mismatch_dim),
                )

            gru_in = H * 2
            if config.use_prosody:
                gru_in += H
            if config.use_identity:
                gru_in += H
            if config.use_mismatch:
                gru_in += config.mismatch_dim
            self.belief_cell = nn.GRUCell(gru_in, H)
            self.prior_proj = nn.Sequential(nn.Linear(H, H), nn.Tanh())
            if config.use_identity:
                self.prior_update = nn.Sequential(
                    nn.Linear(H * 2 + H, H),
                    nn.Tanh(),
                )

        # ---- upsampler & read-out heads (shared) ----
        self.upsampler = SlotUpsampler(H, config.upsample_factor)

        fp_dropout = getattr(config, "frame_phone_dropout", 0.1)
        self.frame_phone_head = nn.Sequential(
            nn.Dropout(fp_dropout),
            nn.Linear(H * 2, H),
            nn.GELU(),
            nn.Dropout(fp_dropout),
            nn.Linear(H, config.phone_vocab_size),
        )
        self.canonical_head = nn.Linear(H, config.phone_vocab_size)
        self._canonical_head_dropout = getattr(config, "canonical_head_dropout", 0.0)
        self.evidence_phone_head = nn.Linear(H, config.phone_vocab_size)

        future_in = H
        if config.use_identity:
            future_in += H
        self.future_head = nn.Sequential(
            nn.Linear(future_in, H), nn.GELU(), nn.Linear(H, H)
        )
        self.recon_head = nn.Linear(H, H)

    # ---- agent-facing inference ---------------------------------------------

    @torch.no_grad()
    def extract_slot_features(
        self,
        evidence: torch.Tensor,
        boundaries: torch.Tensor,
        slot_mask: torch.Tensor,
        num_frames: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Frozen inference returning per-slot beliefs, priors, and CTC logits.

        Used by the RL scheduler agent environment.  No JEPA masking or
        training-only heads are computed.
        """
        out = self.forward(
            evidence, boundaries, slot_mask, num_frames,
            frame_mask=frame_mask, compute_jepa_loss=False,
        )
        return {
            "slots": out["slots"],
            "beliefs": out["beliefs"],
            "priors": out["priors"],
            "slot_mask": out["slot_mask"],
            "canonical_logits": out["canonical_logits"],
            "up_slot_mask": out["up_slot_mask"],
        }

    # ---- helpers -----------------------------------------------------------

    def _pool(
        self,
        projected: torch.Tensor,
        boundaries: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.pooling is not None:
            return self.pooling(projected, boundaries, slot_mask)
        return pool_evidence_mean(projected, boundaries, slot_mask)

    # ---- GRU belief rollout (legacy) ----------------------------------------

    def _belief_rollout_gru(
        self,
        slots: torch.Tensor,
        slot_mask: torch.Tensor,
        identity_h: torch.Tensor | None = None,
        prosody_h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, K, H = slots.shape
        belief = slots.new_zeros(B, H)
        prior = slots.new_zeros(B, H)
        beliefs: list[torch.Tensor] = []
        priors: list[torch.Tensor] = []

        for k in range(K):
            active = slot_mask[:, k].unsqueeze(-1)

            parts = [slots[:, k], prior]
            if prosody_h is not None and self._prosody_proj is not None:
                parts.append(prosody_h[:, k])
            if identity_h is not None:
                parts.append(identity_h)
            if self._mismatch_mlp is not None:
                mm_parts = [slots[:, k], belief]
                if identity_h is not None:
                    mm_parts.append(identity_h)
                mismatch = self._mismatch_mlp(torch.cat(mm_parts, dim=-1))
                parts.append(mismatch)

            gru_input = torch.cat(parts, dim=-1)
            candidate = self.belief_cell(gru_input, belief)
            belief = active * candidate + (1.0 - active) * belief

            if hasattr(self, "prior_update") and identity_h is not None:
                new_prior = self.prior_update(
                    torch.cat([prior, belief, identity_h], dim=-1)
                )
            else:
                new_prior = self.prior_proj(belief)
            prior = active * new_prior + (1.0 - active) * prior

            beliefs.append(belief)
            priors.append(prior)

        return torch.stack(beliefs, dim=1), torch.stack(priors, dim=1)

    # ---- JEPA belief encoding -----------------------------------------------

    def _belief_encode_jepa(
        self,
        slots: torch.Tensor,
        slot_mask: torch.Tensor,
        compute_jepa_loss: bool = True,
    ) -> dict:
        """Run JEPA encoding on syllable slots.

        Masking is applied in **latent space** (after the encoder) rather
        than in input space.  This prevents the encoder's self-attention
        from trivially filling in masked positions, forcing the predictor
        to do the actual prediction work — matching the I-JEPA design
        where the encoder never sees masked positions.

        The encoder runs only once on clean input, producing both beliefs
        (for downstream heads) and context representations (for JEPA).

        Returns dict with beliefs, priors, and JEPA-specific tensors.
        """
        jcfg = self._jepa_cfg

        z_online = self.online_encoder(slots, slot_mask)
        beliefs = z_online

        priors = self.prior_encoder(slots, slot_mask)

        result: dict = {
            "beliefs": beliefs,
            "priors": priors,
            "z_target": None,
            "z_pred": None,
            "jepa_mask": None,
        }

        if compute_jepa_loss:
            B, K, _H = slots.shape
            jepa_mask = block_mask_slots(
                B, K, slots.device,
                mask_ratio=jcfg.mask_ratio,
                min_span=jcfg.mask_min_span,
                max_span=jcfg.mask_max_span,
            )

            z_masked = z_online.clone()
            z_masked[jepa_mask] = self.mask_token.to(z_masked.dtype)

            z_pred = self.jepa_predictor(z_masked, slot_mask)

            with torch.no_grad():
                z_target = self.target_encoder(slots, slot_mask)

            result["z_target"] = z_target
            result["z_pred"] = z_pred
            result["jepa_mask"] = jepa_mask

        return result

    @torch.no_grad()
    def update_ema(self, tau: float | None = None) -> None:
        """EMA update for JEPA target encoder. Call after each training step."""
        if self._use_jepa:
            _tau = tau if tau is not None else self._jepa_cfg.ema_tau
            update_ema(self.online_encoder, self.target_encoder, _tau)

    # ---- forward -----------------------------------------------------------

    def forward(
        self,
        evidence: torch.Tensor,
        boundaries: torch.Tensor,
        slot_mask: torch.Tensor,
        num_frames: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
        compute_jepa_loss: bool | None = None,
        jepa_only: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = evidence.shape

        # --- frame-level branches ---
        uncertainty: torch.Tensor | None = None
        if self.uncertainty_estimator is not None and not jepa_only:
            uncertainty = self.uncertainty_estimator(evidence)

        identity: torch.Tensor | None = None
        identity_h: torch.Tensor | None = None
        if self.identity_encoder is not None and not jepa_only:
            identity = self.identity_encoder(evidence, frame_mask)
            identity_h = self._identity_proj(identity)  # type: ignore[misc]

        projected = self.evidence_proj(evidence)

        prosody: torch.Tensor | None = None
        prosody_h: torch.Tensor | None = None
        if self.prosody_extractor is not None and not jepa_only:
            prosody = self.prosody_extractor(evidence, boundaries, slot_mask)
            prosody_h = self._prosody_proj(prosody)  # type: ignore[misc]

        # --- slot aggregation ---
        if self.enriched_slotizer is not None and uncertainty is not None and prosody is not None:
            slots = self.enriched_slotizer(
                projected, uncertainty, prosody, boundaries, slot_mask
            )
        else:
            slots = self._pool(projected, boundaries, slot_mask)

        # --- belief computation ---
        if self._use_jepa:
            need_jepa = compute_jepa_loss if compute_jepa_loss is not None else self.training
            jepa_out = self._belief_encode_jepa(
                slots, slot_mask, compute_jepa_loss=need_jepa
            )
            beliefs = jepa_out["beliefs"]
            priors = jepa_out["priors"]
        else:
            beliefs, priors = self._belief_rollout_gru(
                slots, slot_mask, identity_h=identity_h, prosody_h=prosody_h
            )

        # --- fast path: skip read-out heads when only JEPA loss is needed ---
        if jepa_only and self._use_jepa:
            out: Dict[str, torch.Tensor] = {
                "slots": slots,
                "beliefs": beliefs,
                "priors": priors,
                "slot_mask": slot_mask,
            }
            if jepa_out["z_pred"] is not None:
                out["z_pred"] = jepa_out["z_pred"]
            if jepa_out["z_target"] is not None:
                out["z_target"] = jepa_out["z_target"]
            if jepa_out["jepa_mask"] is not None:
                out["jepa_mask"] = jepa_out["jepa_mask"]
            return out

        # --- read-out heads ---
        belief_frames = broadcast_to_frames(beliefs, boundaries, slot_mask, T)
        detach_bf = getattr(self.config, "detach_belief_for_frame_phone", False)
        grad_scale = getattr(self.config, "belief_grad_scale", 0.1)
        if detach_bf:
            bf_for_phone = belief_frames.detach()
        elif grad_scale < 1.0:
            bf_for_phone = belief_frames.detach() + grad_scale * (belief_frames - belief_frames.detach())
        else:
            bf_for_phone = belief_frames
        augmented = torch.cat([projected, bf_for_phone], dim=-1)

        up_slots, up_mask = self.upsampler(slots, slot_mask)
        up_beliefs, _ = self.upsampler(beliefs, slot_mask)
        up_priors, _ = self.upsampler(priors, slot_mask)

        if identity_h is not None:
            identity_exp = identity_h.unsqueeze(1).expand(-1, beliefs.shape[1], -1)
            future_input = torch.cat([beliefs, identity_exp], dim=-1)
        else:
            future_input = beliefs

        out = {
            "slots": slots,
            "beliefs": beliefs,
            "priors": priors,
            "belief_frames": belief_frames,
            "slot_mask": slot_mask,
            "up_slot_mask": up_mask,
            "frame_phone_logits": self.frame_phone_head(augmented),
            "evidence_phone_logits": self.evidence_phone_head(up_slots),
            "canonical_logits": self.canonical_head(up_priors),
            "future_pred": self.future_head(future_input),
            "evidence_recon": self.recon_head(beliefs),
        }

        if self._use_jepa:
            if jepa_out["z_pred"] is not None:
                out["z_pred"] = jepa_out["z_pred"]
            if jepa_out["z_target"] is not None:
                out["z_target"] = jepa_out["z_target"]
            if jepa_out["jepa_mask"] is not None:
                out["jepa_mask"] = jepa_out["jepa_mask"]

        if identity is not None:
            out["identity"] = identity
        if prosody is not None:
            out["prosody"] = prosody
        if uncertainty is not None:
            out["uncertainty"] = uncertainty
        return out


# ---------------------------------------------------------------------------
# Dataset & Collator
# ---------------------------------------------------------------------------


class BeliefWMDataset(Dataset):
    """Loads pre-computed HuPER+Sylber features saved by ``wm_prepare.py``."""

    def __init__(
        self,
        features_dir: str,
        split: str,
        metadata_dir: str,
        phone_vocab: Vocabulary,
        text_vocab: Vocabulary,
        evidence_type: str = "logits",
        max_examples: int | None = None,
        teacher_cache: dict[str, list[str]] | None = None,
    ) -> None:
        feat_dir = Path(features_dir) / split
        manifest = read_json(Path(features_dir) / f"{split}_manifest.json")
        seg_ids: list[str] = manifest["segment_ids"]
        if max_examples:
            seg_ids = seg_ids[:max_examples]
        self.segment_ids = seg_ids
        self.feat_dir = feat_dir
        self.phone_vocab = phone_vocab
        self.text_vocab = text_vocab
        self.evidence_type = evidence_type
        self.teacher_cache = teacher_cache

        meta_path = Path(metadata_dir) / f"{split}.jsonl"
        self.metadata_map: dict = {}
        if meta_path.exists():
            self.metadata_map = {r["segment_id"]: r for r in load_jsonl(meta_path)}

    def __len__(self) -> int:
        return len(self.segment_ids)

    def __getitem__(self, idx: int) -> Dict:
        seg_id = self.segment_ids[idx]
        feats = torch.load(self.feat_dir / f"{seg_id}.pt", weights_only=False)
        meta = self.metadata_map.get(seg_id, feats)

        if self.evidence_type == "logits":
            evidence = feats["huper_logits"].float()
        else:
            if "huper_hidden" not in feats:
                raise KeyError(
                    f"Feature file for '{seg_id}' is missing 'huper_hidden'. "
                    f"Available keys: {list(feats.keys())}. "
                    f"Run: python scripts/update_features.py "
                    f"--features-dir {self.feat_dir.parent} --splits {self.feat_dir.name}"
                )
            evidence = feats["huper_hidden"].float()

        text = meta.get("text", feats.get("text", ""))
        text_chars = meta.get("text_chars", feats.get("text_chars", list(text)))
        canonical = meta.get("canonical_phones", feats.get("canonical_phones", []))
        if self.teacher_cache is not None and seg_id in self.teacher_cache:
            teacher = self.teacher_cache[seg_id]
        else:
            teacher = meta.get("teacher_phones", feats.get("teacher_phones", []))
        if not teacher:
            teacher = canonical

        item = {
            "segment_id": seg_id,
            "evidence": evidence,
            "num_frames": evidence.shape[0],
            "boundaries": feats["sylber_boundaries"],
            "num_syllables": feats["sylber_boundaries"].shape[0],
            "text": text,
            "text_ids": torch.tensor(
                self.text_vocab.encode(text_chars), dtype=torch.long
            ),
            "canonical_phones": canonical,
            "canonical_ids": torch.tensor(
                self.phone_vocab.encode(canonical), dtype=torch.long
            ),
            "teacher_phones": teacher,
            "teacher_ids": torch.tensor(
                self.phone_vocab.encode(teacher), dtype=torch.long
            ),
        }

        if "mel_target" in feats:
            item["mel_target"] = feats["mel_target"].float()
            item["mel_length"] = int(feats["mel_length"])

        return item


class BeliefWMCollator:
    def __call__(self, batch: List[Dict]) -> Dict:
        B = len(batch)
        max_T = max(it["num_frames"] for it in batch)
        max_K = max(it["num_syllables"] for it in batch)
        E = batch[0]["evidence"].shape[-1]

        evidence = torch.zeros(B, max_T, E)
        frame_mask = torch.zeros(B, max_T)
        boundaries = torch.zeros(B, max_K, 2, dtype=torch.long)
        slot_mask = torch.zeros(B, max_K)

        for i, it in enumerate(batch):
            T, K = it["num_frames"], it["num_syllables"]
            evidence[i, :T] = it["evidence"]
            frame_mask[i, :T] = 1.0
            boundaries[i, :K] = it["boundaries"]
            slot_mask[i, :K] = 1.0

        out = {
            "segment_ids": [it["segment_id"] for it in batch],
            "evidence": evidence,
            "frame_mask": frame_mask,
            "boundaries": boundaries,
            "slot_mask": slot_mask,
            "num_frames": torch.tensor([it["num_frames"] for it in batch]),
            "num_syllables": torch.tensor([it["num_syllables"] for it in batch]),
            "texts": [it["text"] for it in batch],
            "text_ids": [it["text_ids"] for it in batch],
            "canonical_phones": [it["canonical_phones"] for it in batch],
            "canonical_ids": [it["canonical_ids"] for it in batch],
            "teacher_phones": [it["teacher_phones"] for it in batch],
            "teacher_ids": [it["teacher_ids"] for it in batch],
        }

        if "mel_target" in batch[0]:
            mel_dim = batch[0]["mel_target"].shape[-1]
            max_mel = max(it["mel_length"] for it in batch)
            mel_target = torch.zeros(B, max_mel, mel_dim)
            mel_lengths = torch.zeros(B, dtype=torch.long)
            for i, it in enumerate(batch):
                ml = it["mel_length"]
                mel_target[i, :ml] = it["mel_target"]
                mel_lengths[i] = ml
            out["mel_target"] = mel_target
            out["mel_lengths"] = mel_lengths

        return out


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def compute_ctc_loss(
    logits: torch.Tensor,
    input_lengths: torch.Tensor,
    target_sequences: List[torch.Tensor],
    blank_id: int,
) -> torch.Tensor:
    valid = [i for i, t in enumerate(target_sequences) if t.numel() > 0]
    if not valid:
        return logits.new_zeros((), requires_grad=True)
    sel_logits = logits[valid]
    sel_lengths = input_lengths[valid].to(logits.device)
    flat = torch.cat([target_sequences[i] for i in valid]).to(logits.device)
    tgt_len = torch.tensor(
        [target_sequences[i].numel() for i in valid],
        device=logits.device,
        dtype=torch.long,
    )
    lp = F.log_softmax(sel_logits, dim=-1).transpose(0, 1)
    return F.ctc_loss(lp, flat, sel_lengths, tgt_len, blank=blank_id, zero_infinity=True)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _phone_error_rate(
    refs: List[List[str]], preds: List[List[str]]
) -> float:
    total_edits = 0
    total_len = 0
    for ref, pred in zip(refs, preds):
        if ref:
            total_edits += levenshtein_distance(ref, pred)
            total_len += len(ref)
    return total_edits / max(total_len, 1)


@torch.no_grad()
def evaluate_belief_wm(
    model: BeliefWorldModel,
    dataloader,
    phone_vocab: Vocabulary,
    device: torch.device,
) -> Dict:
    model.eval()
    canon_refs, canon_preds = [], []
    teacher_refs, teacher_preds = [], []
    future_losses: list[float] = []
    recon_losses: list[float] = []
    belief_cosines: list[float] = []

    for batch in dataloader:
        ev = batch["evidence"].to(device)
        bd = batch["boundaries"].to(device)
        sm = batch["slot_mask"].to(device)
        nf = batch["num_frames"].to(device)
        fm_mask = batch.get("frame_mask")
        if fm_mask is not None:
            fm_mask = fm_mask.to(device)

        out = model(ev, bd, sm, nf, frame_mask=fm_mask)
        up_factor = model.config.upsample_factor

        canon_ids = out["canonical_logits"].argmax(dim=-1).cpu().tolist()
        phone_ids = out["frame_phone_logits"].argmax(dim=-1).cpu().tolist()

        for i in range(len(batch["segment_ids"])):
            up_len = int(batch["num_syllables"][i].item()) * up_factor
            pred_c = phone_vocab.decode_ctc(canon_ids[i][:up_len])
            canon_refs.append(batch["canonical_phones"][i])
            canon_preds.append(pred_c)

            T_i = int(batch["num_frames"][i].item())
            pred_t = phone_vocab.decode_ctc(phone_ids[i][:T_i])
            teacher_refs.append(batch["teacher_phones"][i])
            teacher_preds.append(pred_t)

        H = out["future_pred"].shape[-1]
        if out["future_pred"].shape[1] > 1:
            fm = sm[:, 1:].unsqueeze(-1)
            fl = (
                F.mse_loss(
                    out["future_pred"][:, :-1] * fm,
                    out["slots"][:, 1:] * fm,
                    reduction="sum",
                )
                / (fm.sum().clamp_min(1.0) * H)
            )
            future_losses.append(fl.item())

        rm = sm.unsqueeze(-1)
        rl = (
            F.mse_loss(
                out["evidence_recon"] * rm,
                out["slots"] * rm,
                reduction="sum",
            )
            / (rm.sum().clamp_min(1.0) * H)
        )
        recon_losses.append(rl.item())

        # Belief evolution rate: cosine similarity between adjacent beliefs
        beliefs = out["beliefs"]
        if beliefs.shape[1] > 1:
            b_prev = F.normalize(beliefs[:, :-1], dim=-1)
            b_next = F.normalize(beliefs[:, 1:], dim=-1)
            adj_mask = sm[:, 1:].unsqueeze(-1)
            cos_sim = (b_prev * b_next * adj_mask).sum() / adj_mask.sum().clamp_min(1.0)
            belief_cosines.append(cos_sim.item())

    metrics: Dict[str, object] = {
        "canonical_per": _phone_error_rate(canon_refs, canon_preds),
        "teacher_per": _phone_error_rate(teacher_refs, teacher_preds),
        "future_mse": sum(future_losses) / max(len(future_losses), 1),
        "recon_mse": sum(recon_losses) / max(len(recon_losses), 1),
        "belief_evolution_cosine": (
            sum(belief_cosines) / max(len(belief_cosines), 1)
        ),
        "num_examples": len(canon_refs),
    }
    return metrics
