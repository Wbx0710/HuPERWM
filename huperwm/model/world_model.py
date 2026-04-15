"""Belief World Model — v3 architecture.

Evidence (HuPER hidden states, 1024-dim) → SyllableSlotPooler → slots
    → ComparisonRefinementEncoder → beliefs, priors, distortion_vectors

Read-out heads (shared):
    frame_phone_head    — frame-level phone CTC (evidence + belief context)
    evidence_phone_head — slot-level CTC on raw slots
    canonical_head      — slot-level CTC on priors (canonical phones)
    future_head         — next-slot prediction auxiliary
    recon_head          — slot reconstruction auxiliary

POMDP extensions (gated by config):
    belief_var_weight > 0 — enables KL divergence loss from belief_logvar_head
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from huperwm.model.conformer import compute_sigreg_loss  # re-exported for convenience
from huperwm.model.encoder import (
    ComparisonRefinementConfig,
    ComparisonRefinementEncoder,
    convergence_loss,
)
from huperwm.model.pooling import SyllableSlotPooler


@dataclass
class WorldModelConfig:
    # Evidence input
    evidence_dim: int = 1024        # 1024 for HuPER hidden states; 46 for logits
    hidden_dim: int = 256
    phone_vocab_size: int = 90
    upsample_factor: int = 4
    dropout: float = 0.1

    # Pooling
    boundary_attn_heads: int = 4

    # Comparison Refinement Encoder (prior + belief backbone)
    prior_layers: int = 3
    prior_heads: int = 8
    prior_ff_dim: int = 1024
    prior_conv_kernel: int = 15
    num_refinements: int = 2
    refinement_heads: int = 4
    refinement_ff_dim: int = 512
    refinement_conv_kernel: int = 15
    convergence_loss_weight: float = 0.2

    # Frame-phone head gradient controls
    belief_grad_scale: float = 0.1
    frame_phone_dropout: float = 0.1
    canonical_head_dropout: float = 0.0

    # POMDP extension: VAE-style belief variance (gated — 0.0 = disabled)
    belief_var_weight: float = 0.0


class SlotUpsampler(nn.Module):
    """Repeat each slot F times and add learned sub-slot position embeddings.

    Converts (B, K, H) slot features to (B, K*F, H) for CTC decoding.
    """

    def __init__(self, hidden_dim: int, factor: int = 4) -> None:
        super().__init__()
        self.factor = factor
        self.pos_embed = nn.Parameter(torch.randn(factor, hidden_dim) * 0.02)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())

    def forward(
        self, slots: torch.Tensor, slot_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (upsampled, up_mask) with shapes (B, K*F, H) and (B, K*F)."""
        B, K, H = slots.shape
        F_ = self.factor
        repeated = slots.unsqueeze(2).expand(-1, -1, F_, -1)  # (B, K, F, H)
        pos = self.pos_embed.view(1, 1, F_, H)
        up = self.proj((repeated + pos).reshape(B, K * F_, H))
        up_mask = slot_mask.unsqueeze(-1).expand(-1, -1, F_).reshape(B, K * F_)
        return up, up_mask


def broadcast_to_frames(
    slot_values: torch.Tensor,
    boundaries: torch.Tensor,
    slot_mask: torch.Tensor,
    total_frames: int,
) -> torch.Tensor:
    """Broadcast (B, K, H) slot values back to (B, T, H) frame resolution."""
    B, K, H = slot_values.shape
    tidx = torch.arange(total_frames, device=slot_values.device).view(1, -1, 1)
    starts = boundaries[:, :, 0].unsqueeze(1)
    ends = boundaries[:, :, 1].unsqueeze(1)
    in_slot = (tidx >= starts) & (tidx < ends) & (slot_mask.unsqueeze(1) > 0)
    w = in_slot.float()
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return torch.einsum("btk,bkh->bth", w, slot_values)


class BeliefWorldModel(nn.Module):
    """Belief World Model using ComparisonRefinementEncoder (v3)."""

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        H = config.hidden_dim
        E = config.evidence_dim

        # Evidence projection: E → H with two-layer MLP + layer norms.
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

        # Pooling: boundary-gated cross-attention + causal inter-slot refinement.
        self.pooling = SyllableSlotPooler(H, config.boundary_attn_heads, config.dropout)

        # Belief encoder: causal prior + iterative comparison refinement.
        enc_cfg = ComparisonRefinementConfig(
            hidden_dim=H,
            prior_layers=config.prior_layers,
            prior_heads=config.prior_heads,
            prior_ff_dim=config.prior_ff_dim,
            prior_conv_kernel=config.prior_conv_kernel,
            num_refinements=config.num_refinements,
            refinement_heads=config.refinement_heads,
            refinement_ff_dim=config.refinement_ff_dim,
            refinement_conv_kernel=config.refinement_conv_kernel,
            dropout=config.dropout,
            convergence_loss_weight=config.convergence_loss_weight,
            use_belief_var=(config.belief_var_weight > 0.0),
        )
        self.encoder = ComparisonRefinementEncoder(enc_cfg)

        # Upsampler and read-out heads.
        self.upsampler = SlotUpsampler(H, config.upsample_factor)

        fp_drop = config.frame_phone_dropout
        self.frame_phone_head = nn.Sequential(
            nn.Dropout(fp_drop),
            nn.Linear(H * 2, H),
            nn.GELU(),
            nn.Dropout(fp_drop),
            nn.Linear(H, config.phone_vocab_size),
        )
        self.canonical_head = nn.Linear(H, config.phone_vocab_size)
        self.evidence_phone_head = nn.Linear(H, config.phone_vocab_size)
        self.future_head = nn.Sequential(
            nn.Linear(H, H), nn.GELU(), nn.Linear(H, H)
        )
        self.recon_head = nn.Linear(H, H)

    @torch.no_grad()
    def extract_slot_features(
        self,
        evidence: torch.Tensor,
        boundaries: torch.Tensor,
        slot_mask: torch.Tensor,
        num_frames: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Frozen inference for the RL agent — returns per-slot features.

        Skips training-only heads; includes distortion_vectors (H-dim per-dimension
        signed error between belief and prior) as the primary uncertainty signal
        for the ActiveAgent's comparison gate.
        """
        out = self.forward(evidence, boundaries, slot_mask, num_frames, frame_mask=frame_mask)
        result = {
            "slots": out["slots"],
            "beliefs": out["beliefs"],
            "priors": out["priors"],
            "slot_mask": out["slot_mask"],
            "canonical_logits": out["canonical_logits"],
            "up_slot_mask": out["up_slot_mask"],
            "distortion_vectors": out["distortion_vectors"],  # (B, K, H) per-dim error
        }
        if out.get("belief_logvar") is not None:
            result["belief_logvar"] = out["belief_logvar"]
        return result

    def forward(
        self,
        evidence: torch.Tensor,
        boundaries: torch.Tensor,
        slot_mask: torch.Tensor,
        num_frames: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = evidence.shape

        projected = self.evidence_proj(evidence)  # (B, T, H)
        slots = self.pooling(projected, boundaries, slot_mask)  # (B, K, H)

        beliefs, priors, error_vectors, belief_logvar = self.encoder(slots, slot_mask)

        # Per-dim distortion vector: directional uncertainty signal for the agent.
        distortion_vectors = error_vectors[-1]  # (B, K, H)

        # Frame-phone head: broadcast beliefs back to frame resolution.
        belief_frames = broadcast_to_frames(beliefs, boundaries, slot_mask, T)
        gs = self.config.belief_grad_scale
        if gs < 1.0:
            bf_for_phone = belief_frames.detach() + gs * (belief_frames - belief_frames.detach())
        else:
            bf_for_phone = belief_frames
        augmented = torch.cat([projected, bf_for_phone], dim=-1)  # (B, T, 2H)

        up_slots, up_mask = self.upsampler(slots, slot_mask)
        up_beliefs, _ = self.upsampler(beliefs, slot_mask)
        up_priors, _ = self.upsampler(priors, slot_mask)

        out: Dict[str, torch.Tensor] = {
            "slots": slots,
            "beliefs": beliefs,
            "priors": priors,
            "belief_frames": belief_frames,
            "slot_mask": slot_mask,
            "up_slot_mask": up_mask,
            "distortion_vectors": distortion_vectors,
            # comparison_errors (list of H-dim error_vectors) used by convergence_loss.
            "comparison_errors": error_vectors,
            "frame_phone_logits": self.frame_phone_head(augmented),
            "evidence_phone_logits": self.evidence_phone_head(up_slots),
            "canonical_logits": self.canonical_head(up_priors),
            "future_pred": self.future_head(beliefs),
            "evidence_recon": self.recon_head(beliefs),
        }
        if belief_logvar is not None:
            out["belief_logvar"] = belief_logvar
        return out

    def belief_kl_loss(
        self,
        beliefs: torch.Tensor,
        priors: torch.Tensor,
        belief_logvar: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> torch.Tensor:
        """KL[ q(z|slots) || p(z|prior) ] with prior treated as N(prior, I).

        Uses the analytic KL between two Gaussians:
            KL = 0.5 * ( logvar_p - logvar_q + exp(logvar_q - logvar_p)
                         + (mu_q - mu_p)^2 / exp(logvar_p) - 1 )
        With logvar_p = 0 (unit variance prior) this simplifies to:
            KL = 0.5 * ( -logvar_q + exp(logvar_q) + (belief - prior)^2 - 1 )

        Args:
            beliefs:      (B, K, H) — posterior mean.
            priors:       (B, K, H) — prior mean (treated as N(prior, I)).
            belief_logvar:(B, K, H) — posterior log-variance.
            slot_mask:    (B, K)    — 1 for valid slots.

        Returns:
            Scalar KL loss averaged over valid slots and dimensions.
        """
        kl = 0.5 * (
            -belief_logvar
            + belief_logvar.exp()
            + (beliefs - priors).pow(2)
            - 1.0
        )  # (B, K, H)
        mask = slot_mask.unsqueeze(-1)  # (B, K, 1)
        n_valid = mask.sum().clamp(min=1) * beliefs.shape[-1]
        return (kl * mask).sum() / n_valid
