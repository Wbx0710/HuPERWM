"""Comparison-Refinement Encoder for HuperJEPA v3.

Implements the active-perception belief encoder inspired by
Heald & Nusbaum (2014) "Speech perception as an active cognitive process".

Architecture (Figure 1B + 1C combined)
======================================
1. **CausalConformerEncoder** produces a streaming prior (bottom-up, Figure 1C).
2. **ComparisonRefinementBlocks** iterate:
      error  = ||belief - prior||   (Comparison / Error Signal)
      gate   = σ(MLP(error))        (Attention modulation)
      fused  = gate·slots + (1-gate)·prior   (evidence vs hypothesis blend)
      belief = ConformerBlock(belief + fused)  (hypothesis refinement)
3. Final error after the last iteration replaces the separate
   WordDistortionModule — it IS the convergence/distortion signal.

This module replaces ConformerDAAMEncoder, WordDistortionModule, JEPAPredictor,
and target_encoder from the v2 architecture.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from wm_jepa import CausalConformerEncoder, ConformerBlock, JEPAConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ComparisonRefinementConfig:
    """Configuration for ComparisonRefinementEncoder."""

    hidden_dim: int = 256
    prior_layers: int = 3
    prior_heads: int = 8
    prior_ff_dim: int = 1024
    prior_conv_kernel: int = 15
    num_refinements: int = 2
    refinement_heads: int = 4
    refinement_ff_dim: int = 512
    refinement_conv_kernel: int = 15
    dropout: float = 0.1
    max_slots: int = 200
    convergence_loss_weight: float = 0.2


def _build_prior_jepa_config(cfg: ComparisonRefinementConfig) -> JEPAConfig:
    """Build a JEPAConfig that CausalConformerEncoder can consume."""
    return JEPAConfig(
        hidden_dim=cfg.hidden_dim,
        prior_layers=cfg.prior_layers,
        prior_heads=cfg.prior_heads,
        encoder_ff_dim=cfg.prior_ff_dim,
        encoder_conv_kernel=cfg.prior_conv_kernel,
        dropout=cfg.dropout,
        max_slots=cfg.max_slots,
    )


# ---------------------------------------------------------------------------
# ComparisonRefinementBlock
# ---------------------------------------------------------------------------


class ComparisonRefinementBlock(nn.Module):
    """Single iteration of comparison-gated belief refinement.

    Each call performs: compare → gate → modulate → fuse → refine.
    """

    def __init__(self, cfg: ComparisonRefinementConfig) -> None:
        super().__init__()
        H = cfg.hidden_dim

        self.gate_net = nn.Sequential(
            nn.Linear(1, H // 4),
            nn.GELU(),
            nn.Linear(H // 4, H),
            nn.Sigmoid(),
        )
        self.conformer = ConformerBlock(
            H,
            cfg.refinement_heads,
            cfg.refinement_ff_dim,
            cfg.refinement_conv_kernel,
            cfg.dropout,
            causal=True,
        )
        self.fusion_norm = nn.LayerNorm(H)
        self.output_norm = nn.LayerNorm(H)

    def forward(
        self,
        slots: torch.Tensor,
        belief: torch.Tensor,
        prior: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            slots:     (B, K, H) — raw pooled evidence.
            belief:    (B, K, H) — current hypothesis.
            prior:     (B, K, H) — causal streaming prior (fixed).
            slot_mask: (B, K)    — 1 for valid slots.

        Returns:
            refined: (B, K, H) — updated hypothesis.
            error:   (B, K, 1) — comparison divergence.
        """
        error = (belief - prior).norm(dim=-1, keepdim=True)  # (B, K, 1)

        gate = self.gate_net(error)  # (B, K, H)
        modulated = gate * slots + (1 - gate) * prior  # (B, K, H)

        fused = self.fusion_norm(belief + modulated)

        pad_mask = slot_mask < 0.5
        refined = self.conformer(fused, key_padding_mask=pad_mask)
        return self.output_norm(refined), error


# ---------------------------------------------------------------------------
# ComparisonRefinementEncoder
# ---------------------------------------------------------------------------


class ComparisonRefinementEncoder(nn.Module):
    """Active belief encoder: causal prior + iterative comparison refinement.

    Replaces ConformerDAAMEncoder, WordDistortionModule, JEPAPredictor, and
    target_encoder from v2.
    """

    def __init__(self, cfg: ComparisonRefinementConfig) -> None:
        super().__init__()
        self.cfg = cfg
        jcfg = _build_prior_jepa_config(cfg)
        self.causal_encoder = CausalConformerEncoder(jcfg)
        self.refinement_blocks = nn.ModuleList(
            [ComparisonRefinementBlock(cfg) for _ in range(cfg.num_refinements)]
        )
        self.output_norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        slots: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            slots:     (B, K, H) — pooled syllable-slot features.
            slot_mask: (B, K)    — 1 for valid slots.

        Returns:
            beliefs: (B, K, H) — refined hypothesis after N iterations.
            priors:  (B, K, H) — causal streaming prior.
            errors:  list of (B, K, 1) — error at each iteration.
        """
        priors = self.causal_encoder(slots, slot_mask)

        belief = priors.clone()
        errors: list[torch.Tensor] = []
        for block in self.refinement_blocks:
            belief, error = block(slots, belief, priors, slot_mask)
            errors.append(error)

        beliefs = self.output_norm(belief)
        return beliefs, priors, errors


# ---------------------------------------------------------------------------
# Convergence loss
# ---------------------------------------------------------------------------


def convergence_loss(
    errors: list[torch.Tensor],
    slot_mask: torch.Tensor,
) -> torch.Tensor:
    """Encourage monotonically decreasing error across refinement iterations.

    Penalises when error at iteration *i* exceeds the (detached) error at
    iteration *i-1*.  Only valid (non-padded) slots contribute.

    Args:
        errors:    list of (B, K, 1) tensors from each refinement iteration.
        slot_mask: (B, K) — 1 for valid slots.

    Returns:
        Scalar loss.
    """
    if len(errors) < 2:
        return errors[0].new_zeros(())
    loss = errors[0].new_zeros(())
    mask = slot_mask.unsqueeze(-1)  # (B, K, 1)
    n_valid = mask.sum().clamp(min=1)
    for i in range(1, len(errors)):
        violation = F.relu(errors[i] - errors[i - 1].detach())
        loss = loss + (violation * mask).sum() / n_valid
    return loss / (len(errors) - 1)
