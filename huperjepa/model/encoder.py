"""Comparison-Refinement Encoder — the core belief encoder for HuperJEPA v3.

Architecture inspired by Heald & Nusbaum (2014) "Speech perception as an
active cognitive process" (Fig. B + C combined):

1. CausalConformerEncoder  — bottom-up streaming prior  (Fig. C path)
2. ComparisonRefinementBlocks × N — iterative hypothesis refinement:
      error  = ||belief − prior||₂          (comparison / error signal)
      gate   = σ(MLP(error))               (attentional modulation)
      fused  = gate·slots + (1−gate)·prior  (evidence vs hypothesis blend)
      belief = CausalConformerBlock(belief + fused)  (top-down refinement)
3. Final error (errors[-1]) serves as the convergence / distortion signal
   passed to the ActiveAgent — low error → belief has converged → ready to emit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from huperjepa.model.conformer import CausalConformerEncoder, ConformerBlock


@dataclass
class ComparisonRefinementConfig:
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


class ComparisonRefinementBlock(nn.Module):
    """Single iteration of comparison-gated belief refinement.

    Computes the L₂ error between the current hypothesis (belief) and the
    causal prior, uses it to gate a blend of raw slots vs prior, fuses the
    result with the previous belief, then refines through a causal Conformer.
    """

    def __init__(self, cfg: ComparisonRefinementConfig) -> None:
        super().__init__()
        H = cfg.hidden_dim

        # Gate network: scalar error → per-dimension blend weight.
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
        slots: torch.Tensor,     # (B, K, H) raw pooled evidence
        belief: torch.Tensor,    # (B, K, H) current hypothesis
        prior: torch.Tensor,     # (B, K, H) causal streaming prior (fixed)
        slot_mask: torch.Tensor, # (B, K) 1 for valid slots
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (refined_belief, error)."""
        error = (belief - prior).norm(dim=-1, keepdim=True)  # (B, K, 1)

        gate = self.gate_net(error)                           # (B, K, H)
        modulated = gate * slots + (1 - gate) * prior         # (B, K, H)

        fused = self.fusion_norm(belief + modulated)

        pad_mask = slot_mask < 0.5
        refined = self.conformer(fused, key_padding_mask=pad_mask)
        return self.output_norm(refined), error


class ComparisonRefinementEncoder(nn.Module):
    """Active belief encoder: causal prior + iterative comparison refinement."""

    def __init__(self, cfg: ComparisonRefinementConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.causal_encoder = CausalConformerEncoder(
            hidden_dim=cfg.hidden_dim,
            n_layers=cfg.prior_layers,
            n_heads=cfg.prior_heads,
            ff_dim=cfg.prior_ff_dim,
            conv_kernel=cfg.prior_conv_kernel,
            dropout=cfg.dropout,
            max_slots=cfg.max_slots,
        )
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
            errors:  list[N] of (B, K, 1) — error at each refinement iteration.
        """
        priors = self.causal_encoder(slots, slot_mask)

        belief = priors.clone()
        errors: list[torch.Tensor] = []
        for block in self.refinement_blocks:
            belief, error = block(slots, belief, priors, slot_mask)
            errors.append(error)

        return self.output_norm(belief), priors, errors


def convergence_loss(
    errors: list[torch.Tensor],
    slot_mask: torch.Tensor,
) -> torch.Tensor:
    """Penalise when error at iteration i exceeds the (detached) error at i−1.

    Encourages monotonically decreasing error across refinement iterations,
    i.e. each refinement pass should bring the belief closer to the prior.

    Args:
        errors:    list of (B, K, 1) tensors from each refinement iteration.
        slot_mask: (B, K) — 1 for valid slots.

    Returns:
        Scalar loss.
    """
    if len(errors) < 2:
        return errors[0].new_zeros(())
    loss = errors[0].new_zeros(())
    mask = slot_mask.unsqueeze(-1)
    n_valid = mask.sum().clamp(min=1)
    for i in range(1, len(errors)):
        violation = F.relu(errors[i] - errors[i - 1].detach())
        loss = loss + (violation * mask).sum() / n_valid
    return loss / (len(errors) - 1)
