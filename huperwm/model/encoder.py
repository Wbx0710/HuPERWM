"""Comparison-Refinement Encoder — the core belief encoder for HuperJEPA.

Architecture inspired by Heald & Nusbaum (2014) "Speech perception as an
active cognitive process" (Fig. B + C combined):

1. CausalConformerEncoder  — bottom-up streaming prior  (Fig. C path)
2. ComparisonRefinementBlocks × N — iterative hypothesis refinement:
      error_vec = belief − prior               (per-dimension signed difference)
      error     = ||error_vec||₂               (scalar L2, used only inside gate)
      gate      = σ(MLP(error))               (attentional modulation)
      fused     = gate·slots + (1−gate)·prior  (evidence vs hypothesis blend)
      belief    = CausalConformerBlock(belief + fused)  (top-down refinement)
3. Final error_vec (error_vectors[-1]) serves as the H-dim distortion signal
   passed to the ActiveAgent — each dimension encodes where belief diverges
   from prior, giving directional uncertainty for the comparison gate.
4. Optional belief_logvar_head (use_belief_var=True) produces per-slot
   log-variance for VAE-style KL regularisation in the world model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from huperwm.model.conformer import CausalConformerEncoder, ConformerBlock


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
    # POMDP extension: produce per-slot log-variance for KL regularisation.
    use_belief_var: bool = False


class ComparisonRefinementBlock(nn.Module):
    """Single iteration of comparison-gated belief refinement.

    Computes the signed per-dimension error between the current hypothesis
    (belief) and the causal prior, uses its L2 norm to gate a blend of raw
    slots vs prior, fuses the result with the previous belief, then refines
    through a causal Conformer.

    Returns:
        refined_belief: (B, K, H)
        error_vec:      (B, K, H) — per-dimension signed difference (belief − prior)
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
        slots: torch.Tensor,     # (B, K, H) raw pooled evidence
        belief: torch.Tensor,    # (B, K, H) current hypothesis
        prior: torch.Tensor,     # (B, K, H) causal streaming prior (fixed)
        slot_mask: torch.Tensor, # (B, K) 1 for valid slots
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (refined_belief, error_vec)."""
        error_vec = belief - prior                            # (B, K, H) signed difference
        error_scalar = error_vec.norm(dim=-1, keepdim=True)  # (B, K, 1) L2 norm for gate

        gate = self.gate_net(error_scalar)                    # (B, K, H)
        modulated = gate * slots + (1 - gate) * prior         # (B, K, H)

        fused = self.fusion_norm(belief + modulated)

        pad_mask = slot_mask < 0.5
        refined = self.conformer(fused, key_padding_mask=pad_mask)
        return self.output_norm(refined), error_vec


class ComparisonRefinementEncoder(nn.Module):
    """Active belief encoder: causal prior + iterative comparison refinement.

    When cfg.use_belief_var=True, an additional linear head maps the final
    belief to a per-dimension log-variance, enabling VAE-style KL training.
    """

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

        # Optional: log-variance head for POMDP belief distribution.
        self.belief_logvar_head: nn.Linear | None = None
        if cfg.use_belief_var:
            self.belief_logvar_head = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

    def forward(
        self,
        slots: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor | None]:
        """
        Args:
            slots:     (B, K, H) — pooled syllable-slot features.
            slot_mask: (B, K)    — 1 for valid slots.

        Returns:
            beliefs:       (B, K, H) — refined hypothesis after N iterations.
            priors:        (B, K, H) — causal streaming prior.
            error_vectors: list[N] of (B, K, H) — per-dim signed error per iteration.
            belief_logvar: (B, K, H) or None — log-variance if use_belief_var.
        """
        priors = self.causal_encoder(slots, slot_mask)

        belief = priors.clone()
        error_vectors: list[torch.Tensor] = []
        for block in self.refinement_blocks:
            belief, error_vec = block(slots, belief, priors, slot_mask)
            error_vectors.append(error_vec)

        belief = self.output_norm(belief)

        belief_logvar: torch.Tensor | None = None
        if self.belief_logvar_head is not None:
            belief_logvar = self.belief_logvar_head(belief)

        return belief, priors, error_vectors, belief_logvar


def convergence_loss(
    error_vectors: list[torch.Tensor],
    slot_mask: torch.Tensor,
) -> torch.Tensor:
    """Penalise when belief-prior divergence at iteration i exceeds iteration i−1.

    Encourages monotonically decreasing distortion across refinement passes,
    i.e. each comparison-refinement step should bring belief closer to prior.

    Args:
        error_vectors: list of (B, K, H) per-dimension signed errors from each
                       refinement iteration. Norms are computed internally.
        slot_mask:     (B, K) — 1 for valid slots.

    Returns:
        Scalar loss.
    """
    if len(error_vectors) < 2:
        return error_vectors[0].new_zeros(())

    # Compute scalar L2 norms from H-dim error vectors.
    errors = [ev.norm(dim=-1, keepdim=True) for ev in error_vectors]  # list of (B, K, 1)

    loss = errors[0].new_zeros(())
    mask = slot_mask.unsqueeze(-1)
    n_valid = mask.sum().clamp(min=1)
    for i in range(1, len(errors)):
        violation = F.relu(errors[i] - errors[i - 1].detach())
        loss = loss + (violation * mask).sum() / n_valid
    return loss / (len(errors) - 1)
