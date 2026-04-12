"""ActiveAgent — comparison-gated dual-pathway RL scheduler agent (v3).

Implements the active perception model from Heald & Nusbaum (2014):

B-path (top-down hypothesis):
    belief → hypothesis_proj → h_top

C-path (bottom-up evidence accumulation):
    [prior, syl_feat] → evidence_encoder → evidence_gru → h_bot

Comparison gate (using the convergence error signal):
    error → MLP → Sigmoid → gate
    h_fused = gate · h_bot + (1 - gate) · h_top

Low error (belief ≈ prior) → gate → 0 → trust top-down hypothesis → EMIT.
High error (belief ≠ prior) → gate → 1 → trust bottom-up evidence → WAIT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ActiveAgentConfig:
    belief_dim: int = 256       # must match WorldModelConfig.hidden_dim
    syl_feat_dim: int = 3       # log_duration, relative_position, steps_since_emit
    agent_hidden: int = 128
    gru_layers: int = 1
    dropout: float = 0.1


class ActiveAgent(nn.Module):
    """Comparison-gated dual-pathway agent for streaming word emission."""

    def __init__(self, cfg: ActiveAgentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        H = cfg.belief_dim
        D = cfg.agent_hidden

        # B-path: project refined belief (hypothesis) into agent hidden space.
        self.hypothesis_proj = nn.Sequential(
            nn.Linear(H, D),
            nn.LayerNorm(D),
            nn.GELU(),
        )

        # C-path: encode causal prior + timing features, accumulate via GRU.
        self.evidence_encoder = nn.Sequential(
            nn.Linear(H + cfg.syl_feat_dim, D),
            nn.LayerNorm(D),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.evidence_gru = nn.GRU(D, D, num_layers=cfg.gru_layers, batch_first=True)

        # Comparison gate: error signal → per-dimension blend weight.
        self.comparison_gate = nn.Sequential(
            nn.Linear(1, D // 4),
            nn.GELU(),
            nn.Linear(D // 4, D),
            nn.Sigmoid(),
        )

        self.policy_head = nn.Linear(D, 2)  # [WAIT, EMIT] logits
        self.value_head = nn.Linear(D, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.value_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(
        self,
        belief: torch.Tensor,     # (B, [T,] H)
        prior: torch.Tensor,      # (B, [T,] H)
        syl_feat: torch.Tensor,   # (B, [T,] 3)
        error: torch.Tensor,      # (B, [T,] 1) — comparison convergence error
        hidden: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (B, [T,] 2) — [WAIT, EMIT] action logits.
            value:  (B, [T,] 1) — state value for critic baseline.
            hidden: updated GRU hidden state.
        """
        squeeze = belief.dim() == 2
        if squeeze:
            belief = belief.unsqueeze(1)
            prior = prior.unsqueeze(1)
            syl_feat = syl_feat.unsqueeze(1)
            error = error.unsqueeze(1)

        h_top = self.hypothesis_proj(belief)

        x_bot = self.evidence_encoder(torch.cat([prior, syl_feat], dim=-1))
        h_bot, hidden = self.evidence_gru(x_bot, hidden)

        gate = self.comparison_gate(error)
        h_fused = gate * h_bot + (1 - gate) * h_top

        logits = self.policy_head(h_fused)
        value = self.value_head(h_fused)

        if squeeze:
            logits = logits.squeeze(1)
            value = value.squeeze(1)

        return logits, value, hidden

    def get_action(
        self,
        belief: torch.Tensor,
        prior: torch.Tensor,
        syl_feat: torch.Tensor,
        error: torch.Tensor,
        hidden: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> Tuple[int, float, float, torch.Tensor]:
        """Sample a single action for environment stepping.

        Returns: (action, log_prob, value, new_hidden).
        """
        logits, value, hidden = self.forward(belief, prior, syl_feat, error, hidden)
        dist = torch.distributions.Categorical(logits=logits.squeeze(0))
        if deterministic:
            action = logits.squeeze(0).argmax().item()
        else:
            action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action, device=logits.device)).item()
        return action, log_prob, value.squeeze().item(), hidden
