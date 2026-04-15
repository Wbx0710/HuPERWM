"""ActiveAgent — comparison-gated dual-pathway RL scheduler agent.

Implements the active perception model from Heald & Nusbaum (2014):

B-path (top-down hypothesis):
    belief → hypothesis_proj → h_top

C-path (bottom-up evidence accumulation):
    [prior, syl_feat] + action_embed(prev_action) → evidence_encoder → GRU → h_bot

Comparison gate (using the H-dim per-dimension error vector):
    error_vec → MLP → Sigmoid → gate
    h_fused = gate · h_bot + (1 - gate) · h_top

The H-dim comparison gate provides directional insight into WHERE belief
is uncertain (which dimensions of the embedding space diverge from prior),
enabling finer-grained gating than a scalar L2 norm.

Low error (belief ≈ prior) → gate → 0 → trust top-down hypothesis → EMIT.
High error (belief ≠ prior) → gate → 1 → trust bottom-up evidence → WAIT.

Action conditioning implements T(s'|s,a) dependence: the agent's accumulated
evidence (h_bot) is conditioned on whether a word was just emitted, letting
the GRU reset its context at word boundaries.
"""

from __future__ import annotations

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

        # Comparison gate: H-dim signed error vector → per-dimension blend weight.
        # The full error_vec gives directional uncertainty (which embedding dims diverge).
        self.comparison_gate = nn.Sequential(
            nn.Linear(H, D // 4),
            nn.GELU(),
            nn.Linear(D // 4, D),
            nn.Sigmoid(),
        )

        # Action conditioning: T(s'|s,a) — inject prev WAIT/EMIT into GRU input.
        self.boundary_embed = nn.Embedding(2, D)  # {WAIT=0, EMIT=1} → R^D

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
        belief: torch.Tensor,        # (B, [T,] H)
        prior: torch.Tensor,         # (B, [T,] H)
        syl_feat: torch.Tensor,      # (B, [T,] 3)
        error: torch.Tensor,         # (B, [T,] H) — per-dim signed error (belief − prior)
        hidden: torch.Tensor | None = None,
        prev_action: torch.Tensor | None = None,  # (B, [T,]) int64 — WAIT=0, EMIT=1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            belief:      (B, [T,] H) — refined belief from world model.
            prior:       (B, [T,] H) — causal streaming prior.
            syl_feat:    (B, [T,] 3) — [log_dur, rel_pos, steps_since_emit].
            error:       (B, [T,] H) — per-dimension signed error vector
                         (belief − prior) from the comparison refinement encoder.
            hidden:      GRU hidden state from previous step (or None).
            prev_action: (B, [T,]) int64 — action taken at previous step;
                         WAIT=0 at t=0. Used for T(s'|s,a) conditioning.

        Returns:
            logits: (B, [T,] 2) — [WAIT, EMIT] action logits.
            value:  (B, [T,] 1) — state value estimate for GAE.
            hidden: updated GRU hidden state.
        """
        squeeze = belief.dim() == 2
        if squeeze:
            belief = belief.unsqueeze(1)
            prior = prior.unsqueeze(1)
            syl_feat = syl_feat.unsqueeze(1)
            error = error.unsqueeze(1)
            if prev_action is not None:
                prev_action = prev_action.unsqueeze(1)

        h_top = self.hypothesis_proj(belief)

        x_bot = self.evidence_encoder(torch.cat([prior, syl_feat], dim=-1))

        # Action conditioning: previous WAIT/EMIT tells the GRU whether a word
        # boundary just occurred, enabling context resets at emission points.
        if prev_action is not None:
            x_bot = x_bot + self.boundary_embed(prev_action)

        h_bot, hidden = self.evidence_gru(x_bot, hidden)

        # H-dim gate: each dimension of error_vec independently modulates
        # the corresponding dimension of (h_bot - h_top) blend.
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
        prev_action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> Tuple[int, float, float, torch.Tensor]:
        """Sample a single action for environment stepping.

        Returns: (action, log_prob, value, new_hidden).
        """
        logits, value, hidden = self.forward(
            belief, prior, syl_feat, error, hidden, prev_action
        )
        dist = torch.distributions.Categorical(logits=logits.squeeze(0))
        if deterministic:
            action = logits.squeeze(0).argmax().item()
        else:
            action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action, device=logits.device)).item()
        return action, log_prob, value.squeeze().item(), hidden
