"""Word Distortion Module for HuperJEPA v2.

This module implements the WordDistortionModule, which tracks a word-level
"distortion" signal that decreases as more acoustic evidence accumulates for
the current word. When distortion drops below a learned threshold, the agent
is encouraged to EMIT.

Core intuition:
    - beliefs[k] are computed by a *bidirectional* encoder (can peek at future)
    - priors[k] are computed by a *causal* encoder (only sees past — streaming)
    - point_dist(k) = ||beliefs[k] - priors[k]||₂
      → high when future context greatly changes the current interpretation
        (word is not yet complete / predictable from context alone)
      → low  when beliefs ≈ priors  (word is causally predictable → EMIT)

The WordDistortionModule accumulates evidence about the current word via a
GRU (word_state), and combines it with point_dist to produce a scalar
distortion ∈ [0, 1].

Training supervision (Stage 2):
    - distortion at oracle_emit positions should be LOW  (≈ 0.2)
    - distortion at non-emit positions should be HIGH   (≈ 0.7)
    via distortion_loss() — an MSE against (1 - oracle_emit).

Usage:
    mod = WordDistortionModule(cfg)
    word_state = mod.reset(device)         # start of utterance / after EMIT

    for k in range(K):
        word_state, dist = mod.step(beliefs[k], priors[k], word_state)
        # dist ∈ [0, 1]; low → EMIT encouraged

    # batch mode:
    distortions, word_states = mod.forward_sequence(beliefs, priors, slot_mask)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DistortionConfig:
    """Configuration for WordDistortionModule."""

    hidden_dim: int = 256
    """Dimension of belief/prior vectors (must match BeliefWMConfig.hidden_dim)."""

    distortion_loss_weight: float = 0.5
    """Weight for distortion alignment loss during Stage-2 training."""

    init_threshold: float = 0.3
    """Initial value for the learnable emit threshold.
    During GRPO training, when distortion < threshold, an additional emit
    reward incentive is added (+bonus_reward).
    """

    bonus_reward: float = 0.1
    """Bonus reward added to an EMIT action when distortion < threshold."""


# ---------------------------------------------------------------------------
# WordDistortionModule
# ---------------------------------------------------------------------------


class WordDistortionModule(nn.Module):
    """Maintains a word-level distortion signal that decreases over time.

    The module owns:
      - word_gru:  GRUCell that accumulates evidence about the current word.
      - distortion_head: projects (word_state ‖ point_dist) → scalar ∈ [0, 1].
      - threshold:  learnable scalar; EMIT is encouraged when dist < threshold.

    The module is intended to be integrated into BeliefWorldModel and called
    at each syllable step.
    """

    def __init__(self, cfg: DistortionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_dim

        # GRU accumulates evidence about the current word.
        # Input: belief[k]   Hidden: word_state
        self.word_gru = nn.GRUCell(H, H)

        # Project (word_state ‖ point_dist_scalar) → distortion ∈ [0, 1].
        self.distortion_head = nn.Sequential(
            nn.Linear(H + 1, H // 4),
            nn.LayerNorm(H // 4),
            nn.GELU(),
            nn.Linear(H // 4, 1),
            nn.Sigmoid(),
        )

        # Learnable threshold: when distortion < threshold → encourage EMIT.
        self.threshold = nn.Parameter(torch.tensor(cfg.init_threshold))

        self._init_weights()

    def _init_weights(self) -> None:
        # Initialise GRU so that it starts near identity (stable).
        for name, p in self.word_gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        # Distortion head output bias → ~0.7 (high distortion at start).
        last_linear = self.distortion_head[-2]  # Linear before Sigmoid
        nn.init.zeros_(last_linear.weight)
        nn.init.constant_(last_linear.bias, 0.85)  # Sigmoid(0.85) ≈ 0.70

    # ------------------------------------------------------------------
    # Core per-step computation
    # ------------------------------------------------------------------

    def step(
        self,
        belief_k: torch.Tensor,   # (H,) or (B, H)
        prior_k: torch.Tensor,    # (H,) or (B, H)
        word_state: torch.Tensor, # (H,) or (B, H)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a single syllable slot.

        Args:
            belief_k:   Belief vector at slot k   — bidirectional encoding.
            prior_k:    Prior vector at slot k    — causal/streaming encoding.
            word_state: Accumulated word state (GRU hidden).

        Returns:
            new_word_state: Updated GRU hidden, same shape as word_state.
            distortion:     Scalar in [0, 1]; low when word is predictable.
        """
        # Point-wise belief-prior divergence (key distortion signal).
        # |belief - prior| large → future context matters → word incomplete.
        # |belief - prior| small → prior is sufficient  → word predictable.
        point_dist = (belief_k - prior_k).norm(dim=-1, keepdim=True)  # (..., 1)

        # Accumulate evidence about the current word.
        new_word_state = self.word_gru(belief_k, word_state)

        # Compute distortion from accumulated state + point signal.
        feat = torch.cat([new_word_state, point_dist], dim=-1)  # (..., H+1)
        distortion = self.distortion_head(feat)  # (..., 1)

        return new_word_state, distortion

    # ------------------------------------------------------------------
    # Batch / sequence mode (for training)
    # ------------------------------------------------------------------

    def forward_sequence(
        self,
        beliefs: torch.Tensor,        # (B, K, H)
        priors: torch.Tensor,         # (B, K, H)
        slot_mask: torch.Tensor,      # (B, K)   — 1 for valid slots
        oracle_emit: torch.Tensor | None = None,  # (B, K) — for reset-aware mode
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a full sequence of syllable slots in batch.

        In training we process the sequence without oracle-aware resets
        (the GRU just runs through; the distortion loss still works because
        oracle_emit provides the supervision target).

        Args:
            beliefs:     (B, K, H) — bidirectional slot representations.
            priors:      (B, K, H) — causal slot representations.
            slot_mask:   (B, K)    — 1 for valid slots, 0 for padding.
            oracle_emit: (B, K)    — optional; reserved for future reset-aware mode.

        Returns:
            distortions:  (B, K, 1) — distortion scalar per slot.
            word_states:  (B, K, H) — GRU hidden state per slot.
        """
        B, K, H = beliefs.shape
        word_state = self.reset(beliefs.device, batch_size=B)  # (B, H)

        distortions: list[torch.Tensor] = []
        word_states: list[torch.Tensor] = []

        for k in range(K):
            word_state, dist = self.step(beliefs[:, k], priors[:, k], word_state)
            distortions.append(dist)   # (B, 1)
            word_states.append(word_state)  # (B, H)

            # Mask out padding slots: don't propagate meaningless state.
            valid = slot_mask[:, k].unsqueeze(-1)  # (B, 1)
            # word_state is already updated; keep it as-is for valid slots
            # (padding slots should ideally not update, but for simplicity
            #  we allow it — they don't contribute to the loss via slot_mask).

        distortions_t = torch.stack(distortions, dim=1)  # (B, K, 1)
        word_states_t = torch.stack(word_states, dim=1)  # (B, K, H)
        return distortions_t, word_states_t

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def distortion_loss(
        self,
        distortions: torch.Tensor,   # (B, K, 1) or (B, K)
        oracle_emit: torch.Tensor,   # (B, K) — float, 1 at word-end slots
        slot_mask: torch.Tensor,     # (B, K) — 1 for valid slots
    ) -> torch.Tensor:
        """Supervised distortion alignment loss.

        Target: distortion ≈ (1 - oracle_emit)
            → oracle_emit=1 (word end) → target distortion ≈ 0   (LOW)
            → oracle_emit=0 (mid-word) → target distortion ≈ 1   (HIGH)

        Only valid (non-padded) slots contribute.

        Returns:
            Scalar MSE loss.
        """
        d = distortions.squeeze(-1)  # (B, K)
        target = 1.0 - oracle_emit.float()  # (B, K): 0 at emit, 1 elsewhere
        valid = slot_mask > 0.5  # (B, K) bool

        if not valid.any():
            return d.sum() * 0.0  # zero grad, keeps graph

        return F.mse_loss(d[valid], target[valid])

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        device: torch.device,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Return a zeroed initial word_state (GRU hidden).

        Call this:
          - At the start of each utterance.
          - In the environment after an EMIT action (to start fresh for
            the next word).
        """
        return torch.zeros(batch_size, self.cfg.hidden_dim, device=device)

    # ------------------------------------------------------------------
    # Threshold helpers (for Agent / Env)
    # ------------------------------------------------------------------

    @property
    def emit_threshold(self) -> float:
        """Clamped threshold value in a reasonable range [0.05, 0.95]."""
        return float(self.threshold.clamp(0.05, 0.95).item())

    def below_threshold(self, distortion: torch.Tensor) -> torch.Tensor:
        """Return bool mask: True where distortion < learned threshold."""
        return distortion.squeeze(-1) < self.threshold.clamp(0.05, 0.95)
