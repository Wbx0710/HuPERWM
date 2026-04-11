"""RL Scheduler Agent for streaming ASR word emission.

The agent observes per-syllable JEPA beliefs, language priors, and timing
features, then decides *when* to emit a word.  The environment wraps a
frozen BeliefWorldModel and decodes CTC logits accumulated between
successive emit actions.

Architecture
============
SchedulerAgent
    state_encoder  : MLP mapping (belief, prior, syl_feats) → hidden
    history_gru    : GRU tracking emission context across slots
    policy_head    : Linear → 2 logits  [wait, emit]
    value_head     : Linear → 1 scalar  (PPO critic)

ASRSchedulerEnv
    Holds a single utterance's pre-extracted slot features.
    Steps through slots left-to-right.  On 'emit', decodes CTC buffer
    and computes word-level reward against the ground-truth transcript.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from wm_common import levenshtein_distance

import torch
import torch.nn as nn
import torch.nn.functional as F

from wm_common import Vocabulary

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WAIT = 0
EMIT = 1


@dataclass
class AgentConfig:
    belief_dim: int = 256
    syl_feat_dim: int = 3  # log_duration, relative_position, steps_since_emit
    agent_hidden: int = 128
    gru_layers: int = 1
    dropout: float = 0.1
    # Extra belief/prior interaction features:
    #   True  → 2 extra dims: cosine_sim(belief, prior), entropy(softmax(belief))
    #   False → original 3-dim syl_feat only (backward-compat)
    use_extra_features: bool = False


# ---------------------------------------------------------------------------
# SchedulerAgent
# ---------------------------------------------------------------------------


class SchedulerAgent(nn.Module):
    """Binary wait/emit policy with a value baseline for PPO.

    Optional extra features (cfg.use_extra_features=True):
      - cosine_sim(belief, prior): captures how well the belief matches the
        language prior; low similarity → the acoustic evidence is surprising →
        may signal a word boundary.
      - entropy of softmax(belief): high entropy → uncertain belief → possibly
        at a word boundary or end of utterance.
    These two scalars are appended to syl_feat, increasing syl_feat_dim from 3
    to 5 when the option is active.
    """

    def __init__(self, cfg: AgentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        extra = 2 if cfg.use_extra_features else 0
        state_dim = cfg.belief_dim * 2 + cfg.syl_feat_dim + extra

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, cfg.agent_hidden),
            nn.LayerNorm(cfg.agent_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.agent_hidden, cfg.agent_hidden),
            nn.GELU(),
        )

        self.history_gru = nn.GRU(
            cfg.agent_hidden, cfg.agent_hidden,
            num_layers=cfg.gru_layers, batch_first=True,
        )

        self.policy_head = nn.Linear(cfg.agent_hidden, 2)
        self.value_head = nn.Linear(cfg.agent_hidden, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.value_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    @staticmethod
    def _compute_extra_features(
        belief: torch.Tensor,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        """Compute 2 extra scalar features from belief and prior tensors.

        Args:
            belief: (..., H)
            prior:  (..., H)
        Returns:
            (..., 2) — [cosine_sim, belief_entropy]
        """
        cos_sim = F.cosine_similarity(belief, prior, dim=-1, eps=1e-8).unsqueeze(-1)
        # Entropy of the normalised belief as a probability distribution.
        belief_probs = F.softmax(belief, dim=-1)
        # Clamp to avoid log(0); sum over H.
        entropy = -(belief_probs * (belief_probs + 1e-9).log()).sum(dim=-1, keepdim=True)
        # Normalise entropy to [0, 1] by dividing by log(H).
        log_H = math.log(max(belief.shape[-1], 2))
        entropy = entropy / log_H
        return torch.cat([cos_sim, entropy], dim=-1)  # (..., 2)

    def forward(
        self,
        belief: torch.Tensor,
        prior: torch.Tensor,
        syl_feat: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for a single time-step (or a sequence).

        Args:
            belief:   (B, [T,] H)
            prior:    (B, [T,] H)
            syl_feat: (B, [T,] syl_feat_dim)
            hidden:   (gru_layers, B, agent_hidden) or None

        Returns:
            logits: (B, [T,] 2)  — action logits [wait, emit]
            value:  (B, [T,] 1)  — state value
            hidden: (gru_layers, B, agent_hidden) — updated GRU state
        """
        squeeze_time = belief.dim() == 2
        if squeeze_time:
            belief = belief.unsqueeze(1)
            prior = prior.unsqueeze(1)
            syl_feat = syl_feat.unsqueeze(1)

        if self.cfg.use_extra_features:
            extra = self._compute_extra_features(belief, prior)
            x = torch.cat([belief, prior, syl_feat, extra], dim=-1)
        else:
            x = torch.cat([belief, prior, syl_feat], dim=-1)

        x = self.state_encoder(x)
        x, hidden = self.history_gru(x, hidden)

        logits = self.policy_head(x)
        value = self.value_head(x)

        if squeeze_time:
            logits = logits.squeeze(1)
            value = value.squeeze(1)

        return logits, value, hidden

    def get_action(
        self,
        belief: torch.Tensor,
        prior: torch.Tensor,
        syl_feat: torch.Tensor,
        hidden: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> Tuple[int, float, float, torch.Tensor]:
        """Sample a single action for environment stepping.

        Returns: (action, log_prob, value, new_hidden)
        """
        logits, value, hidden = self.forward(belief, prior, syl_feat, hidden)
        dist = torch.distributions.Categorical(logits=logits.squeeze(0))
        if deterministic:
            action = logits.squeeze(0).argmax().item()
        else:
            action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action, device=logits.device)).item()
        return action, log_prob, value.squeeze().item(), hidden


# ---------------------------------------------------------------------------
# CTC word decoder utilities
# ---------------------------------------------------------------------------


def decode_ctc_to_phones(
    logits: torch.Tensor,
    vocab: Vocabulary,
) -> List[str]:
    """Greedy CTC decode a (T, V) logit tensor → phone list."""
    ids = logits.argmax(dim=-1).tolist()
    return vocab.decode_ctc(ids)


def phones_to_words(phones: List[str], silence: str = "SIL") -> List[str]:
    """Split a phone sequence into words at silence boundaries.

    Falls back to treating each phone sequence between silences as one
    word-token (joined phones).  This works as a simple baseline even
    without a full lexicon.
    """
    if not phones:
        return []
    words: List[str] = []
    current: List[str] = []
    for ph in phones:
        if ph.upper() == silence or ph == "|" or ph == " ":
            if current:
                words.append(" ".join(current))
                current = []
        else:
            current.append(ph)
    if current:
        words.append(" ".join(current))
    return words


# ---------------------------------------------------------------------------
# ASR Scheduler Environment
# ---------------------------------------------------------------------------


@dataclass
class EnvConfig:
    wait_penalty: float = -0.01
    correct_reward: float = 1.0
    wrong_penalty: float = -0.5
    incomplete_penalty: float = 0.0  # disabled: budget-tracking caused over-emitting
    upsample_factor: int = 4
    # Reward mode:
    #   "per_slot"   — original per-step binary reward (default, backward-compat)
    #   "episode_f1" — zero step rewards; terminal F1 against oracle as sole signal
    #   "hybrid"     — small per-step credit (+0.2 / -0.2) plus terminal F1 bonus
    #   "word_match" — acoustic word-recognition reward: when EMIT, decode
    #                  accumulated CTC logits and compare phones against the
    #                  expected GT word (from CMU dict).  reward ∈ [0, 1]
    #                  scaled by correct_reward; missing words penalised by
    #                  missing_word_penalty at episode end.
    reward_mode: str = "per_slot"
    f1_reward_scale: float = 1.0   # scale applied to the terminal F1 value
    # Tolerance window for soft-F1 matching (slots).  0 = exact; 1 = ±1 slot
    # match counts as a true positive.  Useful when oracle is approximate.
    f1_match_window: int = 0
    # Penalty per uncovered GT word at episode end (word_match mode).
    missing_word_penalty: float = 0.5


class ASRSchedulerEnv:
    """Single-utterance RL environment for the scheduler agent.

    Reward modes (set via ``env_cfg.reward_mode``):

    "per_slot" (default, backward-compat):
        Binary per-step reward: EMIT at an oracle slot → +correct_reward;
        EMIT elsewhere → wrong_penalty; WAIT → wait_penalty.

    "episode_f1":
        Zero step rewards; terminal F1 between agent emit slots and oracle
        slots is added at episode end.  Directly optimises the eval metric.

    "hybrid":
        Small per-step credit (±0.2) for credit assignment plus terminal F1.

    "word_match" (recommended for human-brain ASR simulation):
        Simulates the human streaming-ASR process:
        1. Evidence accumulation: agent observes beliefs/priors per slot.
        2. Lexical matching: when EMIT, decode accumulated CTC logits →
           ARPABET phone sequence; compare (stress-normalized) against the
           expected GT word's phones from the CMU Pronouncing Dictionary.
        3. Boundary commitment: reward = max(0, 1 - phone_PER) * correct_reward
           — smooth reward in [0, correct_reward] proportional to word accuracy.
        4. Latency–accuracy tradeoff: the agent learns WHEN to commit based
           on accumulated acoustic evidence, not on oracle proximity.
        Terminal: −missing_word_penalty × uncovered words (under-emit penalty).
    """

    def __init__(
        self,
        beliefs: torch.Tensor,
        priors: torch.Tensor,
        boundaries: torch.Tensor,
        canonical_logits: torch.Tensor,
        up_slot_mask: torch.Tensor,
        slot_mask: torch.Tensor,
        gt_words: List[str],
        phone_vocab: Vocabulary,
        env_cfg: EnvConfig | None = None,
        oracle_emit: torch.Tensor | None = None,
        word_phones_list: Optional[List[List[int]]] = None,
    ) -> None:
        self.beliefs = beliefs          # (K, H)
        self.priors = priors            # (K, H)
        self.boundaries = boundaries    # (K, 2)
        self.canonical_logits = canonical_logits  # (K*F, V)
        self.up_slot_mask = up_slot_mask          # (K*F,)
        self.slot_mask = slot_mask                # (K,)
        self.gt_words = gt_words
        self.vocab = phone_vocab
        self.cfg = env_cfg or EnvConfig()
        # oracle_emit: (K,) float tensor, 1.0 at word-boundary slots
        self.oracle_emit = oracle_emit
        # word_phones_list[i] = CMU phone IDs for gt_words[i]; [] for OOV words.
        # Used in "word_match" reward mode.
        self.word_phones_list: List[List[int]] = word_phones_list or []

        self.num_slots = int(slot_mask.sum().item())
        self.upsample_factor = self.cfg.upsample_factor
        self.total_gt_words = len(gt_words)
        # Number of oracle emit slots (used as recall denominator in oracle mode).
        if oracle_emit is not None:
            self.total_oracle_slots = int(oracle_emit[:self.num_slots].sum().item())
            # Pre-compute oracle slot set for F1 reward modes.
            self._oracle_slot_set: set = {
                t for t in range(self.num_slots)
                if oracle_emit[t].item() > 0.5
            }
        else:
            self.total_oracle_slots = self.total_gt_words
            self._oracle_slot_set = set()

        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.emit_start_slot = 0  # start of current accumulation window
        self.gt_word_idx = 0      # word pointer for legacy CTC and word_match modes
        self.emitted_words: List[str] = []
        self.agent_emit_slots: List[int] = []  # used in episode_f1 / hybrid modes
        self.done = False

    def _decode_buffer(self) -> str:
        """Decode CTC logits from emit_start_slot to current slot t."""
        start_frame = self.emit_start_slot * self.upsample_factor
        end_frame = (self.t + 1) * self.upsample_factor
        logit_slice = self.canonical_logits[start_frame:end_frame]
        mask_slice = self.up_slot_mask[start_frame:end_frame]

        valid_len = int(mask_slice.sum().item())
        if valid_len == 0:
            return ""
        logit_slice = logit_slice[:valid_len]

        phones = decode_ctc_to_phones(logit_slice, self.vocab)
        words = phones_to_words(phones)
        return " ".join(words) if words else ""

    def _compute_syl_feat(self) -> torch.Tensor:
        """Build the 3-dim syllable feature vector for slot self.t."""
        device = self.beliefs.device
        if self.t >= self.num_slots:
            return torch.zeros(3, device=device)

        bnd = self.boundaries[self.t]
        duration = (bnd[1] - bnd[0]).float().clamp(min=1.0)
        log_dur = duration.log()

        rel_pos = torch.tensor(
            self.t / max(self.num_slots - 1, 1), device=device, dtype=torch.float
        )
        steps_since = torch.tensor(
            self.t - self.emit_start_slot, device=device, dtype=torch.float
        )
        return torch.stack([log_dur, rel_pos, steps_since])

    def observe(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Current observation: (belief_t, prior_t, syl_feat_t)."""
        return (
            self.beliefs[self.t],
            self.priors[self.t],
            self._compute_syl_feat(),
        )

    # ------------------------------------------------------------------
    # Phone-level helpers for word_match reward mode
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_phone(ph: str) -> str:
        """Strip stress digit: 'AH0' → 'AH', 'T' → 'T'."""
        return ph.rstrip("012")

    def _decode_ctc_phone_ids(self, start_slot: int, end_slot: int) -> List[int]:
        """Greedy CTC decode of canonical_logits[start_slot..end_slot] → phone IDs.

        Returns a deduplicated (blank-collapsed, repeat-collapsed) list of
        phone vocabulary token IDs, mirroring what the phone string decoder does.
        """
        F = self.upsample_factor
        start_frame = start_slot * F
        end_frame = (end_slot + 1) * F
        logit_slice = self.canonical_logits[start_frame:end_frame]
        mask_slice = self.up_slot_mask[start_frame:end_frame]
        valid_len = int(mask_slice.sum().item())
        if valid_len == 0:
            return []

        raw_ids = logit_slice[:valid_len].argmax(dim=-1).tolist()
        blank = self.vocab.blank_id
        # CTC collapse: remove blanks and consecutive repeats.
        phone_ids: List[int] = []
        prev = blank
        for pid in raw_ids:
            if pid == blank:
                prev = blank
                continue
            if pid != prev:
                phone_ids.append(pid)
            prev = pid
        return phone_ids

    def _phone_per(
        self,
        decoded_ids: List[int],
        expected_ids: List[int],
    ) -> float:
        """Stress-normalized phone error rate in [0, 1].

        Strips stress digits from both sequences before computing edit distance.
        Returns 0.0 for empty expected (OOV word) — treat as neutral.
        """
        if not expected_ids:
            return 0.0   # OOV: can't penalise, give neutral 0-PER
        tokens = self.vocab.tokens
        dec_base = [self._normalize_phone(tokens[pid]) for pid in decoded_ids if pid < len(tokens)]
        exp_base = [self._normalize_phone(tokens[pid]) for pid in expected_ids if pid < len(tokens)]
        if not exp_base:
            return 0.0
        dist = levenshtein_distance(dec_base, exp_base)
        return min(dist / len(exp_base), 1.0)

    def step(self, action: int) -> Tuple[float, bool]:
        """Execute action, return (reward, done)."""
        if self.done:
            return 0.0, True

        reward = 0.0
        mode = self.cfg.reward_mode

        if action == EMIT:
            self.agent_emit_slots.append(self.t)

            if mode == "per_slot":
                if self.oracle_emit is not None:
                    # Pure per-slot oracle reward — no word-budget tracking.
                    is_correct = (
                        self.t < self.num_slots
                        and self.oracle_emit[self.t].item() > 0.5
                    )
                    reward = (
                        self.cfg.correct_reward if is_correct
                        else self.cfg.wrong_penalty
                    )
                else:
                    # Legacy CTC-decode reward (kept for backward compatibility).
                    decoded = self._decode_buffer()
                    if self.gt_word_idx < self.total_gt_words:
                        gt = self.gt_words[self.gt_word_idx]
                        reward = (
                            self.cfg.correct_reward if _fuzzy_match(decoded, gt)
                            else self.cfg.wrong_penalty
                        )
                        self.gt_word_idx += 1
                    else:
                        reward = self.cfg.wrong_penalty
                    self.emitted_words.append(decoded)

            elif mode == "hybrid":
                # Small per-step signal for credit assignment; terminal F1 added below.
                is_correct = (
                    self.oracle_emit is not None
                    and self.t < self.num_slots
                    and self.oracle_emit[self.t].item() > 0.5
                )
                reward = 0.2 if is_correct else -0.2

            elif mode == "word_match":
                # Acoustic word-recognition reward: decode accumulated CTC logits
                # and compare phones against the expected GT word's CMU phones.
                if self.gt_word_idx < self.total_gt_words:
                    expected_ids = (
                        self.word_phones_list[self.gt_word_idx]
                        if self.gt_word_idx < len(self.word_phones_list)
                        else []
                    )
                    decoded_ids = self._decode_ctc_phone_ids(
                        self.emit_start_slot, self.t
                    )
                    per = self._phone_per(decoded_ids, expected_ids)
                    # Smooth reward: 1.0 → correct_reward, 0.0 → 0; no negative here
                    # (the episode-end missing_word_penalty covers under-emit)
                    reward = max(0.0, 1.0 - per) * self.cfg.correct_reward
                    self.gt_word_idx += 1
                else:
                    # Over-emit: agent emitted more words than GT
                    reward = self.cfg.wrong_penalty

            # episode_f1: no intermediate emit reward (deferred to terminal)

            self.emit_start_slot = self.t + 1

        else:  # WAIT
            if mode == "per_slot":
                reward = self.cfg.wait_penalty
            # episode_f1 / hybrid / word_match: no wait penalty (promote any commit)

        self.t += 1
        if self.t >= self.num_slots:
            if mode == "per_slot":
                # incomplete_penalty only applies in legacy CTC mode.
                if self.oracle_emit is None:
                    remaining = self.total_gt_words - self.gt_word_idx
                    if remaining > 0:
                        reward += self.cfg.incomplete_penalty * remaining
            elif mode in ("episode_f1", "hybrid"):
                # Terminal F1 reward: compare agent emits to oracle slots.
                reward += self._compute_terminal_f1()
            elif mode == "word_match":
                # Penalise uncovered GT words (under-emission / missing recall).
                missing = max(0, self.total_gt_words - self.gt_word_idx)
                if missing > 0:
                    reward -= self.cfg.missing_word_penalty * missing
            self.done = True

        return reward, self.done

    def _compute_terminal_f1(self) -> float:
        """Compute F1 between agent_emit_slots and oracle slots at episode end.

        Uses cfg.f1_match_window for soft matching (0 = exact slot match).
        Returns the F1 value scaled by cfg.f1_reward_scale.
        """
        if not self._oracle_slot_set:
            # No oracle slots means the utterance has no words; reward 0.
            return 0.0

        agent_set = set(self.agent_emit_slots)
        if not agent_set:
            # Agent never emitted → recall = 0 → F1 = 0.
            return 0.0

        window = self.cfg.f1_match_window
        if window == 0:
            tp = len(agent_set & self._oracle_slot_set)
        else:
            # Soft matching: greedy assign each agent emit to nearest unmatched
            # oracle slot within ±window.
            oracle_sorted = sorted(self._oracle_slot_set)
            used_oracle: set = set()
            tp = 0
            for ae in sorted(agent_set):
                for oe in oracle_sorted:
                    if oe in used_oracle:
                        continue
                    if abs(ae - oe) <= window:
                        tp += 1
                        used_oracle.add(oe)
                        break

        precision = tp / len(agent_set)
        recall = tp / len(self._oracle_slot_set)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        return f1 * self.cfg.f1_reward_scale


def _fuzzy_match(decoded: str, ground_truth: str, threshold: float = 0.6) -> bool:
    """Phone-sequence level fuzzy match between decoded and GT word.

    Uses character-level edit distance ratio.  For phone-level outputs
    where we don't have a lexicon, this provides a lenient match.
    """
    if not decoded and not ground_truth:
        return True
    if not decoded or not ground_truth:
        return False
    d_lower = decoded.lower().strip()
    g_lower = ground_truth.lower().strip()
    if d_lower == g_lower:
        return True
    max_len = max(len(d_lower), len(g_lower))
    dist = _edit_distance(d_lower, g_lower)
    return (1.0 - dist / max_len) >= threshold


def _edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------------------
# PPO utilities
# ---------------------------------------------------------------------------


@dataclass
class Transition:
    belief: torch.Tensor
    prior: torch.Tensor
    syl_feat: torch.Tensor
    action: int
    log_prob: float
    value: float
    reward: float


@dataclass
class EpisodeBuffer:
    transitions: List[Transition] = field(default_factory=list)
    oracle_labels: Optional[torch.Tensor] = None

    def append(self, t: Transition) -> None:
        self.transitions.append(t)

    def __len__(self) -> int:
        return len(self.transitions)


def compute_gae(
    rewards: List[float],
    values: List[float],
    gamma: float = 0.99,
    lam: float = 0.95,
    last_value: float = 0.0,
) -> Tuple[List[float], List[float]]:
    """Generalized Advantage Estimation.

    Returns (advantages, returns) both as lists of length T.
    """
    T = len(rewards)
    advantages = [0.0] * T
    returns = [0.0] * T
    gae = 0.0
    next_value = last_value
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
        next_value = values[t]
    return advantages, returns


def ppo_clip_loss(
    logits: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """PPO clipped surrogate policy loss."""
    dist = torch.distributions.Categorical(logits=logits)
    new_log_probs = dist.log_prob(actions)
    ratio = (new_log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return -torch.min(surr1, surr2).mean()


def ppo_value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    """MSE value loss for the critic."""
    return F.mse_loss(values.squeeze(-1), returns)


def ppo_entropy_bonus(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of the policy for exploration."""
    dist = torch.distributions.Categorical(logits=logits)
    return dist.entropy().mean()


# ---------------------------------------------------------------------------
# GRPO utilities
# ---------------------------------------------------------------------------


def grpo_update(
    agent: "SchedulerAgent",
    optimizer: torch.optim.Optimizer,
    groups: "List[List[EpisodeBuffer]]",
    device: torch.device,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.02,
    grpo_mini_epochs: int = 1,
) -> Dict[str, float]:
    """Group Relative Policy Optimization (GRPO) update.

    Each element of ``groups`` is a list of G ``EpisodeBuffer`` objects
    collected from the **same** utterance using different stochastic rollouts.
    The episode-level reward (sum of step rewards) is normalised within the
    group to produce a group-relative advantage scalar, which is then used as
    a clipped REINFORCE signal applied to every transition in the episode.

    No value function / critic is needed, eliminating the unstable MSE
    optimisation that plagued the PPO baseline.

    Args:
        agent:           SchedulerAgent (raw, not DDP-wrapped).
        optimizer:       AdamW or similar.
        groups:          List of G-episode groups; len(groups) = num utterances
                         per update.  G = len(groups[0]).
        device:          Compute device.
        clip_eps:        PPO-style ratio clipping epsilon (0 → plain REINFORCE).
        entropy_coef:    Entropy regularisation coefficient.
        grpo_mini_epochs: Number of times to iterate over all groups per call.
                          Keep at 1 to avoid staleness of old_log_probs.

    Returns:
        Dict with "policy_loss" and "entropy" averaged over all updates.
    """
    if not groups:
        return {"policy_loss": 0.0, "entropy": 0.0}

    total_policy_loss = 0.0
    total_entropy = 0.0
    n_updates = 0

    agent.train()

    for _mini_epoch in range(grpo_mini_epochs):
        group_order = torch.randperm(len(groups)).tolist()

        for g_idx in group_order:
            group = groups[g_idx]
            G = len(group)
            if G == 0:
                continue

            # Episode-level rewards for group-relative normalisation.
            ep_rewards = [
                sum(t.reward for t in ep.transitions)
                for ep in group
            ]
            r_mean = sum(ep_rewards) / G
            r_var = sum((r - r_mean) ** 2 for r in ep_rewards) / max(G - 1, 1)
            r_std = math.sqrt(r_var) + 1e-8

            # Accumulate gradients across all G episodes before stepping.
            optimizer.zero_grad()
            group_policy_loss = 0.0

            for ep, ep_reward in zip(group, ep_rewards):
                if len(ep.transitions) == 0:
                    continue

                # Scalar advantage for every transition in this episode.
                adv = (ep_reward - r_mean) / r_std

                b = torch.stack([t.belief   for t in ep.transitions]).unsqueeze(0).to(device)  # (1,T,H)
                p = torch.stack([t.prior    for t in ep.transitions]).unsqueeze(0).to(device)  # (1,T,H)
                s = torch.stack([t.syl_feat for t in ep.transitions]).unsqueeze(0).to(device)  # (1,T,3)
                a = torch.tensor(
                    [t.action   for t in ep.transitions],
                    dtype=torch.long, device=device,
                )  # (T,)
                old_lp = torch.tensor(
                    [t.log_prob for t in ep.transitions],
                    dtype=torch.float32, device=device,
                )  # (T,)

                # Full-sequence forward pass with correct GRU hidden state.
                logits, _, _ = agent(b, p, s)
                logits = logits.squeeze(0)  # (T, 2)

                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(a)  # (T,)
                ent = dist.entropy().mean()

                if clip_eps > 0.0:
                    ratio = (new_lp - old_lp).exp()
                    surr1 = ratio * adv
                    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv
                    p_loss = -torch.min(surr1, surr2).mean()
                else:
                    p_loss = -(new_lp * adv).mean()

                loss = (p_loss - entropy_coef * ent) / G
                loss.backward()

                group_policy_loss += p_loss.item()
                total_entropy += ent.item() / G  # normalize within group

            nn.utils.clip_grad_norm_(agent.parameters(), 5.0)
            optimizer.step()

            total_policy_loss += group_policy_loss / max(G, 1)
            n_updates += 1

    n_updates = max(n_updates, 1)
    return {
        "policy_loss": total_policy_loss / n_updates,
        "entropy":     total_entropy     / n_updates,
    }


# ---------------------------------------------------------------------------
# Sequential and batched rollout collection
# ---------------------------------------------------------------------------


def collect_episode(
    agent: SchedulerAgent,
    env: "ASRSchedulerEnv",
    device: torch.device,
    temperature: float = 1.0,
) -> EpisodeBuffer:
    """Roll out one episode and collect transitions (sequential, batch=1).

    Args:
        temperature: Softmax temperature applied to policy logits before
            sampling.  Values > 1.0 increase action diversity (useful for
            GRPO rollout collection when the policy is near-deterministic);
            1.0 (default) is standard sampling.
    """
    agent.eval()
    env.reset()
    buf = EpisodeBuffer(
        oracle_labels=env.oracle_emit[:env.num_slots].clone()
        if env.oracle_emit is not None else None
    )
    hidden = None

    while not env.done:
        belief, prior, syl_feat = env.observe()
        b = belief.unsqueeze(0).to(device)
        p = prior.unsqueeze(0).to(device)
        s = syl_feat.unsqueeze(0).to(device)

        with torch.no_grad():
            logits, value_t, hidden = agent(b, p, s, hidden)
        logits = logits.squeeze(0)  # (2,)
        dist = torch.distributions.Categorical(logits=logits / temperature)
        action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action, device=logits.device)).item()
        value = value_t.squeeze().item()

        reward, done = env.step(action)

        buf.append(Transition(
            belief=belief, prior=prior, syl_feat=syl_feat,
            action=action, log_prob=log_prob, value=value, reward=reward,
        ))

    return buf


@torch.no_grad()
def _collect_word_match_batched(
    agent: SchedulerAgent,
    env: "ASRSchedulerEnv",
    device: torch.device,
    n_rollouts: int,
    temperature: float = 1.0,
) -> List[EpisodeBuffer]:
    """Batched rollout collection for word_match reward mode.

    Runs all N rollouts in a single batched GRU forward per time step (same as
    the non-word_match batched path).  The only part that cannot be vectorised —
    per-rollout CTC decoding for the word_match reward — is handled by a tight
    Python loop over emitting rollouts at each step.  Since CTC decode is just
    argmax + blank-collapse on a small logit slice (cheap CPU work), the total
    cost is dominated by the batched GPU forward, giving ~N× speedup over the
    fully sequential fallback.

    Reward semantics are identical to ``ASRSchedulerEnv.step`` in word_match
    mode: on EMIT, decode accumulated CTC logits and compare to GT word phones;
    on last slot, penalise missed GT words.
    """
    K = env.num_slots
    N = n_rollouts

    # ── Pre-compute time-invariant data ──────────────────────────────────────
    beliefs_K = env.beliefs[:K].to(device)   # (K, H)
    priors_K  = env.priors[:K].to(device)    # (K, H)
    bnd = env.boundaries[:K].to(device)      # (K, 2)

    durations = (bnd[:, 1] - bnd[:, 0]).float().clamp(min=1.0).log()  # (K,)
    rel_pos   = torch.arange(K, device=device).float() / max(K - 1, 1)  # (K,)

    # CTC decode state (accessed directly to avoid per-call method overhead)
    canonical_logits = env.canonical_logits  # (num_frames, vocab)
    up_slot_mask     = env.up_slot_mask      # (num_frames,)
    F_up             = env.upsample_factor
    blank_id         = env.vocab.blank_id
    vocab_tokens     = env.vocab.tokens
    vocab_len        = len(vocab_tokens)

    # Pre-compute per-slot frame ranges and valid frame counts (shared across rollouts)
    slot_sf   = [t * F_up for t in range(K)]
    slot_ef   = [(t + 1) * F_up for t in range(K)]
    slot_vlen = [int(up_slot_mask[slot_sf[t]:slot_ef[t]].sum().item()) for t in range(K)]

    total_gt_words   = env.total_gt_words
    word_phones_list = env.word_phones_list
    cfg              = env.cfg
    oracle           = env.oracle_emit
    oracle_labels    = oracle[:K].clone() if oracle is not None else None

    # ── Per-rollout mutable state ─────────────────────────────────────────────
    steps_since  = torch.zeros(N, device=device)   # steps since last emit
    emit_start   = [0] * N                          # start slot of current word window
    gt_idx       = [0] * N                          # GT word pointer per rollout

    # Storage (indexed by time step, then assembled into EpisodeBuffers)
    s_beliefs:    List[torch.Tensor] = []  # (K,) each: belief at step t (shared)
    s_priors:     List[torch.Tensor] = []
    s_syl_feats:  List[torch.Tensor] = []  # (K, N, 3)
    s_actions:    List[torch.Tensor] = []  # (K, N)
    s_log_probs:  List[torch.Tensor] = []  # (K, N)
    s_values:     List[torch.Tensor] = []  # (K, N)
    s_rewards:    List[List[float]]  = [[] for _ in range(N)]  # (N, K)

    hidden: Optional[torch.Tensor] = None

    for t in range(K):
        # ── Batched syl_feat ─────────────────────────────────────────────────
        syl_feat_t = torch.stack([
            durations[t].expand(N),    # (N,)
            rel_pos[t].expand(N),      # (N,)
            steps_since,               # (N,)
        ], dim=-1)  # (N, 3)

        # ── Batched GRU forward: batch=N, seq_len=1 ──────────────────────────
        b_t = beliefs_K[t].unsqueeze(0).expand(N, -1).unsqueeze(1)   # (N, 1, H)
        p_t = priors_K[t].unsqueeze(0).expand(N, -1).unsqueeze(1)    # (N, 1, H)
        s_t = syl_feat_t.unsqueeze(1)                                  # (N, 1, 3)

        logits_t, values_t, hidden = agent(b_t, p_t, s_t, hidden)
        # logits_t: (N, 1, 2);  values_t: (N, 1, 1)

        logits_sq   = logits_t.squeeze(1)                              # (N, 2)
        sample_dist = torch.distributions.Categorical(logits=logits_sq / temperature)
        actions_t   = sample_dist.sample()                             # (N,)
        lp_dist     = torch.distributions.Categorical(logits=logits_sq)
        log_probs_t = lp_dist.log_prob(actions_t)                     # (N,)

        # ── Per-rollout word_match reward (CPU, cheap) ───────────────────────
        is_last = (t == K - 1)
        rewards_step: List[float] = []
        # One bulk transfer instead of N individual .item() GPU syncs
        actions_list_t: List[int] = actions_t.cpu().tolist()

        for n, a in enumerate(actions_list_t):
            r = 0.0

            if a == EMIT:
                if gt_idx[n] < total_gt_words:
                    expected_ids: List[int] = (
                        word_phones_list[gt_idx[n]]
                        if gt_idx[n] < len(word_phones_list) else []
                    )
                    # Greedy CTC decode of canonical_logits[emit_start[n]..t]
                    sf          = emit_start[n] * F_up
                    ef          = slot_ef[t]
                    logit_slice = canonical_logits[sf:ef]
                    valid_len   = int(up_slot_mask[sf:ef].sum().item())
                    if valid_len > 0:
                        raw_ids = logit_slice[:valid_len].argmax(dim=-1).tolist()
                        decoded_ids: List[int] = []
                        prev = blank_id
                        for pid in raw_ids:
                            if pid == blank_id:
                                prev = blank_id
                                continue
                            if pid != prev:
                                decoded_ids.append(pid)
                            prev = pid
                    else:
                        decoded_ids = []

                    # Stress-normalised PER
                    dec_base = [vocab_tokens[pid].rstrip("012")
                                for pid in decoded_ids if pid < vocab_len]
                    exp_base = [vocab_tokens[pid].rstrip("012")
                                for pid in expected_ids if pid < vocab_len]
                    if exp_base:
                        per = min(levenshtein_distance(dec_base, exp_base) / len(exp_base), 1.0)
                    else:
                        per = 0.0

                    r = max(0.0, 1.0 - per) * cfg.correct_reward
                    gt_idx[n] += 1
                else:
                    r = cfg.wrong_penalty  # over-emit

                emit_start[n] = t + 1

            # Terminal: penalise missed GT words on last slot
            if is_last and gt_idx[n] < total_gt_words:
                r -= cfg.missing_word_penalty * (total_gt_words - gt_idx[n])

            rewards_step.append(r)

        # ── Update per-rollout steps_since_emit ───────────────────────────────
        emitting    = (actions_t == EMIT)
        steps_since = torch.where(emitting, torch.zeros_like(steps_since), steps_since + 1)

        # ── Store step data ───────────────────────────────────────────────────
        s_beliefs.append(beliefs_K[t].cpu())
        s_priors.append(priors_K[t].cpu())
        s_syl_feats.append(syl_feat_t.cpu())          # (N, 3)
        s_actions.append(actions_t.cpu())              # (N,)
        s_log_probs.append(log_probs_t.cpu())          # (N,)
        s_values.append(values_t.squeeze(-1).squeeze(-1).cpu())  # (N,)
        for n in range(N):
            s_rewards[n].append(rewards_step[n])

    # ── Assemble EpisodeBuffers ───────────────────────────────────────────────
    buffers: List[EpisodeBuffer] = []
    for n in range(N):
        buf = EpisodeBuffer(oracle_labels=oracle_labels)
        for t in range(K):
            buf.append(Transition(
                belief=s_beliefs[t],
                prior=s_priors[t],
                syl_feat=s_syl_feats[t][n],
                action=s_actions[t][n].item(),
                log_prob=s_log_probs[t][n].item(),
                value=s_values[t][n].item(),
                reward=s_rewards[n][t],
            ))
        buffers.append(buf)

    return buffers


@torch.no_grad()
def collect_episodes_batched(
    agent: SchedulerAgent,
    env: "ASRSchedulerEnv",
    device: torch.device,
    n_rollouts: int,
    temperature: float = 1.0,
) -> List[EpisodeBuffer]:
    """Run *n_rollouts* rollouts of the same utterance in a single batched forward.

    Instead of n_rollouts sequential episodes (each with batch=1), this runs
    all rollouts simultaneously with batch=n_rollouts at every time step.
    Reduces Python overhead by ~n_rollouts× and increases GPU utilisation
    from ~13% to 50%+ for typical agent sizes.

    Supports reward_mode: ``per_slot``, ``hybrid``, ``episode_f1``.
    Falls back to sequential ``collect_episode`` for ``word_match`` (CTC decode
    per rollout is hard to vectorise; use hybrid instead for best performance).

    Args:
        temperature: Softmax temperature applied to policy logits before
            sampling.  Values > 1.0 diversify rollouts when the policy is
            near-deterministic, preventing GRPO group advantages from
            collapsing to zero.  The temperature is applied at sample time
            only; stored log_probs use the *unscaled* logits so that the
            GRPO ratio computation remains correct.

    Returns:
        List of n_rollouts ``EpisodeBuffer`` objects, identical in structure to
        sequential collection and compatible with ``grpo_update``.
    """
    cfg = env.cfg

    K = env.num_slots
    if K == 0:
        return [EpisodeBuffer() for _ in range(n_rollouts)]

    if cfg.reward_mode == "word_match":
        # Batch the GRU forward across all N rollouts; only serialize the
        # per-rollout CTC decode reward (cheap CPU argmax + levenshtein).
        # Speedup vs fully sequential: ~N× (N=32 → ~20× wall-clock reduction).
        return _collect_word_match_batched(agent, env, device, n_rollouts, temperature)

    N = n_rollouts

    # ── Pre-compute time-invariant features ──────────────────────────────────
    beliefs_K = env.beliefs[:K].to(device)   # (K, H)
    priors_K  = env.priors[:K].to(device)    # (K, H)
    bnd = env.boundaries[:K].to(device)      # (K, 2)

    durations = (bnd[:, 1] - bnd[:, 0]).float().clamp(min=1.0).log()  # (K,)
    rel_pos   = torch.arange(K, device=device).float() / max(K - 1, 1) # (K,)

    # Oracle info for reward computation
    oracle = env.oracle_emit        # (K,) or None
    oracle_mask = (
        oracle[:K].to(device) if oracle is not None
        else torch.zeros(K, device=device)
    )  # (K,) float: 1.0 at oracle slots

    # ── Per-rollout mutable state ─────────────────────────────────────────────
    # steps_since_emit: t - last_emit_slot (the only per-rollout varying feature)
    steps_since = torch.zeros(N, device=device)  # (N,)

    # Track emitted slots per rollout for episode_f1 terminal reward
    agent_emit_sets: List[set] = [set() for _ in range(N)]

    # Storage: one entry per time step
    all_beliefs:   List[torch.Tensor] = []
    all_priors:    List[torch.Tensor] = []
    all_syl_feats: List[torch.Tensor] = []
    all_actions:   List[torch.Tensor] = []   # (N,) int
    all_log_probs: List[torch.Tensor] = []   # (N,) float
    all_values:    List[torch.Tensor] = []   # (N,) float
    all_rewards:   List[torch.Tensor] = []   # (N,) float

    hidden: Optional[torch.Tensor] = None

    for t in range(K):
        # ── Build batched syl_feat ────────────────────────────────────────────
        syl_feat_t = torch.stack([
            durations[t].expand(N),    # (N,) — same for all rollouts
            rel_pos[t].expand(N),      # (N,)
            steps_since,               # (N,) — differs per rollout
        ], dim=-1)  # (N, 3)

        # ── Batched agent forward: (N, 1, dim) ───────────────────────────────
        b_t = beliefs_K[t].unsqueeze(0).expand(N, -1).unsqueeze(1)  # (N, 1, H)
        p_t = priors_K[t].unsqueeze(0).expand(N, -1).unsqueeze(1)   # (N, 1, H)
        s_t = syl_feat_t.unsqueeze(1)                                # (N, 1, 3)

        logits_t, values_t, hidden = agent(b_t, p_t, s_t, hidden)
        # logits_t: (N, 1, 2),  values_t: (N, 1, 1)

        logits_sq = logits_t.squeeze(1)  # (N, 2)
        # Sample actions using temperature-scaled logits for diversity.
        # Store log_probs from the *unscaled* logits so GRPO ratio is correct.
        sample_dist = torch.distributions.Categorical(logits=logits_sq / temperature)
        actions_t   = sample_dist.sample()                                     # (N,)
        lp_dist     = torch.distributions.Categorical(logits=logits_sq)
        log_probs_t = lp_dist.log_prob(actions_t)                              # (N,)

        # ── Per-rollout rewards ───────────────────────────────────────────────
        emitting  = (actions_t == EMIT)  # (N,) bool
        is_oracle = oracle_mask[t].item() > 0.5

        if cfg.reward_mode == "per_slot":
            correct = emitting &  is_oracle
            wrong   = emitting & (not is_oracle)
            rewards_t = torch.where(
                correct.bool(),
                torch.full((N,), cfg.correct_reward, device=device),
                torch.where(
                    wrong.bool(),
                    torch.full((N,), cfg.wrong_penalty, device=device),
                    torch.full((N,), cfg.wait_penalty,  device=device),
                ),
            )
        elif cfg.reward_mode == "hybrid":
            correct = emitting & is_oracle
            wrong   = emitting & (not is_oracle)
            rewards_t = torch.where(
                correct.bool(),
                torch.full((N,), 0.2, device=device),
                torch.where(
                    wrong.bool(),
                    torch.full((N,), -0.2, device=device),
                    torch.zeros(N, device=device),
                ),
            )
        else:  # episode_f1 — zero intermediate
            rewards_t = torch.zeros(N, device=device)

        # Track emitted slots for terminal F1 computation
        for i in range(N):
            if emitting[i]:
                agent_emit_sets[i].add(t)

        # ── Update per-rollout state ──────────────────────────────────────────
        steps_since = torch.where(
            emitting,
            torch.zeros(N, device=device),
            steps_since + 1.0,
        )

        all_beliefs.append(beliefs_K[t].cpu())
        all_priors.append(priors_K[t].cpu())
        all_syl_feats.append(syl_feat_t.cpu())
        all_actions.append(actions_t.cpu())
        all_log_probs.append(log_probs_t.cpu())
        all_values.append(values_t.squeeze(-1).squeeze(-1).cpu())
        all_rewards.append(rewards_t.cpu())

    # ── Terminal rewards (episode_f1 / hybrid) ────────────────────────────────
    if cfg.reward_mode in ("episode_f1", "hybrid") and env._oracle_slot_set:
        oracle_set = env._oracle_slot_set
        window = cfg.f1_match_window
        for i in range(N):
            ae = agent_emit_sets[i]
            if not ae:
                f1 = 0.0
            elif not oracle_set:
                f1 = 0.0
            else:
                if window == 0:
                    tp = len(ae & oracle_set)
                else:
                    oracle_s = sorted(oracle_set)
                    used: set = set()
                    tp = 0
                    for a_slot in sorted(ae):
                        for o_slot in oracle_s:
                            if o_slot in used:
                                continue
                            if abs(a_slot - o_slot) <= window:
                                tp += 1
                                used.add(o_slot)
                                break
                prec = tp / len(ae)
                rec  = tp / len(oracle_set)
                f1   = 2.0 * prec * rec / max(prec + rec, 1e-8)
            all_rewards[-1][i] = all_rewards[-1][i] + f1 * cfg.f1_reward_scale

    # ── Assemble EpisodeBuffer per rollout ────────────────────────────────────
    episodes: List[EpisodeBuffer] = []
    for i in range(N):
        ep = EpisodeBuffer(
            oracle_labels=oracle[:K].clone() if oracle is not None else None
        )
        for t in range(K):
            ep.append(Transition(
                belief   = all_beliefs[t],
                prior    = all_priors[t],
                syl_feat = all_syl_feats[t][i],
                action   = int(all_actions[t][i].item()),
                log_prob = float(all_log_probs[t][i].item()),
                value    = float(all_values[t][i].item()),
                reward   = float(all_rewards[t][i].item()),
            ))
        episodes.append(ep)
    return episodes
