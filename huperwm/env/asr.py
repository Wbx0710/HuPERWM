"""ASR Scheduler Environment and policy-gradient rollout utilities.

The environment steps through syllable slots left-to-right. On each step the
ActiveAgent receives (belief, prior, syl_feat, error_vec, prev_action) and
outputs a WAIT/EMIT action.  On EMIT, CTC logits accumulated since the last
emit are decoded and compared against the expected ground-truth word
(word_match mode).

Reward modes:
    "per_slot"   — per-step binary reward against oracle slots.
    "episode_f1" — zero step rewards; terminal F1 against oracle.
    "hybrid"     — small per-step credit (±0.2) + terminal F1.
    "word_match" — acoustic word recognition: (1 − phonePER) per EMIT.

POMDP extensions:
    info_gain_scale > 0 — WAIT receives a bonus reward proportional to the
        decrease in L2 distortion norm from the current slot to the next,
        implementing the POMDP principle that information-gathering actions
        have intrinsic value.

Update algorithm (grpo_update):
    Group Relative Policy Optimization + Generalized Advantage Estimation.
    G rollouts of the same utterance form a group.  Per-step advantages are
    computed via GAE (Schulman et al. 2016) using the value_head predictions,
    then group-normalised to remove utterance-level difficulty bias.  A value
    function loss (MSE) trains the critic alongside the policy.

    Per-step GAE is superior to episode-level GRPO for long sequences with
    sparse rewards: each WAIT/EMIT decision gets its own credit-assigned
    advantage, and the value baseline reduces variance at every step.
    Reference: Ni et al. (2022) "Recurrent Model-Free RL Can Be a Strong
    Baseline for Many POMDPs" — confirms GRU + GAE matches specialised POMDP
    algorithms on 18/21 benchmarks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from huperwm.data.vocab import Vocabulary, levenshtein_distance
from huperwm.model.agent import ActiveAgent

WAIT = 0
EMIT = 1


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------


@dataclass
class EnvConfig:
    wait_penalty: float = -0.01
    correct_reward: float = 1.0
    wrong_penalty: float = -0.5
    incomplete_penalty: float = 0.0
    upsample_factor: int = 4
    reward_mode: str = "per_slot"   # per_slot | episode_f1 | hybrid | word_match
    f1_reward_scale: float = 1.0
    f1_match_window: int = 0
    missing_word_penalty: float = 0.5
    # POMDP: reward WAIT when distortion norm decreases (information gathering).
    info_gain_scale: float = 0.0


# ---------------------------------------------------------------------------
# Transition and episode buffer
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
    error: Optional[torch.Tensor] = None  # (H,) distortion vector
    prev_action: Optional[int] = None     # action taken at previous step


@dataclass
class EpisodeBuffer:
    transitions: List[Transition] = field(default_factory=list)
    oracle_labels: Optional[torch.Tensor] = None

    def append(self, t: Transition) -> None:
        self.transitions.append(t)

    def __len__(self) -> int:
        return len(self.transitions)


# ---------------------------------------------------------------------------
# CTC decode helpers
# ---------------------------------------------------------------------------


def decode_ctc_to_phones(logits: torch.Tensor, vocab: Vocabulary) -> List[str]:
    """Greedy CTC decode a (T, V) logit tensor → phone list."""
    return vocab.decode_ctc(logits.argmax(dim=-1).tolist())


def phones_to_words(phones: List[str], silence: str = "SIL") -> List[str]:
    """Split a phone sequence into words at silence boundaries."""
    if not phones:
        return []
    words: List[str] = []
    current: List[str] = []
    for ph in phones:
        if ph.upper() == silence or ph in ("|", " "):
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


def _fuzzy_match(decoded: str, ground_truth: str, threshold: float = 0.6) -> bool:
    if not decoded and not ground_truth:
        return True
    if not decoded or not ground_truth:
        return False
    d = decoded.lower().strip()
    g = ground_truth.lower().strip()
    if d == g:
        return True
    max_len = max(len(d), len(g))
    dist = levenshtein_distance(list(d), list(g))
    return (1.0 - dist / max_len) >= threshold


class ASRSchedulerEnv:
    """Single-utterance RL environment for the streaming word scheduler."""

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
        distortion_vectors: torch.Tensor | None = None,  # (K, H) per-dim errors
    ) -> None:
        self.beliefs = beliefs
        self.priors = priors
        self.boundaries = boundaries
        self.canonical_logits = canonical_logits
        self.up_slot_mask = up_slot_mask
        self.slot_mask = slot_mask
        self.gt_words = gt_words
        self.vocab = phone_vocab
        self.cfg = env_cfg or EnvConfig()
        self.oracle_emit = oracle_emit
        self.word_phones_list: List[List[int]] = word_phones_list or []
        self.distortion_vectors = distortion_vectors

        self.num_slots = int(slot_mask.sum().item())
        self.upsample_factor = self.cfg.upsample_factor
        self.total_gt_words = len(gt_words)

        if oracle_emit is not None:
            self.total_oracle_slots = int(oracle_emit[:self.num_slots].sum().item())
            self._oracle_slot_set: set = {
                t for t in range(self.num_slots) if oracle_emit[t].item() > 0.5
            }
        else:
            self.total_oracle_slots = self.total_gt_words
            self._oracle_slot_set = set()

        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.emit_start_slot = 0
        self.gt_word_idx = 0
        self.emitted_words: List[str] = []
        self.agent_emit_slots: List[int] = []
        self.done = False
        self.last_action: int = WAIT

    def _decode_buffer(self) -> str:
        start_frame = self.emit_start_slot * self.upsample_factor
        end_frame = (self.t + 1) * self.upsample_factor
        logit_slice = self.canonical_logits[start_frame:end_frame]
        mask_slice = self.up_slot_mask[start_frame:end_frame]
        valid_len = int(mask_slice.sum().item())
        if valid_len == 0:
            return ""
        phones = decode_ctc_to_phones(logit_slice[:valid_len], self.vocab)
        words = phones_to_words(phones)
        return " ".join(words) if words else ""

    def _compute_syl_feat(self) -> torch.Tensor:
        device = self.beliefs.device
        if self.t >= self.num_slots:
            return torch.zeros(3, device=device)
        bnd = self.boundaries[self.t]
        log_dur = (bnd[1] - bnd[0]).float().clamp(min=1.0).log()
        rel_pos = torch.tensor(self.t / max(self.num_slots - 1, 1), device=device, dtype=torch.float)
        steps_since = torch.tensor(self.t - self.emit_start_slot, device=device, dtype=torch.float)
        return torch.stack([log_dur, rel_pos, steps_since])

    def _get_error(self) -> torch.Tensor | None:
        """Return distortion_vectors[t] (H-dim) or None."""
        if self.distortion_vectors is not None and self.t < self.distortion_vectors.shape[0]:
            return self.distortion_vectors[self.t]  # (H,)
        return None

    def observe(self) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor | None, int,
    ]:
        """Return (belief_t, prior_t, syl_feat_t, error_t, prev_action).

        prev_action is the action taken at step t-1 (WAIT=0 at t=0).
        error_t is H-dim distortion vector or None.
        """
        return (
            self.beliefs[self.t],
            self.priors[self.t],
            self._compute_syl_feat(),
            self._get_error(),
            self.last_action,
        )

    @staticmethod
    def _normalize_phone(ph: str) -> str:
        return ph.rstrip("012")

    def _decode_ctc_phone_ids(self, start_slot: int, end_slot: int) -> List[int]:
        F = self.upsample_factor
        logit_slice = self.canonical_logits[start_slot * F:(end_slot + 1) * F]
        mask_slice = self.up_slot_mask[start_slot * F:(end_slot + 1) * F]
        valid_len = int(mask_slice.sum().item())
        if valid_len == 0:
            return []
        raw_ids = logit_slice[:valid_len].argmax(dim=-1).tolist()
        blank = self.vocab.blank_id
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

    def _phone_per(self, decoded_ids: List[int], expected_ids: List[int]) -> float:
        if not expected_ids:
            return 0.0
        tokens = self.vocab.tokens
        dec_base = [self._normalize_phone(tokens[pid]) for pid in decoded_ids if pid < len(tokens)]
        exp_base = [self._normalize_phone(tokens[pid]) for pid in expected_ids if pid < len(tokens)]
        if not exp_base:
            return 0.0
        return min(levenshtein_distance(dec_base, exp_base) / len(exp_base), 1.0)

    def _f1_reward(self) -> float:
        if not self._oracle_slot_set:
            return 0.0
        agent_set = set(self.agent_emit_slots)
        if not agent_set:
            return 0.0
        window = self.cfg.f1_match_window
        if window == 0:
            tp = len(agent_set & self._oracle_slot_set)
        else:
            oracle_sorted = sorted(self._oracle_slot_set)
            used: set = set()
            tp = 0
            for ae in sorted(agent_set):
                for oe in oracle_sorted:
                    if oe not in used and abs(ae - oe) <= window:
                        tp += 1
                        used.add(oe)
                        break
        precision = tp / len(agent_set)
        recall = tp / len(self._oracle_slot_set)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        return f1 * self.cfg.f1_reward_scale

    def _info_gain_reward(self) -> float:
        """POMDP information gain: reward WAIT when distortion norm decreases next step."""
        if self.cfg.info_gain_scale <= 0.0 or self.distortion_vectors is None:
            return 0.0
        if self.t + 1 >= self.distortion_vectors.shape[0]:
            return 0.0
        d_now = self.distortion_vectors[self.t].norm().item()
        d_next = self.distortion_vectors[self.t + 1].norm().item()
        return self.cfg.info_gain_scale * max(0.0, d_now - d_next)

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
                    is_correct = self.t < self.num_slots and self.oracle_emit[self.t].item() > 0.5
                    reward = self.cfg.correct_reward if is_correct else self.cfg.wrong_penalty
                else:
                    decoded = self._decode_buffer()
                    if self.gt_word_idx < self.total_gt_words:
                        gt = self.gt_words[self.gt_word_idx]
                        reward = self.cfg.correct_reward if _fuzzy_match(decoded, gt) else self.cfg.wrong_penalty
                        self.gt_word_idx += 1
            elif mode == "hybrid":
                if self.oracle_emit is not None:
                    is_correct = self.t < self.num_slots and self.oracle_emit[self.t].item() > 0.5
                    reward = 0.2 if is_correct else -0.2
            elif mode == "word_match":
                if self.gt_word_idx < self.total_gt_words:
                    expected_ids = (
                        self.word_phones_list[self.gt_word_idx]
                        if self.gt_word_idx < len(self.word_phones_list) else []
                    )
                    decoded_ids = self._decode_ctc_phone_ids(self.emit_start_slot, self.t)
                    per = self._phone_per(decoded_ids, expected_ids)
                    reward = max(0.0, 1.0 - per) * self.cfg.correct_reward
                    self.gt_word_idx += 1
                else:
                    reward = self.cfg.wrong_penalty
            self.emit_start_slot = self.t + 1
        else:  # WAIT
            if mode == "per_slot":
                reward = self.cfg.wait_penalty
            # POMDP: information-gathering bonus for WAIT when uncertainty decreases.
            reward += self._info_gain_reward()

        self.last_action = action
        self.t += 1
        if self.t >= self.num_slots:
            self.done = True
            if mode == "episode_f1":
                reward = self._f1_reward()
            elif mode == "hybrid":
                reward += self._f1_reward()
            elif mode == "word_match" and self.gt_word_idx < self.total_gt_words:
                reward -= self.cfg.missing_word_penalty * (self.total_gt_words - self.gt_word_idx)

        return reward, self.done


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------


def collect_episode(
    agent: ActiveAgent,
    env: ASRSchedulerEnv,
    device: torch.device,
    temperature: float = 1.0,
) -> EpisodeBuffer:
    """Roll out one episode sequentially (batch=1).

    Inputs to the agent are shaped (1, dim) — 2D — matching the squeeze path
    in ActiveAgent.forward() which handles single-step input without a time axis.
    """
    agent.eval()
    env.reset()
    buf = EpisodeBuffer(
        oracle_labels=env.oracle_emit[:env.num_slots].clone()
        if env.oracle_emit is not None else None
    )
    hidden = None

    while not env.done:
        belief, prior, syl_feat, error_obs, prev_act = env.observe()
        b = belief.unsqueeze(0).to(device)      # (1, H)
        p = prior.unsqueeze(0).to(device)       # (1, H)
        s = syl_feat.unsqueeze(0).to(device)    # (1, 3)
        err = (
            error_obs.unsqueeze(0).to(device)   # (1, H)
            if error_obs is not None
            else torch.zeros(1, agent.cfg.belief_dim, device=device)
        )
        prev_action_t = torch.tensor([prev_act], dtype=torch.long, device=device)

        with torch.no_grad():
            logits, value_t, hidden = agent(b, p, s, err, hidden, prev_action_t)
        logits = logits.squeeze(0)
        dist = torch.distributions.Categorical(logits=logits / temperature)
        action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action, device=logits.device)).item()
        value = value_t.squeeze().item()

        reward, done = env.step(action)
        buf.append(Transition(
            belief=belief, prior=prior, syl_feat=syl_feat,
            action=action, log_prob=log_prob, value=value, reward=reward,
            error=error_obs, prev_action=prev_act,
        ))

    return buf


@torch.no_grad()
def _collect_word_match_batched(
    agent: ActiveAgent,
    env: ASRSchedulerEnv,
    device: torch.device,
    n_rollouts: int,
    temperature: float = 1.0,
) -> List[EpisodeBuffer]:
    """Batched rollout for word_match mode: N rollouts in a single forward per step."""
    K = env.num_slots
    N = n_rollouts
    H = agent.cfg.belief_dim

    beliefs_K = env.beliefs[:K].to(device)
    priors_K = env.priors[:K].to(device)
    bnd = env.boundaries[:K].to(device)

    distortions_K = (
        env.distortion_vectors[:K].to(device)  # (K, H)
        if env.distortion_vectors is not None
        else None
    )

    durations = (bnd[:, 1] - bnd[:, 0]).float().clamp(min=1.0).log()
    rel_pos = torch.arange(K, device=device).float() / max(K - 1, 1)

    canonical_logits = env.canonical_logits
    up_slot_mask = env.up_slot_mask
    F_up = env.upsample_factor
    blank_id = env.vocab.blank_id
    vocab_tokens = env.vocab.tokens
    vocab_len = len(vocab_tokens)

    slot_sf = [t * F_up for t in range(K)]
    slot_ef = [(t + 1) * F_up for t in range(K)]

    total_gt_words = env.total_gt_words
    word_phones_list = env.word_phones_list
    cfg = env.cfg
    oracle = env.oracle_emit
    oracle_labels = oracle[:K].clone() if oracle is not None else None

    steps_since = torch.zeros(N, device=device)
    emit_start = [0] * N
    gt_idx = [0] * N
    prev_actions = torch.zeros(N, dtype=torch.long, device=device)

    s_beliefs: List[torch.Tensor] = []
    s_priors: List[torch.Tensor] = []
    s_syl_feats: List[torch.Tensor] = []
    s_errors: List[Optional[torch.Tensor]] = []
    s_prev_actions: List[torch.Tensor] = []
    s_actions: List[torch.Tensor] = []
    s_log_probs: List[torch.Tensor] = []
    s_values: List[torch.Tensor] = []
    s_rewards: List[List[float]] = [[] for _ in range(N)]

    hidden: Optional[torch.Tensor] = None

    for t in range(K):
        syl_feat_t = torch.stack([durations[t].expand(N), rel_pos[t].expand(N), steps_since], dim=-1)
        b_t = beliefs_K[t].unsqueeze(0).expand(N, -1).unsqueeze(1)
        p_t = priors_K[t].unsqueeze(0).expand(N, -1).unsqueeze(1)
        s_t = syl_feat_t.unsqueeze(1)

        d_t = (
            distortions_K[t].unsqueeze(0).expand(N, -1).unsqueeze(1)  # (N, 1, H)
            if distortions_K is not None
            else torch.zeros(N, 1, H, device=device)
        )

        prev_act_t = prev_actions.unsqueeze(1)

        logits_t, values_t, hidden = agent(b_t, p_t, s_t, d_t, hidden, prev_act_t)
        logits_sq = logits_t.squeeze(1)
        sample_dist = torch.distributions.Categorical(logits=logits_sq / temperature)
        actions_t = sample_dist.sample()
        lp_dist = torch.distributions.Categorical(logits=logits_sq)
        log_probs_t = lp_dist.log_prob(actions_t)

        is_last = (t == K - 1)
        rewards_step: List[float] = []
        actions_list_t: List[int] = actions_t.cpu().tolist()

        for n, a in enumerate(actions_list_t):
            r = 0.0
            if a == EMIT:
                if gt_idx[n] < total_gt_words:
                    expected_ids: List[int] = (
                        word_phones_list[gt_idx[n]] if gt_idx[n] < len(word_phones_list) else []
                    )
                    sf = emit_start[n] * F_up
                    ef = slot_ef[t]
                    logit_slice = canonical_logits[sf:ef]
                    valid_len = int(up_slot_mask[sf:ef].sum().item())
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

                    dec_base = [vocab_tokens[pid].rstrip("012") for pid in decoded_ids if pid < vocab_len]
                    exp_base = [vocab_tokens[pid].rstrip("012") for pid in expected_ids if pid < vocab_len]
                    per = (min(levenshtein_distance(dec_base, exp_base) / len(exp_base), 1.0)
                           if exp_base else 0.0)
                    r = max(0.0, 1.0 - per) * cfg.correct_reward
                    gt_idx[n] += 1
                else:
                    r = cfg.wrong_penalty
                emit_start[n] = t + 1
            else:
                # POMDP: info gain reward for WAIT when distortion norm decreases.
                if cfg.info_gain_scale > 0.0 and distortions_K is not None and t + 1 < K:
                    d_now = distortions_K[t].norm().item()
                    d_next = distortions_K[t + 1].norm().item()
                    r += cfg.info_gain_scale * max(0.0, d_now - d_next)

            if is_last and gt_idx[n] < total_gt_words:
                r -= cfg.missing_word_penalty * (total_gt_words - gt_idx[n])
            rewards_step.append(r)

        emitting = (actions_t == EMIT)
        steps_since = torch.where(emitting, torch.zeros_like(steps_since), steps_since + 1)

        s_beliefs.append(beliefs_K[t].cpu())
        s_priors.append(priors_K[t].cpu())
        s_syl_feats.append(syl_feat_t.cpu())
        s_errors.append(distortions_K[t].cpu() if distortions_K is not None else None)
        s_prev_actions.append(prev_actions.cpu())
        s_actions.append(actions_t.cpu())
        s_log_probs.append(log_probs_t.cpu())
        s_values.append(values_t.squeeze(-1).squeeze(-1).cpu())
        for n in range(N):
            s_rewards[n].append(rewards_step[n])

        prev_actions = actions_t.clone()

    buffers: List[EpisodeBuffer] = []
    for n in range(N):
        buf = EpisodeBuffer(oracle_labels=oracle_labels)
        for t in range(K):
            err_t = s_errors[t]
            buf.append(Transition(
                belief=s_beliefs[t],
                prior=s_priors[t],
                syl_feat=s_syl_feats[t][n],
                action=s_actions[t][n].item(),
                log_prob=s_log_probs[t][n].item(),
                value=s_values[t][n].item(),
                reward=s_rewards[n][t],
                error=err_t,
                prev_action=s_prev_actions[t][n].item(),
            ))
        buffers.append(buf)
    return buffers


@torch.no_grad()
def collect_episodes_batched(
    agent: ActiveAgent,
    env: ASRSchedulerEnv,
    device: torch.device,
    n_rollouts: int,
    temperature: float = 1.0,
) -> List[EpisodeBuffer]:
    """Run n_rollouts of the same utterance in a single batched forward per step.

    Falls back to the word_match-specific batched path when in word_match mode
    (CTC decode per rollout cannot be vectorised but the GRU forward can).
    """
    if env.cfg.reward_mode == "word_match":
        return _collect_word_match_batched(agent, env, device, n_rollouts, temperature)

    K = env.num_slots
    N = n_rollouts
    H = agent.cfg.belief_dim

    beliefs_K = env.beliefs[:K].to(device)
    priors_K = env.priors[:K].to(device)
    bnd = env.boundaries[:K].to(device)

    distortions_K = (
        env.distortion_vectors[:K].to(device)  # (K, H)
        if env.distortion_vectors is not None
        else None
    )

    durations = (bnd[:, 1] - bnd[:, 0]).float().clamp(min=1.0).log()
    rel_pos = torch.arange(K, device=device).float() / max(K - 1, 1)

    oracle = env.oracle_emit
    oracle_set = env._oracle_slot_set
    oracle_labels = oracle[:K].clone() if oracle is not None else None
    cfg = env.cfg
    total_gt_words = env.total_gt_words

    steps_since = torch.zeros(N, device=device)
    gt_idx_v = [0] * N
    emit_slots: List[List[int]] = [[] for _ in range(N)]
    prev_actions = torch.zeros(N, dtype=torch.long, device=device)

    s_beliefs: List[torch.Tensor] = []
    s_priors: List[torch.Tensor] = []
    s_syl_feats: List[torch.Tensor] = []
    s_errors: List[Optional[torch.Tensor]] = []
    s_prev_actions: List[torch.Tensor] = []
    s_actions: List[torch.Tensor] = []
    s_log_probs: List[torch.Tensor] = []
    s_values: List[torch.Tensor] = []
    s_rewards: List[List[float]] = [[] for _ in range(N)]

    hidden: Optional[torch.Tensor] = None

    for t in range(K):
        syl_feat_t = torch.stack([durations[t].expand(N), rel_pos[t].expand(N), steps_since], dim=-1)
        b_t = beliefs_K[t].unsqueeze(0).expand(N, -1).unsqueeze(1)
        p_t = priors_K[t].unsqueeze(0).expand(N, -1).unsqueeze(1)
        s_t = syl_feat_t.unsqueeze(1)

        d_t = (
            distortions_K[t].unsqueeze(0).expand(N, -1).unsqueeze(1)
            if distortions_K is not None
            else torch.zeros(N, 1, H, device=device)
        )

        prev_act_t = prev_actions.unsqueeze(1)

        logits_t, values_t, hidden = agent(b_t, p_t, s_t, d_t, hidden, prev_act_t)
        logits_sq = logits_t.squeeze(1)
        sample_dist = torch.distributions.Categorical(logits=logits_sq / temperature)
        actions_t = sample_dist.sample()
        lp_dist = torch.distributions.Categorical(logits=logits_sq)
        log_probs_t = lp_dist.log_prob(actions_t)

        is_last = (t == K - 1)
        rewards_step: List[float] = []
        actions_list_t: List[int] = actions_t.cpu().tolist()

        for n, a in enumerate(actions_list_t):
            r = 0.0
            if a == EMIT:
                emit_slots[n].append(t)
                if cfg.reward_mode == "per_slot" and oracle is not None:
                    r = cfg.correct_reward if t in oracle_set else cfg.wrong_penalty
                elif cfg.reward_mode == "hybrid" and oracle is not None:
                    r = 0.2 if t in oracle_set else -0.2
            else:
                if cfg.reward_mode == "per_slot":
                    r = cfg.wait_penalty
                # POMDP: info gain reward for WAIT.
                if cfg.info_gain_scale > 0.0 and distortions_K is not None and t + 1 < K:
                    d_now = distortions_K[t].norm().item()
                    d_next = distortions_K[t + 1].norm().item()
                    r += cfg.info_gain_scale * max(0.0, d_now - d_next)

            if is_last:
                if cfg.reward_mode == "episode_f1":
                    agent_set = set(emit_slots[n])
                    if oracle_set and agent_set:
                        tp = len(agent_set & oracle_set)
                        pr = tp / len(agent_set)
                        rc = tp / len(oracle_set)
                        r = 2 * pr * rc / max(pr + rc, 1e-8) * cfg.f1_reward_scale
                elif cfg.reward_mode == "hybrid":
                    agent_set = set(emit_slots[n])
                    if oracle_set and agent_set:
                        tp = len(agent_set & oracle_set)
                        pr = tp / len(agent_set)
                        rc = tp / len(oracle_set)
                        r += 2 * pr * rc / max(pr + rc, 1e-8) * cfg.f1_reward_scale
            rewards_step.append(r)

        emitting = (actions_t == EMIT)
        steps_since = torch.where(emitting, torch.zeros_like(steps_since), steps_since + 1)

        s_beliefs.append(beliefs_K[t].cpu())
        s_priors.append(priors_K[t].cpu())
        s_syl_feats.append(syl_feat_t.cpu())
        s_errors.append(distortions_K[t].cpu() if distortions_K is not None else None)
        s_prev_actions.append(prev_actions.cpu())
        s_actions.append(actions_t.cpu())
        s_log_probs.append(log_probs_t.cpu())
        s_values.append(values_t.squeeze(-1).squeeze(-1).cpu())
        for n in range(N):
            s_rewards[n].append(rewards_step[n])

        prev_actions = actions_t.clone()

    buffers: List[EpisodeBuffer] = []
    for n in range(N):
        buf = EpisodeBuffer(oracle_labels=oracle_labels)
        for t in range(K):
            err_t = s_errors[t]
            buf.append(Transition(
                belief=s_beliefs[t],
                prior=s_priors[t],
                syl_feat=s_syl_feats[t][n],
                action=s_actions[t][n].item(),
                log_prob=s_log_probs[t][n].item(),
                value=s_values[t][n].item(),
                reward=s_rewards[n][t],
                error=err_t,
                prev_action=s_prev_actions[t][n].item(),
            ))
        buffers.append(buf)
    return buffers


# ---------------------------------------------------------------------------
# GAE + GRPO update
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: List[float],
    values: List[float],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-step Generalized Advantage Estimates and n-step returns.

    GAE (Schulman et al. 2016) interpolates between high-variance Monte Carlo
    returns (lam=1) and low-variance TD(0) (lam=0).  lam=0.95 is a well-tuned
    default that provides strong variance reduction while preserving enough
    multi-step context for credit assignment in sparse-reward problems.

    Args:
        rewards: Step rewards [r_0, ..., r_{T-1}].
        values:  Per-step value predictions [V_0, ..., V_{T-1}].
        gamma:   Discount factor (default 0.99).
        lam:     GAE lambda (default 0.95).

    Returns:
        advantages: (T,) per-step advantage estimates A_t.
        returns:    (T,) n-step returns = A_t + V_t (value training targets).
    """
    T = len(rewards)
    adv = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    returns = adv + torch.tensor(values, dtype=torch.float32)
    return adv, returns


def grpo_update(
    agent: ActiveAgent,
    optimizer: torch.optim.Optimizer,
    groups: List[List[EpisodeBuffer]],
    device: torch.device,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.02,
    grpo_mini_epochs: int = 1,
    gamma: float = 0.99,
    lam: float = 0.95,
    value_coef: float = 0.5,
) -> Dict[str, float]:
    """GRPO with per-step GAE advantage estimation.

    Each element of *groups* is G episodes from the **same** utterance.
    Per-step advantages are computed via GAE using the stored value predictions,
    then group-normalised (group-relative) to remove utterance-level difficulty
    bias. A value function loss (MSE to GAE returns) trains the critic.

    Compared to episode-level GRPO, per-step GAE:
      - Assigns credit correctly to individual WAIT/EMIT decisions.
      - Uses the value baseline to reduce advantage variance at every step.
      - Handles long sequences with sparse rewards more gracefully.

    Args:
        gamma:      Discount factor for future rewards.
        lam:        GAE lambda (0 = TD(0), 1 = Monte Carlo).
        value_coef: Coefficient for value function loss relative to policy loss.
    """
    if not groups:
        return {"policy_loss": 0.0, "entropy": 0.0, "value_loss": 0.0}

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_updates = 0

    agent.train()

    for _mini_epoch in range(grpo_mini_epochs):
        for g_idx in torch.randperm(len(groups)).tolist():
            group = groups[g_idx]
            G = len(group)
            if G == 0:
                continue

            # Step 1: Compute per-step GAE advantages for every episode.
            all_advs: List[torch.Tensor] = []
            all_returns: List[torch.Tensor] = []
            for ep in group:
                rewards = [t.reward for t in ep.transitions]
                values = [t.value for t in ep.transitions]
                ep_advs, ep_returns = compute_gae(rewards, values, gamma, lam)
                all_advs.append(ep_advs)
                all_returns.append(ep_returns)

            # Step 2: Group-relative normalisation of per-step advantages.
            # Subtracting the group mean removes utterance-level difficulty bias;
            # dividing by std keeps gradients well-scaled.
            all_advs_cat = torch.cat(all_advs)
            adv_mean = all_advs_cat.mean()
            adv_std = all_advs_cat.std() + 1e-8

            optimizer.zero_grad()
            group_policy_loss = 0.0
            group_value_loss = 0.0

            for ep, ep_advs, ep_returns in zip(group, all_advs, all_returns):
                if len(ep.transitions) == 0:
                    continue

                normed_advs = ((ep_advs - adv_mean) / adv_std).to(device)
                ep_returns = ep_returns.to(device)

                b = torch.stack([t.belief for t in ep.transitions]).unsqueeze(0).to(device)
                p = torch.stack([t.prior for t in ep.transitions]).unsqueeze(0).to(device)
                s = torch.stack([t.syl_feat for t in ep.transitions]).unsqueeze(0).to(device)
                a = torch.tensor([t.action for t in ep.transitions], dtype=torch.long, device=device)
                old_lp = torch.tensor([t.log_prob for t in ep.transitions], dtype=torch.float32, device=device)

                _e_list = [t.error for t in ep.transitions]
                H = agent.cfg.belief_dim
                err = (
                    torch.stack(_e_list).unsqueeze(0).to(device)
                    if _e_list[0] is not None
                    else torch.zeros(1, b.shape[1], H, device=device)
                )

                _pa_list = [t.prev_action if t.prev_action is not None else WAIT
                            for t in ep.transitions]
                prev_act = torch.tensor(_pa_list, dtype=torch.long, device=device).unsqueeze(0)

                logits, value_preds, _ = agent(b, p, s, err, prev_action=prev_act)
                logits = logits.squeeze(0)           # (T, 2)
                value_preds = value_preds.squeeze()  # (T,) or scalar if T=1

                dist_p = torch.distributions.Categorical(logits=logits)
                new_lp = dist_p.log_prob(a)
                ent = dist_p.entropy().mean()

                if clip_eps > 0.0:
                    ratio = (new_lp - old_lp).exp()
                    surr1 = ratio * normed_advs
                    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * normed_advs
                    p_loss = -torch.min(surr1, surr2).mean()
                else:
                    p_loss = -(new_lp * normed_advs).mean()

                v_loss = F.mse_loss(value_preds.view(-1), ep_returns.view(-1))

                loss = (p_loss + value_coef * v_loss - entropy_coef * ent) / G
                loss.backward()

                group_policy_loss += p_loss.item()
                group_value_loss += v_loss.item()
                total_entropy += ent.item() / G

            nn.utils.clip_grad_norm_(agent.parameters(), 5.0)
            optimizer.step()

            total_policy_loss += group_policy_loss / max(G, 1)
            total_value_loss += group_value_loss / max(G, 1)
            n_updates += 1

    n_updates = max(n_updates, 1)
    return {
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "entropy": total_entropy / n_updates,
    }
