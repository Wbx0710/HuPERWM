"""Train the RL Scheduler Agent for streaming ASR word emission.

Two-phase training:
  Phase 1 — Imitation Learning (IL): binary cross-entropy on oracle emit labels
  Phase 2 — PPO Fine-tuning: on-policy rollouts with word-level accuracy reward

Single-GPU:
    python train_wm_agent.py --phase il ...

Multi-GPU (DDP via torchrun):
    torchrun --nproc_per_node=4 train_wm_agent.py --phase il ...
    torchrun --nproc_per_node=4 train_wm_agent.py --phase ppo ...

All 16 GB of agent data is preloaded into RAM at startup (requires ~16 GB per
process, so ensure sufficient RAM; the machine has 251 GB).  Preloading
eliminates shard-level random I/O and makes GPU the bottleneck, not disk.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from wm_common import Vocabulary
from wm_agent import (
    EMIT,
    WAIT,
    collect_episode,
    collect_episodes_batched,
    ActiveAgent,
    ActiveAgentConfig,
    AgentConfig,
    ASRSchedulerEnv,
    EnvConfig,
    EpisodeBuffer,
    SchedulerAgent,
    Transition,
    compute_gae,
    grpo_update,
    ppo_clip_loss,
    ppo_entropy_bonus,
    ppo_value_loss,
)
from wm_agent_data import AgentCollator, AgentDataset


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------


def setup_ddp() -> tuple[int, int, int]:
    """Initialise DDP if launched via torchrun, else return rank=0, world=1."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    return rank, world_size, local_rank


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def reduce_mean(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """All-reduce a scalar tensor and average across ranks."""
    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / world_size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_syl_features(
    boundaries: torch.Tensor,
    slot_mask: torch.Tensor,
    num_slots: int,
) -> torch.Tensor:
    """Build (K, 3) syllable feature matrix for a single utterance."""
    K = slot_mask.shape[0]
    feats = torch.zeros(K, 3, device=boundaries.device)
    for k in range(num_slots):
        dur = (boundaries[k, 1] - boundaries[k, 0]).float().clamp(min=1.0)
        feats[k, 0] = dur.log()
        feats[k, 1] = k / max(num_slots - 1, 1)
    return feats


def compute_steps_since_emit(oracle_emit: torch.Tensor) -> torch.Tensor:
    """Vectorised steps-since-last-emit for a batch of oracle label sequences.

    Uses torch.cummax to compute, for each slot k, the index of the most
    recent emit label at or before k, then subtracts to get step count.

    Args:
        oracle_emit: (B, max_K) float — 1.0 at oracle emit positions

    Returns:
        (B, max_K) float — steps since the last emit (or k+1 if no prior emit)
    """
    B, K = oracle_emit.shape
    device = oracle_emit.device
    k_idx = torch.arange(K, device=device, dtype=torch.float)  # (K,)

    # At emit positions, record the position index; elsewhere record -1.
    emit_positions = torch.where(
        oracle_emit > 0.5,
        k_idx.unsqueeze(0).expand(B, -1),
        torch.full((B, K), -1.0, device=device),
    )  # (B, K)

    # cummax gives the running maximum — i.e., the index of the last emit up to k.
    last_emit_idx, _ = torch.cummax(emit_positions, dim=1)  # (B, K)

    # steps_since = current position − last emit position
    # When no emit has occurred yet, last_emit_idx = -1 → steps = k+1.
    steps_since = k_idx.unsqueeze(0) - last_emit_idx  # (B, K)
    return steps_since


def build_syl_features_batch(
    boundaries: torch.Tensor,
    num_slots: torch.Tensor,
    oracle_emit: torch.Tensor | None = None,
) -> torch.Tensor:
    """Vectorised batch syllable feature builder. Runs entirely on GPU.

    Args:
        boundaries:   (B, max_K, 2)  int64
        num_slots:    (B,)            int64
        oracle_emit:  (B, max_K) float — oracle emit labels (optional).
                      When provided, the third feature is ``steps_since_emit``
                      derived from oracle labels, matching the runtime feature
                      computed by ``ASRSchedulerEnv._compute_syl_feat()``.
                      When None, the third feature is zeros (backward-compat).

    Returns:
        (B, max_K, 3) float — [log_duration, relative_position, steps_since_emit]
    """
    B, max_K, _ = boundaries.shape
    dur = (boundaries[:, :, 1] - boundaries[:, :, 0]).float().clamp(min=1.0)
    log_dur = dur.log()                                               # (B, max_K)

    k_idx = torch.arange(max_K, device=boundaries.device, dtype=torch.float).unsqueeze(0)
    denom = (num_slots.float() - 1.0).clamp(min=1.0).unsqueeze(1)    # (B, 1)
    rel_pos = k_idx / denom                                           # (B, max_K)

    if oracle_emit is not None:
        steps_since = compute_steps_since_emit(
            oracle_emit.to(boundaries.device)
        )  # (B, max_K)
    else:
        steps_since = torch.zeros(B, max_K, device=boundaries.device)

    return torch.stack([log_dur, rel_pos, steps_since], dim=-1)       # (B, max_K, 3)


# ---------------------------------------------------------------------------
# Phase 1: Imitation Learning
# ---------------------------------------------------------------------------


def train_il_epoch(
    agent: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    world_size: int = 1,
    pos_weight: float = 5.0,
    epoch: int = 0,
    show_progress: bool = True,
) -> Dict[str, float]:
    """One epoch of imitation learning with cross-entropy on both WAIT+EMIT heads.

    Uses cross-entropy over the full [WAIT, EMIT] distribution instead of
    binary BCE on the emit logit alone.  BCE only provided gradients to the
    EMIT head (policy_head row 1), leaving the WAIT head (row 0) at its
    near-zero orthogonal initialisation.  With both heads at ~0, the Categorical
    policy produces P(EMIT)≈0.5 everywhere, causing emit_ratio≈1/oracle_density
    regardless of training (observed 1.40 in v5b).  Cross-entropy trains both
    rows simultaneously, driving P(EMIT|non-oracle) down to ~0.1.
    """
    agent.train()
    total_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0, device=device)
    total_samples = torch.tensor(0, device=device)
    total_emit_tp = torch.tensor(0, device=device)
    total_emit_pos = torch.tensor(0, device=device)
    total_emit_pred = torch.tensor(0, device=device)

    # class_weight: WAIT=1.0, EMIT=pos_weight.
    # pos_weight > 1 up-weights false negatives (missed oracle emits) so the
    # model learns to fire at word boundaries; use 4.0 for good recall.
    class_weight = torch.tensor([1.0, pos_weight], device=device)

    pbar = tqdm(
        loader,
        desc=f"IL epoch {epoch:3d}",
        unit="batch",
        disable=not show_progress,
        dynamic_ncols=True,
    )

    for batch in pbar:
        beliefs = batch["beliefs"].to(device)
        priors = batch["priors"].to(device)
        slot_mask = batch["slot_mask"].to(device)
        oracle = batch["oracle_emit"].to(device)
        boundaries = batch["boundaries"].to(device)
        num_slots = batch["num_slots"].to(device)

        # Vectorised syllable features — keep oracle steps_since_emit so the
        # model can learn the timing signal from acoustic context.
        syl_feats = build_syl_features_batch(boundaries, num_slots, oracle)  # (B, K, 3)

        distortions = batch["distortions"].to(device) if "distortions" in batch else None
        raw = agent.module if isinstance(agent, DDP) else agent
        if isinstance(raw, ActiveAgent):
            error = distortions if distortions is not None else torch.zeros(beliefs.shape[0], beliefs.shape[1], 1, device=device)
            logits, _values, _ = agent(beliefs, priors, syl_feats, error)
        else:
            logits, _values, _ = agent(beliefs, priors, syl_feats, distortion=distortions)  # (B, K, 2)

        # Cross-entropy on full [WAIT, EMIT] distribution with class weighting.
        actions = oracle.long()  # (B, K): 0=WAIT, 1=EMIT
        loss_per_token = F.cross_entropy(
            logits.view(-1, 2), actions.view(-1),
            weight=class_weight, reduction='none',
        )  # (B*K,)
        mask_flat = slot_mask.view(-1)
        loss = (loss_per_token * mask_flat).sum() / mask_flat.sum().clamp(min=1)

        optimizer.zero_grad()
        loss.backward()
        raw = agent.module if isinstance(agent, DDP) else agent
        nn.utils.clip_grad_norm_(raw.parameters(), 5.0)
        optimizer.step()

        preds = logits.argmax(dim=-1).float()  # (B, K): argmax over [WAIT, EMIT]
        mask = slot_mask.bool()
        n = mask.sum()
        total_loss += loss.detach() * n
        total_correct += ((preds == oracle) & mask).sum()
        total_samples += n
        total_emit_tp += ((preds == 1) & (oracle == 1) & mask).sum()
        total_emit_pos += (oracle == 1).sum()
        total_emit_pred += ((preds == 1) & mask).sum()

        # Live metrics in progress bar (rank-0 view, not yet all-reduced)
        if show_progress:
            cur_n = total_samples.item()
            pbar.set_postfix(
                loss=f"{total_loss.item() / max(cur_n, 1):.4f}",
                acc=f"{total_correct.item() / max(cur_n, 1):.3f}",
                recall=f"{total_emit_tp.item() / max(total_emit_pos.item(), 1):.3f}",
            )

    # All-reduce statistics across ranks
    for t in [total_loss, total_correct, total_samples,
              total_emit_tp, total_emit_pos, total_emit_pred]:
        reduce_mean(t, world_size)  # in-place sum then divide

    n = total_samples.item()
    acc = total_correct.item() / max(n, 1)
    recall = total_emit_tp.item() / max(total_emit_pos.item(), 1)
    precision = total_emit_tp.item() / max(total_emit_pred.item(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "il_loss": total_loss.item() / max(n, 1),
        "il_acc": acc,
        "il_emit_recall": recall,
        "il_emit_precision": precision,
        "il_emit_f1": f1,
    }


@torch.no_grad()
def eval_il(
    agent: nn.Module,
    loader: DataLoader,
    device: torch.device,
    world_size: int = 1,
    show_progress: bool = True,
) -> Dict[str, float]:
    agent.eval()
    total_correct = torch.tensor(0, device=device)
    total_samples = torch.tensor(0, device=device)
    total_emit_tp = torch.tensor(0, device=device)
    total_emit_pos = torch.tensor(0, device=device)
    total_emit_pred = torch.tensor(0, device=device)

    for batch in tqdm(loader, desc="  eval", unit="batch",
                      disable=not show_progress, dynamic_ncols=True, leave=False):
        beliefs = batch["beliefs"].to(device)
        priors = batch["priors"].to(device)
        slot_mask = batch["slot_mask"].to(device)
        oracle = batch["oracle_emit"].to(device)
        boundaries = batch["boundaries"].to(device)
        num_slots = batch["num_slots"].to(device)

        syl_feats = build_syl_features_batch(boundaries, num_slots, oracle)

        distortions = batch["distortions"].to(device) if "distortions" in batch else None
        raw = agent.module if isinstance(agent, DDP) else agent
        if isinstance(raw, ActiveAgent):
            error = distortions if distortions is not None else torch.zeros(beliefs.shape[0], beliefs.shape[1], 1, device=device)
            logits, _, _ = agent(beliefs, priors, syl_feats, error)
        else:
            logits, _, _ = agent(beliefs, priors, syl_feats, distortion=distortions)  # (B, K, 2)
        preds = logits.argmax(dim=-1).float()  # (B, K): argmax over [WAIT, EMIT]
        mask = slot_mask.bool()
        total_correct += ((preds == oracle) & mask).sum()
        total_samples += mask.sum()
        total_emit_tp += ((preds == 1) & (oracle == 1) & mask).sum()
        total_emit_pos += (oracle == 1).sum()
        total_emit_pred += ((preds == 1) & mask).sum()

    for t in [total_correct, total_samples, total_emit_tp, total_emit_pos, total_emit_pred]:
        reduce_mean(t, world_size)

    acc = total_correct.item() / max(total_samples.item(), 1)
    recall = total_emit_tp.item() / max(total_emit_pos.item(), 1)
    precision = total_emit_tp.item() / max(total_emit_pred.item(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "val_acc": acc,
        "val_emit_recall": recall,
        "val_emit_precision": precision,
        "val_emit_f1": f1,
    }


# ---------------------------------------------------------------------------
# Phase 2: PPO
# ---------------------------------------------------------------------------


# collect_episode is defined in wm_agent and imported above.


def ppo_update(
    agent: SchedulerAgent,
    optimizer: torch.optim.Optimizer,
    episodes: List[EpisodeBuffer],
    device: torch.device,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    gamma: float = 0.99,
    lam: float = 0.95,
    ppo_mini_epochs: int = 4,
    mini_batch_size: int = 256,
) -> Dict[str, float]:
    """PPO update with sequential GRU forward — each episode is processed as a
    full sequence so new_log_probs are computed with the same sequential hidden
    state as old_log_probs collected during rollout.

    This fixes the GRU-inconsistency bug where the old approach shuffled all
    transitions and called forward with hidden=None, causing up to 62-70% of
    policy ratios to be clipped even on the very first update.
    """
    if not episodes:
        return {"ppo_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    # Pre-compute GAE returns/advantages and gather per-episode tensors.
    ep_beliefs: List[torch.Tensor] = []       # (T_i, H)
    ep_priors: List[torch.Tensor] = []        # (T_i, H)
    ep_syl_feats: List[torch.Tensor] = []     # (T_i, 3)
    ep_actions: List[torch.Tensor] = []       # (T_i,)
    ep_old_lp: List[torch.Tensor] = []        # (T_i,)
    ep_advantages: List[torch.Tensor] = []    # (T_i,)
    ep_returns: List[torch.Tensor] = []       # (T_i,)

    for ep in episodes:
        rewards = [t.reward for t in ep.transitions]
        values  = [t.value  for t in ep.transitions]
        advs, rets = compute_gae(rewards, values, gamma, lam)

        ep_beliefs.append(torch.stack([t.belief   for t in ep.transitions]))
        ep_priors.append(torch.stack([t.prior     for t in ep.transitions]))
        ep_syl_feats.append(torch.stack([t.syl_feat for t in ep.transitions]))
        ep_actions.append(torch.tensor([t.action  for t in ep.transitions], dtype=torch.long))
        ep_old_lp.append(torch.tensor([t.log_prob for t in ep.transitions], dtype=torch.float32))
        ep_advantages.append(torch.tensor(advs, dtype=torch.float32))
        ep_returns.append(torch.tensor(rets, dtype=torch.float32))

    # Global advantage normalisation across all episodes.
    all_adv = torch.cat(ep_advantages)
    adv_mean = all_adv.mean()
    adv_std  = all_adv.std() + 1e-8
    ep_advantages = [(a - adv_mean) / adv_std for a in ep_advantages]

    n_ep = len(episodes)
    total_policy_loss = 0.0
    total_value_loss  = 0.0
    total_entropy     = 0.0
    n_updates = 0

    agent.train()
    for mini_epoch_idx in range(ppo_mini_epochs):
        # Shuffle episode order each mini-epoch for diversity.
        ep_perm = torch.randperm(n_ep).tolist()
        for ep_idx in ep_perm:
            # Full-sequence forward: GRU hidden starts at None (same as rollout).
            b = ep_beliefs[ep_idx].to(device).unsqueeze(0)    # (1, T, H)
            p = ep_priors[ep_idx].to(device).unsqueeze(0)     # (1, T, H)
            s = ep_syl_feats[ep_idx].to(device).unsqueeze(0)  # (1, T, 3)
            a   = ep_actions[ep_idx].to(device)               # (T,)
            olp = ep_old_lp[ep_idx].to(device)                # (T,)
            adv = ep_advantages[ep_idx].to(device)            # (T,)
            ret = ep_returns[ep_idx].to(device)               # (T,)
            _ep = episodes[ep_idx]
            _d_list = [t.distortion for t in _ep.transitions]
            d_ppo = (torch.stack(_d_list).unsqueeze(0).to(device)
                     if _d_list[0] is not None else None)

            raw_for_check = agent.module if isinstance(agent, DDP) else agent
            if isinstance(raw_for_check, ActiveAgent):
                err_ppo = d_ppo if d_ppo is not None else torch.zeros(1, b.shape[1], 1, device=device)
                logits, values_pred, _ = agent(b, p, s, err_ppo)
            else:
                logits, values_pred, _ = agent(b, p, s, distortion=d_ppo)
            logits      = logits.squeeze(0)           # (T, 2)
            values_pred = values_pred.squeeze(0).squeeze(-1)  # (T,)

            p_loss = ppo_clip_loss(logits, a, olp, adv, clip_eps)
            v_loss = ppo_value_loss(values_pred, ret)
            ent    = ppo_entropy_bonus(logits)

            loss = p_loss + value_coef * v_loss - entropy_coef * ent

            optimizer.zero_grad()
            loss.backward()
            raw = agent.module if isinstance(agent, DDP) else agent
            nn.utils.clip_grad_norm_(raw.parameters(), 5.0)
            optimizer.step()

            total_policy_loss += p_loss.item()
            total_value_loss  += v_loss.item()
            total_entropy     += ent.item()
            n_updates += 1

    n_updates = max(n_updates, 1)
    return {
        "policy_loss": total_policy_loss / n_updates,
        "value_loss":  total_value_loss  / n_updates,
        "entropy":     total_entropy     / n_updates,
    }


@torch.no_grad()
def evaluate_agent(
    agent: nn.Module,
    dataset: AgentDataset,
    phone_vocab: Vocabulary,
    device: torch.device,
    env_cfg: EnvConfig,
    max_episodes: int = 200,
) -> Dict[str, float]:
    """Evaluate the agent, returning reward + precision/recall/F1 + word accuracy.

    Primary metrics:
    - precision/recall/F1: oracle-slot matching (independent of reward_mode).
    - word_acc: average (1-phonePER) per EMIT in word_match mode, i.e. the
      fraction of each committed segment that correctly decoded to the expected
      GT word phoneme sequence.  Available when word_phone_ids are present.
    """
    agent.eval()
    total_reward = 0.0
    total_correct = 0    # EMIT on oracle=1 slot
    total_emitted = 0
    total_oracle_slots = 0
    total_word_acc = 0.0    # sum of (1 - phonePER) across all word-match emits
    total_word_emit = 0     # number of within-budget EMITs in word_match mode
    n_episodes = 0

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:max_episodes]

    raw_agent = agent.module if isinstance(agent, DDP) else agent
    is_word_match = env_cfg.reward_mode == "word_match"

    for idx in indices:
        record = dataset[idx]
        ns = record["num_slots"]
        if ns == 0:
            continue

        oracle_emit = record.get("oracle_emit")

        env = ASRSchedulerEnv(
            beliefs=record["beliefs"],
            priors=record["priors"],
            boundaries=record["boundaries"],
            canonical_logits=record["canonical_logits"],
            up_slot_mask=record["up_slot_mask"],
            slot_mask=record["slot_mask"],
            gt_words=record["words"],
            phone_vocab=phone_vocab,
            env_cfg=env_cfg,
            oracle_emit=oracle_emit,
            word_phones_list=record.get("word_phone_ids"),
            distortions=record.get("distortions"),
            word_states=record.get("word_states"),
        )

        # Collect episode; also compute word-match accuracy in parallel.
        word_phone_ids = record.get("word_phone_ids") or []
        word_ptr_eval = 0
        emit_start_eval = 0  # slot where current word's accumulation window began

        ep = collect_episode(raw_agent, env, device)

        total_reward += sum(t.reward for t in ep.transitions)
        total_oracle_slots += env.total_oracle_slots

        # Oracle-slot matching (always computed, independent of reward_mode).
        oracle_set = env._oracle_slot_set if oracle_emit is not None else set()
        for t_idx, t in enumerate(ep.transitions):
            if t.action != EMIT:
                continue
            total_emitted += 1
            if oracle_emit is not None and t_idx in oracle_set:
                total_correct += 1
            # Word-match accuracy: decode CTC and compare against expected phones.
            if is_word_match and word_ptr_eval < len(word_phone_ids):
                expected_ids = word_phone_ids[word_ptr_eval]
                if expected_ids:
                    decoded_ids = env._decode_ctc_phone_ids(
                        emit_start_eval, t_idx
                    )
                    per = env._phone_per(decoded_ids, expected_ids)
                    total_word_acc += max(0.0, 1.0 - per)
                    total_word_emit += 1
                word_ptr_eval += 1
                emit_start_eval = t_idx + 1

        n_episodes += 1

    n_episodes = max(n_episodes, 1)
    precision = total_correct / max(total_emitted, 1)
    recall    = total_correct / max(total_oracle_slots, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    word_acc  = total_word_acc / max(total_word_emit, 1) if total_word_emit > 0 else 0.0
    return {
        "avg_reward":   total_reward / n_episodes,
        "emit_ratio":   total_emitted / max(total_oracle_slots, 1),
        "precision":    precision,
        "recall":       recall,
        "f1":           f1,
        "word_acc":     word_acc,     # avg (1-phonePER) per committed word segment
        "episodes":     n_episodes,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RL Scheduler Agent.")
    p.add_argument("--agent-data-dir", required=True, help="Output of wm_agent_data.py")
    p.add_argument("--metadata-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--phase", choices=["il", "ppo", "grpo", "both"], default="both",
                   help="Training phase: 'il'=imitation learning only, "
                        "'grpo'=GRPO only (requires --resume-from), "
                        "'ppo'=PPO only, "
                        "'both'=IL then GRPO (recommended full pipeline).")

    # Agent architecture
    p.add_argument("--belief-dim", type=int, default=256)
    p.add_argument("--agent-hidden", type=int, default=128,
                   help="Hidden size for the agent MLP and GRU. "
                        "128 for fast training / fine-tuning; 256 for full re-train.")
    p.add_argument("--gru-layers", type=int, default=1,
                   help="Number of GRU layers. 1 is fastest; 2 adds depth for re-train.")
    p.add_argument("--agent-dropout", type=float, default=0.1)
    p.add_argument("--use-extra-features", action="store_true",
                   help="Append 2 extra belief/prior features to the observation: "
                        "cosine_sim(belief, prior) and entropy(softmax(belief)). "
                        "Requires re-training from IL since the input dimension changes.")
    p.add_argument("--use-distortion", action="store_true",
                   help="Append 1 distortion scalar to the observation (HuperJEPA v2). "
                        "Requires agent data extracted from a --use-distortion Stage 2 "
                        "checkpoint (distortions field must be present).")
    p.add_argument("--use-active-agent", action="store_true",
                   help="Use ActiveAgent (v3) with dual-pathway comparison gating instead "
                        "of SchedulerAgent. Requires agent data extracted from a "
                        "belief_type=comparison Stage 2 checkpoint.")

    # IL hyperparameters
    p.add_argument("--il-epochs", type=int, default=30)
    p.add_argument("--il-lr", type=float, default=1e-3)
    p.add_argument("--il-batch-size", type=int, default=32)
    p.add_argument("--il-pos-weight", type=float, default=4.0,
                    help="EMIT class weight in cross-entropy IL loss. "
                         ">1.0 up-weights oracle emit positions (false negatives) so the agent "
                         "learns to fire at word boundaries; 4.0 gives good recall without "
                         "collapsing to always-emit. The WAIT class is always weighted 1.0.")
    p.add_argument("--oracle-min-gap", type=int, default=2,
                    help="Minimum slot gap between consecutive oracle emits (default 2). "
                         "With LibriSpeech word/slot≈0.70 this keeps oracle_density≈70%%, "
                         "ensuring E[always-emit]>E[always-wait] and preventing "
                         "the never-emit collapse seen with sparser oracles.")

    # PPO hyperparameters
    p.add_argument("--ppo-epochs", type=int, default=100)
    p.add_argument("--ppo-lr", type=float, default=3e-4)
    p.add_argument("--ppo-episodes-per-update", type=int, default=64)
    p.add_argument("--ppo-mini-epochs", type=int, default=4)
    p.add_argument("--ppo-mini-batch-size", type=int, default=256)
    p.add_argument("--ppo-clip-eps", type=float, default=0.2)
    p.add_argument("--ppo-entropy-coef", type=float, default=0.01)
    p.add_argument("--ppo-value-coef", type=float, default=0.5)
    p.add_argument("--ppo-gamma", type=float, default=0.99)
    p.add_argument("--ppo-lam", type=float, default=0.95)

    # GRPO hyperparameters
    p.add_argument("--grpo-epochs", type=int, default=150,
                   help="Number of GRPO update epochs.")
    p.add_argument("--grpo-lr", type=float, default=1e-4,
                   help="Learning rate for GRPO phase.")
    p.add_argument("--grpo-utterances-per-update", type=int, default=32,
                   help="Number of utterances sampled per GRPO epoch.")
    p.add_argument("--grpo-rollouts", type=int, default=8,
                   help="G: number of rollouts per utterance for group normalisation.")
    p.add_argument("--grpo-clip-eps", type=float, default=0.2,
                   help="PPO-style ratio clipping epsilon for GRPO (0 = plain REINFORCE).")
    p.add_argument("--grpo-entropy-coef", type=float, default=0.02,
                   help="Entropy regularisation coefficient for GRPO.")
    p.add_argument("--grpo-mini-epochs", type=int, default=1,
                   help="Mini-epochs per GRPO update (keep 1 to avoid stale log-probs).")
    p.add_argument("--rollout-temperature", type=float, default=1.0,
                   help="Softmax temperature for action sampling during GRPO rollout "
                        "collection.  Values > 1 diversify rollouts when the policy is "
                        "near-deterministic, preventing group advantages from collapsing "
                        "to zero.  Stored log_probs use the unscaled logits so the GRPO "
                        "importance-ratio remains correct.  Recommended: 1.5.")

    # Environment reward shaping
    p.add_argument("--wait-penalty", type=float, default=-0.01)
    p.add_argument("--correct-reward", type=float, default=1.0)
    p.add_argument("--wrong-penalty", type=float, default=-1.0,
                   help="Penalty for emitting at a non-oracle slot. With oracle_density=70%%, "
                        "-1.0 keeps E[always-emit]=+0.40>0 (safe start) while "
                        "-2.6 gives E[always-emit]=-0.08 (collapses to never-emit).")
    p.add_argument("--incomplete-penalty", type=float, default=0.0,
                   help="Per-word penalty for uncovered words (oracle mode: 0.0 disabled).")
    p.add_argument("--reward-mode", type=str, default="per_slot",
                   choices=["per_slot", "episode_f1", "hybrid", "word_match"],
                   help="Reward mode for GRPO/PPO training. "
                        "'per_slot': original per-step binary reward (backward-compat). "
                        "'episode_f1': zero step rewards; terminal F1 as sole signal. "
                        "'hybrid': small per-step credit (±0.2) + terminal F1 bonus. "
                        "'word_match': decode CTC at each EMIT; reward = (1-phonePER) "
                        "against expected GT word phones — directly simulates "
                        "human-brain streaming ASR evidence accumulation.")
    p.add_argument("--f1-reward-scale", type=float, default=1.0,
                   help="Scale factor applied to the terminal F1 reward in "
                        "episode_f1 / hybrid modes (default 1.0).")
    p.add_argument("--f1-match-window", type=int, default=0,
                   help="Slot tolerance for soft F1 matching (0=exact, 1=±1 slot). "
                        "Useful when oracle is approximate (proportional interpolation).")
    p.add_argument("--missing-word-penalty", type=float, default=0.5,
                   help="Per-word penalty for GT words not covered by any EMIT in "
                        "word_match mode (terminal reward, default 0.5).")

    # General
    p.add_argument("--lr", type=float, default=None, help="Override LR for current phase")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers. 0 = main process (fastest when preloading).")
    p.add_argument("--no-preload", action="store_true",
                   help="Disable RAM preloading (use lazy shard cache). "
                        "Only needed when RAM < 20 GB.")
    p.add_argument("--max-train-examples", type=int, default=None)
    p.add_argument("--max-val-examples", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--resume-from", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # --- DDP initialisation (no-op when not launched via torchrun) ---
    rank, world_size, local_rank = setup_ddp()
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    output_dir = Path(args.output_dir)
    if is_main(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "agent_train_args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"DDP: rank={rank}/{world_size}, device={device}", flush=True)

    # Barrier so all ranks wait until output_dir exists
    if world_size > 1:
        dist.barrier(device_ids=[local_rank])

    phone_vocab = Vocabulary.from_file(Path(args.metadata_dir) / "phone_vocab.json")

    preload = not args.no_preload
    oracle_min_gap = getattr(args, "oracle_min_gap", 2)
    train_ds = AgentDataset(
        args.agent_data_dir, "train",
        max_examples=args.max_train_examples,
        preload=preload, rank=rank,
        recompute_oracle=True,
        phone_vocab=phone_vocab,
        upsample_factor=4,
        oracle_min_gap=oracle_min_gap,
    )
    val_ds = AgentDataset(
        args.agent_data_dir, "validation",
        max_examples=args.max_val_examples,
        preload=preload, rank=rank,
        recompute_oracle=True,
        phone_vocab=phone_vocab,
        upsample_factor=4,
        oracle_min_gap=oracle_min_gap,
    )
    if is_main(rank):
        print(f"Train: {len(train_ds)} utterances, Val: {len(val_ds)} utterances")

    collator = AgentCollator()

    # IL DataLoader: DistributedSampler splits data evenly across GPUs
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False,
        )
        il_shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        il_shuffle = True

    # num_workers=0 is fastest when data is already in RAM (no IPC overhead)
    train_loader = DataLoader(
        train_ds, batch_size=args.il_batch_size,
        shuffle=il_shuffle, sampler=train_sampler,
        num_workers=args.num_workers, collate_fn=collator,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.il_batch_size,
        shuffle=False, sampler=val_sampler,
        num_workers=args.num_workers, collate_fn=collator,
        pin_memory=True,
    )

    use_active = getattr(args, "use_active_agent", False)
    if use_active:
        cfg = ActiveAgentConfig(
            belief_dim=args.belief_dim,
            agent_hidden=args.agent_hidden,
            gru_layers=args.gru_layers,
            dropout=args.agent_dropout,
        )
        agent_raw = ActiveAgent(cfg).to(device)
    else:
        cfg = AgentConfig(
            belief_dim=args.belief_dim,
            agent_hidden=args.agent_hidden,
            gru_layers=args.gru_layers,
            dropout=args.agent_dropout,
            use_extra_features=getattr(args, "use_extra_features", False),
            use_distortion=getattr(args, "use_distortion", False),
        )
        agent_raw = SchedulerAgent(cfg).to(device)

    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        agent_raw.load_state_dict(ckpt["agent_state_dict"])
        if is_main(rank):
            print(f"Resumed agent weights from {args.resume_from}")

    # Wrap in DDP after loading weights.
    # find_unused_parameters=True: during IL, value_head has no gradient
    # (only policy_head is trained); during PPO both heads are used.
    if world_size > 1:
        agent: nn.Module = DDP(
            agent_raw, device_ids=[local_rank], find_unused_parameters=True,
        )
    else:
        agent = agent_raw

    env_cfg = EnvConfig(
        wait_penalty=args.wait_penalty,
        correct_reward=args.correct_reward,
        wrong_penalty=args.wrong_penalty,
        incomplete_penalty=args.incomplete_penalty,
        reward_mode=args.reward_mode,
        f1_reward_scale=args.f1_reward_scale,
        f1_match_window=args.f1_match_window,
        missing_word_penalty=getattr(args, "missing_word_penalty", 0.5),
    )

    if is_main(rank):
        param_count = sum(p.numel() for p in agent_raw.parameters() if p.requires_grad)
        print(f"Agent parameters: {param_count:,}")

    # ==================================================================
    # Phase 1: Imitation Learning
    # ==================================================================
    run_il   = args.phase in ("il",   "both")
    run_ppo  = args.phase in ("ppo",)           # PPO only with explicit --phase ppo
    run_grpo = args.phase in ("grpo", "both")   # "both" = IL → GRPO (recommended)

    if run_il:
        if is_main(rank):
            print("\n" + "=" * 60)
            print(f"Phase 1: Imitation Learning  (world_size={world_size})")
            print(f"  effective batch = {args.il_batch_size} × {world_size} = "
                  f"{args.il_batch_size * world_size}")
            print("=" * 60)

        il_lr = args.lr if args.lr else args.il_lr
        # Scale LR linearly with number of GPUs (linear scaling rule)
        scaled_lr = il_lr * world_size
        optimizer = AdamW(agent_raw.parameters(), lr=scaled_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.il_epochs, eta_min=scaled_lr * 0.01,
        )

        best_f1 = 0.0
        history_il: list[dict] = []

        for epoch in range(1, args.il_epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_metrics = train_il_epoch(
                agent, train_loader, optimizer, device,
                world_size=world_size, pos_weight=args.il_pos_weight,
                epoch=epoch, show_progress=is_main(rank),
            )
            scheduler.step()

            if epoch % args.eval_every == 0 or epoch == args.il_epochs:
                val_metrics = eval_il(
                    agent, val_loader, device,
                    world_size=world_size, show_progress=is_main(rank),
                )

                if is_main(rank):
                    record = {
                        "epoch": epoch, **train_metrics, **val_metrics,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                    history_il.append(record)
                    print(json.dumps(record, ensure_ascii=True), flush=True)

                    ckpt = {
                        "agent_state_dict": agent_raw.state_dict(),
                        "agent_config": cfg,
                        "epoch": epoch,
                        "history": history_il,
                    }
                    torch.save(ckpt, output_dir / "il_last.pt")

                    if val_metrics["val_emit_f1"] > best_f1:
                        best_f1 = val_metrics["val_emit_f1"]
                        torch.save(ckpt, output_dir / "il_best.pt")
                        print(f"  New best F1: {best_f1:.4f}")
            elif epoch % args.log_every == 0 and is_main(rank):
                record = {"epoch": epoch, **train_metrics, "lr": optimizer.param_groups[0]["lr"]}
                print(json.dumps(record, ensure_ascii=True), flush=True)

        if is_main(rank):
            print(f"IL complete. Best val F1: {best_f1:.4f}")

    # ==================================================================
    # Phase 2: PPO Fine-tuning
    # ==================================================================
    if run_ppo:
        if is_main(rank):
            print("\n" + "=" * 60)
            print(f"Phase 2: PPO Fine-tuning  (world_size={world_size})")
            print(f"  episodes per update (total) = "
                  f"{args.ppo_episodes_per_update} × {world_size} = "
                  f"{args.ppo_episodes_per_update * world_size}")
            print("=" * 60)

        # Load best IL weights on rank 0, then broadcast to all ranks
        if run_il and not args.resume_from:
            best_il = output_dir / "il_best.pt"
            if best_il.exists():
                ckpt = torch.load(best_il, map_location="cpu", weights_only=False)
                agent_raw.load_state_dict(ckpt["agent_state_dict"])
                if is_main(rank):
                    print(f"Starting PPO from IL best checkpoint (F1={best_f1:.4f})")
        elif args.resume_from and not run_il:
            # resume_from already loaded before DDP wrap; re-broadcast just in case
            pass
        if world_size > 1:
            # Broadcast weights from rank 0 to all ranks
            for param in agent_raw.parameters():
                dist.broadcast(param.data, src=0)

        ppo_lr = args.lr if args.lr else args.ppo_lr
        scaled_ppo_lr = ppo_lr * world_size
        optimizer = AdamW(agent_raw.parameters(), lr=scaled_ppo_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.ppo_epochs, eta_min=scaled_ppo_lr * 0.01,
        )

        # PPO must NOT use DDP for the gradient update step.
        # DDP requires every rank to do the same number of forward+backward
        # calls per synchronisation point, but different episodes yield different
        # total transition counts → ranks diverge → NCCL ALLREDUCE timeout.
        #
        # Instead we use the "independent-update + parameter-average" pattern:
        #   1. Each rank runs ppo_update on agent_raw independently.
        #   2. After the update, we all-reduce each parameter tensor so all
        #      ranks converge to the same averaged model.
        # This is equivalent to a federated average step and correctly exploits
        # the diversity of experience collected on different ranks.
        agent = agent_raw  # no DDP wrapper for PPO

        best_reward = -float("inf")
        history_ppo: list[dict] = []

        # Each rank gets its own non-overlapping subset of train indices
        all_indices = list(range(len(train_ds)))
        episodes_per_rank = args.ppo_episodes_per_update

        epoch_pbar = tqdm(
            range(1, args.ppo_epochs + 1),
            desc="PPO",
            unit="epoch",
            disable=not is_main(rank),
            dynamic_ncols=True,
        )

        for epoch in epoch_pbar:
            # Each rank shuffles with a different seed so episodes are diverse
            rng = random.Random(args.seed + epoch * 1000 + rank)
            rng.shuffle(all_indices)
            ep_indices = all_indices[:episodes_per_rank]

            episodes: List[EpisodeBuffer] = []
            epoch_reward = 0.0
            epoch_emit_count = 0
            epoch_gt_words = 0

            ep_pbar = tqdm(
                ep_indices,
                desc=f"  ep collect",
                unit="ep",
                disable=not is_main(rank),
                dynamic_ncols=True,
                leave=False,
            )
            for idx in ep_pbar:
                record = train_ds[idx]
                if record["num_slots"] == 0:
                    continue

                env = ASRSchedulerEnv(
                    beliefs=record["beliefs"],
                    priors=record["priors"],
                    boundaries=record["boundaries"],
                    canonical_logits=record["canonical_logits"],
                    up_slot_mask=record["up_slot_mask"],
                    slot_mask=record["slot_mask"],
                    gt_words=record["words"],
                    phone_vocab=phone_vocab,
                    env_cfg=env_cfg,
                    oracle_emit=record.get("oracle_emit"),
                    word_phones_list=record.get("word_phone_ids"),
                    distortions=record.get("distortions"),
                    word_states=record.get("word_states"),
                )

                # Use a single batched forward (batch=1 for PPO, but the
                # batched API is consistent and avoids per-step Python loops).
                ep_buf = collect_episodes_batched(agent_raw, env, device, 1)[0]
                episodes.append(ep_buf)
                ep_reward = sum(t.reward for t in ep_buf.transitions)
                epoch_reward += ep_reward
                epoch_emit_count += sum(1 for t in ep_buf.transitions if t.action == EMIT)
                epoch_gt_words += env.total_gt_words

                if is_main(rank):
                    ep_pbar.set_postfix(
                        r=f"{ep_reward:.2f}",
                        avg=f"{epoch_reward / max(len(episodes), 1):.2f}",
                    )

            update_metrics = ppo_update(
                agent_raw, optimizer, episodes, device,
                clip_eps=args.ppo_clip_eps,
                entropy_coef=args.ppo_entropy_coef,
                value_coef=args.ppo_value_coef,
                gamma=args.ppo_gamma,
                lam=args.ppo_lam,
                ppo_mini_epochs=args.ppo_mini_epochs,
                mini_batch_size=args.ppo_mini_batch_size,
            )
            # Average parameters across ranks so all ranks share the same
            # model going into the next episode-collection phase.
            if world_size > 1:
                for param in agent_raw.parameters():
                    dist.all_reduce(param.data, op=dist.ReduceOp.AVG)
            scheduler.step()

            n_ep = max(len(episodes), 1)
            avg_reward = epoch_reward / n_ep
            emit_ratio = epoch_emit_count / max(epoch_gt_words, 1)

            # Reduce reward scalar across ranks so main rank sees global average.
            # Fix H-C: use the RETURN VALUE of reduce_mean (which divides by world_size)
            # instead of reward_t (which holds the raw ALL_REDUCE SUM).
            if world_size > 1:
                reward_t = torch.tensor(avg_reward, device=device)
                avg_reward = reduce_mean(reward_t, world_size).item()

            if is_main(rank):
                epoch_pbar.set_postfix(
                    r=f"{avg_reward:.2f}",
                    pl=f"{update_metrics.get('policy_loss', 0):.4f}",
                    ent=f"{update_metrics.get('entropy', 0):.3f}",
                    emit=f"{emit_ratio:.2f}",
                )

            if epoch % args.eval_every == 0 or epoch == args.ppo_epochs:
                # Only rank 0 evaluates (val_ds is fully loaded on all ranks anyway)
                if is_main(rank):
                    raw_agent = agent.module if isinstance(agent, DDP) else agent
                    val_metrics = evaluate_agent(
                        raw_agent, val_ds, phone_vocab, device, env_cfg,
                        max_episodes=min(200, len(val_ds)),
                    )
                    record = {
                        "epoch": epoch,
                        "train_avg_reward": avg_reward,
                        "train_emit_ratio": epoch_emit_count / max(epoch_gt_words, 1),
                        **update_metrics,
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                    history_ppo.append(record)
                    print(json.dumps(record, ensure_ascii=True), flush=True)

                    ckpt = {
                        "agent_state_dict": agent_raw.state_dict(),
                        "agent_config": cfg,
                        "env_config": env_cfg,
                        "epoch": epoch,
                        "history": history_ppo,
                    }
                    torch.save(ckpt, output_dir / "ppo_last.pt")

                    val_f1 = val_metrics.get("f1", 0.0)
                    if val_f1 > best_reward:
                        best_reward = val_f1
                        torch.save(ckpt, output_dir / "ppo_best.pt")
                        print(f"  New best val F1: {best_reward:.4f} "
                              f"(prec={val_metrics.get('precision',0):.3f} "
                              f"rec={val_metrics.get('recall',0):.3f})")
            elif epoch % args.log_every == 0 and is_main(rank):
                record = {
                    "epoch": epoch,
                    "train_avg_reward": avg_reward,
                    "train_emit_ratio": epoch_emit_count / max(epoch_gt_words, 1),
                    **update_metrics,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                print(json.dumps(record, ensure_ascii=True), flush=True)

        if is_main(rank):
            print(f"PPO complete. Best val reward: {best_reward:.4f}")

    # ==================================================================
    # Phase 3: GRPO Fine-tuning
    # ==================================================================
    if run_grpo:
        if is_main(rank):
            print("\n" + "=" * 60)
            print(f"Phase 3: GRPO Fine-tuning  (world_size={world_size})")
            print(f"  utterances per update = {args.grpo_utterances_per_update}"
                  f" × {world_size} ranks")
            print(f"  rollouts per utterance (G) = {args.grpo_rollouts}")
            print("=" * 60)

        # Load weights: if resuming, use the provided checkpoint;
        # otherwise start from the IL best (if IL just ran).
        if run_il and not args.resume_from:
            best_il = output_dir / "il_best.pt"
            if best_il.exists():
                ckpt = torch.load(best_il, map_location="cpu", weights_only=False)
                agent_raw.load_state_dict(ckpt["agent_state_dict"])
                if is_main(rank):
                    print(f"Starting GRPO from IL best checkpoint")
        # For --phase grpo, resume_from is already loaded above.

        if world_size > 1:
            for param in agent_raw.parameters():
                dist.broadcast(param.data, src=0)

        grpo_lr = args.lr if args.lr else args.grpo_lr
        scaled_grpo_lr = grpo_lr * world_size
        optimizer = AdamW(agent_raw.parameters(), lr=scaled_grpo_lr, weight_decay=1e-4)
        # CosineAnnealingWarmRestarts resets LR every T_0 epochs instead of
        # decaying monotonically to 1% by epoch 200 (which killed learning in
        # v5b).  eta_min=10% keeps a usable LR floor throughout training.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=1, eta_min=scaled_grpo_lr * 0.1,
        )

        # GRPO uses the same "independent-update + parameter-average" pattern
        # as PPO: no DDP wrapper, manual all-reduce after each epoch.
        agent = agent_raw

        best_f1 = -float("inf")
        history_grpo: list[dict] = []
        all_indices = list(range(len(train_ds)))

        epoch_pbar = tqdm(
            range(1, args.grpo_epochs + 1),
            desc="GRPO",
            unit="epoch",
            disable=not is_main(rank),
            dynamic_ncols=True,
        )

        for epoch in epoch_pbar:
            # Sample a diverse set of utterances; each rank uses a different seed.
            rng = random.Random(args.seed + epoch * 1000 + rank)
            rng.shuffle(all_indices)
            utt_indices = all_indices[:args.grpo_utterances_per_update]

            groups: List[List[EpisodeBuffer]] = []
            epoch_reward = 0.0
            epoch_emit_count = 0
            epoch_oracle_slots = 0
            n_collected = 0

            for utt_idx in utt_indices:
                record = train_ds[utt_idx]
                if record["num_slots"] == 0:
                    continue

                # Build a single env for this utterance (reused across rollouts).
                env = ASRSchedulerEnv(
                    beliefs=record["beliefs"],
                    priors=record["priors"],
                    boundaries=record["boundaries"],
                    canonical_logits=record["canonical_logits"],
                    up_slot_mask=record["up_slot_mask"],
                    slot_mask=record["slot_mask"],
                    gt_words=record["words"],
                    phone_vocab=phone_vocab,
                    env_cfg=env_cfg,
                    oracle_emit=record.get("oracle_emit"),
                    word_phones_list=record.get("word_phone_ids"),
                    distortions=record.get("distortions"),
                    word_states=record.get("word_states"),
                )
                # Batched rollout: all grpo_rollouts run in a single
                # batch=grpo_rollouts GPU forward per time step.
                # Falls back to sequential for word_match mode.
                group_episodes = collect_episodes_batched(
                    agent_raw, env, device, args.grpo_rollouts,
                    temperature=getattr(args, "rollout_temperature", 1.0),
                )
                for ep_buf in group_episodes:
                    epoch_reward += sum(t.reward for t in ep_buf.transitions)
                    epoch_emit_count += sum(
                        1 for t in ep_buf.transitions if t.action == EMIT
                    )
                epoch_oracle_slots += env.total_oracle_slots * args.grpo_rollouts

                groups.append(group_episodes)
                n_collected += 1

            update_metrics = grpo_update(
                agent_raw, optimizer, groups, device,
                clip_eps=args.grpo_clip_eps,
                entropy_coef=args.grpo_entropy_coef,
                grpo_mini_epochs=args.grpo_mini_epochs,
            )

            # Average parameters across ranks.
            if world_size > 1:
                for param in agent_raw.parameters():
                    dist.all_reduce(param.data, op=dist.ReduceOp.AVG)
            scheduler.step()

            total_transitions = max(
                args.grpo_rollouts * n_collected, 1
            )
            avg_reward = epoch_reward / total_transitions
            emit_ratio = epoch_emit_count / max(epoch_oracle_slots, 1)

            if world_size > 1:
                reward_t = torch.tensor(avg_reward, device=device)
                avg_reward = reduce_mean(reward_t, world_size).item()

            if is_main(rank):
                epoch_pbar.set_postfix(
                    r=f"{avg_reward:.3f}",
                    pl=f"{update_metrics.get('policy_loss', 0):.4f}",
                    ent=f"{update_metrics.get('entropy', 0):.3f}",
                    emit=f"{emit_ratio:.2f}",
                )

            if epoch % args.eval_every == 0 or epoch == args.grpo_epochs:
                if is_main(rank):
                    val_metrics = evaluate_agent(
                        agent_raw, val_ds, phone_vocab, device, env_cfg,
                        max_episodes=min(200, len(val_ds)),
                    )
                    record = {
                        "epoch": epoch,
                        "train_avg_reward": avg_reward,
                        "train_emit_ratio": emit_ratio,
                        **update_metrics,
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                    history_grpo.append(record)
                    print(json.dumps(record, ensure_ascii=True), flush=True)

                    ckpt = {
                        "agent_state_dict": agent_raw.state_dict(),
                        "agent_config": cfg,
                        "env_config": env_cfg,
                        "epoch": epoch,
                        "history": history_grpo,
                    }
                    torch.save(ckpt, output_dir / "grpo_last.pt")

                    val_f1 = val_metrics.get("f1", 0.0)
                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        torch.save(ckpt, output_dir / "grpo_best.pt")
                        word_acc_str = (
                            f" word_acc={val_metrics.get('word_acc', 0):.3f}"
                            if env_cfg.reward_mode == "word_match"
                            else ""
                        )
                        print(
                            f"  New best val F1: {best_f1:.4f} "
                            f"(prec={val_metrics.get('precision', 0):.3f} "
                            f"rec={val_metrics.get('recall', 0):.3f})"
                            f"{word_acc_str}"
                        )
            elif epoch % args.log_every == 0 and is_main(rank):
                record = {
                    "epoch": epoch,
                    "train_avg_reward": avg_reward,
                    "train_emit_ratio": emit_ratio,
                    **update_metrics,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                print(json.dumps(record, ensure_ascii=True), flush=True)

        if is_main(rank):
            print(f"GRPO complete. Best val F1: {best_f1:.4f}")

    if is_main(rank):
        print(f"\nTraining complete → {output_dir}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
