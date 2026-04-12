"""Train the RL Scheduler Agent (ActiveAgent) for streaming ASR word emission.

Two-phase training:
  Phase 1 — Imitation Learning (IL):  cross-entropy on oracle emit labels.
  Phase 2 — GRPO fine-tuning:         on-policy rollouts with word-level reward.

Single-GPU:
    python train_agent.py --phase il ...

Multi-GPU (DDP via torchrun):
    torchrun --nproc_per_node=4 train_agent.py --phase il ...
    torchrun --nproc_per_node=4 train_agent.py --phase grpo ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from huperjepa.data.vocab import Vocabulary
from huperjepa.data.agent import AgentCollator, AgentDataset
from huperjepa.env.asr import (
    EMIT, WAIT,
    ASRSchedulerEnv, EnvConfig, EpisodeBuffer, Transition,
    collect_episode, collect_episodes_batched, grpo_update,
)
from huperjepa.model.agent import ActiveAgent, ActiveAgentConfig


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------


def setup_ddp() -> tuple[int, int, int]:
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def reduce_mean(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / world_size


# ---------------------------------------------------------------------------
# Syllable feature helpers
# ---------------------------------------------------------------------------


def compute_steps_since_emit(oracle_emit: torch.Tensor) -> torch.Tensor:
    """Vectorised steps-since-last-emit from oracle labels (B, K) → (B, K)."""
    B, K = oracle_emit.shape
    device = oracle_emit.device
    k_idx = torch.arange(K, device=device, dtype=torch.float)
    emit_positions = torch.where(
        oracle_emit > 0.5,
        k_idx.unsqueeze(0).expand(B, -1),
        torch.full((B, K), -1.0, device=device),
    )
    last_emit_idx, _ = torch.cummax(emit_positions, dim=1)
    return k_idx.unsqueeze(0) - last_emit_idx


def build_syl_features_batch(
    boundaries: torch.Tensor,
    num_slots: torch.Tensor,
    oracle_emit: torch.Tensor | None = None,
) -> torch.Tensor:
    """Vectorised (B, K, 3) syllable feature matrix for a batch."""
    B, max_K, _ = boundaries.shape
    dur = (boundaries[:, :, 1] - boundaries[:, :, 0]).float().clamp(min=1.0)
    log_dur = dur.log()
    k_idx = torch.arange(max_K, device=boundaries.device, dtype=torch.float).unsqueeze(0)
    denom = (num_slots.float() - 1.0).clamp(min=1.0).unsqueeze(1)
    rel_pos = k_idx / denom
    if oracle_emit is not None:
        steps_since = compute_steps_since_emit(oracle_emit.to(boundaries.device))
    else:
        steps_since = torch.zeros(B, max_K, device=boundaries.device)
    return torch.stack([log_dur, rel_pos, steps_since], dim=-1)


# ---------------------------------------------------------------------------
# Imitation Learning
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
    """One epoch of IL: cross-entropy on [WAIT, EMIT] distribution."""
    agent.train()
    total_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0, device=device)
    total_samples = torch.tensor(0, device=device)
    total_emit_tp = torch.tensor(0, device=device)
    total_emit_pos = torch.tensor(0, device=device)
    total_emit_pred = torch.tensor(0, device=device)

    class_weight = torch.tensor([1.0, pos_weight], device=device)

    pbar = tqdm(loader, desc=f"IL epoch {epoch:3d}", unit="batch",
                disable=not show_progress, dynamic_ncols=True)

    for batch in pbar:
        beliefs = batch["beliefs"].to(device)
        priors = batch["priors"].to(device)
        slot_mask = batch["slot_mask"].to(device)
        oracle = batch["oracle_emit"].to(device)
        boundaries = batch["boundaries"].to(device)
        num_slots = batch["num_slots"].to(device)

        syl_feats = build_syl_features_batch(boundaries, num_slots, oracle)
        distortions = batch["distortions"].to(device) if "distortions" in batch else None
        err = (distortions if distortions is not None
               else torch.zeros(beliefs.shape[0], beliefs.shape[1], 1, device=device))

        logits, _values, _ = agent(beliefs, priors, syl_feats, err)

        actions = oracle.long()
        loss_per_token = F.cross_entropy(
            logits.view(-1, 2), actions.view(-1), weight=class_weight, reduction="none",
        )
        mask_flat = slot_mask.view(-1)
        loss = (loss_per_token * mask_flat).sum() / mask_flat.sum().clamp(min=1)

        optimizer.zero_grad()
        loss.backward()
        raw = agent.module if isinstance(agent, DDP) else agent
        nn.utils.clip_grad_norm_(raw.parameters(), 5.0)
        optimizer.step()

        preds = logits.argmax(dim=-1).float()
        mask = slot_mask.bool()
        n = mask.sum()
        total_loss += loss.detach() * n
        total_correct += ((preds == oracle) & mask).sum()
        total_samples += n
        total_emit_tp += ((preds == 1) & (oracle == 1) & mask).sum()
        total_emit_pos += (oracle == 1).sum()
        total_emit_pred += ((preds == 1) & mask).sum()

        if show_progress:
            cur_n = total_samples.item()
            pbar.set_postfix(
                loss=f"{total_loss.item() / max(cur_n, 1):.4f}",
                acc=f"{total_correct.item() / max(cur_n, 1):.3f}",
                recall=f"{total_emit_tp.item() / max(total_emit_pos.item(), 1):.3f}",
            )

    for t in [total_loss, total_correct, total_samples, total_emit_tp, total_emit_pos, total_emit_pred]:
        reduce_mean(t, world_size)

    n = total_samples.item()
    recall = total_emit_tp.item() / max(total_emit_pos.item(), 1)
    precision = total_emit_tp.item() / max(total_emit_pred.item(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "il_loss": total_loss.item() / max(n, 1),
        "il_acc": total_correct.item() / max(n, 1),
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
        err = (distortions if distortions is not None
               else torch.zeros(beliefs.shape[0], beliefs.shape[1], 1, device=device))

        logits, _, _ = agent(beliefs, priors, syl_feats, err)
        preds = logits.argmax(dim=-1).float()
        mask = slot_mask.bool()
        total_correct += ((preds == oracle) & mask).sum()
        total_samples += mask.sum()
        total_emit_tp += ((preds == 1) & (oracle == 1) & mask).sum()
        total_emit_pos += (oracle == 1).sum()
        total_emit_pred += ((preds == 1) & mask).sum()

    for t in [total_correct, total_samples, total_emit_tp, total_emit_pos, total_emit_pred]:
        reduce_mean(t, world_size)

    recall = total_emit_tp.item() / max(total_emit_pos.item(), 1)
    precision = total_emit_tp.item() / max(total_emit_pred.item(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "val_acc": total_correct.item() / max(total_samples.item(), 1),
        "val_emit_recall": recall,
        "val_emit_precision": precision,
        "val_emit_f1": f1,
    }


@torch.no_grad()
def evaluate_agent(
    agent: ActiveAgent,
    dataset: AgentDataset,
    phone_vocab: Vocabulary,
    device: torch.device,
    env_cfg: EnvConfig,
    max_episodes: int = 200,
) -> Dict[str, float]:
    agent.eval()
    total_reward = 0.0
    total_correct = 0
    total_emitted = 0
    total_oracle_slots = 0
    total_word_acc = 0.0
    total_word_emit = 0
    n_episodes = 0

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:max_episodes]

    is_word_match = env_cfg.reward_mode == "word_match"

    for idx in indices:
        record = dataset[idx]
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
        )

        word_phone_ids = record.get("word_phone_ids") or []
        word_ptr_eval = 0
        emit_start_eval = 0

        ep = collect_episode(agent, env, device)
        total_reward += sum(t.reward for t in ep.transitions)
        total_oracle_slots += env.total_oracle_slots

        oracle_set = env._oracle_slot_set if env.oracle_emit is not None else set()
        for t_idx, t in enumerate(ep.transitions):
            if t.action != EMIT:
                continue
            total_emitted += 1
            if t_idx in oracle_set:
                total_correct += 1
            if is_word_match and word_ptr_eval < len(word_phone_ids):
                expected_ids = word_phone_ids[word_ptr_eval]
                if expected_ids:
                    decoded_ids = env._decode_ctc_phone_ids(emit_start_eval, t_idx)
                    per = env._phone_per(decoded_ids, expected_ids)
                    total_word_acc += max(0.0, 1.0 - per)
                    total_word_emit += 1
                word_ptr_eval += 1
                emit_start_eval = t_idx + 1

        n_episodes += 1

    n_episodes = max(n_episodes, 1)
    precision = total_correct / max(total_emitted, 1)
    recall = total_correct / max(total_oracle_slots, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    word_acc = total_word_acc / max(total_word_emit, 1) if total_word_emit > 0 else 0.0
    return {
        "avg_reward": total_reward / n_episodes,
        "emit_ratio": total_emitted / max(total_oracle_slots, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "word_acc": word_acc,
        "episodes": n_episodes,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ActiveAgent for streaming ASR.")
    p.add_argument("--agent-data-dir", required=True)
    p.add_argument("--metadata-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--phase", choices=["il", "grpo", "both"], default="both")

    # Agent architecture
    p.add_argument("--belief-dim", type=int, default=256)
    p.add_argument("--agent-hidden", type=int, default=128)
    p.add_argument("--gru-layers", type=int, default=1)
    p.add_argument("--agent-dropout", type=float, default=0.1)

    # IL
    p.add_argument("--il-epochs", type=int, default=30)
    p.add_argument("--il-lr", type=float, default=1e-3)
    p.add_argument("--il-batch-size", type=int, default=32)
    p.add_argument("--il-pos-weight", type=float, default=4.0)
    p.add_argument("--oracle-min-gap", type=int, default=2)

    # GRPO
    p.add_argument("--grpo-epochs", type=int, default=150)
    p.add_argument("--grpo-lr", type=float, default=1e-4)
    p.add_argument("--grpo-utterances-per-update", type=int, default=32)
    p.add_argument("--grpo-rollouts", type=int, default=8)
    p.add_argument("--grpo-clip-eps", type=float, default=0.2)
    p.add_argument("--grpo-entropy-coef", type=float, default=0.02)
    p.add_argument("--grpo-mini-epochs", type=int, default=1)
    p.add_argument("--rollout-temperature", type=float, default=1.0)

    # Environment
    p.add_argument("--wait-penalty", type=float, default=-0.01)
    p.add_argument("--correct-reward", type=float, default=1.0)
    p.add_argument("--wrong-penalty", type=float, default=-1.0)
    p.add_argument("--incomplete-penalty", type=float, default=0.0)
    p.add_argument("--reward-mode", choices=["per_slot", "episode_f1", "hybrid", "word_match"],
                   default="per_slot")
    p.add_argument("--f1-reward-scale", type=float, default=1.0)
    p.add_argument("--f1-match-window", type=int, default=0)
    p.add_argument("--missing-word-penalty", type=float, default=0.5)

    # General
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-preload", action="store_true")
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

    if world_size > 1:
        dist.barrier(device_ids=[local_rank])

    phone_vocab = Vocabulary.from_file(Path(args.metadata_dir) / "phone_vocab.json")

    preload = not args.no_preload
    oracle_min_gap = args.oracle_min_gap
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

    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        il_shuffle = False
    else:
        train_sampler = val_sampler = None
        il_shuffle = True

    train_loader = DataLoader(
        train_ds, batch_size=args.il_batch_size,
        shuffle=il_shuffle, sampler=train_sampler,
        num_workers=args.num_workers, collate_fn=collator, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.il_batch_size,
        shuffle=False, sampler=val_sampler,
        num_workers=args.num_workers, collate_fn=collator, pin_memory=True,
    )

    cfg = ActiveAgentConfig(
        belief_dim=args.belief_dim,
        agent_hidden=args.agent_hidden,
        gru_layers=args.gru_layers,
        dropout=args.agent_dropout,
    )
    agent_raw = ActiveAgent(cfg).to(device)

    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        agent_raw.load_state_dict(ckpt["agent_state_dict"])
        if is_main(rank):
            print(f"Resumed agent weights from {args.resume_from}")

    if world_size > 1:
        agent: nn.Module = DDP(agent_raw, device_ids=[local_rank], find_unused_parameters=True)
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
        missing_word_penalty=args.missing_word_penalty,
    )

    if is_main(rank):
        print(f"Agent parameters: {sum(p.numel() for p in agent_raw.parameters() if p.requires_grad):,}")

    run_il = args.phase in ("il", "both")
    run_grpo = args.phase in ("grpo", "both")

    # ------------------------------------------------------------------
    # Phase 1: Imitation Learning
    # ------------------------------------------------------------------
    if run_il:
        if is_main(rank):
            print(f"\n{'=' * 60}\nPhase 1: Imitation Learning  (world_size={world_size})\n{'=' * 60}")

        il_lr = (args.lr if args.lr else args.il_lr) * world_size
        optimizer = AdamW(agent_raw.parameters(), lr=il_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.il_epochs, eta_min=il_lr * 0.01)

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
                val_metrics = eval_il(agent, val_loader, device, world_size=world_size, show_progress=is_main(rank))
                if is_main(rank):
                    record = {"epoch": epoch, **train_metrics, **val_metrics, "lr": optimizer.param_groups[0]["lr"]}
                    history_il.append(record)
                    print(json.dumps(record, ensure_ascii=True), flush=True)
                    ckpt = {"agent_state_dict": agent_raw.state_dict(), "agent_config": cfg, "epoch": epoch, "history": history_il}
                    torch.save(ckpt, output_dir / "il_last.pt")
                    if val_metrics["val_emit_f1"] > best_f1:
                        best_f1 = val_metrics["val_emit_f1"]
                        torch.save(ckpt, output_dir / "il_best.pt")
                        print(f"  New best F1: {best_f1:.4f}")
            elif epoch % args.log_every == 0 and is_main(rank):
                print(json.dumps({"epoch": epoch, **train_metrics, "lr": optimizer.param_groups[0]["lr"]}), flush=True)

        if is_main(rank):
            print(f"IL complete. Best val F1: {best_f1:.4f}")

    # ------------------------------------------------------------------
    # Phase 2: GRPO Fine-tuning
    # ------------------------------------------------------------------
    if run_grpo:
        if is_main(rank):
            print(f"\n{'=' * 60}\nPhase 2: GRPO  (world_size={world_size})\n{'=' * 60}")

        if run_il and not args.resume_from:
            best_il = output_dir / "il_best.pt"
            if best_il.exists():
                ckpt = torch.load(best_il, map_location="cpu", weights_only=False)
                agent_raw.load_state_dict(ckpt["agent_state_dict"])
                if is_main(rank):
                    print("Starting GRPO from IL best checkpoint")

        if world_size > 1:
            for param in agent_raw.parameters():
                dist.broadcast(param.data, src=0)

        grpo_lr = ((args.lr if args.lr else args.grpo_lr)) * world_size
        optimizer = AdamW(agent_raw.parameters(), lr=grpo_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=1, eta_min=grpo_lr * 0.1,
        )

        best_f1 = -float("inf")
        history_grpo: list[dict] = []
        all_indices = list(range(len(train_ds)))

        epoch_pbar = tqdm(range(1, args.grpo_epochs + 1), desc="GRPO", unit="epoch",
                          disable=not is_main(rank), dynamic_ncols=True)

        for epoch in epoch_pbar:
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
                )
                group_episodes = collect_episodes_batched(
                    agent_raw, env, device, args.grpo_rollouts,
                    temperature=args.rollout_temperature,
                )
                for ep_buf in group_episodes:
                    epoch_reward += sum(t.reward for t in ep_buf.transitions)
                    epoch_emit_count += sum(1 for t in ep_buf.transitions if t.action == EMIT)
                epoch_oracle_slots += env.total_oracle_slots * args.grpo_rollouts
                groups.append(group_episodes)
                n_collected += 1

            update_metrics = grpo_update(
                agent_raw, optimizer, groups, device,
                clip_eps=args.grpo_clip_eps,
                entropy_coef=args.grpo_entropy_coef,
                grpo_mini_epochs=args.grpo_mini_epochs,
            )

            if world_size > 1:
                for param in agent_raw.parameters():
                    dist.all_reduce(param.data, op=dist.ReduceOp.AVG)
            scheduler.step()

            total_transitions = max(args.grpo_rollouts * n_collected, 1)
            avg_reward = epoch_reward / total_transitions
            emit_ratio = epoch_emit_count / max(epoch_oracle_slots, 1)

            if world_size > 1:
                avg_reward = reduce_mean(torch.tensor(avg_reward, device=device), world_size).item()

            if is_main(rank):
                epoch_pbar.set_postfix(
                    r=f"{avg_reward:.3f}",
                    pl=f"{update_metrics.get('policy_loss', 0):.4f}",
                    ent=f"{update_metrics.get('entropy', 0):.3f}",
                    emit=f"{emit_ratio:.2f}",
                )

            if epoch % args.eval_every == 0 or epoch == args.grpo_epochs:
                if is_main(rank):
                    val_metrics = evaluate_agent(agent_raw, val_ds, phone_vocab, device, env_cfg,
                                                 max_episodes=min(200, len(val_ds)))
                    record = {
                        "epoch": epoch, "train_avg_reward": avg_reward, "train_emit_ratio": emit_ratio,
                        **update_metrics, **{f"val_{k}": v for k, v in val_metrics.items()},
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                    history_grpo.append(record)
                    print(json.dumps(record, ensure_ascii=True), flush=True)
                    ckpt = {"agent_state_dict": agent_raw.state_dict(), "agent_config": cfg,
                            "env_config": env_cfg, "epoch": epoch, "history": history_grpo}
                    torch.save(ckpt, output_dir / "grpo_last.pt")
                    val_f1 = val_metrics.get("f1", 0.0)
                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        torch.save(ckpt, output_dir / "grpo_best.pt")
                        word_acc_str = (f" word_acc={val_metrics.get('word_acc', 0):.3f}"
                                        if env_cfg.reward_mode == "word_match" else "")
                        print(f"  New best F1: {best_f1:.4f} "
                              f"(prec={val_metrics.get('precision', 0):.3f} "
                              f"rec={val_metrics.get('recall', 0):.3f}){word_acc_str}")
            elif epoch % args.log_every == 0 and is_main(rank):
                print(json.dumps({"epoch": epoch, "train_avg_reward": avg_reward,
                                  "train_emit_ratio": emit_ratio, **update_metrics,
                                  "lr": optimizer.param_groups[0]["lr"]}), flush=True)

        if is_main(rank):
            print(f"GRPO complete. Best val F1: {best_f1:.4f}")

    if is_main(rank):
        print(f"\nTraining complete → {output_dir}")
    cleanup_ddp()


if __name__ == "__main__":
    main()
