#!/usr/bin/env bash
# ==============================================================================
# RL Scheduler Agent Training Pipeline
# ==============================================================================
# Three steps:
#   1. Extract agent training data from a Stage 2 checkpoint
#   2. Imitation learning (IL) on oracle word-boundary labels
#   3. GRPO fine-tuning with word_match reward
#
# Prerequisites:
#   - A trained Stage 2 checkpoint (best.pt from train_jepa_stage2_asr.sh)
#   - Pre-extracted features (artifacts/wm_features_librispeech)
#
# Usage (defaults run v6):
#   bash scripts/train_agent.sh                         # v6: IL → GRPO, word_match
#   SKIP_EXTRACT=1 bash scripts/train_agent.sh          # skip data extraction
#   PHASE=grpo bash scripts/train_agent.sh              # GRPO only (resume from IL)
#   PHASE=ppo  bash scripts/train_agent.sh              # PPO only (legacy)
#   REWARD_MODE=per_slot bash scripts/train_agent.sh    # revert to per-slot reward
#
# Key fixes in v6 vs v5b (see analysis in plan fix_grpo_agent_training):
#   - IL loss: cross-entropy on full [WAIT, EMIT] distribution (was BCE on emit
#     head only → WAIT head received no gradient → P(EMIT)≈0.5 everywhere →
#     emit_ratio=1.40 before GRPO even started)
#   - il_pos_weight=4.0 (was 0.4): properly up-weights oracle emit positions so
#     the EMIT head fires confidently at word boundaries
#   - PHASE=both: re-run IL with fixed loss; start GRPO from a calibrated policy
#   - REWARD_MODE=word_match (explicitly set — v5b used hybrid by mistake)
#   - grpo_entropy_coef=0.1 (was 0.05): stronger entropy regularisation to
#     prevent the 137× entropy collapse seen in v5b
#   - rollout_temperature=1.5: forces diverse GRPO rollouts when policy is
#     near-deterministic, preventing group_std≈0 → advantage≈0 collapse
#   - LR schedule: CosineAnnealingWarmRestarts T_0=50 (was monotonic decay
#     to 8e-8 by epoch 200, preventing any update in the final 150 epochs)
# ==============================================================================
set -euo pipefail

export HF_HOME="${HF_HOME:-/data/bixingwu/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/bixingwu/.cache/huggingface/datasets}"

WORK_DIR="${WORK_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
META="${META:-${WORK_DIR}/artifacts/metadata_librispeech}"
FEATURES_DIR="${FEATURES_DIR:-${WORK_DIR}/artifacts/wm_features_librispeech}"
S2_CKPT="${S2_CKPT:-${WORK_DIR}/runs/jepa_stage2_asr/best.pt}"
AGENT_DATA="${AGENT_DATA:-/data/bixingwu/agent_data}"
AGENT_OUT="${AGENT_OUT:-/data/bixingwu/runs/agent_grpo_v6}"
SKIP_EXTRACT="${SKIP_EXTRACT:-1}"
PHASE="${PHASE:-both}"
GPUS="${GPUS:-2,3,4,5}"
N_GPUS="${N_GPUS:-4}"
# word_match is the correct reward for v6.  hybrid is a valid fallback but
# was the cause of val_word_acc=0.0 in v5b (do NOT override to hybrid).
REWARD_MODE="${REWARD_MODE:-word_match}"

# --- Step 1: Extract agent data ---
if [[ "${SKIP_EXTRACT}" != "1" ]]; then
  echo "[Agent] Step 1: Extracting agent training data from ${S2_CKPT}"
  python "${WORK_DIR}/wm_agent_data.py" \
    --checkpoint "${S2_CKPT}" \
    --features-dir "${FEATURES_DIR}" \
    --metadata-dir "${META}" \
    --output-dir "${AGENT_DATA}" \
    --splits train validation \
    --batch-size 32 \
    --num-workers 4
  echo "[Agent] Step 1 complete → ${AGENT_DATA}"
else
  echo "[Agent] Skipping data extraction (SKIP_EXTRACT=1)"
fi

# --- Resume args: only for PHASE=grpo (skip-IL path) ---
RESUME_ARGS=()
if [[ "${PHASE}" == "grpo" ]]; then
  # Warm-start from a previously trained checkpoint when skipping IL.
  # Default: use the v6 il_best.pt if it exists, else v4 grpo_best.pt.
  IL_CKPT="${IL_CKPT:-${AGENT_OUT}/il_best.pt}"
  if [[ ! -f "${IL_CKPT}" ]]; then
    IL_CKPT="/data/bixingwu/runs/agent_grpo_v4/grpo_best.pt"
  fi
  if [[ -f "${IL_CKPT}" ]]; then
    RESUME_ARGS+=(--resume-from "${IL_CKPT}")
    echo "[Agent] GRPO will resume from: ${IL_CKPT}"
  else
    echo "[Agent] WARNING: no checkpoint found, starting GRPO from scratch"
  fi
fi

# --- Step 2+3: Train agent (v6) ---
echo "[Agent] Training agent v6 (phase=${PHASE}, reward_mode=${REWARD_MODE}, GPUs=${GPUS}, n=${N_GPUS})"
CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --nproc_per_node="${N_GPUS}" \
  "${WORK_DIR}/train_wm_agent.py" \
  --agent-data-dir "${AGENT_DATA}" \
  --metadata-dir "${META}" \
  --output-dir "${AGENT_OUT}" \
  --phase "${PHASE}" \
  --belief-dim 256 \
  --agent-hidden 128 \
  --gru-layers 1 \
  --oracle-min-gap 2 \
  --il-epochs 30 \
  --il-lr 1e-3 \
  --il-batch-size 64 \
  --il-pos-weight 4.0 \
  --grpo-epochs 200 \
  --grpo-lr 1e-4 \
  --lr 2e-6 \
  --grpo-utterances-per-update 256 \
  --grpo-rollouts 32 \
  --grpo-clip-eps 0.2 \
  --grpo-entropy-coef 0.1 \
  --grpo-mini-epochs 1 \
  --rollout-temperature 5.0 \
  --wait-penalty -0.01 \
  --correct-reward 1.0 \
  --wrong-penalty -0.8 \
  --incomplete-penalty 0.0 \
  --reward-mode "${REWARD_MODE}" \
  --f1-reward-scale 1.0 \
  --f1-match-window 1 \
  --missing-word-penalty 0.5 \
  --eval-every 5 \
  --num-workers 0 \
  --device cuda \
  "${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}"

echo "[Agent] Training complete → ${AGENT_OUT}"

# ==============================================================================
# Legacy: v5 config (grpo-only, hybrid reward — kept for reference)
# ==============================================================================
# AGENT_OUT=/data/bixingwu/runs/agent_grpo_v5 \
# PHASE=grpo \
# REWARD_MODE=word_match \
# IL_CKPT=/data/bixingwu/runs/agent_grpo_v5/il_best.pt \
# bash scripts/train_agent.sh
