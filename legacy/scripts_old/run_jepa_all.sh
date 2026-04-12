#!/usr/bin/env bash
# ==============================================================================
# Master Script: Run the complete JEPA training pipeline
# ==============================================================================
# Executes Stage 1 → Stage 2 (ASR) → Eval, with optional Stage 3 (TTS).
#
# Usage:
#   bash scripts/run_jepa_all.sh              # Stage 1 + 2 + eval
#   bash scripts/run_jepa_all.sh --with-tts   # Stage 1 + 2 + 3 + eval
#
# Multi-GPU:
#   NGPUS=4 bash scripts/run_jepa_all.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_jepa_all.sh
#
# Resume from a specific stage (skips completed stages):
#   START_STAGE=2 bash scripts/run_jepa_all.sh
# ==============================================================================
set -euo pipefail

WORK_DIR="${WORK_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
START_STAGE="${START_STAGE:-1}"
WITH_TTS=false

for arg in "$@"; do
  case "${arg}" in
    --with-tts) WITH_TTS=true ;;
  esac
done

elapsed() {
  local start=$1
  local end=$(date +%s)
  local d=$(( (end - start) / 86400 ))
  local h=$(( (end - start) % 86400 / 3600 ))
  local m=$(( (end - start) % 3600 / 60 ))
  echo "${d}d ${h}h ${m}m"
}

PIPELINE_START=$(date +%s)

echo "============================================================"
echo " JEPA Training Pipeline"
echo "============================================================"
echo " Start stage : ${START_STAGE}"
echo " With TTS    : ${WITH_TTS}"
echo " NGPUS       : ${NGPUS:-auto}"
echo " Work dir    : ${WORK_DIR}"
echo "============================================================"
echo ""

# ------------------------------------------------------------------
# Stage 1: JEPA Self-Supervised Pretraining
# ------------------------------------------------------------------
if [[ "${START_STAGE}" -le 1 ]]; then
  echo ">>> Stage 1: JEPA Self-Supervised Pretraining"
  STAGE_START=$(date +%s)
  bash "${WORK_DIR}/scripts/train_jepa_stage1.sh"
  echo ">>> Stage 1 done ($(elapsed ${STAGE_START}))"
  echo ""
fi

# ------------------------------------------------------------------
# Stage 2: ASR with JEPA Backbone
# ------------------------------------------------------------------
if [[ "${START_STAGE}" -le 2 ]]; then
  echo ">>> Stage 2: ASR Training (CTC + JEPA aux)"
  STAGE_START=$(date +%s)
  bash "${WORK_DIR}/scripts/train_jepa_stage2_asr.sh"
  echo ">>> Stage 2 done ($(elapsed ${STAGE_START}))"
  echo ""

  echo ">>> Evaluating Stage 2 ASR model"
  CKPT="${WORK_DIR}/runs/jepa_stage2_asr/best.pt" \
    bash "${WORK_DIR}/scripts/eval_jepa.sh" || true
  echo ""
fi

# ------------------------------------------------------------------
# Stage 3 (optional): ASR + TTS Joint Training
# ------------------------------------------------------------------
if [[ "${WITH_TTS}" == "true" && "${START_STAGE}" -le 3 ]]; then
  echo ">>> Stage 3: ASR + TTS Joint Training"
  STAGE_START=$(date +%s)
  bash "${WORK_DIR}/scripts/train_jepa_stage3_tts.sh"
  echo ">>> Stage 3 done ($(elapsed ${STAGE_START}))"
  echo ""

  echo ">>> Evaluating Stage 3 TTS model"
  CKPT="${WORK_DIR}/runs/jepa_stage3_tts/best.pt" \
    EVIDENCE=hidden \
    bash "${WORK_DIR}/scripts/eval_jepa.sh" || true
  echo ""
fi

echo "============================================================"
echo " Pipeline Complete! Total time: $(elapsed ${PIPELINE_START})"
echo "============================================================"
echo ""
echo " Outputs:"
echo "   Stage 1 (JEPA):  runs/jepa_stage1/"
echo "   Stage 2 (ASR):   runs/jepa_stage2_asr/"
[[ "${WITH_TTS}" == "true" ]] && \
echo "   Stage 3 (TTS):   runs/jepa_stage3_tts/"
echo ""
echo " Quick commands:"
echo "   View ASR metrics:  cat runs/jepa_stage2_asr/eval_validation/metrics.json | python -m json.tool"
[[ "${WITH_TTS}" == "true" ]] && \
echo "   View TTS metrics:  cat runs/jepa_stage3_tts/eval_validation/metrics.json | python -m json.tool"
