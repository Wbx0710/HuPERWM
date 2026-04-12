#!/usr/bin/env bash
# ==============================================================================
# Stage 2: ASR Training with JEPA Backbone
# ==============================================================================
# Loads Stage 1 encoder weights and trains all CTC read-out heads:
#   - Frame-level phone CTC  (evidence + belief context → realized phones)
#   - Slot-level canonical CTC (priors → canonical phones)
#   - Slot-level evidence CTC  (slots → realized phones)
#   - Future slot prediction
#   - Reconstruction loss
#   - JEPA auxiliary loss (keeps encoder representations sharp)
#
# Multi-GPU: automatically uses all visible GPUs.
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_jepa_stage2_asr.sh
#   NGPUS=4 bash scripts/train_jepa_stage2_asr.sh
#
# Key outputs:
#   runs/jepa_stage2_asr/best.pt   (best canonical PER)
#   runs/jepa_stage2_asr/last.pt
# ==============================================================================
set -euo pipefail

export HF_HOME="${HF_HOME:-/data/bixingwu/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/bixingwu/.cache/huggingface/datasets}"
export NLTK_DATA="${NLTK_DATA:-/data/bixingwu/nltk_data}"

WORK_DIR="${WORK_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
META="${META:-${WORK_DIR}/artifacts/metadata_librispeech}"
FEATURES_DIR="${FEATURES_DIR:-${WORK_DIR}/artifacts/wm_features_librispeech}"
OUT="${OUT:-${WORK_DIR}/runs/jepa_stage2_asr}"
NGPUS="${NGPUS:-}"
TEACHER_CACHE_TRAIN="${TEACHER_CACHE_TRAIN:-}"
TEACHER_CACHE_VAL="${TEACHER_CACHE_VAL:-}"

# Stage 1 checkpoint (auto-detect)
S1_DIR="${S1_DIR:-${WORK_DIR}/runs/jepa_stage1}"
S1_CKPT="${S1_CKPT:-}"
if [[ -z "${S1_CKPT}" ]]; then
  if [[ -f "${S1_DIR}/best_stage1.pt" ]]; then
    S1_CKPT="${S1_DIR}/best_stage1.pt"
  fi
fi

# --- Data mode ---
data_args=()
if [[ -d "${FEATURES_DIR}/train" ]]; then
  echo "[Stage 2 ASR] Offline mode: ${FEATURES_DIR}"
  data_args+=(--features-dir "${FEATURES_DIR}")
  num_workers=4
else
  echo "[Stage 2 ASR] Online mode: HuggingFace streaming"
  data_args+=(
    --online-features
    --hf-dataset-name openslr/librispeech_asr
    --hf-dataset-config all
    --hf-train-split train.clean.360
    --hf-val-split validation.clean
    --feature-device cuda
  )
  num_workers=0
fi

device_args=()
if [[ -n "${NGPUS}" ]]; then
  device_args+=(--devices "${NGPUS}")
fi

teacher_args=()
_teacher_jsonl=()
[[ -n "${TEACHER_CACHE_TRAIN}" ]] && _teacher_jsonl+=("${TEACHER_CACHE_TRAIN}")
[[ -n "${TEACHER_CACHE_VAL}" ]] && _teacher_jsonl+=("${TEACHER_CACHE_VAL}")
if [[ ${#_teacher_jsonl[@]} -gt 0 ]]; then
  teacher_args+=(--teacher-cache "${_teacher_jsonl[@]}")
fi

s1_args=()
if [[ -n "${S1_CKPT}" && -f "${S1_CKPT}" ]]; then
  echo "[Stage 2 ASR] Loading Stage 1 weights: ${S1_CKPT}"
  s1_args+=(--stage1-checkpoint "${S1_CKPT}")
else
  echo "[Stage 2 ASR] WARNING: No Stage 1 checkpoint found, training from scratch"
fi

EPOCHS="${EPOCHS:-150}"
RESUME_FROM="${RESUME_FROM:-}"

resume_args=()
if [[ -n "${RESUME_FROM}" && -f "${RESUME_FROM}" ]]; then
  echo "[Stage 2 ASR] Resuming from: ${RESUME_FROM}"
  resume_args+=(--resume-from "${RESUME_FROM}")
fi

echo "[Stage 2 ASR] JEPA + CTC Training → ${OUT} (${EPOCHS} epochs)"
python "${WORK_DIR}/train_wm_belief.py" \
  --metadata-dir "${META}" \
  --output-dir "${OUT}" \
  "${data_args[@]}" \
  --evidence-type logits \
  --hidden-dim 256 \
  --pooling-type mean \
  --upsample-factor 4 \
  --dropout 0.1 \
  --belief-type jepa \
  --jepa-encoder-layers 6 \
  --jepa-encoder-heads 8 \
  --jepa-encoder-ff-dim 1024 \
  --jepa-encoder-conv-kernel 15 \
  --jepa-predictor-layers 2 \
  --jepa-predictor-heads 8 \
  --jepa-prior-layers 2 \
  --jepa-prior-heads 8 \
  --jepa-mask-ratio 0.5 \
  --jepa-ema-tau 0.996 \
  --jepa-ema-tau-end 0.9999 \
  --jepa-predictor-lr-mult 2.0 \
  --jepa-aux-weight 0.3 \
  --vicreg-weight 0.01 \
  --vicreg-var-gamma 1.0 \
  --frame-phone-weight 1.0 \
  --evidence-phone-weight 0.5 \
  --canonical-weight 0.5 \
  --future-weight 0.3 \
  --recon-weight 0.3 \
  --belief-grad-scale 0.0 \
  --frame-phone-dropout 0.1 \
  --batch-size 64 \
  --eval-batch-size 64 \
  --epochs "${EPOCHS}" \
  --lr 3e-4 \
  --min-lr-ratio 0.01 \
  --weight-decay 1e-2 \
  --warmup-steps 200 \
  --max-grad-norm 5.0 \
  --patience 50 \
  --num-workers "${num_workers}" \
  --eval-every-epochs 5 \
  --log-every-steps 50 \
  --precision bf16-mixed \
  "${s1_args[@]+"${s1_args[@]}"}" \
  "${teacher_args[@]+"${teacher_args[@]}"}" \
  "${resume_args[@]+"${resume_args[@]}"}" \
  "${device_args[@]+"${device_args[@]}"}"

echo "[Stage 2 ASR] Complete → ${OUT}"
