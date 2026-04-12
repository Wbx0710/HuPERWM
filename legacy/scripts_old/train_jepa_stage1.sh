#!/usr/bin/env bash
# ==============================================================================
# Stage 1: JEPA Self-Supervised Pretraining
# ==============================================================================
# Trains the online encoder (Conformer+DAAM) and predictor via masked slot
# prediction in latent space. No phone labels or TTS targets required.
#
# Multi-GPU: automatically uses all visible GPUs.
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_jepa_stage1.sh
#   NGPUS=4 bash scripts/train_jepa_stage1.sh
#
# Key outputs:
#   runs/jepa_stage1/best_stage1.pt   (best validation JEPA loss)
#   runs/jepa_stage1/last_stage1.pt
# ==============================================================================
set -euo pipefail

export HF_HOME="${HF_HOME:-/data/bixingwu/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/bixingwu/.cache/huggingface/datasets}"
export NLTK_DATA="${NLTK_DATA:-/data/bixingwu/nltk_data}"

WORK_DIR="${WORK_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
META="${META:-${WORK_DIR}/artifacts/metadata_librispeech}"
FEATURES_DIR="${FEATURES_DIR:-${WORK_DIR}/artifacts/wm_features_librispeech}"
OUT="${OUT:-${WORK_DIR}/runs/jepa_stage1}"
NGPUS="${NGPUS:-}"

# --- Data mode: offline features (fast) or online extraction (slow) ---
data_args=()
if [[ -d "${FEATURES_DIR}/train" ]]; then
  echo "[Stage 1] Offline mode: ${FEATURES_DIR}"
  data_args+=(--features-dir "${FEATURES_DIR}")
  num_workers=4
else
  echo "[Stage 1] Online mode: HuggingFace streaming"
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

echo "[Stage 1] JEPA Self-Supervised Pretraining → ${OUT}"
python "${WORK_DIR}/train_jepa_stage1.py" \
  --metadata-dir "${META}" \
  --output-dir "${OUT}" \
  "${data_args[@]}" \
  --evidence-type logits \
  --hidden-dim 256 \
  --pooling-type mean \
  --dropout 0.1 \
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
  --vicreg-weight 0.01 \
  --vicreg-var-gamma 1.0 \
  --vicreg-warmup-steps 500 \
  --predictor-lr-mult 2.0 \
  --batch-size 64 \
  --eval-batch-size 64 \
  --epochs 50 \
  --lr 1.5e-4 \
  --weight-decay 1e-3 \
  --warmup-steps 500 \
  --max-grad-norm 5.0 \
  --num-workers "${num_workers}" \
  --eval-every-epochs 5 \
  --log-every-steps 50 \
  --precision bf16-mixed \
  "${device_args[@]+"${device_args[@]}"}"

echo "[Stage 1] Complete → ${OUT}"
