#!/usr/bin/env bash
# ==============================================================================
# Stage 3: ASR + TTS Joint Training with JEPA Backbone
# ==============================================================================
# Loads Stage 2 ASR weights and adds TTS flow-matching decoder.
# Uses hidden-state evidence (1024-dim) for richer acoustic representation.
#
# Prerequisites:
#   1. Feature files must contain huper_hidden (1024-dim):
#      python scripts/update_features.py --features-dir $FEATURES_DIR --device cuda
#   2. Mel statistics must be computed:
#      python scripts/compute_mel_stats.py --features-dir $FEATURES_DIR
#
# Multi-GPU:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_jepa_stage3_tts.sh
#
# Key outputs:
#   runs/jepa_stage3_tts/best.pt   (best canonical PER)
#   runs/jepa_stage3_tts/last.pt
# ==============================================================================
set -euo pipefail

export HF_HOME="${HF_HOME:-/data/bixingwu/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/bixingwu/.cache/huggingface/datasets}"
export NLTK_DATA="${NLTK_DATA:-/data/bixingwu/nltk_data}"

WORK_DIR="${WORK_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
META="${META:-${WORK_DIR}/artifacts/metadata_librispeech}"
FEATURES_DIR="${FEATURES_DIR:-${WORK_DIR}/artifacts/wm_features_librispeech}"
OUT="${OUT:-${WORK_DIR}/runs/jepa_stage3_tts}"
NGPUS="${NGPUS:-}"
TEACHER_CACHE_TRAIN="${TEACHER_CACHE_TRAIN:-}"
TEACHER_CACHE_VAL="${TEACHER_CACHE_VAL:-}"

# Stage 2 ASR checkpoint (auto-detect)
S2_DIR="${S2_DIR:-${WORK_DIR}/runs/jepa_stage2_asr}"
RESUME_FROM="${RESUME_FROM:-}"
if [[ -z "${RESUME_FROM}" ]]; then
  if [[ -f "${S2_DIR}/best.pt" ]]; then
    RESUME_FROM="${S2_DIR}/best.pt"
  fi
fi

MEL_STATS="${MEL_STATS:-${FEATURES_DIR}/mel_stats.pt}"

# --- Validate prerequisites ---
if [[ ! -f "${MEL_STATS}" ]]; then
  echo "[ERROR] mel_stats.pt not found at ${MEL_STATS}"
  echo "  Run:  python scripts/compute_mel_stats.py --features-dir ${FEATURES_DIR}"
  exit 1
fi

python -c "
import torch, json, sys, random
manifest = json.loads(open('${FEATURES_DIR}/train_manifest.json').read())
ids = manifest['segment_ids']
random.seed(0)
samples = random.sample(ids, min(10, len(ids)))
missing = [s for s in samples if 'huper_hidden' not in torch.load(f'${FEATURES_DIR}/train/{s}.pt', map_location='cpu', weights_only=False)]
if missing:
    print(f'[ERROR] {len(missing)}/{len(samples)} files missing huper_hidden.', file=sys.stderr)
    print(f'  Run:  python scripts/update_features.py --features-dir ${FEATURES_DIR} --device cuda', file=sys.stderr)
    sys.exit(1)
print('[Stage 3 TTS] Feature validation passed')
" || exit 1

# --- Data mode ---
data_args=()
if [[ -d "${FEATURES_DIR}/train" ]]; then
  echo "[Stage 3 TTS] Offline mode: ${FEATURES_DIR}"
  data_args+=(--features-dir "${FEATURES_DIR}")
  num_workers=4
else
  echo "[Stage 3 TTS] Online mode: HuggingFace streaming"
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

resume_args=()
if [[ -n "${RESUME_FROM}" && -f "${RESUME_FROM}" ]]; then
  echo "[Stage 3 TTS] Resuming from Stage 2: ${RESUME_FROM}"
  resume_args+=(--resume-from "${RESUME_FROM}")
else
  echo "[Stage 3 TTS] WARNING: No Stage 2 checkpoint found, training from scratch"
fi

echo "[Stage 3 TTS] JEPA + ASR + TTS Joint Training → ${OUT}"
python "${WORK_DIR}/train_wm_belief.py" \
  --metadata-dir "${META}" \
  --output-dir "${OUT}" \
  "${data_args[@]}" \
  --evidence-type hidden \
  --hidden-dim 256 \
  --pooling-type mean \
  --upsample-factor 4 \
  --dropout 0.1 \
  --use-identity --identity-dim 128 \
  --use-prosody --prosody-dim 64 \
  --use-uncertainty --uncertainty-dim 32 \
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
  --jepa-aux-weight 0.05 \
  --use-tts --tts-weight 1.0 --tts-dur-weight 0.2 \
  --tts-decoder-dim 512 --tts-decoder-layers 6 \
  --mel-stats-path "${MEL_STATS}" \
  --tts-finetune-encoder \
  --frame-phone-weight 1.0 \
  --evidence-phone-weight 0.5 \
  --canonical-weight 0.5 \
  --future-weight 0.3 \
  --recon-weight 0.3 \
  --batch-size 12 \
  --eval-batch-size 12 \
  --accumulate-grad-batches 2 \
  --epochs 100 \
  --lr 2e-4 \
  --weight-decay 1e-2 \
  --warmup-steps 500 \
  --max-grad-norm 5.0 \
  --num-workers "${num_workers}" \
  --eval-every-epochs 5 \
  --log-every-steps 50 \
  --precision bf16-mixed \
  "${resume_args[@]+"${resume_args[@]}"}" \
  "${teacher_args[@]+"${teacher_args[@]}"}" \
  "${device_args[@]+"${device_args[@]}"}"

echo "[Stage 3 TTS] Complete → ${OUT}"
