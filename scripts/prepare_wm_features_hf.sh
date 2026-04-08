#!/usr/bin/env bash
# Pre-extract HuPER + Sylber features from HuggingFace LibriSpeech.
#
# Saves one .pt file per audio clip to /data/bixingwu/ (RAID, 1.9T free).
# Run this ONCE before offline training. The 104k train clips take ~3-4 hours
# on a single GPU. Subsequent training runs load from disk and are ~5x faster.
#
# Usage:
#   conda activate phn
#   bash scripts/prepare_wm_features_hf.sh
#
# Override output dir:
#   OUT_DIR=/your/path bash scripts/prepare_wm_features_hf.sh
set -euo pipefail

export HF_HOME="${HF_HOME:-/data/bixingwu/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/bixingwu/.cache/huggingface/datasets}"
export NLTK_DATA="${NLTK_DATA:-/data/bixingwu/nltk_data}"

WORK_DIR="${WORK_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
META_DIR="${META_DIR:-${WORK_DIR}/artifacts/metadata_librispeech}"
OUT_DIR="${OUT_DIR:-/data/bixingwu/huperworldmodel/artifacts/wm_features_librispeech}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUT_DIR}"

echo "=============================================="
echo " Offline feature extraction for LibriSpeech"
echo "=============================================="
echo " Output dir : ${OUT_DIR}"
echo " HF cache   : ${HF_DATASETS_CACHE}"
echo " Device     : ${DEVICE}"
echo " Splits     : train (104k clips) + validation (2.7k clips)"
echo " Disk est.  : ~6.5 GB (logits fp16 only)"
echo " Time est.  : ~3-4 hours on a single A6000"
echo "=============================================="

python "${WORK_DIR}/wm_prepare.py" \
  --hf-dataset-name openslr/librispeech_asr \
  --hf-dataset-config all \
  --hf-train-split train.clean.360 \
  --hf-val-split validation.clean \
  --metadata-dir "${META_DIR}" \
  --output-dir "${OUT_DIR}" \
  --splits train validation \
  --evidence-type logits \
  --device "${DEVICE}"

echo ""
echo "Feature extraction complete → ${OUT_DIR}"
echo ""
echo "To start offline training, run:"
echo "  FEATURES_DIR=${OUT_DIR} bash scripts/train_wm_identity.sh"
