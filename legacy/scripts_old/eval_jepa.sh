#!/usr/bin/env bash
# ==============================================================================
# Evaluate a trained JEPA model checkpoint
# ==============================================================================
# Usage:
#   bash scripts/eval_jepa.sh                                      # eval best Stage 2
#   CKPT=runs/jepa_stage3_tts/best.pt bash scripts/eval_jepa.sh   # eval Stage 3
#   SPLIT=test bash scripts/eval_jepa.sh                           # eval on test set
# ==============================================================================
set -euo pipefail

WORK_DIR="${WORK_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
META="${META:-${WORK_DIR}/artifacts/metadata_librispeech}"
FEATURES_DIR="${FEATURES_DIR:-${WORK_DIR}/artifacts/wm_features_librispeech}"
SPLIT="${SPLIT:-validation}"
EVIDENCE="${EVIDENCE:-logits}"

# Auto-detect checkpoint
CKPT="${CKPT:-}"
if [[ -z "${CKPT}" ]]; then
  for candidate in \
    "${WORK_DIR}/runs/jepa_stage2_asr/best.pt" \
    "${WORK_DIR}/runs/jepa_stage3_tts/best.pt" \
    "${WORK_DIR}/runs/jepa_stage2_asr/last.pt"; do
    if [[ -f "${candidate}" ]]; then
      CKPT="${candidate}"
      break
    fi
  done
fi

if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "[ERROR] No checkpoint found. Set CKPT=path/to/best.pt"
  exit 1
fi

OUT_DIR="$(dirname "${CKPT}")/eval_${SPLIT}"

echo "[Eval] Checkpoint: ${CKPT}"
echo "[Eval] Split: ${SPLIT} → ${OUT_DIR}"

python "${WORK_DIR}/eval_wm_belief.py" \
  --features-dir "${FEATURES_DIR}" \
  --metadata-dir "${META}" \
  --checkpoint "${CKPT}" \
  --output-dir "${OUT_DIR}" \
  --split "${SPLIT}" \
  --evidence-type "${EVIDENCE}" \
  --batch-size 16 \
  --num-workers 4

echo "[Eval] Results → ${OUT_DIR}"
echo "[Eval] Summary:"
cat "${OUT_DIR}/metrics.json" | python -m json.tool
