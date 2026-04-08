#!/usr/bin/env bash
# Extract HuPER + Sylber features from LibriSpeech.
set -euo pipefail

export LIBRISPEECH="${LIBRISPEECH:-/data/chenxu/datasets/librispeech/LibriSpeech}"
export WORK_DIR="${WORK_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export META_DIR="${META_DIR:-${WORK_DIR}/artifacts/metadata_librispeech}"
export OUT_DIR="${OUT_DIR:-${WORK_DIR}/artifacts/wm_features_librispeech}"
export SPLITS="${SPLITS:-train validation test}"
export MAX_PER_SPLIT="${MAX_PER_SPLIT:-}"

extra_args=()
if [[ -n "${MAX_PER_SPLIT}" ]]; then
  extra_args+=(--max-examples-per-split "${MAX_PER_SPLIT}")
fi

python "${WORK_DIR}/wm_prepare.py" \
  --librispeech-path "${LIBRISPEECH}" \
  --metadata-dir "${META_DIR}" \
  --output-dir "${OUT_DIR}" \
  --splits ${SPLITS} \
  --evidence-type both \
  "${extra_args[@]+"${extra_args[@]}"}"
