#!/usr/bin/env bash
# Generate metadata (vocabs + JSONL) from raw LibriSpeech.
# When only dev-clean / test-clean are available, --use-dev-as-train
# splits dev-clean into 90 % train + 10 % validation.
set -euo pipefail

export LIBRISPEECH="${LIBRISPEECH:-/data/chenxu/datasets/librispeech/LibriSpeech}"
export WORK_DIR="${WORK_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export OUT_DIR="${OUT_DIR:-${WORK_DIR}/artifacts/metadata_librispeech}"

python "${WORK_DIR}/wm_prepare_meta_librispeech.py" \
  --librispeech-path "${LIBRISPEECH}" \
  --output-dir "${OUT_DIR}" \
  --use-dev-as-train \
  --val-ratio 0.1
