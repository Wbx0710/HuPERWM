#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python extract_agent_data.py \
    --checkpoint  /data/bixingwu/runs/wm_v3/best.pt \
    --features-dir /data/bixingwu/huperworldmodel/artifacts/wm_features_librispeech \
    --metadata-dir /data/bixingwu/huperworldmodel/artifacts/metadata_librispeech \
    --output-dir   /data/bixingwu/agent_data_v3 \
    --splits train validation \
    --evidence-type hidden \
    --batch-size 64 --num-workers 4
