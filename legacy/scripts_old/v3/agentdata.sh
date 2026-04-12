#!/usr/bin/env bash
# HuperJEPA v3 — Extract agent data from Comparison Stage 2 checkpoint

python wm_agent_data.py \
    --checkpoint /data/bixingwu/runs/comparison_stage2_v3/best.pt \
    --features-dir artifacts/wm_features_librispeech \
    --metadata-dir artifacts/metadata_librispeech \
    --output-dir /data/bixingwu/agent_data_v3 \
    --splits train validation \
    --batch-size 32 \
    --num-workers 4
