#!/usr/bin/env bash
# HuperJEPA v3 — ActiveAgent with comparison-gated dual-pathway architecture
#
# Key differences from v2:
#   - --use-active-agent  (dual-pathway comparison-gated agent)
#   - No --use-distortion (error signal comes from comparison encoder, delivered
#     via the 'distortions' key in agent data)
#   - Agent data from comparison Stage 2 checkpoint

CUDA_VISIBLE_DEVICES=0,2,3,4 torchrun --nproc_per_node=4 --master_port=${MASTER_PORT:-29502} \
    train_wm_agent.py \
    --agent-data-dir /data/bixingwu/agent_data_v3 \
    --metadata-dir artifacts/metadata_librispeech \
    --output-dir /data/bixingwu/runs/active_agent_v3 \
    --phase both \
    --belief-dim 256 \
    --agent-hidden 128 \
    --use-active-agent \
    --oracle-min-gap 2 \
    --il-epochs 30 \
    --il-lr 1e-3 \
    --il-batch-size 64 \
    --il-pos-weight 4.0 \
    --grpo-epochs 200 \
    --grpo-lr 1e-4 \
    --grpo-utterances-per-update 256 \
    --grpo-rollouts 32 \
    --grpo-clip-eps 0.2 \
    --grpo-entropy-coef 0.1 \
    --grpo-mini-epochs 1 \
    --rollout-temperature 5.0 \
    --wait-penalty -0.01 \
    --correct-reward 1.0 \
    --wrong-penalty -0.8 \
    --incomplete-penalty 0.0 \
    --reward-mode word_match \
    --f1-reward-scale 1.0 \
    --f1-match-window 1 \
    --missing-word-penalty 0.5 \
    --eval-every 5 \
    --num-workers 0 \
    --device cuda
