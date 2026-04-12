#!/bin/bash
# Usage: [CUDA_VISIBLE_DEVICES=0,1,2,3] bash scripts/train_agent_grpo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS=${CUDA_VISIBLE_DEVICES:-0}
N=$(echo "$GPUS" | tr ',' '\n' | wc -l)
export CUDA_VISIBLE_DEVICES=$GPUS

torchrun --nproc_per_node=$N --master_port=${MASTER_PORT:-29502} train_agent.py \
    --agent-data-dir /data/bixingwu/agent_data_v3 \
    --metadata-dir   /data/bixingwu/huperworldmodel/artifacts/metadata_librispeech \
    --output-dir     /data/bixingwu/runs/agent_v3 \
    --phase grpo \
    --belief-dim 256 --agent-hidden 128 --gru-layers 1 \
    --grpo-epochs 200 --grpo-lr 1e-4 \
    --grpo-utterances-per-update 32 --grpo-rollouts 8 \
    --grpo-clip-eps 0.2 --grpo-entropy-coef 0.02 \
    --rollout-temperature 1.0 \
    --reward-mode word_match \
    --correct-reward 1.0 --wrong-penalty -0.5 --missing-word-penalty 0.5 \
    --eval-every 10 --seed 42
