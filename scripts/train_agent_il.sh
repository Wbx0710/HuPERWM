#!/bin/bash
# Usage: [CUDA_VISIBLE_DEVICES=0,1,2,3] bash scripts/train_agent_il.sh
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS=${CUDA_VISIBLE_DEVICES:-0}
N=$(echo "$GPUS" | tr ',' '\n' | wc -l)
export CUDA_VISIBLE_DEVICES=$GPUS

torchrun --nproc_per_node=$N --master_port=${MASTER_PORT:-29501} train_agent.py \
    --agent-data-dir /data/bixingwu/agent_data_v3 \
    --metadata-dir   /data/bixingwu/huperworldmodel/artifacts/metadata_librispeech \
    --output-dir     /data/bixingwu/runs/agent_v3 \
    --phase il \
    --belief-dim 256 --agent-hidden 128 --gru-layers 1 \
    --il-epochs 30 --il-lr 1e-3 --il-batch-size 64 \
    --il-pos-weight 4.0 --oracle-min-gap 2 \
    --eval-every 5 --seed 42
