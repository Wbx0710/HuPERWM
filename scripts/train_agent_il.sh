#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS=${CUDA_VISIBLE_DEVICES:-0}
N=$(echo "$GPUS" | tr ',' '\n' | wc -l)
export CUDA_VISIBLE_DEVICES=$GPUS

# Auto-select a free port (overridable via MASTER_PORT env var).
get_free_port() {
    python -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)"
}
PORT=${MASTER_PORT:-$(get_free_port)}
echo "[train_agent_il] Using port $PORT"

torchrun --nproc_per_node=$N --master_port=$PORT train_agent.py \
    --agent-data-dir /data/bixingwu/agent_data_pomdp_v1 \
    --metadata-dir   /data/bixingwu/huperworldmodel/artifacts/metadata_librispeech \
    --output-dir     /data/bixingwu/runs/pomdp_v1/il \
    --phase il \
    --belief-dim 256 --agent-hidden 128 --gru-layers 1 \
    --il-epochs 30 --il-lr 1e-3 --il-batch-size 64 \
    --il-pos-weight 4.0 --oracle-min-gap 2 \
    --rollout-eval-every 5 --rollout-eval-episodes 50 \
    --log-every 1 --seed 42
