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
echo "[train_agent_grpo] Using port $PORT"

torchrun --nproc_per_node=$N --master_port=$PORT train_agent.py \
    --agent-data-dir /data/bixingwu/agent_data_pomdp_v1 \
    --metadata-dir   /data/bixingwu/huperworldmodel/artifacts/metadata_librispeech \
    --output-dir     /data/bixingwu/runs/pomdp_v1/grpo \
    --phase grpo \
    --resume-from    /data/bixingwu/runs/pomdp_v1/il/il_best.pt \
    --belief-dim 256 --agent-hidden 128 --gru-layers 1 \
    --grpo-epochs 200 --grpo-lr 1e-4 \
    --grpo-utterances-per-update 32 --grpo-rollouts 8 \
    --grpo-clip-eps 0.2 \
    --grpo-entropy-coef 0.05 --grpo-entropy-coef-end 0.01 \
    --rollout-temperature 1.0 \
    --reward-mode word_match \
    --correct-reward 1.0 --wrong-penalty -0.5 --missing-word-penalty 0.5 \
    --info-gain-scale 0.1 \
    --gae-gamma 0.99 --gae-lambda 0.95 --value-coef 0.5 \
    --eval-every 10 --seed 42
