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
echo "[train_world_model] Using port $PORT"

torchrun --nproc_per_node=$N --master_port=$PORT train_world_model.py \
    --features-dir /data/bixingwu/huperworldmodel/artifacts/wm_features_librispeech \
    --metadata-dir /data/bixingwu/huperworldmodel/artifacts/metadata_librispeech \
    --output-dir   /data/bixingwu/runs/wm_v4 \
    --evidence-type hidden \
    --hidden-dim 256 \
    --prior-layers 3 --prior-heads 8 --prior-ff-dim 1024 --prior-conv-kernel 15 \
    --num-refinements 2 --refinement-heads 4 --refinement-ff-dim 512 \
    --batch-size 64 --epochs 150 --lr 3e-4 --warmup-steps 400 \
    --convergence-loss-weight 0.2 --sigreg-weight 0.05 \
    --diversity-weight 0.1 --diversity-hinge 0.8 \
    --eval-every-epochs 5 --log-every-steps 10 \
    --num-workers 4 --seed 42 --patience 30 \
    --resume-from   /data/bixingwu/runs/wm_v4/last.pt
