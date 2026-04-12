#!/usr/bin/env bash
# HuperJEPA v3 — Stage 2: Comparison-Refinement World Model
#
# Key differences from v2:
#   - belief_type=comparison  (replaces JEPA bidirectional encoder)
#   - pooling_type=enhanced_boundary  (adds inter-slot self-attention)
#   - evidence_type=hidden  (1024-dim HuPER hidden states)
#   - prior_layers=3  (deeper causal prior than v2's 2 layers)
#   - num_refinements=2  (iterative comparison-gated belief refinement)
#   - convergence_loss_weight=0.2  (encourages error decrease across iterations)
#   - sigreg_weight=0.05  (replaces VICReg for anti-collapse)
#   - No JEPA masking, no EMA target encoder, no DAAM

CUDA_VISIBLE_DEVICES=4,5,6,7 python train_wm_belief.py \
    --metadata-dir artifacts/metadata_librispeech \
    --features-dir artifacts/wm_features_librispeech \
    --output-dir /data/bixingwu/runs/comparison_stage2_v3 \
    --evidence-type hidden \
    --hidden-dim 256 \
    --pooling-type enhanced_boundary \
    --boundary-attn-heads 4 \
    --belief-type comparison \
    --jepa-prior-layers 3 \
    --jepa-prior-heads 8 \
    --jepa-encoder-ff-dim 1024 \
    --jepa-encoder-conv-kernel 15 \
    --num-refinements 2 \
    --refinement-heads 4 \
    --refinement-ff-dim 512 \
    --refinement-conv-kernel 15 \
    --convergence-loss-weight 0.2 \
    --sigreg-weight 0.05 \
    --sigreg-projections 64 \
    --frame-phone-weight 1.0 \
    --evidence-phone-weight 0.5 \
    --canonical-weight 0.5 \
    --future-weight 0.3 \
    --recon-weight 0.3 \
    --belief-grad-scale 0.0 \
    --frame-phone-dropout 0.1 \
    --batch-size 64 \
    --eval-batch-size 64 \
    --epochs 150 \
    --lr 3e-4 \
    --min-lr-ratio 0.01 \
    --weight-decay 1e-2 \
    --warmup-steps 200 \
    --max-grad-norm 5.0 \
    --patience 50 \
    --num-workers 4 \
    --eval-every-epochs 5 \
    --log-every-steps 50 \
    --precision bf16-mixed \
    --devices 4
