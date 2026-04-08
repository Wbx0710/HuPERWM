#!/usr/bin/env bash
# === 快速验证脚本 ===
# 用最少数据验证整个训练 pipeline 能跑通。
# 3个阶段可以分别运行，也可以一次全跑。
#
# Usage:
#   conda activate phn
#   bash scripts/quick_test.sh           # 跑全部 3 个阶段
#   bash scripts/quick_test.sh meta      # 只准备 metadata
#   bash scripts/quick_test.sh baseline  # 只跑 baseline 训练
#   bash scripts/quick_test.sh extended  # 只跑 identity+prosody+uncertainty 训练
set -euo pipefail

export HF_HOME="${HF_HOME:-/data/bixingwu/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/bixingwu/.cache/huggingface/datasets}"
export NLTK_DATA="${NLTK_DATA:-/data/bixingwu/nltk_data}"

WORK_DIR="${WORK_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
META="${WORK_DIR}/artifacts/metadata_librispeech"
STAGE="${1:-all}"

# ============================================================
# 阶段 1: 准备 metadata (从 HuggingFace 在线获取)
# ============================================================
if [[ "${STAGE}" == "all" || "${STAGE}" == "meta" ]]; then
  echo "========================================"
  echo "阶段 1: 准备 metadata (phone_vocab + text_vocab)"
  echo "========================================"
  python "${WORK_DIR}/scripts/prepare_meta_from_hf.py" \
    --hf-dataset-name openslr/librispeech_asr \
    --hf-dataset-config all \
    --hf-split validation.clean \
    --output-dir "${META}" \
    --max-examples 200 || {
    # HF streaming datasets may crash on Python exit (PyGILState_Release bug).
    # If output files exist, the generation succeeded — ignore the exit code.
    if [[ -f "${META}/phone_vocab.json" && -f "${META}/text_vocab.json" ]]; then
      echo "⚠ Metadata script exited with error, but output files exist — continuing."
    else
      echo "✗ Metadata generation failed and output files missing."
      exit 1
    fi
  }
  echo ""
fi

# ============================================================
# 阶段 2: Baseline 模型快速 overfit (原始 ASR, 无扩展)
# ============================================================
if [[ "${STAGE}" == "all" || "${STAGE}" == "baseline" ]]; then
  echo "========================================"
  echo "阶段 2: Baseline ASR 快速 overfit 测试"
  echo "========================================"
  python "${WORK_DIR}/train_wm_belief.py" \
    --metadata-dir "${META}" \
    --output-dir "${WORK_DIR}/runs/quick_test_baseline" \
    --online-features \
    --hf-dataset-name openslr/librispeech_asr \
    --hf-dataset-config all \
    --hf-train-split validation.clean \
    --hf-val-split validation.clean \
    --feature-device cuda \
    --evidence-type logits \
    --hidden-dim 256 \
    --pooling-type mean \
    --upsample-factor 4 \
    --batch-size 4 \
    --eval-batch-size 4 \
    --epochs 5 \
    --lr 3e-4 \
    --warmup-steps 10 \
    --num-workers 0 \
    --eval-every-epochs 5 \
    --log-every-steps 2 \
    --max-train-examples 16 \
    --max-val-examples 8 \
    --devices 1 \
    --precision 32-true
  echo ""
  echo "Baseline 测试完成 → runs/quick_test_baseline/"
  echo ""
fi

# ============================================================
# 阶段 3: 扩展模型快速 overfit (Identity + Prosody + Uncertainty + Mismatch)
# ============================================================
if [[ "${STAGE}" == "all" || "${STAGE}" == "extended" ]]; then
  echo "========================================"
  echo "阶段 3: Extended 模型 (Identity+Prosody+Uncertainty) overfit 测试"
  echo "========================================"
  python "${WORK_DIR}/train_wm_belief.py" \
    --metadata-dir "${META}" \
    --output-dir "${WORK_DIR}/runs/quick_test_extended" \
    --online-features \
    --hf-dataset-name openslr/librispeech_asr \
    --hf-dataset-config all \
    --hf-train-split validation.clean \
    --hf-val-split validation.clean \
    --feature-device cuda \
    --evidence-type logits \
    --hidden-dim 256 \
    --pooling-type mean \
    --upsample-factor 4 \
    --use-identity --identity-dim 64 \
    --use-prosody --prosody-dim 32 \
    --use-uncertainty --uncertainty-dim 16 \
    --use-mismatch --mismatch-dim 32 \
    --batch-size 4 \
    --eval-batch-size 4 \
    --epochs 5 \
    --lr 3e-4 \
    --warmup-steps 10 \
    --num-workers 0 \
    --eval-every-epochs 5 \
    --log-every-steps 2 \
    --max-train-examples 16 \
    --max-val-examples 8 \
    --devices 1 \
    --precision 32-true
  echo ""
  echo "Extended 测试完成 → runs/quick_test_extended/"
  echo ""
fi

# ============================================================
# 阶段 4: JEPA Stage 1 自监督预训练快速测试
# ============================================================
if [[ "${STAGE}" == "all" || "${STAGE}" == "jepa_s1" ]]; then
  echo "========================================"
  echo "阶段 4: JEPA Stage 1 自监督预训练 (快速测试)"
  echo "========================================"
  python "${WORK_DIR}/train_jepa_stage1.py" \
    --metadata-dir "${META}" \
    --output-dir "${WORK_DIR}/runs/quick_test_jepa_s1" \
    --online-features \
    --hf-dataset-name openslr/librispeech_asr \
    --hf-dataset-config all \
    --hf-train-split validation.clean \
    --hf-val-split validation.clean \
    --feature-device cuda \
    --evidence-type logits \
    --hidden-dim 256 \
    --pooling-type mean \
    --jepa-encoder-layers 2 \
    --jepa-encoder-heads 4 \
    --jepa-encoder-ff-dim 512 \
    --jepa-predictor-layers 1 \
    --jepa-prior-layers 1 \
    --jepa-mask-ratio 0.5 \
    --batch-size 4 \
    --eval-batch-size 4 \
    --epochs 5 \
    --lr 1.5e-4 \
    --warmup-steps 10 \
    --num-workers 0 \
    --eval-every-epochs 5 \
    --log-every-steps 2 \
    --max-train-examples 16 \
    --max-val-examples 8 \
    --devices 1 \
    --precision 32-true
  echo ""
  echo "JEPA Stage 1 测试完成 → runs/quick_test_jepa_s1/"
  echo ""
fi

# ============================================================
# 阶段 5: JEPA Stage 2 (用 Stage 1 权重初始化 + CTC 训练)
# ============================================================
if [[ "${STAGE}" == "all" || "${STAGE}" == "jepa_s2" ]]; then
  echo "========================================"
  echo "阶段 5: JEPA Stage 2 (CTC + JEPA aux) 快速测试"
  echo "========================================"
  S1_CKPT="${WORK_DIR}/runs/quick_test_jepa_s1/best_stage1.pt"
  S1_ARG=()
  if [[ -f "${S1_CKPT}" ]]; then
    S1_ARG+=(--stage1-checkpoint "${S1_CKPT}")
    echo "使用 Stage 1 checkpoint: ${S1_CKPT}"
  else
    echo "未找到 Stage 1 checkpoint, 从头训练 JEPA Stage 2"
  fi
  python "${WORK_DIR}/train_wm_belief.py" \
    --metadata-dir "${META}" \
    --output-dir "${WORK_DIR}/runs/quick_test_jepa_s2" \
    --online-features \
    --hf-dataset-name openslr/librispeech_asr \
    --hf-dataset-config all \
    --hf-train-split validation.clean \
    --hf-val-split validation.clean \
    --feature-device cuda \
    --evidence-type logits \
    --hidden-dim 256 \
    --pooling-type mean \
    --upsample-factor 4 \
    --belief-type jepa \
    --jepa-encoder-layers 2 \
    --jepa-encoder-heads 4 \
    --jepa-encoder-ff-dim 512 \
    --jepa-predictor-layers 1 \
    --jepa-prior-layers 1 \
    --jepa-aux-weight 0.1 \
    --batch-size 4 \
    --eval-batch-size 4 \
    --epochs 5 \
    --lr 3e-4 \
    --warmup-steps 10 \
    --num-workers 0 \
    --eval-every-epochs 5 \
    --log-every-steps 2 \
    --max-train-examples 16 \
    --max-val-examples 8 \
    --devices 1 \
    --precision 32-true \
    "${S1_ARG[@]+"${S1_ARG[@]}"}"
  echo ""
  echo "JEPA Stage 2 测试完成 → runs/quick_test_jepa_s2/"
  echo ""
fi

echo "==========================="
echo "全部阶段完成!"
echo "==========================="