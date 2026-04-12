python wm_agent_data.py \
    --checkpoint /data/bixingwu/runs/jepa_stage2_v2/best.pt \
    --features-dir artifacts/wm_features_librispeech \
    --metadata-dir artifacts/metadata_librispeech \
    --output-dir /data/bixingwu/agent_data_v2 \
    --splits train validation \
    --batch-size 32 \
    --num-workers 4