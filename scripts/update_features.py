"""Update pre-extracted features: add hidden states + re-extract mel.

This script updates existing .pt feature files with:
1. ``huper_hidden`` — WavLM last-layer hidden states (1024-dim), required for
   TTS with rich acoustic-phonetic conditioning.
2. ``mel_target`` — re-extracted log-mel using magnitude spectrum (not power),
   matching the HiFi-GAN convention for vocoder compatibility.

Supports resuming partial runs: each file is checked individually.

After running this script, also run ``compute_mel_stats.py`` to produce
the normalisation statistics needed for TTS training.

Usage:
    conda activate phn
    python scripts/update_features.py \
        --features-dir /data/bixingwu/huperworldmodel/artifacts/wm_features_librispeech \
        --hf-dataset-name openslr/librispeech_asr \
        --hf-dataset-config all \
        --splits train validation \
        --device cuda

    # Then compute mel stats:
    python scripts/compute_mel_stats.py \
        --features-dir /data/bixingwu/huperworldmodel/artifacts/wm_features_librispeech
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2Processor, WavLMForCTC

from wm_common import MIN_WAVLM_INPUT_SAMPLES, ensure_min_audio_length
from wm_tts import MelExtractor


def parse_args():
    p = argparse.ArgumentParser(description="Add hidden states + re-extract mel.")
    p.add_argument("--features-dir", required=True)
    p.add_argument("--hf-dataset-name", default="openslr/librispeech_asr")
    p.add_argument("--hf-dataset-config", default="all")
    p.add_argument("--splits", nargs="+", default=["train", "validation"])
    p.add_argument("--hf-train-split", default="train.clean.360")
    p.add_argument("--hf-val-split", default="validation.clean")
    p.add_argument("--huper-repo", default="huper29/huper_recognizer")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--force-mel",
        action="store_true",
        help="Re-extract mel even if mel_target already exists.",
    )
    p.add_argument(
        "--force-hidden",
        action="store_true",
        help="Re-extract hidden states even if huper_hidden already exists.",
    )
    return p.parse_args()


def decode_audio(example) -> np.ndarray | None:
    audio_data = example["audio"]
    raw_bytes = audio_data.get("bytes")
    raw_path = audio_data.get("path")
    if raw_bytes:
        audio_np, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    elif raw_path:
        audio_np, sr = sf.read(raw_path, dtype="float32")
    else:
        return None
    if audio_np.ndim > 1:
        audio_np = audio_np[:, 0]
    if sr != 16000:
        import torchaudio

        audio_t = torch.from_numpy(audio_np).unsqueeze(0)
        audio_t = torchaudio.functional.resample(audio_t, sr, 16000)
        audio_np = audio_t.squeeze(0).numpy()
    return audio_np


def main():
    args = parse_args()
    features_dir = Path(args.features_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    split_map = {
        "train": args.hf_train_split,
        "validation": args.hf_val_split,
    }

    print("Loading HuPER model …")
    processor = Wav2Vec2Processor.from_pretrained(args.huper_repo)
    huper_model = WavLMForCTC.from_pretrained(args.huper_repo).to(device).eval()

    mel_extractor = MelExtractor().to(device)

    from datasets import Audio, load_dataset

    for split in args.splits:
        hf_split = split_map.get(split, split)
        manifest_path = features_dir / f"{split}_manifest.json"
        if not manifest_path.exists():
            print(f"[{split}] No manifest, skipping.")
            continue

        manifest = json.loads(manifest_path.read_text())
        segment_ids = manifest["segment_ids"]
        split_dir = features_dir / split

        # --- Scan ALL files to find which ones need updating ---
        need_hidden_ids: list[str] = []
        need_mel_ids: list[str] = []
        for seg_id in segment_ids:
            pt_path = split_dir / f"{seg_id}.pt"
            if not pt_path.exists():
                continue
            feats = torch.load(pt_path, map_location="cpu", weights_only=False)
            if args.force_hidden or "huper_hidden" not in feats:
                need_hidden_ids.append(seg_id)
            if args.force_mel or "mel_target" not in feats:
                need_mel_ids.append(seg_id)

        need_update_ids = set(need_hidden_ids) | set(need_mel_ids)
        if not need_update_ids:
            print(f"[{split}] All {len(segment_ids)} files already up-to-date, skipping.")
            continue

        print(f"[{split}] {len(segment_ids)} total files:")
        print(f"  Need huper_hidden: {len(need_hidden_ids)}")
        print(f"  Need mel_target:   {len(need_mel_ids)}")
        print(f"  Files to process:  {len(need_update_ids)}")

        # --- Build ID→index map using fast column access ---
        print(f"[{split}] Loading HF split '{hf_split}' …")
        ds = load_dataset(args.hf_dataset_name, args.hf_dataset_config, split=hf_split)
        ds = ds.cast_column("audio", Audio(decode=False))

        update_set = need_update_ids
        if "id" in ds.column_names:
            all_ids = ds["id"]
            id_to_idx = {sid: idx for idx, sid in enumerate(all_ids) if sid in update_set}
        else:
            id_to_idx = {str(idx): idx for idx in range(len(ds)) if str(idx) in update_set}
        print(f"[{split}] Matched {len(id_to_idx)}/{len(update_set)} IDs in HF dataset.")

        need_hidden_set = set(need_hidden_ids)
        need_mel_set = set(need_mel_ids)

        done = 0
        skipped = 0
        for seg_id in segment_ids:
            if seg_id not in need_update_ids:
                continue
            if seg_id not in id_to_idx:
                skipped += 1
                continue

            pt_path = split_dir / f"{seg_id}.pt"
            audio_np = decode_audio(ds[id_to_idx[seg_id]])
            if audio_np is None:
                skipped += 1
                continue

            feats = torch.load(pt_path, weights_only=False)
            changed = False

            if seg_id in need_hidden_set:
                inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = huper_model(**inputs, output_hidden_states=True)
                feats["huper_hidden"] = out.hidden_states[-1][0].cpu().half()
                changed = True

            if seg_id in need_mel_set:
                audio_tensor = torch.from_numpy(audio_np).float()
                audio_tensor = ensure_min_audio_length(audio_tensor, MIN_WAVLM_INPUT_SAMPLES)
                with torch.no_grad():
                    mel = mel_extractor(audio_tensor.unsqueeze(0).to(device))
                feats["mel_target"] = mel.squeeze(0).cpu().half()
                feats["mel_length"] = mel.shape[1]
                changed = True

            if changed:
                torch.save(feats, pt_path)
                done += 1

            if done % 500 == 0 and done > 0:
                print(f"  [{split}] {done}/{len(need_update_ids)} done, {skipped} skipped")

        print(f"[{split}] Done: {done} updated, {skipped} skipped (no audio match).\n")

    print("Feature update complete.")
    print(
        "Next step: run compute_mel_stats.py to compute normalisation statistics:\n"
        f"  python scripts/compute_mel_stats.py --features-dir {args.features_dir}"
    )


if __name__ == "__main__":
    main()
