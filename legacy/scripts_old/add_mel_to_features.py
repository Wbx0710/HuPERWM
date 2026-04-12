"""Add mel spectrograms to existing pre-extracted .pt feature files.

This avoids re-extracting HuPER + Sylber features (which takes hours).
Only the lightweight mel extraction (pure STFT, no GPU needed) is performed.

Usage:
    conda activate phn
    python scripts/add_mel_to_features.py \
        --features-dir /data/bixingwu/huperworldmodel/artifacts/wm_features_librispeech \
        --hf-dataset-name openslr/librispeech_asr \
        --hf-dataset-config all \
        --splits train validation
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

from wm_tts import MelExtractor


def parse_args():
    p = argparse.ArgumentParser(description="Add mel spectrograms to .pt features.")
    p.add_argument("--features-dir", required=True)
    p.add_argument("--hf-dataset-name", default="openslr/librispeech_asr")
    p.add_argument("--hf-dataset-config", default="all")
    p.add_argument("--splits", nargs="+", default=["train", "validation"])
    p.add_argument("--hf-train-split", default="train.clean.360")
    p.add_argument("--hf-val-split", default="validation.clean")
    return p.parse_args()


SPLIT_MAP = {}


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

    split_map = {
        "train": args.hf_train_split,
        "validation": args.hf_val_split,
    }

    mel_extractor = MelExtractor()

    from datasets import Audio, load_dataset

    for split in args.splits:
        hf_split = split_map.get(split, split)
        manifest_path = features_dir / f"{split}_manifest.json"
        if not manifest_path.exists():
            print(f"[{split}] No manifest found, skipping.")
            continue

        manifest = json.loads(manifest_path.read_text())
        segment_ids = set(manifest["segment_ids"])
        split_dir = features_dir / split

        need_update = [
            s for s in manifest["segment_ids"]
            if "mel_target" not in torch.load(split_dir / f"{s}.pt", map_location="cpu", weights_only=False)
        ]
        if not need_update:
            print(f"[{split}] All files already have mel_target, skipping.")
            continue
        need_set = set(need_update)
        print(f"[{split}] {len(need_update)}/{len(segment_ids)} files need mel_target.")

        print(f"[{split}] Loading HF split '{hf_split}' ...")
        ds = load_dataset(args.hf_dataset_name, args.hf_dataset_config, split=hf_split)
        ds = ds.cast_column("audio", Audio(decode=False))

        if "id" in ds.column_names:
            all_ids = ds["id"]
            id_to_idx = {sid: idx for idx, sid in enumerate(all_ids) if sid in need_set}
        else:
            id_to_idx = {str(idx): idx for idx in range(len(ds)) if str(idx) in need_set}

        print(f"[{split}] Matched {len(id_to_idx)}/{len(segment_ids)} segment IDs.")

        done = 0
        skipped = 0
        for seg_id in manifest["segment_ids"]:
            pt_path = split_dir / f"{seg_id}.pt"
            if seg_id not in id_to_idx:
                skipped += 1
                continue

            audio_np = decode_audio(ds[id_to_idx[seg_id]])
            if audio_np is None:
                skipped += 1
                continue

            audio_tensor = torch.from_numpy(audio_np).float()
            with torch.no_grad():
                mel = mel_extractor(audio_tensor)  # (1, T_mel, 80)

            feats = torch.load(pt_path, weights_only=False)
            feats["mel_target"] = mel.squeeze(0).half()
            feats["mel_length"] = mel.shape[1]
            torch.save(feats, pt_path)

            done += 1
            if done % 500 == 0:
                print(f"  [{split}] {done}/{len(segment_ids)} done")

        print(f"[{split}] Done: {done} updated, {skipped} skipped.")


if __name__ == "__main__":
    main()
