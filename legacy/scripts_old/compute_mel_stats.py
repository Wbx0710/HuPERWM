"""Compute per-bin mel mean and std from pre-extracted feature files.

Saves a .pt file with keys ``mean`` and ``std``, each of shape ``(mel_dim,)``.
These statistics are loaded by ``MelNormalizer`` during TTS training.

Usage:
    conda activate phn
    python scripts/compute_mel_stats.py \
        --features-dir /data/bixingwu/huperworldmodel/artifacts/wm_features_librispeech \
        --split train \
        --output mel_stats.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Compute mel normalization statistics.")
    p.add_argument("--features-dir", required=True)
    p.add_argument("--split", default="train", help="Which split to compute stats from.")
    p.add_argument(
        "--output",
        default=None,
        help="Output .pt path.  Defaults to <features-dir>/mel_stats.pt.",
    )
    p.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit examples for faster computation.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    features_dir = Path(args.features_dir)
    split_dir = features_dir / args.split

    manifest_path = features_dir / f"{args.split}_manifest.json"
    if not manifest_path.exists():
        print(f"No manifest at {manifest_path}")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    segment_ids = manifest["segment_ids"]
    if args.max_examples:
        segment_ids = segment_ids[: args.max_examples]

    count = 0
    mean = None
    m2 = None

    for i, seg_id in enumerate(segment_ids):
        pt_path = split_dir / f"{seg_id}.pt"
        if not pt_path.exists():
            continue
        feats = torch.load(pt_path, map_location="cpu", weights_only=False)
        if "mel_target" not in feats:
            continue

        mel = feats["mel_target"].float()  # (T, mel_dim)
        T = mel.shape[0]

        if mean is None:
            mel_dim = mel.shape[1]
            mean = torch.zeros(mel_dim, dtype=torch.float64)
            m2 = torch.zeros(mel_dim, dtype=torch.float64)

        for t in range(T):
            count += 1
            frame = mel[t].double()
            delta = frame - mean
            mean += delta / count
            delta2 = frame - mean
            m2 += delta * delta2

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(segment_ids)} processed ({count} frames)")

    if count < 2:
        print("Not enough mel frames to compute statistics.")
        sys.exit(1)

    std = (m2 / count).sqrt().float()
    mean = mean.float()

    out_path = args.output or str(features_dir / "mel_stats.pt")
    torch.save({"mean": mean, "std": std, "count": count}, out_path)
    print(f"Saved mel stats to {out_path}")
    print(f"  frames: {count}")
    print(f"  mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  std  range: [{std.min():.4f}, {std.max():.4f}]")


if __name__ == "__main__":
    main()
