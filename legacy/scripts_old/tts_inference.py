"""Generate speech from a trained BWM + TTS decoder checkpoint.

Loads validation examples, runs the BWM forward pass to extract belief / prior /
identity representations, generates mel spectrograms via flow-matching, and
converts to waveforms with either HiFi-GAN or Griffin-Lim.

Usage (HiFi-GAN):
    conda activate phn
    python scripts/tts_inference.py \
        --checkpoint runs/wm_tts_librispeech/best.pt \
        --features-dir /data/bixingwu/huperworldmodel/artifacts/wm_features_librispeech \
        --metadata-dir artifacts/metadata_librispeech \
        --output-dir runs/wm_tts_librispeech/samples \
        --vocoder-path pretrained/hifigan/g_02500000 \
        --num-samples 8

Usage (Griffin-Lim fallback — no extra download needed):
    python scripts/tts_inference.py \
        --checkpoint runs/wm_tts_librispeech/best.pt \
        --features-dir /data/bixingwu/huperworldmodel/artifacts/wm_features_librispeech \
        --metadata-dir artifacts/metadata_librispeech \
        --output-dir runs/wm_tts_librispeech/samples \
        --num-samples 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from wm_common import Vocabulary
from wm_core import BeliefWMCollator, BeliefWMConfig, BeliefWMDataset, BeliefWorldModel
from wm_tts import FlowMatchingTTSDecoder, TTSConfig
from wm_vocoder import load_vocoder


def parse_args():
    p = argparse.ArgumentParser(description="TTS inference from BWM checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--features-dir", required=True)
    p.add_argument("--metadata-dir", required=True)
    p.add_argument("--output-dir", default="runs/wm_tts_librispeech/samples")
    p.add_argument("--split", default="validation")
    p.add_argument("--evidence-type", default="hidden")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=32, help="ODE integration steps")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--duration-scale", type=float, default=1.0,
                    help="Multiply predicted durations by this factor (>1 = longer).")
    p.add_argument("--vocoder-path", type=str, default=None,
                    help="Path to HiFi-GAN generator checkpoint.  "
                         "Falls back to Griffin-Lim if not provided.")
    p.add_argument("--mel-stats-path", type=str, default=None,
                    help="mel_stats.pt for normalisation. Auto-detected from features-dir.")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def plot_mel_comparison(gt_mel, gen_mel, seg_id, text, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), dpi=150)

    axes[0].imshow(gt_mel.T, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title(f"Ground Truth — {seg_id}", fontsize=10)
    axes[0].set_ylabel("Mel bin")

    axes[1].imshow(gen_mel.T, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title(f"Generated (flow-matching) — {seg_id}", fontsize=10)
    axes[1].set_ylabel("Mel bin")
    axes[1].set_xlabel("Frame")

    fig.suptitle(text[:80] + ("…" if len(text) > 80 else ""), fontsize=9, y=0.02)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Auto-detect mel stats ---
    mel_stats_path = args.mel_stats_path
    if mel_stats_path is None:
        candidate = Path(args.features_dir) / "mel_stats.pt"
        if candidate.exists():
            mel_stats_path = str(candidate)

    # --- Load checkpoint ---
    print("Loading checkpoint …")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config: BeliefWMConfig = ckpt["config"]

    model = BeliefWorldModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    tts_cfg: TTSConfig = ckpt["tts_config"]
    tts_decoder = FlowMatchingTTSDecoder(tts_cfg)
    tts_decoder.load_state_dict(ckpt["tts_decoder_state_dict"])
    tts_decoder.to(device).eval()

    if mel_stats_path:
        tts_decoder.mel_normalizer.load_stats(mel_stats_path)
        print(f"  Mel stats loaded from {mel_stats_path}")
    else:
        print("  Warning: no mel_stats.pt found — mel denormalisation disabled.")

    print(f"  BWM config: hidden_dim={config.hidden_dim}, evidence_dim={config.evidence_dim}, "
          f"identity={config.use_identity}")
    print(f"  TTS config: decoder_dim={tts_cfg.decoder_dim}, layers={tts_cfg.decoder_layers}")

    # --- Load vocoder ---
    vocoder = load_vocoder(
        vocoder_path=args.vocoder_path,
        device=str(device),
        n_fft=tts_cfg.mel_sr // 16,
        hop_length=tts_cfg.mel_hop,
        n_mels=tts_cfg.mel_dim,
        sr=tts_cfg.mel_sr,
    )

    # --- Dataset ---
    phone_vocab = Vocabulary.from_file(Path(args.metadata_dir) / "phone_vocab.json")
    text_vocab = Vocabulary.from_file(Path(args.metadata_dir) / "text_vocab.json")

    ds = BeliefWMDataset(
        args.features_dir, args.split, args.metadata_dir,
        phone_vocab, text_vocab, evidence_type=args.evidence_type,
        max_examples=args.num_samples,
    )
    loader = DataLoader(
        ds, batch_size=1, shuffle=False, collate_fn=BeliefWMCollator(),
    )

    print(f"\nGenerating {args.num_samples} samples: "
          f"{args.n_steps} ODE steps, temp={args.temperature}, "
          f"dur_scale={args.duration_scale}\n")

    results = []
    for i, batch in enumerate(loader):
        seg_id = batch["segment_ids"][0]
        text = batch["texts"][0]

        ev = batch["evidence"].to(device)
        bd = batch["boundaries"].to(device)
        sm = batch["slot_mask"].to(device)
        nf = batch["num_frames"].to(device)
        fm = batch.get("frame_mask")
        if fm is not None:
            fm = fm.to(device)

        with torch.no_grad():
            outputs = model(ev, bd, sm, nf, frame_mask=fm)

            beliefs = outputs["beliefs"]
            priors = outputs["priors"]
            identity = outputs.get("identity")
            if identity is None:
                identity = torch.zeros(1, config.identity_dim, device=device)

            gen_mel, gen_lengths = tts_decoder.generate(
                beliefs=beliefs, priors=priors, identity=identity,
                slot_mask=sm, n_steps=args.n_steps, temperature=args.temperature,
                duration_scale=args.duration_scale,
            )

        gen_mel_np = gen_mel[0, :gen_lengths[0].item()].cpu()
        gen_wav = vocoder(gen_mel_np)

        sf.write(str(output_dir / f"{seg_id}_gen.wav"), gen_wav.numpy(), tts_cfg.mel_sr)

        gt_mel = None
        if "mel_target" in batch:
            gt_mel_full = batch["mel_target"][0]
            ml = batch["mel_lengths"][0].item()
            gt_mel = gt_mel_full[:ml]
            gt_wav = vocoder(gt_mel)
            sf.write(str(output_dir / f"{seg_id}_gt_resynth.wav"), gt_wav.numpy(), tts_cfg.mel_sr)

        plot_path = output_dir / f"{seg_id}_mel.png"
        plot_gt = gt_mel.numpy() if gt_mel is not None else gen_mel_np.numpy()
        plot_mel_comparison(plot_gt, gen_mel_np.numpy(), seg_id, text, plot_path)

        dur_sec_gen = gen_lengths[0].item() * tts_cfg.mel_hop / tts_cfg.mel_sr
        print(f"  [{i+1}/{args.num_samples}] {seg_id}")
        print(f"    text: {text[:70]}{'…' if len(text) > 70 else ''}")
        print(f"    generated: {gen_lengths[0].item()} mel frames ({dur_sec_gen:.2f}s)")
        if gt_mel is not None:
            dur_sec_gt = gt_mel.shape[0] * tts_cfg.mel_hop / tts_cfg.mel_sr
            print(f"    ground truth: {gt_mel.shape[0]} mel frames ({dur_sec_gt:.2f}s)")
        print()

        results.append({
            "segment_id": seg_id,
            "text": text,
            "gen_mel_frames": gen_lengths[0].item(),
            "gt_mel_frames": gt_mel.shape[0] if gt_mel is not None else None,
        })

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Done! {len(results)} samples saved to {output_dir}")
    print(f"  *_gen.wav         — generated speech")
    print(f"  *_gt_resynth.wav  — ground truth (vocoder re-synthesis for fair comparison)")
    print(f"  *_mel.png         — mel spectrogram comparison")


if __name__ == "__main__":
    main()
