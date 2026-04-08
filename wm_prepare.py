"""Extract HuPER evidence features and Sylber syllable boundaries.

Runs frozen HuPER (WavLM-Large CTC) and Sylber on each audio example,
saving per-example .pt files with frame-level logits/hidden-states and
syllable boundary indices.  Both models output at ~50 Hz so frame indices
are directly aligned.

Supports two data sources:
  1. HuggingFace dataset (--dataset-path)   – GigaSpeech, etc.
  2. Raw LibriSpeech     (--librispeech-path) – standard speaker/chapter layout

Usage (LibriSpeech):
    conda activate phn
    python wm_prepare.py \
        --librispeech-path /data/chenxu/datasets/librispeech/LibriSpeech \
        --metadata-dir artifacts/metadata_librispeech \
        --output-dir artifacts/wm_features_librispeech

Usage (HuggingFace):
    conda activate phn
    python wm_prepare.py \
        --dataset-path /data/chenxu/gigaspeech/xs \
        --metadata-dir artifacts/metadata_xs \
        --output-dir artifacts/wm_features_xs
"""

from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2Processor, WavLMForCTC

from wm_common import (
    MIN_WAVLM_INPUT_SAMPLES,
    ensure_min_audio_length,
    load_jsonl,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract HuPER + Sylber features.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset-path", type=str, help="HuggingFace dataset on disk.")
    src.add_argument("--librispeech-path", type=str,
                     help="Root of raw LibriSpeech directory.")
    src.add_argument(
        "--hf-dataset-name", type=str,
        help="HuggingFace dataset name for online loading (e.g. openslr/librispeech_asr). "
             "Audio is decoded with soundfile to avoid torchcodec dependency.",
    )
    p.add_argument("--hf-dataset-config", type=str, default="all",
                   help="HF dataset config, used with --hf-dataset-name.")
    p.add_argument("--hf-train-split", type=str, default="train.clean.360",
                   help="HF split name that maps to the 'train' output directory.")
    p.add_argument("--hf-val-split", type=str, default="validation.clean",
                   help="HF split name that maps to the 'validation' output directory.")
    p.add_argument("--hf-test-split", type=str, default="test.clean",
                   help="HF split name that maps to the 'test' output directory.")
    p.add_argument("--metadata-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--huper-repo", type=str, default="huper29/huper_recognizer")
    p.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    p.add_argument(
        "--evidence-type",
        choices=["logits", "hidden", "both"],
        default="both",
        help="Which HuPER outputs to save.",
    )
    p.add_argument("--max-examples-per-split", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data source iterators
# ---------------------------------------------------------------------------


def _iter_huggingface(dataset_split) -> list[tuple[str, np.ndarray]]:
    """Yield (segment_id, audio_array) from a HuggingFace dataset split."""
    examples = []
    for idx in range(len(dataset_split)):
        ex = dataset_split[idx]
        seg_id = ex["segment_id"]
        audio = np.asarray(ex["audio"]["array"], dtype=np.float32)
        examples.append((seg_id, audio))
    return examples


def _iter_librispeech_split(
    metadata_records: list[dict],
) -> list[tuple[str, np.ndarray]]:
    """Yield (segment_id, audio_array) from metadata records with audio_path."""
    examples = []
    for rec in metadata_records:
        seg_id = rec["segment_id"]
        audio_path = rec["audio_path"]
        audio, sr = sf.read(audio_path, dtype="float32")
        if sr != 16000:
            import torchaudio
            audio_t = torch.from_numpy(audio).unsqueeze(0)
            audio_t = torchaudio.functional.resample(audio_t, sr, 16000)
            audio = audio_t.squeeze(0).numpy()
        examples.append((seg_id, audio))
    return examples


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_huper_features(
    model: WavLMForCTC,
    processor: Wav2Vec2Processor,
    audio: torch.Tensor,
    device: torch.device,
    evidence_type: str,
) -> dict:
    audio_np = audio.squeeze().numpy()
    inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    need_hidden = evidence_type in ("hidden", "both")
    outputs = model(**inputs, output_hidden_states=need_hidden)

    result = {"num_frames": outputs.logits.shape[1]}
    if evidence_type in ("logits", "both"):
        result["huper_logits"] = outputs.logits[0].cpu().half()
    if need_hidden:
        result["huper_hidden"] = outputs.hidden_states[-1][0].cpu().half()
    return result


@torch.no_grad()
def extract_sylber_boundaries(segmenter, audio: torch.Tensor) -> torch.Tensor:
    wav = audio.unsqueeze(0) if audio.dim() == 1 else audio
    std = wav.std()
    if std < 1e-6:
        std = torch.tensor(1.0)
    wav = (wav - wav.mean()) / std

    result = segmenter(wav=[wav], in_second=False)
    segments = result[0]["segments"]

    if len(segments) == 0:
        num_frames = max(1, audio.shape[-1] // 320)
        return torch.tensor([[0, num_frames]], dtype=torch.long)

    return torch.from_numpy(np.asarray(segments, dtype=np.int64))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)

    print("Loading HuPER model …")
    processor = Wav2Vec2Processor.from_pretrained(args.huper_repo)
    huper_model = WavLMForCTC.from_pretrained(args.huper_repo).to(device).eval()

    print("Loading Sylber segmenter …")
    from sylber import Segmenter
    segmenter = Segmenter(device=str(device))

    # --- Source-specific one-time setup ---
    hf_dataset = None
    g2p = None
    _hf_split_map: dict[str, str] = {}

    if args.dataset_path:
        from datasets import load_from_disk
        hf_dataset = load_from_disk(args.dataset_path)
    elif args.hf_dataset_name:
        from g2p_en import G2p
        g2p = G2p()
        _hf_split_map = {
            "train": args.hf_train_split,
            "validation": args.hf_val_split,
            "test": args.hf_test_split,
        }
        print("G2P initialized — canonical phones will be generated on-the-fly.")

    stats: dict = {}

    for split in args.splits:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        meta_path = Path(args.metadata_dir) / f"{split}.jsonl"
        metadata_map: dict = {}
        metadata_records: list[dict] = []
        if meta_path.exists():
            metadata_records = load_jsonl(meta_path)
            metadata_map = {r["segment_id"]: r for r in metadata_records}

        # --- Resolve data source for this split ---
        # hf_lazy_ds: used for the new --hf-dataset-name path (lazy, one sample at a time)
        # examples:   pre-loaded list for legacy paths
        hf_lazy_ds = None
        examples: list[tuple[str, np.ndarray]] = []

        if args.librispeech_path:
            if not metadata_records:
                print(f"  [{split}] No metadata JSONL found, skipping.")
                continue
            examples = _iter_librispeech_split(metadata_records)
        elif args.dataset_path:
            assert hf_dataset is not None
            if split not in hf_dataset:
                print(f"  [{split}] Not found in HF dataset, skipping.")
                continue
            examples = _iter_huggingface(hf_dataset[split])
        else:
            # --hf-dataset-name: lazy load, audio decoded with soundfile (no torchcodec)
            from datasets import load_dataset, Audio as HFAudio
            hf_split_name = _hf_split_map.get(split, split)
            print(f"  [{split}] Loading HF split '{hf_split_name}' …")
            hf_lazy_ds = load_dataset(
                args.hf_dataset_name, args.hf_dataset_config, split=hf_split_name
            )
            hf_lazy_ds = hf_lazy_ds.cast_column("audio", HFAudio(decode=False))

        limit = len(hf_lazy_ds) if hf_lazy_ds is not None else len(examples)
        if args.max_examples_per_split is not None:
            limit = min(limit, args.max_examples_per_split)

        segment_ids: list[str] = []

        for idx in range(limit):
            # --- Fetch audio and optional text for this example ---
            hf_text: str | None = None

            if hf_lazy_ds is not None:
                ex = hf_lazy_ds[idx]
                seg_id = ex.get("id", str(idx))
                hf_text = ex.get("text", "")
                audio_data = ex["audio"]
                raw_bytes = audio_data.get("bytes")
                raw_path = audio_data.get("path")
                if raw_bytes:
                    audio_np, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
                elif raw_path:
                    audio_np, sr = sf.read(raw_path, dtype="float32")
                else:
                    print(f"  [{split}] No audio for index {idx}, skipping.")
                    continue
                if audio_np.ndim > 1:
                    audio_np = audio_np[:, 0]
                if sr != 16000:
                    import torchaudio
                    audio_t = torch.from_numpy(audio_np).unsqueeze(0)
                    audio_t = torchaudio.functional.resample(audio_t, sr, 16000)
                    audio_np = audio_t.squeeze(0).numpy()
            else:
                seg_id, audio_np = examples[idx]

            audio = torch.from_numpy(audio_np).float()
            audio = ensure_min_audio_length(audio, MIN_WAVLM_INPUT_SAMPLES)

            huper = extract_huper_features(
                huper_model, processor, audio, device, args.evidence_type
            )
            bounds = extract_sylber_boundaries(segmenter, audio)

            num_frames = huper["num_frames"]
            bounds = bounds.clamp(min=0, max=num_frames)
            valid = bounds[:, 1] > bounds[:, 0]
            bounds = bounds[valid]
            if bounds.shape[0] == 0:
                bounds = torch.tensor([[0, num_frames]], dtype=torch.long)

            save_dict: dict = {
                "segment_id": seg_id,
                "num_frames": num_frames,
                "num_syllables": int(bounds.shape[0]),
                "sylber_boundaries": bounds,
            }
            if "huper_logits" in huper:
                save_dict["huper_logits"] = huper["huper_logits"]
            if "huper_hidden" in huper:
                save_dict["huper_hidden"] = huper["huper_hidden"]

            # --- Attach text and phone labels ---
            if hf_text is not None:
                # HF source: text comes directly from the dataset; phones from G2P
                save_dict["text"] = hf_text
                save_dict["text_chars"] = list(hf_text)
                if g2p is not None:
                    phones_raw = g2p(hf_text)
                    canonical = [
                        p for p in phones_raw
                        if p.strip() and not re.match(r"^[^\w]+$", p)
                    ]
                    save_dict["canonical_phones"] = canonical
                save_dict["teacher_phones"] = []
            elif seg_id in metadata_map:
                meta = metadata_map[seg_id]
                save_dict["text"] = meta["text"]
                save_dict["text_chars"] = meta.get("text_chars", list(meta.get("text", "")))
                save_dict["canonical_phones"] = meta.get("canonical_phones", [])
                save_dict["teacher_phones"] = meta.get("teacher_phones", [])
                if "syllable_count" in meta:
                    save_dict["syllable_count_g2p"] = meta["syllable_count"]

            torch.save(save_dict, split_dir / f"{seg_id}.pt")
            segment_ids.append(seg_id)

            if (idx + 1) % 200 == 0:
                print(f"  [{split}] {idx + 1}/{limit}")

        write_json(
            output_dir / f"{split}_manifest.json",
            {
                "segment_ids": segment_ids,
                "evidence_type": args.evidence_type,
                "num_examples": len(segment_ids),
            },
        )
        stats[split] = len(segment_ids)
        print(f"[{split}] {len(segment_ids)} examples → {split_dir}")

    src_path = args.dataset_path or args.librispeech_path or args.hf_dataset_name
    write_json(
        output_dir / "prepare_config.json",
        {
            "data_source": src_path,
            "metadata_dir": args.metadata_dir,
            "huper_repo": args.huper_repo,
            "evidence_type": args.evidence_type,
            "stats": stats,
        },
    )
    print(f"Done.  Output → {output_dir}")


if __name__ == "__main__":
    main()
