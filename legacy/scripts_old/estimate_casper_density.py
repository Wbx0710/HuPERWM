"""Estimate oracle_density and syllable statistics for a HuggingFace speech dataset.

Designed primarily for evaluating CASPER (CASPER-SSSD/CASPER) as an alternative to
LibriSpeech, whose high oracle_density (~72.8%) causes the SchedulerAgent to degenerate
into an "always-emit" policy.

What this script does
---------------------
1. Streams N audio examples from the specified HuggingFace dataset.
2. Runs Sylber to get syllable boundaries (K slots per utterance).
3. Counts words from the transcript and estimates oracle_emit positions using
   the same proportional heuristic as ``label_word_boundaries`` (Tier-3 fallback).
   (Full forced-alignment is skipped here for speed; this gives a good estimate.)
4. Reports:
   - oracle_density = oracle_emit.sum() / K   (averaged over utterances)
   - word/slot ratio = num_words / num_slots
   - syllable count distribution
   - Comparison table at oracle_min_gap = 2, 3, 4

Usage
-----
    conda activate phn
    python scripts/estimate_casper_density.py \\
        --hf-dataset CASPER-SSSD/CASPER \\
        --hf-split train \\
        --max-examples 500 \\
        --oracle-min-gap 2 3 4 \\
        --device cuda

    # Compare with LibriSpeech:
    python scripts/estimate_casper_density.py \\
        --hf-dataset openslr/librispeech_asr \\
        --hf-config all \\
        --hf-split validation.clean \\
        --max-examples 500 \\
        --dataset-label LibriSpeech-val-clean

Requirements
------------
    pip install datasets soundfile torchaudio
    # Sylber: follow project README (sylber package must be importable)
"""

from __future__ import annotations

import argparse
import io
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wm_common import MIN_WAVLM_INPUT_SAMPLES, ensure_min_audio_length


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate oracle_density / syllable stats for a HF speech dataset."
    )
    p.add_argument(
        "--hf-dataset", default="CASPER-SSSD/CASPER",
        help="HuggingFace dataset name  (default: CASPER-SSSD/CASPER)",
    )
    p.add_argument(
        "--hf-config", default=None,
        help="Dataset config / subset name (optional, e.g. 'all' for LibriSpeech)",
    )
    p.add_argument(
        "--hf-split", default="train",
        help="Dataset split to use (default: train)",
    )
    p.add_argument(
        "--max-examples", type=int, default=500,
        help="Max utterances to process (default: 500)",
    )
    p.add_argument(
        "--oracle-min-gap", type=int, nargs="+", default=[2, 3, 4],
        help="oracle_min_gap values to evaluate (default: 2 3 4)",
    )
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for Sylber (default: cuda if available)",
    )
    p.add_argument(
        "--dataset-label", default=None,
        help="Human-readable dataset label for output (default: derived from --hf-dataset)",
    )
    p.add_argument(
        "--audio-field", default=None,
        help="Name of the audio column in the dataset "
             "(auto-detected if not specified: tries 'audio', 'speech')",
    )
    p.add_argument(
        "--text-field", default=None,
        help="Name of the text/transcript column "
             "(auto-detected: tries 'text', 'sentence', 'transcription', 'normalized_text')",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Syllable boundary extraction (same as wm_online_data.py)
# ---------------------------------------------------------------------------

def _extract_sylber_boundaries(segmenter, audio: torch.Tensor) -> torch.Tensor:
    wav = audio.unsqueeze(0) if audio.dim() == 1 else audio
    std = wav.std()
    if std < 1e-6:
        std = torch.tensor(1.0, device=wav.device)
    wav = (wav - wav.mean()) / std
    result = segmenter(wav=[wav], in_second=False)
    segments = result[0]["segments"]
    if len(segments) == 0:
        num_frames = max(1, audio.shape[-1] // 320)
        return torch.tensor([[0, num_frames]], dtype=torch.long)
    return torch.from_numpy(np.asarray(segments, dtype=np.int64))


# ---------------------------------------------------------------------------
# Oracle density estimation (Tier-3 proportional heuristic — fast)
# ---------------------------------------------------------------------------

def _proportional_oracle_emit(num_slots: int, num_words: int, min_gap: int) -> float:
    """Estimate oracle_density via proportional interpolation (Tier-3 heuristic).

    Places oracle_emit markers at uniformly spaced slot indices, one per word,
    with min_gap enforced.  Returns the fraction of slots that are emit positions.

    This matches the fallback logic in ``label_word_boundaries`` (wm_agent_data.py).
    """
    if num_slots == 0 or num_words == 0:
        return 0.0

    emit_slots: list[int] = []
    step = num_slots / num_words
    last = -min_gap  # allow first emit at index 0 if min_gap=0

    for w in range(num_words):
        raw = step * (w + 1) - 1  # last slot of the w-th word (0-indexed)
        candidate = int(round(raw))
        candidate = max(candidate, last + min_gap)
        candidate = min(candidate, num_slots - 1)
        emit_slots.append(candidate)
        last = candidate

    # De-duplicate (in case clamping pushed multiple to the same slot).
    emit_slots = sorted(set(emit_slots))
    return len(emit_slots) / num_slots


# ---------------------------------------------------------------------------
# Audio loading helpers
# ---------------------------------------------------------------------------

def _load_audio_from_example(ex: dict, audio_field: str) -> tuple[np.ndarray, int]:
    """Return (audio_np float32, sample_rate) from a HF dataset example."""
    import soundfile as sf

    audio_entry = ex[audio_field]
    if isinstance(audio_entry, dict):
        # Decoded entry: {"array": np.ndarray, "sampling_rate": int}
        if "array" in audio_entry:
            return audio_entry["array"].astype(np.float32), audio_entry["sampling_rate"]
        # Encoded (bytes) entry
        audio_bytes = audio_entry.get("bytes")
        audio_path  = audio_entry.get("path")
        if audio_bytes is not None:
            return sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio_path is not None:
            return sf.read(audio_path, dtype="float32")
        raise ValueError(f"Cannot decode audio from entry: {list(audio_entry.keys())}")
    if isinstance(audio_entry, np.ndarray):
        # Raw numpy array — assume 16 kHz
        return audio_entry.astype(np.float32), 16000
    raise ValueError(f"Unexpected audio field type: {type(audio_entry)}")


def _detect_field(ex: dict, candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in ex:
            return c
    available = list(ex.keys())
    raise KeyError(
        f"Could not auto-detect {label} field. "
        f"Available columns: {available}. "
        f"Set --{'audio' if label == 'audio' else 'text'}-field explicitly."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    dataset_label = args.dataset_label or args.hf_dataset
    device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"  Oracle Density Estimator")
    print(f"  Dataset : {dataset_label}")
    print(f"  Split   : {args.hf_split}")
    print(f"  Samples : {args.max_examples}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Load dataset (streaming to avoid downloading entire corpus)
    # ------------------------------------------------------------------
    print("Loading HuggingFace dataset (streaming) …")
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Missing: pip install datasets")

    load_kwargs: dict = dict(split=args.hf_split, streaming=True)
    if args.hf_config:
        load_kwargs["name"] = args.hf_config
    # Respect HF_TOKEN env var so gated datasets (e.g. CASPER) work without
    # an explicit huggingface-cli login.
    import os as _os
    hf_token = _os.environ.get("HF_TOKEN")
    if hf_token:
        load_kwargs["token"] = hf_token

    try:
        ds_stream = load_dataset(args.hf_dataset, **load_kwargs)
    except Exception as e:
        sys.exit(f"Failed to load dataset '{args.hf_dataset}': {e}")

    # Disable HF automatic audio decoding (avoids torchcodec dependency).
    # Our _load_audio_from_example already handles raw {"bytes", "path"} entries
    # via soundfile, so we just need datasets to skip its own decode step.
    try:
        from datasets import Audio as _HfAudio
        for _field in ["audio", "speech"]:
            try:
                ds_stream = ds_stream.cast_column(_field, _HfAudio(decode=False))
            except Exception:
                pass
    except ImportError:
        pass

    # ------------------------------------------------------------------
    # 2. Load Sylber
    # ------------------------------------------------------------------
    print("Loading Sylber segmenter …")
    try:
        from sylber import Segmenter
        segmenter = Segmenter(device=str(device))
    except ImportError:
        sys.exit("Missing: sylber package. Follow the project README to install it.")

    # ------------------------------------------------------------------
    # 3. Process utterances
    # ------------------------------------------------------------------
    audio_field: str | None = args.audio_field
    text_field: str | None  = args.text_field

    stats: list[dict] = []   # one dict per utterance
    errors = 0

    print(f"Processing up to {args.max_examples} utterances …\n")
    for i, ex in enumerate(ds_stream):
        if i >= args.max_examples:
            break

        # Auto-detect fields on first example.
        if audio_field is None:
            audio_field = _detect_field(ex, ["audio", "speech"], "audio")
        if text_field is None:
            text_field = _detect_field(
                ex,
                ["text", "sentence", "transcription", "normalized_text", "norm_text"],
                "text",
            )

        try:
            audio_np, sr = _load_audio_from_example(ex, audio_field)
        except Exception as e:
            warnings.warn(f"[{i}] Audio load error: {e}")
            errors += 1
            continue

        # Convert to mono float32 @ 16 kHz.
        if audio_np.ndim > 1:
            audio_np = audio_np[:, 0]
        if sr != 16000:
            try:
                import torchaudio
                t = torch.from_numpy(audio_np).unsqueeze(0)
                t = torchaudio.functional.resample(t, sr, 16000)
                audio_np = t.squeeze(0).numpy()
            except ImportError:
                # Fallback: skip resampling warning but continue.
                warnings.warn(f"[{i}] sr={sr}≠16000 and torchaudio unavailable; skipping resample.")

        audio = torch.from_numpy(audio_np).float()
        audio = ensure_min_audio_length(audio, MIN_WAVLM_INPUT_SAMPLES)

        text = ex.get(text_field, "") or ""
        num_words = len(text.strip().split()) if text.strip() else 0

        try:
            with torch.no_grad():
                bounds = _extract_sylber_boundaries(segmenter, audio.to(device)).cpu()
        except Exception as e:
            warnings.warn(f"[{i}] Sylber error: {e}")
            errors += 1
            continue

        num_slots = int(bounds.shape[0])

        # Duration in seconds.
        duration_s = audio_np.shape[0] / 16000.0

        stats.append({
            "num_slots": num_slots,
            "num_words": num_words,
            "duration_s": duration_s,
        })

        if (i + 1) % 50 == 0:
            n = len(stats)
            avg_density = np.mean([
                _proportional_oracle_emit(s["num_slots"], s["num_words"], min_gap=2)
                for s in stats if s["num_slots"] > 0
            ])
            print(f"  [{i+1:>4}/{args.max_examples}]  "
                  f"processed={n}  errors={errors}  "
                  f"avg_oracle_density(gap=2)={avg_density:.3f}")

    # ------------------------------------------------------------------
    # 4. Compute and report statistics
    # ------------------------------------------------------------------
    n = len(stats)
    if n == 0:
        print("\nNo valid examples processed. Check dataset/field names.")
        return

    num_slots_all  = np.array([s["num_slots"]  for s in stats], dtype=float)
    num_words_all  = np.array([s["num_words"]  for s in stats], dtype=float)
    duration_all   = np.array([s["duration_s"] for s in stats], dtype=float)

    word_slot_ratio = num_words_all / num_slots_all.clip(1)

    print(f"\n{'='*60}")
    print(f"  Results for: {dataset_label}  (N={n}, errors={errors})")
    print(f"{'='*60}\n")

    print(f"  Duration (s)")
    print(f"    mean   : {duration_all.mean():.2f}")
    print(f"    median : {np.median(duration_all):.2f}")
    print(f"    p5/p95 : {np.percentile(duration_all, 5):.2f} / {np.percentile(duration_all, 95):.2f}")

    print(f"\n  Syllable slots (K per utterance)")
    print(f"    mean   : {num_slots_all.mean():.1f}")
    print(f"    median : {np.median(num_slots_all):.1f}")
    print(f"    p5/p95 : {np.percentile(num_slots_all, 5):.1f} / {np.percentile(num_slots_all, 95):.1f}")

    print(f"\n  Words per utterance")
    print(f"    mean   : {num_words_all.mean():.1f}")
    print(f"    median : {np.median(num_words_all):.1f}")

    print(f"\n  Word/slot ratio  (lower → sparser → Agent harder to game)")
    print(f"    mean   : {word_slot_ratio.mean():.3f}")
    print(f"    median : {np.median(word_slot_ratio):.3f}")
    print(f"    p5/p95 : {np.percentile(word_slot_ratio, 5):.3f} / {np.percentile(word_slot_ratio, 95):.3f}")

    # oracle_density at various min_gap values.
    print(f"\n  Oracle Density  (oracle_emit.sum() / K, averaged over utterances)")
    print(f"  {'min_gap':>8}  {'oracle_density':>15}  {'note':}")
    for gap in sorted(args.oracle_min_gap):
        densities = np.array([
            _proportional_oracle_emit(s["num_slots"], s["num_words"], min_gap=gap)
            for s in stats if s["num_slots"] > 0
        ])
        note = ""
        if densities.mean() > 0.65:
            note = "⚠  too high — Agent may always-emit"
        elif densities.mean() < 0.35:
            note = "✓  good — sufficient negative examples"
        else:
            note = "✓  acceptable"
        print(f"  {gap:>8}       {densities.mean():>8.3f}        {note}")

    # LibriSpeech reference for comparison.
    print(f"\n  ── LibriSpeech reference (gap=2): ~0.728 (too high)")

    print(f"\n{'='*60}")
    print(f"  Recommendation:")
    density_gap2 = np.mean([
        _proportional_oracle_emit(s["num_slots"], s["num_words"], min_gap=2)
        for s in stats if s["num_slots"] > 0
    ])
    if density_gap2 < 0.55:
        print(f"  oracle_density={density_gap2:.3f} @ gap=2 → {dataset_label} is suitable.")
        print(f"  Consider switching training data to: {args.hf_dataset}")
    else:
        best_gap = None
        for gap in sorted(args.oracle_min_gap):
            d = np.mean([
                _proportional_oracle_emit(s["num_slots"], s["num_words"], min_gap=gap)
                for s in stats if s["num_slots"] > 0
            ])
            if d < 0.55 and (best_gap is None or gap < best_gap):
                best_gap = gap
        if best_gap is not None:
            print(f"  oracle_density @ gap=2 is {density_gap2:.3f} (high).")
            print(f"  But gap={best_gap} brings it below 0.55 — increase oracle_min_gap.")
            print(f"  In scripts/train_agent.sh: --oracle-min-gap {best_gap}")
        else:
            print(f"  oracle_density remains high even at large gaps.")
            print(f"  Consider using a dataset with more natural conversational speech.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
