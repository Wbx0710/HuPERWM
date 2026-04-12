"""Generate metadata (phone/text vocabs + per-split JSONL) from raw LibriSpeech.

Scans the standard LibriSpeech directory layout::

    LibriSpeech/
      dev-clean/
        <speaker>/<chapter>/<speaker>-<chapter>-<utt>.flac
        <speaker>/<chapter>/<speaker>-<chapter>.trans.txt
      test-clean/
      train-clean-100/
      ...

Produces an output metadata directory with:
  - phone_vocab.json
  - text_vocab.json
  - train.jsonl, validation.jsonl, test.jsonl

Usage:
    conda activate phn
    python wm_prepare_meta_librispeech.py \
        --librispeech-path /data/chenxu/datasets/librispeech/LibriSpeech \
        --output-dir artifacts/metadata_librispeech
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import soundfile as sf
from g2p_en import G2p

from wm_common import write_json, write_jsonl


LIBRISPEECH_SPLIT_MAP = {
    "train-clean-100": "train",
    "train-clean-360": "train",
    "train-other-500": "train",
    "dev-clean": "validation",
    "dev-other": "validation",
    "test-clean": "test",
    "test-other": "test",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare LibriSpeech metadata for WM.")
    p.add_argument("--librispeech-path", type=str, required=True,
                    help="Root of LibriSpeech (contains dev-clean/, test-clean/, etc.)")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--split-map", type=str, nargs="*", default=None,
                    help="Override split mapping, e.g. 'dev-clean=train test-clean=test'")
    p.add_argument("--use-dev-as-train", action="store_true",
                    help="When no train-* splits exist, use dev-clean as train "
                         "(with a portion held out for validation).")
    p.add_argument("--val-ratio", type=float, default=0.1,
                    help="Fraction of dev-clean to hold out for validation "
                         "when --use-dev-as-train is set.")
    return p.parse_args()


def scan_librispeech(root: Path) -> dict[str, list[dict]]:
    """Scan LibriSpeech directory and return {ls_split: [records]}."""
    splits: dict[str, list[dict]] = {}
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        records = _scan_split(sub)
        if records:
            splits[sub.name] = records
            print(f"  Scanned {sub.name}: {len(records)} utterances")
    return splits


def _scan_split(split_dir: Path) -> list[dict]:
    records: list[dict] = []
    for trans_file in sorted(split_dir.rglob("*.trans.txt")):
        chapter_dir = trans_file.parent
        lines = trans_file.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            parts = line.strip().split(maxsplit=1)
            if len(parts) < 2:
                continue
            seg_id, text = parts[0], parts[1]
            flac = chapter_dir / f"{seg_id}.flac"
            if not flac.exists():
                continue
            info = sf.info(str(flac))
            records.append({
                "segment_id": seg_id,
                "audio_path": str(flac),
                "text": text,
                "duration_sec": info.duration,
                "sample_rate": info.samplerate,
            })
    return records


def resolve_split_mapping(
    available: dict[str, list[dict]],
    override: list[str] | None,
    use_dev_as_train: bool,
    val_ratio: float,
) -> dict[str, list[dict]]:
    """Map LibriSpeech split names → {train, validation, test}."""
    if override:
        mapping = {}
        for item in override:
            src, dst = item.split("=")
            mapping[src.strip()] = dst.strip()
    else:
        mapping = dict(LIBRISPEECH_SPLIT_MAP)

    result: dict[str, list[dict]] = {}
    for ls_name, records in available.items():
        target = mapping.get(ls_name)
        if target is None:
            print(f"  Warning: skipping unknown split '{ls_name}'")
            continue
        result.setdefault(target, []).extend(records)

    if "train" not in result and use_dev_as_train and "validation" in result:
        all_val = result["validation"]
        n_val = max(1, int(len(all_val) * val_ratio))
        result["train"] = all_val[n_val:]
        result["validation"] = all_val[:n_val]
        print(f"  Split validation → train ({len(result['train'])}) + "
              f"validation ({len(result['validation'])})")

    return result


_STRESS_RE = re.compile(r"\d+$")


def run_g2p(records: list[dict], g2p: G2p) -> list[dict]:
    """Add canonical_phones, text_chars, syllable_count to each record."""
    for rec in records:
        text = rec["text"]
        rec["text_chars"] = list(text)

        phones_raw = g2p(text)
        phones = [p for p in phones_raw if p.strip() and not re.match(r"^[^\w]+$", p)]
        rec["canonical_phones"] = phones

        vowels = [p for p in phones if _STRESS_RE.search(p)]
        rec["syllable_count"] = max(1, len(vowels))

        rec["teacher_phones"] = []

    return records


def build_phone_vocab(all_records: list[dict]) -> dict:
    counter: Counter = Counter()
    for rec in all_records:
        counter.update(rec["canonical_phones"])
    sorted_phones = sorted(counter.keys())
    tokens = ["<pad>", "<blank>", "<unk>"] + sorted_phones
    return {"tokens": tokens}


def build_text_vocab(all_records: list[dict]) -> dict:
    chars: set[str] = set()
    for rec in all_records:
        chars.update(rec["text_chars"])
    sorted_chars = sorted(chars)
    tokens = ["<pad>", "<blank>", "<unk>"] + sorted_chars
    return {"tokens": tokens}


def main() -> None:
    args = parse_args()
    root = Path(args.librispeech_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Scanning LibriSpeech …")
    raw_splits = scan_librispeech(root)
    if not raw_splits:
        print("ERROR: no LibriSpeech splits found.", file=sys.stderr)
        sys.exit(1)

    splits = resolve_split_mapping(
        raw_splits, args.split_map, args.use_dev_as_train, args.val_ratio
    )
    print(f"Resolved splits: { {k: len(v) for k, v in splits.items()} }")

    print("Running G2P …")
    g2p = G2p()
    all_records: list[dict] = []
    for name, records in splits.items():
        run_g2p(records, g2p)
        all_records.extend(records)
        print(f"  {name}: {len(records)} records processed")

    print("Building vocabularies …")
    phone_vocab = build_phone_vocab(all_records)
    text_vocab = build_text_vocab(all_records)
    write_json(output_dir / "phone_vocab.json", phone_vocab)
    write_json(output_dir / "text_vocab.json", text_vocab)
    print(f"  Phone vocab: {len(phone_vocab['tokens'])} tokens")
    print(f"  Text vocab:  {len(text_vocab['tokens'])} tokens")

    for name, records in splits.items():
        jsonl_records = []
        for rec in records:
            jsonl_records.append({
                "segment_id": rec["segment_id"],
                "audio_path": rec["audio_path"],
                "text": rec["text"],
                "text_chars": rec["text_chars"],
                "canonical_phones": rec["canonical_phones"],
                "teacher_phones": rec["teacher_phones"],
                "syllable_count": rec["syllable_count"],
                "duration_sec": rec["duration_sec"],
            })
        write_jsonl(output_dir / f"{name}.jsonl", jsonl_records)
        print(f"  Wrote {name}.jsonl ({len(jsonl_records)} records)")

    write_json(output_dir / "meta.json", {
        "librispeech_path": args.librispeech_path,
        "splits": {k: len(v) for k, v in splits.items()},
        "phone_vocab_size": len(phone_vocab["tokens"]),
        "text_vocab_size": len(text_vocab["tokens"]),
    })
    print(f"Done → {output_dir}")


if __name__ == "__main__":
    main()
