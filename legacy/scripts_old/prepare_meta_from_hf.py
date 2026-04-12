"""Generate phone_vocab.json + text_vocab.json from a HuggingFace speech dataset.

This avoids needing a local LibriSpeech directory — it downloads a small split
from HuggingFace, runs G2P on the transcripts, and writes the vocab files that
train_wm_belief.py needs for online-features mode.

Usage:
    conda activate phn
    python scripts/prepare_meta_from_hf.py \
        --hf-dataset-name openslr/librispeech_asr \
        --hf-dataset-config all \
        --hf-split validation.clean \
        --output-dir artifacts/metadata_librispeech \
        --max-examples 200
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wm_common import write_json, write_jsonl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-dataset-name", default="openslr/librispeech_asr")
    p.add_argument("--hf-dataset-config", default="all")
    p.add_argument("--hf-split", default="validation.clean")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-examples", type=int, default=500)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading HF dataset (streaming, text-only to avoid full download) …")
    from datasets import load_dataset

    ds_iter = load_dataset(
        args.hf_dataset_name,
        args.hf_dataset_config,
        split=args.hf_split,
        streaming=True,
    ).select_columns(["id", "text"])

    print("Running G2P …")
    from g2p_en import G2p

    g2p = G2p()
    phone_counter: Counter = Counter()
    char_set: set[str] = set()
    records = []

    for i, ex in enumerate(ds_iter):
        if i >= args.max_examples:
            break
        text = ex.get("text", "")
        seg_id = ex.get("id", str(i))
        text_chars = list(text)
        char_set.update(text_chars)

        phones_raw = g2p(text)
        phones = [p for p in phones_raw if p.strip() and not re.match(r"^[^\w]+$", p)]
        phone_counter.update(phones)

        records.append({
            "segment_id": seg_id,
            "text": text,
            "text_chars": text_chars,
            "canonical_phones": phones,
            "teacher_phones": [],
        })

    del ds_iter

    n = len(records)
    print(f"  Collected {n} examples from {args.hf_split}")

    sorted_phones = sorted(phone_counter.keys())
    phone_vocab = {"tokens": ["<pad>", "<blank>", "<unk>"] + sorted_phones}
    sorted_chars = sorted(char_set)
    text_vocab = {"tokens": ["<pad>", "<blank>", "<unk>"] + sorted_chars}

    write_json(out / "phone_vocab.json", phone_vocab)
    write_json(out / "text_vocab.json", text_vocab)
    write_jsonl(out / "validation.jsonl", records)
    write_jsonl(out / "train.jsonl", records)

    write_json(out / "meta.json", {
        "source": f"{args.hf_dataset_name}/{args.hf_split}",
        "n_examples": n,
        "phone_vocab_size": len(phone_vocab["tokens"]),
        "text_vocab_size": len(text_vocab["tokens"]),
    })
    print(f"  Phone vocab: {len(phone_vocab['tokens'])} tokens")
    print(f"  Text vocab: {len(text_vocab['tokens'])} tokens")
    print(f"Done → {out}")


if __name__ == "__main__":
    main()
    import os
    os._exit(0)
