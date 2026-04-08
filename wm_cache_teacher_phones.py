"""Build cached pseudo phone teachers from frozen HuPER (greedy CTC).

Mode A — from ``wm_prepare.py`` outputs (``huper_logits`` in each ``.pt``)::

    python wm_cache_teacher_phones.py \\
        --features-dir artifacts/wm_features_librispeech \\
        --metadata-dir artifacts/metadata_librispeech \\
        --phone-vocab artifacts/metadata_librispeech/phone_vocab.json \\
        --splits train validation \\
        --output-dir artifacts/teacher_cache_librispeech

Mode B — batched HuggingFace LibriSpeech forward (default batch size 10).
Use ``--hf-num-procs 8 --hf-device-ids 0,1,2,3,4,5,6,7`` for 8 GPUs::

    python wm_cache_teacher_phones.py \\
        --phone-vocab artifacts/metadata_librispeech/phone_vocab.json \\
        --hf-dataset-name openslr/librispeech_asr \\
        --hf-dataset-config all \\
        --hf-split train.clean.360 \\
        --output-dir artifacts/teacher_cache_hf \\
        --hf-num-procs 8 \\
        --hf-batch-size 10 \\
        --hf-device-ids 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from g2p_en import G2p
from transformers import AutoConfig, Wav2Vec2Processor, WavLMForCTC

from wm_common import (
    MIN_WAVLM_INPUT_SAMPLES,
    Vocabulary,
    ensure_min_audio_length,
    write_json,
    load_jsonl,
    read_json,
    levenshtein_distance,
)
from wm_teacher import (
    base_phone_sequence,
    build_cache_rows_for_split,
    logits_to_teacher_phones_for_vocab,
    teacher_sanity_summary,
    write_teacher_cache_jsonl,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache HuPER greedy teacher phones.")
    p.add_argument("--phone-vocab", type=str, required=True)
    p.add_argument("--features-dir", type=str, default=None)
    p.add_argument(
        "--metadata-dir",
        type=str,
        default=None,
        help="Required with --features-dir (for duration, canonical, segment order).",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        help="With --features-dir: split names for subdirs and {split}.jsonl.",
    )
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--huper-repo", type=str, default="huper29/huper_recognizer")
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument(
        "--hf-dataset-name",
        type=str,
        default=None,
        help="If omitted but --hf-split is set (and not using --features-dir), "
        "defaults to openslr/librispeech_asr (same as train_wm_belief.py).",
    )
    p.add_argument("--hf-dataset-config", type=str, default="all")
    p.add_argument("--hf-split", type=str, default=None)
    p.add_argument("--hf-device", type=str, default="cuda")
    p.add_argument(
        "--hf-num-procs",
        type=int,
        default=1,
        help="Number of parallel processes (GPUs). Each loads a strided shard of the split.",
    )
    p.add_argument(
        "--hf-batch-size",
        type=int,
        default=10,
        help="HuPER forward batch size per GPU (default 10 ≈ 10× former batch-1 throughput).",
    )
    p.add_argument(
        "--hf-device-ids",
        type=str,
        default=None,
        help="Comma-separated CUDA device ids, one per proc (default: 0,1,…,num_procs-1). "
        "Ignored when num_procs=1; then --hf-device is used (e.g. cuda:1).",
    )
    return p.parse_args()


def _hf_cuda_device_index(hf_device: str) -> int:
    if hf_device == "cuda" or not hf_device.startswith("cuda:"):
        return 0
    return int(hf_device.split(":", 1)[1])


def _parse_hf_device_ids(arg_s: str | None, num_procs: int) -> list[int]:
    if arg_s is None:
        return list(range(num_procs))
    parts = [int(x.strip()) for x in arg_s.split(",") if x.strip()]
    if len(parts) != num_procs:
        raise SystemExit(
            f"--hf-device-ids must list {num_procs} integers, got {len(parts)} ({parts})."
        )
    return parts


def _backbone_for_lengths(model: torch.nn.Module):
    return getattr(model, "wavlm", None) or getattr(model, "wav2vec2", None)


def _ctc_frame_lengths(model: torch.nn.Module, attention_mask: torch.Tensor) -> torch.Tensor:
    bb = _backbone_for_lengths(model)
    if bb is None:
        raise RuntimeError("Cannot infer CTC lengths: model has no wavlm/wav2vec2.")
    ilen = attention_mask.sum(dim=-1)
    return bb._get_feat_extract_output_lengths(ilen)


def _attention_mask_from_lengths(
    waveforms: list[np.ndarray], input_values: torch.Tensor
) -> torch.Tensor:
    """Some processor builds omit ``attention_mask``; infer 1/0 pad mask from sample lengths."""
    dev = input_values.device
    b, t = input_values.shape
    lens = torch.tensor([len(w) for w in waveforms], device=dev, dtype=torch.long)
    idx = torch.arange(t, device=dev).unsqueeze(0).expand(b, -1)
    return (idx < lens.unsqueeze(1)).long()


def _prepare_waveform_mono_16k(audio_np: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    if sr != 16000:
        import torchaudio

        audio_t = torch.from_numpy(audio_np).unsqueeze(0)
        audio_t = torchaudio.functional.resample(audio_t, sr, 16000)
        audio_np = audio_t.squeeze(0).numpy()
    dur = float(len(audio_np) / 16000.0)
    audio_t = torch.from_numpy(audio_np).float()
    audio_t = ensure_min_audio_length(audio_t, MIN_WAVLM_INPUT_SAMPLES)
    return audio_t.numpy(), dur


@torch.no_grad()
def _hf_shard_worker(rank: int, world_size: int, device_id: int, config: dict) -> None:
    """Run one shard (strided global indices) with batched HuPER; write partial JSONL."""
    tag = config["tag"]
    split_name = config["hf_split"]
    out_dir = Path(config["out_dir"])
    use_cuda = config["use_cuda"] and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")

    phone_vocab = Vocabulary.from_file(Path(config["phone_vocab_path"]))
    id2label = config["id2label"]
    batch_size = max(1, int(config["hf_batch_size"]))

    processor = Wav2Vec2Processor.from_pretrained(config["huper_repo"])
    model = WavLMForCTC.from_pretrained(config["huper_repo"]).to(device).eval()
    g2p = G2p()

    ds = load_dataset(
        config["hf_dataset_name"],
        config["hf_dataset_config"],
        split=split_name,
    )
    n_total = len(ds) if config["max_examples"] is None else min(
        len(ds), int(config["max_examples"])
    )
    global_indices = list(range(rank, n_total, world_size))
    n_local = len(global_indices)

    rows: list[dict] = []
    b = 0
    while b < n_local:
        chunk_globals = global_indices[b : b + batch_size]
        waveforms: list[np.ndarray] = []
        metas: list[tuple] = []
        for g_idx in chunk_globals:
            ex = ds[g_idx]
            seg_id = ex.get("id", ex.get("utt_id", str(g_idx)))
            text = ex.get("text", "")
            audio_np = np.asarray(ex["audio"]["array"], dtype=np.float32)
            sr = int(ex["audio"]["sampling_rate"])
            wav_np, dur = _prepare_waveform_mono_16k(audio_np, sr)
            waveforms.append(wav_np)
            metas.append((g_idx, seg_id, text, dur))

        inputs = processor(
            waveforms,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_values = inputs["input_values"]
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = _attention_mask_from_lengths(waveforms, input_values)
        out = model(**inputs)
        logits_batched = out.logits.float()
        attn = inputs["attention_mask"]
        lengths = _ctc_frame_lengths(model, attn)

        for j in range(logits_batched.shape[0]):
            tlen = int(lengths[j].item())
            tlen = min(tlen, logits_batched.shape[1])
            seq = logits_batched[j, :tlen].cpu()
            teacher = logits_to_teacher_phones_for_vocab(seq, id2label, phone_vocab)
            g_idx, seg_id, text, dur = metas[j]
            phones_raw = g2p(text)
            canonical = [
                p for p in phones_raw if p.strip() and not re.match(r"^[^\w]+$", p)
            ]
            ed = levenshtein_distance(
                base_phone_sequence(teacher), base_phone_sequence(canonical)
            )
            ratio = (len(teacher) / dur) if dur > 0 else float("nan")
            rows.append(
                {
                    "_orig_idx": g_idx,
                    "segment_id": seg_id,
                    "teacher_phones": teacher,
                    "n_teacher": len(teacher),
                    "duration_sec": dur,
                    "len_over_dur": ratio,
                    "edit_dist_base": ed,
                }
            )

        prev_b = b
        b += len(chunk_globals)
        if rank == 0 and (b // 1000) > (prev_b // 1000) and b > 0:
            print(
                f"  [{split_name}] rank0 shard progress {b}/{n_local} (examples)",
                flush=True,
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / f"{tag}_shard{rank}_partial.jsonl"
    with open(partial_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        f"[{split_name}] rank {rank}: wrote {len(rows)} rows → {partial_path}",
        flush=True,
    )


def _merge_hf_partials(out_dir: Path, tag: str, world_size: int) -> list[dict]:
    merged: list[dict] = []
    for rank in range(world_size):
        p = out_dir / f"{tag}_shard{rank}_partial.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"Missing shard output: {p}")
        merged.extend(load_jsonl(p))
    merged.sort(key=lambda r: int(r["_orig_idx"]))
    for r in merged:
        del r["_orig_idx"]
    return merged


def _finalize_hf_cache(
    rows: list[dict],
    out_dir: Path,
    tag: str,
    split_name: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"{tag}.jsonl"
    write_teacher_cache_jsonl(cache_path, rows)
    summary = teacher_sanity_summary(rows)
    write_json(out_dir / f"sanity_{tag}.json", summary)
    print(f"[HF {split_name}] merged {len(rows)} rows → {cache_path}")
    print(
        f"  nonempty_rate={summary['teacher_nonempty_rate']:.4f}  "
        f"phones/sec mean={summary['len_over_duration_sec']['mean']:.2f}  "
        f"edit_base mean={summary['edit_dist_base_teacher_vs_canonical']['mean']:.2f}  "
        f"tok_unk_rate={summary['teacher_token_unk_rate']:.4f}"
    )


def _build_hf_worker_config(args, id2label: dict, out_dir: Path, tag: str) -> dict:
    return {
        "tag": tag,
        "out_dir": str(out_dir),
        "phone_vocab_path": str(Path(args.phone_vocab).resolve()),
        "id2label": id2label,
        "huper_repo": args.huper_repo,
        "hf_dataset_name": args.hf_dataset_name,
        "hf_dataset_config": args.hf_dataset_config,
        "hf_split": args.hf_split,
        "max_examples": args.max_examples,
        "hf_batch_size": args.hf_batch_size,
        "use_cuda": str(args.hf_device).lower() != "cpu",
    }


def _cache_from_hf_split(args, id2label: dict, out_dir: Path) -> None:
    tag = args.hf_split.replace(".", "_")
    config = _build_hf_worker_config(args, id2label, out_dir, tag)

    if args.hf_num_procs < 1:
        raise SystemExit("--hf-num-procs must be >= 1")

    if args.hf_num_procs == 1:
        dev_id = _hf_cuda_device_index(args.hf_device) if config["use_cuda"] else 0
        _hf_shard_worker(0, 1, dev_id, config)
        partial = out_dir / f"{tag}_shard0_partial.jsonl"
        rows = load_jsonl(partial)
        for r in rows:
            del r["_orig_idx"]
        partial.unlink(missing_ok=True)
        _finalize_hf_cache(rows, out_dir, tag, args.hf_split)
        return

    if not torch.cuda.is_available():
        raise SystemExit("Multi-GPU mode requires CUDA; use --hf-num-procs 1 for CPU.")

    device_ids = _parse_hf_device_ids(args.hf_device_ids, args.hf_num_procs)
    if args.hf_num_procs > torch.cuda.device_count():
        raise SystemExit(
            f"--hf-num-procs ({args.hf_num_procs}) > visible CUDA devices "
            f"({torch.cuda.device_count()})."
        )

    ctx = mp.get_context("spawn")
    procs: list[mp.Process] = []
    for rank in range(args.hf_num_procs):
        p = ctx.Process(
            target=_hf_shard_worker,
            args=(rank, args.hf_num_procs, device_ids[rank], config),
        )
        p.start()
        procs.append(p)
    exit_codes = []
    for p in procs:
        p.join()
        exit_codes.append(p.exitcode)
    if any(c != 0 for c in exit_codes):
        raise SystemExit(f"HF worker(s) failed, exit codes={exit_codes}")

    rows = _merge_hf_partials(out_dir, tag, args.hf_num_procs)
    for rank in range(args.hf_num_procs):
        (out_dir / f"{tag}_shard{rank}_partial.jsonl").unlink(missing_ok=True)
    _finalize_hf_cache(rows, out_dir, tag, args.hf_split)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    phone_vocab = Vocabulary.from_file(Path(args.phone_vocab))

    cfg = AutoConfig.from_pretrained(args.huper_repo)
    id2label = cfg.id2label
    if not id2label:
        raise ValueError(f"No id2label in config for {args.huper_repo}")

    use_pt = args.features_dir is not None
    if (
        not use_pt
        and args.hf_split is not None
        and args.hf_dataset_name is None
    ):
        args.hf_dataset_name = "openslr/librispeech_asr"

    use_hf = args.hf_dataset_name is not None and args.hf_split is not None

    if use_hf and use_pt:
        raise SystemExit("Use either --features-dir or --hf-dataset-name/--hf-split, not both.")
    if use_hf:
        _cache_from_hf_split(args, dict(id2label), out_dir)
        print(f"Done. Output dir → {out_dir}")
        return

    if not use_pt:
        raise SystemExit(
            "Provide --features-dir (+ --metadata-dir), or --hf-split "
            "(LibriSpeech HF defaults: openslr/librispeech_asr + config all), "
            "or explicitly --hf-dataset-name with --hf-split."
        )
    if not args.metadata_dir:
        raise SystemExit("--metadata-dir is required with --features-dir.")

    feat_root = Path(args.features_dir)
    meta_dir = Path(args.metadata_dir)

    for split in args.splits:
        split_feat = feat_root / split
        if not split_feat.is_dir():
            print(f"[skip] No feature directory: {split_feat}")
            continue

        meta_path = meta_dir / f"{split}.jsonl"
        if not meta_path.exists():
            print(f"[skip] No metadata {meta_path}")
            continue

        manifest_path = feat_root / f"{split}_manifest.json"
        if not manifest_path.exists():
            print(f"[skip] No manifest {manifest_path}")
            continue

        manifest = read_json(manifest_path)
        segment_ids: list[str] = list(manifest["segment_ids"])
        if args.max_examples is not None:
            segment_ids = segment_ids[: args.max_examples]

        metadata_records = load_jsonl(meta_path)
        metadata_by_id = {r["segment_id"]: r for r in metadata_records}

        rows = build_cache_rows_for_split(
            features_split_dir=split_feat,
            segment_ids=segment_ids,
            metadata_by_id=metadata_by_id,
            phone_vocab=phone_vocab,
            id2label=id2label,
        )

        cache_path = out_dir / f"{split}.jsonl"
        write_teacher_cache_jsonl(cache_path, rows)
        summary = teacher_sanity_summary(rows)
        write_json(out_dir / f"sanity_{split}.json", summary)

        print(f"[{split}] wrote {len(rows)} rows → {cache_path}")
        print(
            f"  nonempty_rate={summary['teacher_nonempty_rate']:.4f}  "
            f"phones/sec mean={summary['len_over_duration_sec']['mean']:.2f}  "
            f"edit_base mean={summary['edit_dist_base_teacher_vs_canonical']['mean']:.2f}  "
            f"tok_unk_rate={summary['teacher_token_unk_rate']:.4f}"
        )

    print(f"Done. Output dir → {out_dir}")


if __name__ == "__main__":
    main()
