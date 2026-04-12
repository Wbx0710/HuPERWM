"""Frozen HuPER greedy CTC decode, teacher→WM vocab mapping, and cache I/O.

HuPER outputs unstressed phone symbols (e.g. AH, AA) while wm_prepare_meta uses
stressed g2p_en tokens (AH0, AH1).  We map each HuPER phone to a WM vocab token by
exact match first, then {base}{0,1,2} stress fallbacks, else <unk>.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import torch

from huperjepa.data.vocab import UNK_TOKEN, Vocabulary, levenshtein_distance, load_jsonl


def config_id2label_to_int_map(id2label: Mapping) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for k, v in id2label.items():
        idx = int(k) if isinstance(k, str) else int(k)
        out[idx] = str(v).strip()
    return out


def _is_ctc_blank_symbol(sym: str) -> bool:
    return sym == "" or sym.isspace() or sym == "|"


def _blank_ids_from_id2label(int_labels: Mapping[int, str]) -> set[int]:
    return {i for i, lab in int_labels.items() if _is_ctc_blank_symbol(lab)}


def greedy_ctc_decode_huper(
    logits: torch.Tensor,
    id2label: Mapping,
) -> List[str]:
    """Greedy frame-wise argmax + CTC collapse for HuPER logits [T, C]."""
    int_labels = config_id2label_to_int_map(id2label)
    blank_ids = _blank_ids_from_id2label(int_labels)

    if logits.dim() != 2:
        raise ValueError(f"Expected logits [T, C], got {tuple(logits.shape)}")
    pred = logits.argmax(dim=-1).tolist()

    out: List[str] = []
    prev_nonblank: int | None = None
    for tid in pred:
        if tid in blank_ids:
            prev_nonblank = None
            continue
        if prev_nonblank is not None and tid == prev_nonblank:
            continue
        prev_nonblank = tid
        sym = int_labels.get(tid, "")
        if sym and not _is_ctc_blank_symbol(sym):
            out.append(sym)
    return out


def map_huper_phones_to_wm_vocab(phones: Sequence[str], vocab: Vocabulary) -> List[str]:
    mapped: List[str] = []
    for p in phones:
        if p in vocab.token_to_id:
            mapped.append(p)
            continue
        hit: str | None = None
        for stress in ("0", "1", "2"):
            cand = f"{p}{stress}"
            if cand in vocab.token_to_id:
                hit = cand
                break
        mapped.append(hit if hit is not None else UNK_TOKEN)
    return mapped


def logits_to_teacher_phones_for_vocab(
    logits: torch.Tensor,
    id2label: Mapping,
    phone_vocab: Vocabulary,
) -> List[str]:
    raw = greedy_ctc_decode_huper(logits, id2label)
    return map_huper_phones_to_wm_vocab(raw, phone_vocab)


def phone_to_base(s: str) -> str:
    """Strip ARPAbet stress digit for edit-distance comparisons."""
    if len(s) >= 2 and s[-1].isdigit():
        return s[:-1]
    return s


def base_phone_sequence(seq: Sequence[str]) -> List[str]:
    return [phone_to_base(p) for p in seq]


def load_teacher_phone_cache(paths: Iterable[str | Path]) -> Dict[str, List[str]]:
    """Merge JSONL teacher caches; each line: {\"segment_id\", \"teacher_phones\"}."""
    merged: Dict[str, List[str]] = {}
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"Teacher cache not found: {p}")
        for row in load_jsonl(p):
            sid = row["segment_id"]
            merged[sid] = list(row["teacher_phones"])
    return merged


def _percentile_sorted(xs_sorted: List[float], q: float) -> float:
    if not xs_sorted:
        return float("nan")
    i = min(len(xs_sorted) - 1, max(0, int(round(q * (len(xs_sorted) - 1)))))
    return float(xs_sorted[i])


def teacher_sanity_summary(
    rows: Sequence[Mapping],
) -> dict:
    """Aggregate sanity stats from pre-built row dicts (see wm_cache_teacher_phones)."""
    n = len(rows)
    nonempty = sum(1 for r in rows if r.get("n_teacher", 0) > 0)
    ratios = sorted(r["len_over_dur"] for r in rows if math.isfinite(r["len_over_dur"]))
    eds_sorted = sorted(int(r["edit_dist_base"]) for r in rows)
    all_toks = [p for r in rows for p in r["teacher_phones"]]
    n_unk_tok = sum(1 for p in all_toks if p == UNK_TOKEN)

    return {
        "n_utterances": n,
        "n_teacher_nonempty": int(nonempty),
        "teacher_nonempty_rate": float(nonempty / n) if n else 0.0,
        "teacher_token_unk_rate": float(n_unk_tok / len(all_toks)) if all_toks else 0.0,
        "len_over_duration_sec": {
            "mean": float(sum(ratios) / len(ratios)) if ratios else float("nan"),
            "p50": _percentile_sorted(ratios, 0.50),
            "p90": _percentile_sorted(ratios, 0.90),
            "min": float(ratios[0]) if ratios else float("nan"),
            "max": float(ratios[-1]) if ratios else float("nan"),
        },
        "edit_dist_base_teacher_vs_canonical": {
            "mean": float(sum(eds_sorted) / len(eds_sorted)) if eds_sorted else float("nan"),
            "p50": _percentile_sorted([float(x) for x in eds_sorted], 0.50),
            "p90": _percentile_sorted([float(x) for x in eds_sorted], 0.90),
            "min": int(eds_sorted[0]) if eds_sorted else 0,
            "max": int(eds_sorted[-1]) if eds_sorted else 0,
        },
    }


def build_cache_rows_for_split(
    *,
    features_split_dir: Path,
    segment_ids: Sequence[str],
    metadata_by_id: Mapping[str, Mapping],
    phone_vocab: Vocabulary,
    id2label: Mapping,
) -> List[dict]:
    rows: List[dict] = []
    for seg_id in segment_ids:
        pt_path = features_split_dir / f"{seg_id}.pt"
        if not pt_path.exists():
            continue
        try:
            feats = torch.load(pt_path, weights_only=False, map_location="cpu")
        except TypeError:
            feats = torch.load(pt_path, map_location="cpu")
        if "huper_logits" not in feats:
            raise KeyError(f"{pt_path} has no huper_logits")
        logits = feats["huper_logits"].float()
        teacher = logits_to_teacher_phones_for_vocab(logits, id2label, phone_vocab)

        meta = metadata_by_id.get(seg_id, {})
        duration = float(meta.get("duration_sec", 0.0) or 0.0)
        canonical = list(meta.get("canonical_phones", []))

        t_base = base_phone_sequence(teacher)
        c_base = base_phone_sequence(canonical)
        ed = levenshtein_distance(t_base, c_base)

        ratio = (len(teacher) / duration) if duration > 0 else float("nan")
        rows.append(
            {
                "segment_id": seg_id,
                "teacher_phones": teacher,
                "n_teacher": len(teacher),
                "duration_sec": duration,
                "len_over_dur": ratio,
                "edit_dist_base": ed,
            }
        )
    return rows


def write_teacher_cache_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {
                        "segment_id": r["segment_id"],
                        "teacher_phones": r["teacher_phones"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
