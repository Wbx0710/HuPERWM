"""Agent data: oracle labeling, AgentDataset, AgentCollator, and extraction pipeline.

Oracle labeling assigns word-boundary emit labels to syllable slots using a
three-tier priority:
  1. CTC forced alignment (torchaudio + CMU dict) — most accurate.
  2. CTC blank-gap heuristic — no dict needed, still acoustic.
  3. Proportional interpolation — last resort, no acoustic grounding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from huperjepa.data.vocab import Vocabulary

_CMU_DICT: Optional[dict] = None


def _get_cmu_dict() -> dict:
    global _CMU_DICT
    if _CMU_DICT is None:
        from nltk.corpus import cmudict
        _CMU_DICT = cmudict.dict()
    return _CMU_DICT


def _normalize_phone(ph: str) -> str:
    return ph.rstrip("012")


def word_to_phone_ids(
    word: str,
    phone_vocab: Vocabulary,
    cmu_dict: Optional[dict] = None,
) -> List[int]:
    """Map a GT word → list of phone vocab token IDs using CMU Pronouncing Dict."""
    if cmu_dict is None:
        cmu_dict = _get_cmu_dict()
    variants = cmu_dict.get(word.lower())
    if not variants:
        return []
    phones = variants[0]
    ids: List[int] = []
    for ph in phones:
        if ph in phone_vocab.token_to_id:
            ids.append(phone_vocab.token_to_id[ph])
        else:
            base = _normalize_phone(ph)
            for stress in ("1", "0", "2", ""):
                cand = base + stress if stress else base
                if cand in phone_vocab.token_to_id:
                    ids.append(phone_vocab.token_to_id[cand])
                    break
    return ids


def build_word_phone_ids(
    words: List[str],
    phone_vocab: Vocabulary,
    cmu_dict: Optional[dict] = None,
) -> List[List[int]]:
    if cmu_dict is None:
        cmu_dict = _get_cmu_dict()
    return [word_to_phone_ids(w, phone_vocab, cmu_dict) for w in words]


# ---------------------------------------------------------------------------
# Oracle labeling
# ---------------------------------------------------------------------------


def label_word_boundaries(
    canonical_logits: torch.Tensor,
    up_slot_mask: torch.Tensor,
    slot_mask: torch.Tensor,
    phone_vocab: Vocabulary,
    upsample_factor: int,
    text: str,
    min_gap: int = 2,
) -> tuple[torch.Tensor, list[str], list[list[int]]]:
    """Assign oracle emit labels to syllable slots (three-tier priority).

    Returns:
        oracle_emit:    (K,) float — 1.0 at word-end slots.
        words:          GT word list.
        word_phone_ids: Per-word phone ID lists from CMU dict.
    """
    words = text.strip().split()
    num_words = len(words)
    num_slots = int(slot_mask.sum().item())
    K = slot_mask.shape[0]
    empty = torch.zeros(K, dtype=torch.float32)

    if not words or num_slots == 0:
        return empty, words, [[] for _ in words]

    cmu_dict = _get_cmu_dict()
    word_phone_ids = build_word_phone_ids(words, phone_vocab, cmu_dict)

    oracle_fa = _label_forced_align(
        canonical_logits, up_slot_mask, num_slots, K,
        phone_vocab, upsample_factor, word_phone_ids, min_gap,
    )
    if oracle_fa is not None:
        return oracle_fa, words, word_phone_ids

    valid_len = int(up_slot_mask.sum().item())
    oracle_heuristic = _label_blank_gap(
        canonical_logits, valid_len, num_slots, K,
        phone_vocab.blank_id, upsample_factor, num_words, min_gap,
    )
    if oracle_heuristic is not None:
        return oracle_heuristic, words, word_phone_ids

    return _label_proportional(num_slots, K, num_words, min_gap), words, word_phone_ids


def _alignment_to_positions(token_seq: List[int], blank_id: int) -> List[int]:
    positions: List[int] = []
    pos = -1
    prev_non_blank = blank_id
    for tok in token_seq:
        if tok == blank_id:
            positions.append(pos)
            prev_non_blank = blank_id
        elif tok != prev_non_blank:
            pos += 1
            positions.append(pos)
            prev_non_blank = tok
        else:
            positions.append(pos)
    return positions


def _label_forced_align(
    canonical_logits: torch.Tensor,
    up_slot_mask: torch.Tensor,
    num_slots: int,
    K: int,
    phone_vocab: Vocabulary,
    upsample_factor: int,
    word_phone_ids: List[List[int]],
    min_gap: int,
) -> Optional[torch.Tensor]:
    from torchaudio.functional import forced_align as ta_forced_align

    all_phones: List[int] = []
    word_last_ph_idx: List[int] = []
    for ph_ids in word_phone_ids:
        if ph_ids:
            all_phones.extend(ph_ids)
            word_last_ph_idx.append(len(all_phones) - 1)
        else:
            word_last_ph_idx.append(-1)

    known = sum(1 for i in word_last_ph_idx if i >= 0)
    if known < max(1, len(word_phone_ids) // 2):
        return None

    valid_len = int(up_slot_mask.sum().item())
    if valid_len < max(2 * len(all_phones) - 1, 1) or valid_len == 0:
        return None

    try:
        log_probs = F.log_softmax(canonical_logits[:valid_len].float(), dim=-1)
        targets = torch.tensor([all_phones], dtype=torch.int32)
        in_lens = torch.tensor([valid_len], dtype=torch.int32)
        tgt_lens = torch.tensor([len(all_phones)], dtype=torch.int32)

        alignments, _ = ta_forced_align(
            log_probs.unsqueeze(0), targets, in_lens, tgt_lens,
            blank=phone_vocab.blank_id,
        )
        token_seq = alignments[0].tolist()
        frame_positions = _alignment_to_positions(token_seq, phone_vocab.blank_id)

        oracle = torch.zeros(K, dtype=torch.float32)
        effective_gap = min(min_gap, max(1, num_slots // max(len(word_phone_ids), 1)))
        prev_emit = -effective_gap

        for w_idx, last_ph_idx in enumerate(word_last_ph_idx):
            if last_ph_idx < 0:
                frac = (w_idx + 1) / len(word_phone_ids)
                slot = min(int(round(frac * num_slots)) - 1, num_slots - 1)
            else:
                last_frame = max(
                    (f for f, p in enumerate(frame_positions) if p == last_ph_idx), default=-1
                )
                if last_frame < 0:
                    frac = (w_idx + 1) / len(word_phone_ids)
                    slot = min(int(round(frac * num_slots)) - 1, num_slots - 1)
                else:
                    slot = min(last_frame // upsample_factor, num_slots - 1)

            slot = max(max(slot, 0), prev_emit + effective_gap)
            if slot < num_slots:
                oracle[slot] = 1.0
                prev_emit = slot

        return oracle
    except Exception:
        return None


def _label_blank_gap(
    canonical_logits: torch.Tensor,
    valid_len: int,
    num_slots: int,
    K: int,
    blank_id: int,
    upsample_factor: int,
    num_words: int,
    min_gap: int,
) -> Optional[torch.Tensor]:
    if valid_len == 0:
        return None

    frame_ids = canonical_logits[:valid_len].argmax(dim=-1).tolist()
    gap_threshold = max(2, upsample_factor // 2)
    word_end_frames: List[int] = []
    in_content = False
    last_content_frame = -1
    blank_run = 0

    for f_idx, fid in enumerate(frame_ids):
        if fid == blank_id:
            blank_run += 1
            if blank_run >= gap_threshold and in_content and last_content_frame >= 0:
                word_end_frames.append(last_content_frame)
                in_content = False
                last_content_frame = -1
        else:
            blank_run = 0
            in_content = True
            last_content_frame = f_idx
    if in_content and last_content_frame >= 0:
        word_end_frames.append(last_content_frame)

    if len(word_end_frames) < max(1, num_words // 2):
        return None

    if len(word_end_frames) != num_words:
        n = len(word_end_frames)
        idxs = [int(round(i * (n - 1) / max(num_words - 1, 1))) for i in range(num_words)]
        word_end_frames = [word_end_frames[min(i, n - 1)] for i in idxs]

    effective_gap = min(min_gap, max(1, num_slots // max(num_words, 1)))
    oracle = torch.zeros(K, dtype=torch.float32)
    prev_emit = -effective_gap
    for frame_idx in word_end_frames:
        slot = min(frame_idx // upsample_factor, num_slots - 1)
        slot = max(slot, prev_emit + effective_gap)
        if slot < num_slots:
            oracle[slot] = 1.0
            prev_emit = slot
    return oracle


def _label_proportional(num_slots: int, K: int, num_words: int, min_gap: int) -> torch.Tensor:
    oracle = torch.zeros(K, dtype=torch.float32)
    effective_gap = min(min_gap, max(1, num_slots // max(num_words, 1)))
    prev_emit = -effective_gap
    for w_idx in range(num_words):
        frac = (w_idx + 1) / num_words
        slot = min(int(round(frac * num_slots)) - 1, num_slots - 1)
        slot = max(max(slot, 0), prev_emit + effective_gap)
        if slot < num_slots:
            oracle[slot] = 1.0
            prev_emit = slot
    return oracle


# ---------------------------------------------------------------------------
# AgentDataset and AgentCollator
# ---------------------------------------------------------------------------

SHARD_SIZE = 1000


class AgentDataset(torch.utils.data.Dataset):
    """Loads shard-based agent feature files produced by ``extract_agent_data.py``.

    Two modes:
    - preload=True (default): all shards loaded into RAM at init time (~16GB for
      full train split). Eliminates I/O during training; recommended when RAM >= 64GB.
    - preload=False: lazy per-shard caching. Use when RAM is scarce.

    Pass ``recompute_oracle=True`` + ``phone_vocab`` to refresh oracle labels
    using the current labeling logic without re-running extraction.
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        max_examples: int | None = None,
        preload: bool = True,
        rank: int = 0,
        recompute_oracle: bool = False,
        phone_vocab: Vocabulary | None = None,
        upsample_factor: int = 4,
        oracle_min_gap: int = 2,
    ) -> None:
        self._recompute_oracle = recompute_oracle
        self._phone_vocab = phone_vocab
        self._upsample_factor = upsample_factor
        self._oracle_min_gap = oracle_min_gap
        data_path = Path(data_dir)
        with open(data_path / f"{split}_manifest.json") as f:
            manifest = json.load(f)
        records: list[dict] = manifest["records"]
        if max_examples:
            records = records[:max_examples]
        self.records = records
        self.feat_dir = data_path / split
        self._preload = preload
        self._flat: list[dict] | None = None

        if preload:
            num_shards = manifest.get("num_shards", 0)
            if rank == 0:
                print(f"[AgentDataset] Preloading {num_shards} shards into RAM...", flush=True)
            all_records: list[dict] = []
            for si in range(num_shards):
                shard_path = self.feat_dir / f"shard_{si:05d}.pt"
                if shard_path.exists():
                    all_records.extend(torch.load(shard_path, weights_only=False))
            seg_to_flat = {rec["segment_id"]: rec for rec in all_records}
            self._flat = [seg_to_flat.get(r["segment_id"]) for r in records]
            if rank == 0:
                loaded = sum(1 for x in self._flat if x is not None)
                print(f"[AgentDataset] Preloaded {loaded}/{len(records)} records.", flush=True)
        else:
            self._cache: dict[int, list] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _maybe_recompute_oracle(self, item: dict) -> dict:
        if not self._recompute_oracle or self._phone_vocab is None:
            return item
        text = item.get("text", "")
        canon = item.get("canonical_logits")
        up_mask = item.get("up_slot_mask")
        s_mask = item.get("slot_mask")
        if canon is None or up_mask is None or s_mask is None or not text:
            return item
        oracle, words, word_phone_ids = label_word_boundaries(
            canon, up_mask, s_mask, self._phone_vocab,
            self._upsample_factor, text, min_gap=self._oracle_min_gap,
        )
        item = dict(item)
        item["oracle_emit"] = oracle
        item["words"] = words
        item["word_phone_ids"] = word_phone_ids
        return item

    def __getitem__(self, idx: int) -> Dict:
        if self._flat is not None:
            item = self._flat[idx]
            if item is None:
                raise KeyError(f"Missing preloaded record at index {idx}")
            return self._maybe_recompute_oracle(item)
        rec = self.records[idx]
        shard_idx = rec["shard"]
        local_idx = rec["local_idx"]
        if shard_idx not in self._cache:
            shard_path = self.feat_dir / f"shard_{shard_idx:05d}.pt"
            self._cache[shard_idx] = torch.load(shard_path, weights_only=False)
        return self._maybe_recompute_oracle(self._cache[shard_idx][local_idx])


class AgentCollator:
    """Collate agent records into padded batches (slot-level padding)."""

    def __call__(self, batch: list[Dict]) -> Dict:
        B = len(batch)
        max_K = max(it["num_slots"] for it in batch)
        H = batch[0]["beliefs"].shape[-1]
        V = batch[0]["canonical_logits"].shape[-1]

        upsample_factor = 4
        for it in batch:
            K = it["num_slots"]
            if K > 0 and it["canonical_logits"].shape[0] > 0:
                upsample_factor = it["canonical_logits"].shape[0] // K
                break
        max_up = max_K * upsample_factor

        beliefs = torch.zeros(B, max_K, H)
        priors = torch.zeros(B, max_K, H)
        boundaries = torch.zeros(B, max_K, 2, dtype=torch.long)
        slot_mask = torch.zeros(B, max_K)
        oracle_emit = torch.zeros(B, max_K)
        canonical_logits = torch.zeros(B, max_up, V)
        up_slot_mask = torch.zeros(B, max_up)
        has_distortions = "distortions" in batch[0]
        distortions = torch.zeros(B, max_K, 1) if has_distortions else None

        for i, it in enumerate(batch):
            K = it["num_slots"]
            beliefs[i, :K] = it["beliefs"]
            priors[i, :K] = it["priors"]
            boundaries[i, :K] = it["boundaries"]
            slot_mask[i, :K] = 1.0
            oracle_emit[i, :K] = it["oracle_emit"]
            up_len = it["canonical_logits"].shape[0]
            canonical_logits[i, :up_len] = it["canonical_logits"]
            up_slot_mask[i, :up_len] = it["up_slot_mask"]
            if distortions is not None and "distortions" in it:
                distortions[i, :K] = it["distortions"]

        out = {
            "segment_ids": [it["segment_id"] for it in batch],
            "beliefs": beliefs,
            "priors": priors,
            "boundaries": boundaries,
            "slot_mask": slot_mask,
            "oracle_emit": oracle_emit,
            "canonical_logits": canonical_logits,
            "up_slot_mask": up_slot_mask,
            "words": [it["words"] for it in batch],
            "word_phone_ids": [it.get("word_phone_ids", []) for it in batch],
            "texts": [it["text"] for it in batch],
            "num_slots": torch.tensor([it["num_slots"] for it in batch]),
            "upsample_factor": upsample_factor,
        }
        if distortions is not None:
            out["distortions"] = distortions
        return out


# ---------------------------------------------------------------------------
# Extraction pipeline
# ---------------------------------------------------------------------------


def extract_agent_features(
    model,
    dataloader: DataLoader,
    phone_vocab: Vocabulary,
    device: torch.device,
    output_dir: Path,
    split: str,
    shard_size: int = SHARD_SIZE,
) -> int:
    """Run frozen model on the dataset and save agent data as shard files."""
    model.eval()
    save_dir = output_dir / split
    save_dir.mkdir(parents=True, exist_ok=True)

    upsample_factor = model.config.upsample_factor
    count = 0
    shard_idx = 0
    manifest = []
    current_shard: list[dict] = []

    def _flush(shard: list[dict], idx: int) -> None:
        torch.save(shard, save_dir / f"shard_{idx:05d}.pt")

    for batch in dataloader:
        ev = batch["evidence"].to(device)
        bd = batch["boundaries"].to(device)
        sm = batch["slot_mask"].to(device)
        nf = batch["num_frames"].to(device)
        fm = batch.get("frame_mask")
        if fm is not None:
            fm = fm.to(device)

        out = model.extract_slot_features(ev, bd, sm, nf, frame_mask=fm)

        B = ev.shape[0]
        for i in range(B):
            seg_id = batch["segment_ids"][i]
            num_syl = int(batch["num_syllables"][i].item())
            text = batch["texts"][i]
            up_len = num_syl * upsample_factor

            beliefs_i = out["beliefs"][i, :num_syl].contiguous().cpu()
            priors_i = out["priors"][i, :num_syl].contiguous().cpu()
            bd_i = batch["boundaries"][i, :num_syl].contiguous().cpu()
            sm_i = out["slot_mask"][i, :num_syl].contiguous().cpu()
            canon_logits_i = out["canonical_logits"][i, :up_len].contiguous().cpu()
            up_mask_i = out["up_slot_mask"][i, :up_len].contiguous().cpu()
            dist_i = None
            if "distortions" in out:
                dist_i = out["distortions"][i, :num_syl].contiguous().cpu()

            oracle_emit, words, word_phone_ids = label_word_boundaries(
                canon_logits_i, up_mask_i, sm_i, phone_vocab, upsample_factor, text,
            )
            oracle_emit = oracle_emit[:num_syl].contiguous()

            record: dict = {
                "segment_id": seg_id,
                "beliefs": beliefs_i,
                "priors": priors_i,
                "boundaries": bd_i,
                "canonical_logits": canon_logits_i,
                "up_slot_mask": up_mask_i,
                "slot_mask": sm_i,
                "oracle_emit": oracle_emit,
                "words": words,
                "word_phone_ids": word_phone_ids,
                "text": text,
                "num_slots": num_syl,
            }
            if dist_i is not None:
                record["distortions"] = dist_i

            current_shard.append(record)
            manifest.append({
                "segment_id": seg_id,
                "shard": shard_idx,
                "local_idx": len(current_shard) - 1,
                "num_words": len(words),
                "num_slots": num_syl,
            })
            count += 1

            if len(current_shard) >= shard_size:
                _flush(current_shard, shard_idx)
                print(f"  [{split}] Saved shard {shard_idx} ({count} total)", flush=True)
                shard_idx += 1
                current_shard = []

    if current_shard:
        _flush(current_shard, shard_idx)
        print(f"  [{split}] Saved shard {shard_idx} ({count} total)", flush=True)

    with open(output_dir / f"{split}_manifest.json", "w") as f:
        json.dump({
            "segment_ids": [m["segment_id"] for m in manifest],
            "records": manifest,
            "num_shards": shard_idx + 1,
            "shard_size": shard_size,
        }, f, indent=2)

    return count
