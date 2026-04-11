"""Extract aligned slot features and oracle word-boundary labels for agent training.

Loads a frozen Stage 2 BeliefWorldModel checkpoint, runs inference on the
dataset, and produces per-utterance records containing:
    - beliefs  (K, H)
    - priors   (K, H)
    - boundaries (K, 2)
    - canonical_logits (K*F, V)
    - up_slot_mask (K*F,)
    - slot_mask (K,)
    - oracle_emit (K,) — 1 at last syllable of each word, 0 otherwise
    - words  List[str]
    - segment_id  str

Oracle labeling strategy
========================
Three-tier priority for finding the true word-end slot:

1. CTC Forced Alignment (primary, uses CMU Pronouncing Dictionary):
   - Map each GT word to its canonical ARPABET phone sequence.
   - Run ``torchaudio.functional.forced_align`` on ``canonical_logits``.
   - Extract the last frame aligned to each word's last phone → slot.
   - This gives the ACOUSTIC word boundary, grounding the oracle in
     real speech evidence rather than proportional time estimates.

2. CTC Blank-Gap Heuristic (secondary, no dict needed):
   - Find long blank-dominated stretches between phone clusters in the
     greedy CTC decode; treat them as word boundaries.

3. Proportional Interpolation (last resort):
   - Uniformly space one emit per word across the utterance.

The function also returns ``word_phone_ids``: per-word lists of phone
vocabulary token IDs from the CMU dict.  These are stored in agent
records and used at GRPO time to compute a phone-level word-recognition
reward (``reward_mode="word_match"``), replacing the per-slot oracle
proximity signal with an *acoustic word accuracy* reward.

``recompute_oracle=True`` (default in ``train_wm_agent.py``) refreshes
both the oracle and ``word_phone_ids`` at load time without re-running
data extraction.

Usage:
    python wm_agent_data.py \\
        --checkpoint runs/jepa_stage2_asr/best.pt \\
        --features-dir artifacts/wm_features_librispeech \\
        --metadata-dir artifacts/metadata_librispeech \\
        --output-dir artifacts/agent_data \\
        --splits train validation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wm_common import Vocabulary, levenshtein_distance, normalize_text_for_char_ctc
from wm_core import (
    BeliefWMCollator,
    BeliefWMConfig,
    BeliefWMDataset,
    BeliefWorldModel,
)


# ---------------------------------------------------------------------------
# Oracle word-boundary labeling
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CMU dict helpers — phone sequence lookup + normalization
# ---------------------------------------------------------------------------

_CMU_DICT: Optional[dict] = None  # lazy-loaded singleton


def _get_cmu_dict() -> dict:
    """Lazy-load NLTK CMU Pronouncing Dictionary (singleton)."""
    global _CMU_DICT
    if _CMU_DICT is None:
        from nltk.corpus import cmudict
        _CMU_DICT = cmudict.dict()
    return _CMU_DICT


def _normalize_phone(ph: str) -> str:
    """Strip stress digit from ARPABET: 'AH0' → 'AH', 'T' → 'T'."""
    return ph.rstrip("012")


def word_to_phone_ids(
    word: str,
    phone_vocab: Vocabulary,
    cmu_dict: Optional[dict] = None,
) -> List[int]:
    """Map a single GT word to a list of phone vocabulary token IDs.

    Uses the CMU Pronouncing Dictionary for the lookup, selecting the first
    (most common) pronunciation variant.  Stress variants are resolved by
    trying the exact phone first, then each stress level in order (1, 0, 2,
    unstressed).  Returns an empty list for OOV words.

    Args:
        word:       GT word string (any case).
        phone_vocab: Vocabulary with ARPABET tokens.
        cmu_dict:   Pre-loaded CMU dict (loaded lazily if None).
    """
    if cmu_dict is None:
        cmu_dict = _get_cmu_dict()
    variants = cmu_dict.get(word.lower())
    if not variants:
        return []
    phones = variants[0]   # first pronunciation variant
    ids: List[int] = []
    for ph in phones:
        if ph in phone_vocab.token_to_id:
            ids.append(phone_vocab.token_to_id[ph])
        else:
            base = _normalize_phone(ph)
            found = False
            for stress in ("1", "0", "2", ""):
                cand = base + stress if stress else base
                if cand in phone_vocab.token_to_id:
                    ids.append(phone_vocab.token_to_id[cand])
                    found = True
                    break
            if not found:
                pass  # skip phones not representable in this vocab
    return ids


def build_word_phone_ids(
    words: List[str],
    phone_vocab: Vocabulary,
    cmu_dict: Optional[dict] = None,
) -> List[List[int]]:
    """Map a list of GT words to their per-word phone ID sequences."""
    if cmu_dict is None:
        cmu_dict = _get_cmu_dict()
    return [word_to_phone_ids(w, phone_vocab, cmu_dict) for w in words]


# ---------------------------------------------------------------------------
# Oracle word-boundary labeling — three-tier priority
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
    """Assign oracle emit labels to syllable slots.

    Three-tier priority (highest to lowest):
    1. CTC forced alignment via CMU dict (most accurate — true acoustic boundary).
    2. CTC blank-gap heuristic (no dict — coarser but acoustic).
    3. Proportional interpolation (last resort — no acoustic grounding).

    In all cases, min_gap and monotonicity are enforced.

    Returns:
        oracle_emit:     (K,) float tensor — 1.0 at word-end slots.
        words:           GT word list.
        word_phone_ids:  Per-word list of phone vocabulary token IDs (from
                         CMU dict); empty list for OOV words.  Used at GRPO
                         time for the ``word_match`` reward mode.
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

    # ---- Tier 1: forced alignment ----------------------------------------
    oracle_fa = _label_forced_align(
        canonical_logits, up_slot_mask, num_slots, K,
        phone_vocab, upsample_factor, word_phone_ids, min_gap,
    )
    if oracle_fa is not None:
        return oracle_fa, words, word_phone_ids

    # ---- Tier 2: CTC blank-gap heuristic ---------------------------------
    valid_len = int(up_slot_mask.sum().item())
    oracle_heuristic = _label_blank_gap(
        canonical_logits, valid_len, num_slots, K,
        phone_vocab.blank_id, upsample_factor, num_words, min_gap,
    )
    if oracle_heuristic is not None:
        return oracle_heuristic, words, word_phone_ids

    # ---- Tier 3: proportional fallback -----------------------------------
    return _label_proportional(num_slots, K, num_words, min_gap), words, word_phone_ids


def _alignment_to_positions(token_seq: List[int], blank_id: int) -> List[int]:
    """Map each frame in a CTC alignment to its target phone position (0-based).

    The alignment is monotone: phone positions never decrease.  Consecutive
    frames with the same non-blank token are assigned the same position (one
    phone emission extends over multiple frames).  A blank token keeps the
    current position without advancing.  The position counter increments when
    a new non-blank token is seen (after a blank or after a different phone).

    Returns:
        List of length len(token_seq); each value is the 0-based index of the
        target phone assigned to that frame (-1 before the first phone).
    """
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
    """Tier-1 oracle: CTC forced alignment with CMU phone sequences.

    Returns oracle tensor on success; None if alignment cannot be run
    (too few frames, too few words in dict, or torchaudio error).
    """
    from torchaudio.functional import forced_align as ta_forced_align

    # Concatenate all known phone sequences.
    all_phones: List[int] = []
    word_last_ph_idx: List[int] = []  # index of last phone for each word
    for ph_ids in word_phone_ids:
        if ph_ids:
            all_phones.extend(ph_ids)
            word_last_ph_idx.append(len(all_phones) - 1)
        else:
            word_last_ph_idx.append(-1)  # OOV

    # Need at least half the words to be in dict.
    known = sum(1 for i in word_last_ph_idx if i >= 0)
    if known < max(1, len(word_phone_ids) // 2):
        return None

    valid_len = int(up_slot_mask.sum().item())
    # CTC requires: T >= 2*N - 1  (minimum length for N targets)
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
        token_seq = alignments[0].tolist()           # (valid_len,)
        frame_positions = _alignment_to_positions(token_seq, phone_vocab.blank_id)

        oracle = torch.zeros(K, dtype=torch.float32)
        effective_gap = min(min_gap, max(1, num_slots // max(len(word_phone_ids), 1)))
        prev_emit = -effective_gap

        for w_idx, last_ph_idx in enumerate(word_last_ph_idx):
            if last_ph_idx < 0:
                # OOV: proportional estimate for this word.
                frac = (w_idx + 1) / len(word_phone_ids)
                slot = min(int(round(frac * num_slots)) - 1, num_slots - 1)
            else:
                # Find last frame assigned to this phone position.
                last_frame = max(
                    (f for f, p in enumerate(frame_positions) if p == last_ph_idx),
                    default=-1,
                )
                if last_frame < 0:
                    frac = (w_idx + 1) / len(word_phone_ids)
                    slot = min(int(round(frac * num_slots)) - 1, num_slots - 1)
                else:
                    slot = min(last_frame // upsample_factor, num_slots - 1)

            slot = max(slot, 0)
            slot = max(slot, prev_emit + effective_gap)
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
    """Tier-2 oracle: blank-gap heuristic (no external dictionary needed).

    Detects word-end frames as the last content frame before a long blank run.
    Returns None if fewer than half the expected word boundaries are found.
    """
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

    # Resample to match word count.
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


def _label_proportional(
    num_slots: int, K: int, num_words: int, min_gap: int
) -> torch.Tensor:
    """Tier-3 oracle: proportional interpolation (last resort, no acoustics)."""
    oracle = torch.zeros(K, dtype=torch.float32)
    effective_gap = min(min_gap, max(1, num_slots // max(num_words, 1)))
    prev_emit = -effective_gap
    for w_idx in range(num_words):
        frac = (w_idx + 1) / num_words
        target_slot = min(int(round(frac * num_slots)) - 1, num_slots - 1)
        target_slot = max(target_slot, 0)
        target_slot = max(target_slot, prev_emit + effective_gap)
        if target_slot < num_slots:
            oracle[target_slot] = 1.0
            prev_emit = target_slot
    return oracle


def frame_to_slot(frame_idx: int, upsample_factor: int, num_slots: int) -> int:
    return min(frame_idx // upsample_factor, num_slots - 1)


# ---------------------------------------------------------------------------
# Extraction pipeline — shard-based storage
# ---------------------------------------------------------------------------

SHARD_SIZE = 1000  # utterances per shard file


def extract_features(
    model: BeliefWorldModel,
    dataloader: DataLoader,
    phone_vocab: Vocabulary,
    device: torch.device,
    output_dir: Path,
    split: str,
    shard_size: int = SHARD_SIZE,
) -> int:
    """Run frozen model on the dataset and save agent data as shard files.

    Instead of one .pt file per utterance (104k files), records are batched
    into shard files of `shard_size` utterances each (~1000 files total for
    the train split).  This avoids inode exhaustion and home-directory quota
    problems from many small files, and is faster to write.

    Each shard: list of dicts, tensors clipped to valid length (no padding).
    """
    model.eval()
    save_dir = output_dir / split
    save_dir.mkdir(parents=True, exist_ok=True)

    upsample_factor = model.config.upsample_factor
    count = 0
    shard_idx = 0
    manifest = []
    current_shard: list[dict] = []

    def _flush_shard(shard: list[dict], idx: int) -> None:
        path = save_dir / f"shard_{idx:05d}.pt"
        torch.save(shard, path)

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

            # Clip to valid length and ensure contiguous memory layout
            beliefs_i = out["beliefs"][i, :num_syl].contiguous().cpu()
            priors_i = out["priors"][i, :num_syl].contiguous().cpu()
            bd_i = batch["boundaries"][i, :num_syl].contiguous().cpu()
            sm_i = out["slot_mask"][i, :num_syl].contiguous().cpu()
            canon_logits_i = out["canonical_logits"][i, :up_len].contiguous().cpu()
            up_mask_i = out["up_slot_mask"][i, :up_len].contiguous().cpu()

            oracle_emit, words, word_phone_ids = label_word_boundaries(
                canon_logits_i, up_mask_i, sm_i, phone_vocab,
                upsample_factor, text,
            )
            oracle_emit = oracle_emit[:num_syl].contiguous()

            record = {
                "segment_id": seg_id,
                "beliefs": beliefs_i,
                "priors": priors_i,
                "boundaries": bd_i,
                "canonical_logits": canon_logits_i,
                "up_slot_mask": up_mask_i,
                "slot_mask": sm_i,
                "oracle_emit": oracle_emit,
                "words": words,
                "word_phone_ids": word_phone_ids,  # List[List[int]] — phone IDs per word
                "text": text,
                "num_slots": num_syl,
            }
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
                _flush_shard(current_shard, shard_idx)
                print(f"  [{split}] Saved shard {shard_idx} ({count} total)", flush=True)
                shard_idx += 1
                current_shard = []

    # Flush remaining records
    if current_shard:
        _flush_shard(current_shard, shard_idx)
        print(f"  [{split}] Saved shard {shard_idx} ({count} total)", flush=True)

    with open(output_dir / f"{split}_manifest.json", "w") as f:
        json.dump({
            "segment_ids": [m["segment_id"] for m in manifest],
            "records": manifest,
            "num_shards": shard_idx + 1,
            "shard_size": shard_size,
        }, f, indent=2)

    return count


# ---------------------------------------------------------------------------
# AgentDataset — loads pre-extracted agent records
# ---------------------------------------------------------------------------


class AgentDataset(torch.utils.data.Dataset):
    """Loads shard-based agent feature files produced by this script.

    Two modes:
    - preload=True  (default): load all shards into RAM at __init__ time.
      Eliminates random-shard I/O during training; requires ~16GB RAM for
      the full train split (recommended when RAM >= 64GB).
    - preload=False: lazy per-shard caching (one shard per DataLoader worker).
      Use when RAM is scarce.

    Pass ``recompute_oracle=True`` together with a ``phone_vocab`` to discard
    stored oracle_emit labels and recompute them on-the-fly using the current
    ``label_word_boundaries`` logic (useful after changing labeling strategy
    without re-running data extraction).
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        max_examples: int | None = None,
        preload: bool = True,
        rank: int = 0,
        recompute_oracle: bool = False,
        phone_vocab=None,
        upsample_factor: int = 4,
        oracle_min_gap: int = 2,
    ) -> None:
        self._recompute_oracle = recompute_oracle
        self._phone_vocab = phone_vocab
        self._upsample_factor = upsample_factor
        self._oracle_min_gap = oracle_min_gap
        data_path = Path(data_dir)
        manifest_path = data_path / f"{split}_manifest.json"
        with open(manifest_path) as f:
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
            # Build a direct index: dataset_idx → record (bypasses shard lookup)
            seg_to_flat = {rec["segment_id"]: rec for rec in all_records}
            self._flat = [seg_to_flat.get(r["segment_id"]) for r in records]
            if rank == 0:
                loaded = sum(1 for x in self._flat if x is not None)
                print(f"[AgentDataset] Preloaded {loaded}/{len(records)} records.", flush=True)
        else:
            # Lazy shard cache (per-process, not safe across DataLoader workers)
            self._cache: dict[int, list] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _maybe_recompute_oracle(self, item: dict) -> dict:
        """Return item with oracle_emit and word_phone_ids recomputed.

        Uses the current ``label_word_boundaries`` logic (forced alignment
        primary, blank-gap heuristic secondary, proportional fallback).
        Falls back gracefully if required fields are missing.
        """
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
            self._upsample_factor, text,
            min_gap=self._oracle_min_gap,
        )
        # Return a shallow copy so we don't mutate the preloaded tensors.
        item = dict(item)
        item["oracle_emit"] = oracle
        item["words"] = words
        item["word_phone_ids"] = word_phone_ids
        return item

    def __getitem__(self, idx: int) -> Dict:
        if self._flat is not None:
            item = self._flat[idx]
            if item is None:
                # Fallback: should not happen with correct data
                raise KeyError(f"Missing preloaded record at index {idx}")
            return self._maybe_recompute_oracle(item)
        # Lazy path
        rec = self.records[idx]
        shard_idx = rec["shard"]
        local_idx = rec["local_idx"]
        if shard_idx not in self._cache:
            shard_path = self.feat_dir / f"shard_{shard_idx:05d}.pt"
            self._cache[shard_idx] = torch.load(shard_path, weights_only=False)
        return self._maybe_recompute_oracle(self._cache[shard_idx][local_idx])


class AgentCollator:
    """Collate agent records into padded batches.

    Unlike the regular BeliefWMCollator, this pads *slot-level* tensors
    since the agent operates at slot granularity.
    """

    def __call__(self, batch: list[Dict]) -> Dict:
        B = len(batch)
        max_K = max(it["num_slots"] for it in batch)
        H = batch[0]["beliefs"].shape[-1]
        V = batch[0]["canonical_logits"].shape[-1]

        # Infer upsample_factor from stored (already-clipped) tensor shapes
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

        for i, it in enumerate(batch):
            K = it["num_slots"]
            beliefs[i, :K] = it["beliefs"]          # already clipped to K
            priors[i, :K] = it["priors"]
            boundaries[i, :K] = it["boundaries"]
            slot_mask[i, :K] = 1.0
            oracle_emit[i, :K] = it["oracle_emit"]
            up_len = it["canonical_logits"].shape[0]  # already clipped to K*F
            canonical_logits[i, :up_len] = it["canonical_logits"]
            up_slot_mask[i, :up_len] = it["up_slot_mask"]

        return {
            "segment_ids": [it["segment_id"] for it in batch],
            "beliefs": beliefs,
            "priors": priors,
            "boundaries": boundaries,
            "slot_mask": slot_mask,
            "oracle_emit": oracle_emit,
            "canonical_logits": canonical_logits,
            "up_slot_mask": up_slot_mask,
            "words": [it["words"] for it in batch],
            # word_phone_ids: List[List[List[int]]] — [batch][word][phone_id]
            # Not padded into a tensor: each utterance has a different number of
            # words and each word has a different number of phones.  Kept as a
            # nested Python list for direct use in ASRSchedulerEnv.
            "word_phone_ids": [it.get("word_phone_ids", []) for it in batch],
            "texts": [it["text"] for it in batch],
            "num_slots": torch.tensor([it["num_slots"] for it in batch]),
            "upsample_factor": upsample_factor,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract agent training data from Stage 2 model.")
    p.add_argument("--checkpoint", required=True, help="Stage 2 best.pt")
    p.add_argument("--features-dir", required=True)
    p.add_argument("--metadata-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "validation"])
    p.add_argument("--evidence-type", choices=["logits", "hidden"], default="logits")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config: BeliefWMConfig = ckpt["config"]

    phone_vocab = Vocabulary.from_file(Path(args.metadata_dir) / "phone_vocab.json")
    text_vocab = Vocabulary.from_file(Path(args.metadata_dir) / "text_vocab.json")

    model = BeliefWorldModel(config)
    state = ckpt["model_state_dict"]
    current = model.state_dict()
    for name, param in state.items():
        if name in current and current[name].shape == param.shape:
            current[name].copy_(param)
    model.load_state_dict(current)
    model.to(device)
    model.eval()
    print(f"Loaded model from {args.checkpoint} (belief_type={config.belief_type})")

    for split in args.splits:
        print(f"Extracting {split}...")
        ds = BeliefWMDataset(
            args.features_dir, split, args.metadata_dir,
            phone_vocab, text_vocab,
            evidence_type=args.evidence_type,
            max_examples=args.max_examples,
        )
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=BeliefWMCollator(),
            pin_memory=True,
        )
        count = extract_features(model, loader, phone_vocab, device, output_dir, split)
        print(f"  {split}: {count} utterances saved to {output_dir / split}")

    print(f"Agent data extraction complete → {output_dir}")


if __name__ == "__main__":
    main()
