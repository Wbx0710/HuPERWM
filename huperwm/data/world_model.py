"""Dataset, collator, loss helpers, and evaluation for the Belief World Model."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from huperwm.data.vocab import Vocabulary, levenshtein_distance, load_jsonl, read_json


class BeliefWMDataset(Dataset):
    """Loads pre-computed HuPER + Sylber features saved by ``prepare_features.py``."""

    def __init__(
        self,
        features_dir: str,
        split: str,
        metadata_dir: str,
        phone_vocab: Vocabulary,
        text_vocab: Vocabulary,
        evidence_type: str = "hidden",  # "hidden" (1024-dim) or "logits" (46-dim)
        max_examples: int | None = None,
        teacher_cache: dict[str, list[str]] | None = None,
    ) -> None:
        feat_dir = Path(features_dir) / split
        manifest = read_json(Path(features_dir) / f"{split}_manifest.json")
        seg_ids: list[str] = manifest["segment_ids"]
        if max_examples:
            seg_ids = seg_ids[:max_examples]
        self.segment_ids = seg_ids
        self.feat_dir = feat_dir
        self.phone_vocab = phone_vocab
        self.text_vocab = text_vocab
        self.evidence_type = evidence_type
        self.teacher_cache = teacher_cache

        meta_path = Path(metadata_dir) / f"{split}.jsonl"
        self.metadata_map: dict = {}
        if meta_path.exists():
            self.metadata_map = {r["segment_id"]: r for r in load_jsonl(meta_path)}

    def __len__(self) -> int:
        return len(self.segment_ids)

    def __getitem__(self, idx: int) -> Dict:
        seg_id = self.segment_ids[idx]
        feats = torch.load(self.feat_dir / f"{seg_id}.pt", weights_only=False)
        meta = self.metadata_map.get(seg_id, feats)

        if self.evidence_type == "logits":
            evidence = feats["huper_logits"].float()
        else:
            if "huper_hidden" not in feats:
                raise KeyError(
                    f"Feature file for '{seg_id}' is missing 'huper_hidden'. "
                    f"Re-extract with: python prepare_features.py --evidence-type hidden"
                )
            evidence = feats["huper_hidden"].float()

        text = meta.get("text", feats.get("text", ""))
        text_chars = meta.get("text_chars", feats.get("text_chars", list(text)))
        canonical = meta.get("canonical_phones", feats.get("canonical_phones", []))
        if self.teacher_cache is not None and seg_id in self.teacher_cache:
            teacher = self.teacher_cache[seg_id]
        else:
            teacher = meta.get("teacher_phones", feats.get("teacher_phones", []))
        if not teacher:
            teacher = canonical

        return {
            "segment_id": seg_id,
            "evidence": evidence,
            "num_frames": evidence.shape[0],
            "boundaries": feats["sylber_boundaries"],
            "num_syllables": feats["sylber_boundaries"].shape[0],
            "text": text,
            "text_ids": torch.tensor(self.text_vocab.encode(text_chars), dtype=torch.long),
            "canonical_phones": canonical,
            "canonical_ids": torch.tensor(self.phone_vocab.encode(canonical), dtype=torch.long),
            "teacher_phones": teacher,
            "teacher_ids": torch.tensor(self.phone_vocab.encode(teacher), dtype=torch.long),
        }


class BeliefWMCollator:
    def __call__(self, batch: List[Dict]) -> Dict:
        B = len(batch)
        max_T = max(it["num_frames"] for it in batch)
        max_K = max(it["num_syllables"] for it in batch)
        E = batch[0]["evidence"].shape[-1]

        evidence = torch.zeros(B, max_T, E)
        frame_mask = torch.zeros(B, max_T)
        boundaries = torch.zeros(B, max_K, 2, dtype=torch.long)
        slot_mask = torch.zeros(B, max_K)

        for i, it in enumerate(batch):
            T, K = it["num_frames"], it["num_syllables"]
            evidence[i, :T] = it["evidence"]
            frame_mask[i, :T] = 1.0
            boundaries[i, :K] = it["boundaries"]
            slot_mask[i, :K] = 1.0

        return {
            "segment_ids": [it["segment_id"] for it in batch],
            "evidence": evidence,
            "frame_mask": frame_mask,
            "boundaries": boundaries,
            "slot_mask": slot_mask,
            "num_frames": torch.tensor([it["num_frames"] for it in batch]),
            "num_syllables": torch.tensor([it["num_syllables"] for it in batch]),
            "texts": [it["text"] for it in batch],
            "text_ids": [it["text_ids"] for it in batch],
            "canonical_phones": [it["canonical_phones"] for it in batch],
            "canonical_ids": [it["canonical_ids"] for it in batch],
            "teacher_phones": [it["teacher_phones"] for it in batch],
            "teacher_ids": [it["teacher_ids"] for it in batch],
        }


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def compute_ctc_loss(
    logits: torch.Tensor,
    input_lengths: torch.Tensor,
    target_sequences: List[torch.Tensor],
    blank_id: int,
) -> torch.Tensor:
    valid = [i for i, t in enumerate(target_sequences) if t.numel() > 0]
    if not valid:
        return logits.new_zeros((), requires_grad=True)
    sel_logits = logits[valid]
    sel_lengths = input_lengths[valid].to(logits.device)
    flat = torch.cat([target_sequences[i] for i in valid]).to(logits.device)
    tgt_len = torch.tensor(
        [target_sequences[i].numel() for i in valid], device=logits.device, dtype=torch.long
    )
    lp = F.log_softmax(sel_logits, dim=-1).transpose(0, 1)
    return F.ctc_loss(lp, flat, sel_lengths, tgt_len, blank=blank_id, zero_infinity=True)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _phone_error_rate(refs: List[List[str]], preds: List[List[str]]) -> float:
    total_edits = 0
    total_len = 0
    for ref, pred in zip(refs, preds):
        if ref:
            total_edits += levenshtein_distance(ref, pred)
            total_len += len(ref)
    return total_edits / max(total_len, 1)


@torch.no_grad()
def evaluate_belief_wm(
    model,
    dataloader,
    phone_vocab: Vocabulary,
    device: torch.device,
) -> Dict:
    """Evaluate BeliefWorldModel — returns PER metrics and auxiliary losses."""
    model.eval()
    canon_refs, canon_preds = [], []
    teacher_refs, teacher_preds = [], []
    future_losses: list[float] = []
    recon_losses: list[float] = []
    belief_cosines: list[float] = []

    for batch in dataloader:
        ev = batch["evidence"].to(device)
        bd = batch["boundaries"].to(device)
        sm = batch["slot_mask"].to(device)
        nf = batch["num_frames"].to(device)
        fm = batch.get("frame_mask")
        if fm is not None:
            fm = fm.to(device)

        out = model(ev, bd, sm, nf, frame_mask=fm)
        up_factor = model.config.upsample_factor

        canon_ids = out["canonical_logits"].argmax(dim=-1).cpu().tolist()
        phone_ids = out["frame_phone_logits"].argmax(dim=-1).cpu().tolist()

        for i in range(len(batch["segment_ids"])):
            up_len = int(batch["num_syllables"][i].item()) * up_factor
            pred_c = phone_vocab.decode_ctc(canon_ids[i][:up_len])
            canon_refs.append(batch["canonical_phones"][i])
            canon_preds.append(pred_c)

            T_i = int(batch["num_frames"][i].item())
            pred_t = phone_vocab.decode_ctc(phone_ids[i][:T_i])
            teacher_refs.append(batch["teacher_phones"][i])
            teacher_preds.append(pred_t)

        H = out["future_pred"].shape[-1]
        if out["future_pred"].shape[1] > 1:
            fm_s = sm[:, 1:].unsqueeze(-1)
            fl = (
                F.mse_loss(
                    out["future_pred"][:, :-1] * fm_s,
                    out["slots"][:, 1:] * fm_s,
                    reduction="sum",
                )
                / (fm_s.sum().clamp_min(1.0) * H)
            )
            future_losses.append(fl.item())

        rm = sm.unsqueeze(-1)
        rl = (
            F.mse_loss(
                out["evidence_recon"] * rm,
                out["slots"].detach() * rm,
                reduction="sum",
            )
            / (rm.sum().clamp_min(1.0) * H)
        )
        recon_losses.append(rl.item())

        beliefs = out["beliefs"]
        if beliefs.shape[1] > 1:
            b_prev = F.normalize(beliefs[:, :-1], dim=-1)
            b_next = F.normalize(beliefs[:, 1:], dim=-1)
            adj_mask = sm[:, 1:].unsqueeze(-1)
            cos_sim = (b_prev * b_next * adj_mask).sum() / adj_mask.sum().clamp_min(1.0)
            belief_cosines.append(cos_sim.item())

    return {
        "canonical_per": _phone_error_rate(canon_refs, canon_preds),
        "teacher_per": _phone_error_rate(teacher_refs, teacher_preds),
        "future_mse": sum(future_losses) / max(len(future_losses), 1),
        "recon_mse": sum(recon_losses) / max(len(recon_losses), 1),
        "belief_evolution_cosine": sum(belief_cosines) / max(len(belief_cosines), 1),
        "num_examples": len(canon_refs),
    }
