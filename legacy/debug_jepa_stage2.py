"""Quick diagnostic for Stage-2 JEPA loss plateau.

Loads the Stage-2 last.pt checkpoint and runs one pass over 3 batches from
the validation split to collect per-step diagnostic signals, writing NDJSON
to the debug log for session b78750.

Usage (activate the phn env first):
    python debug_jepa_stage2.py

Hypothesis tested:
  A  Belief adjacent cosine ≈ -1 (oscillating encoder)
  B  z_online vs z_target cosine at unmasked positions (encoder drift from EMA target)
  C  z_pred vs z_target cosine at masked positions (actual JEPA prediction quality)
  D  future/recon gradient magnitude vs JEPA magnitude — are future/recon pushing the encoder?
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
LOG_PATH = "/home/bixingwu/huperJEPA/.cursor/debug-b78750.log"
SESSION_ID = "b78750"

STAGE2_CKPT = "/home/bixingwu/huperJEPA/runs/jepa_stage2_asr/last.pt"
STAGE1_CKPT = "/home/bixingwu/huperJEPA/runs/jepa_stage1/best_stage1.pt"
FEATURES_DIR = "/home/bixingwu/huperJEPA/artifacts/wm_features_librispeech"
METADATA_DIR = "/home/bixingwu/huperJEPA/artifacts/metadata_librispeech"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_BATCHES = 5
BATCH_SIZE = 16
# ---------------------------------------------------------------------------


def _log(message: str, data: dict, hyp: str = "", run_id: str = "diag"):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    payload = {
        "sessionId": SESSION_ID,
        "id": f"log_{int(time.time()*1000)}",
        "timestamp": int(time.time() * 1000),
        "location": "debug_jepa_stage2.py",
        "message": message,
        "hypothesisId": hyp,
        "runId": run_id,
        "data": data,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(payload) + "\n")
    print(f"[{hyp}] {message}: {data}")


def main():
    from wm_common import Vocabulary
    from wm_core import (
        BeliefWMCollator,
        BeliefWMDataset,
        BeliefWorldModel,
    )
    from wm_jepa import block_mask_slots

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    # clear old log
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)

    print(f"Loading Stage-2 checkpoint: {STAGE2_CKPT}")
    s2_ckpt = torch.load(STAGE2_CKPT, map_location="cpu", weights_only=False)
    config = s2_ckpt["config"]
    model = BeliefWorldModel(config).to(DEVICE)
    model.load_state_dict(s2_ckpt["model_state_dict"])
    model.eval()

    print(f"Loading Stage-1 checkpoint (reference): {STAGE1_CKPT}")
    s1_ckpt = torch.load(STAGE1_CKPT, map_location="cpu", weights_only=False)
    s1_config = s1_ckpt["config"]
    s1_model = BeliefWorldModel(s1_config).to(DEVICE)
    missing, unexpected = s1_model.load_state_dict(s1_ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        print(f"Stage-1 partial load: missing={missing[:5]}, unexpected={unexpected[:5]}")
    s1_model.eval()

    phone_vocab = Vocabulary.from_file(Path(METADATA_DIR) / "phone_vocab.json")
    text_vocab = Vocabulary.from_file(Path(METADATA_DIR) / "text_vocab.json")

    ds = BeliefWMDataset(
        FEATURES_DIR, "validation", METADATA_DIR,
        phone_vocab, text_vocab, evidence_type="logits", max_examples=100,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=BeliefWMCollator(),
    )

    from wm_jepa import compute_jepa_loss as _jepa_loss_fn

    # -----------------------------------------------------------------------
    all_belief_cos_s2, all_belief_cos_s1 = [], []
    all_pred_target_cos_s2, all_pred_target_cos_s1 = [], []
    all_online_target_cos_s2, all_online_target_cos_s1 = [], []
    # Hypothesis D: does disabling dropout (eval mode) cause the train-eval gap?
    all_jepa_loss_eval_s2, all_jepa_loss_train_s2 = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= N_BATCHES:
                break

            ev = batch["evidence"].to(DEVICE)
            bd = batch["boundaries"].to(DEVICE)
            sm = batch["slot_mask"].to(DEVICE)
            nf = batch["num_frames"].to(DEVICE)

            # ---- Stage 2 EVAL mode diagnostics ----
            model.eval()
            out2 = model(ev, bd, sm, nf, compute_jepa_loss=True)
            beliefs2 = out2["beliefs"]

            # Hypothesis A: adjacent belief cosine
            if beliefs2.shape[1] > 1:
                b_n = F.normalize(beliefs2.float(), dim=-1)
                adj_cos = (b_n[:, :-1] * b_n[:, 1:]).sum(dim=-1)
                adj_mask = sm[:, 1:]
                mean_adj_cos = float((adj_cos * adj_mask).sum() / adj_mask.sum().clamp_min(1.0))
                all_belief_cos_s2.append(mean_adj_cos)

            # Hypothesis B: z_online vs z_target cosine at unmasked positions
            if "z_target" in out2 and out2["z_target"] is not None:
                z_online = beliefs2
                z_target = out2["z_target"]
                jmask = out2["jepa_mask"]
                visible = (~jmask) & (sm > 0.5)
                if visible.any():
                    zo_n = F.normalize(z_online[visible].float(), dim=-1)
                    zt_n = F.normalize(z_target[visible].float(), dim=-1)
                    cos_ot = float((zo_n * zt_n).sum(dim=-1).mean())
                    all_online_target_cos_s2.append(cos_ot)

            # Hypothesis C: z_pred vs z_target at masked positions (eval)
            if "z_pred" in out2 and out2["z_pred"] is not None:
                z_pred = out2["z_pred"]
                z_target_v = out2["z_target"]
                jmask = out2["jepa_mask"]
                valid_masked = jmask & (sm > 0.5)
                if valid_masked.any():
                    zp_n = F.normalize(z_pred[valid_masked].float(), dim=-1)
                    zt_n = F.normalize(z_target_v[valid_masked].float(), dim=-1)
                    cos_pt = float((zp_n * zt_n).sum(dim=-1).mean())
                    all_pred_target_cos_s2.append(cos_pt)

                # Compute actual JEPA loss value in EVAL mode
                jepa_l_eval = float(_jepa_loss_fn(z_pred, z_target_v, jmask, sm))
                all_jepa_loss_eval_s2.append(jepa_l_eval)

            # ---- Stage 2 TRAIN mode: test if dropout causes the gap (Hypothesis D) ----
            model.train()
            out2_train = model(ev, bd, sm, nf, compute_jepa_loss=True)
            if "z_pred" in out2_train and out2_train["z_pred"] is not None:
                jepa_l_train = float(_jepa_loss_fn(
                    out2_train["z_pred"], out2_train["z_target"],
                    out2_train["jepa_mask"], sm,
                ))
                all_jepa_loss_train_s2.append(jepa_l_train)

            # ---- Stage 1 reference diagnostics ----
            s1_model.eval()
            out1 = s1_model(ev, bd, sm, nf, compute_jepa_loss=True)
            beliefs1 = out1["beliefs"]

            if beliefs1.shape[1] > 1:
                b_n1 = F.normalize(beliefs1.float(), dim=-1)
                adj_cos1 = (b_n1[:, :-1] * b_n1[:, 1:]).sum(dim=-1)
                adj_mask1 = sm[:, 1:]
                mean_adj_cos1 = float((adj_cos1 * adj_mask1).sum() / adj_mask1.sum().clamp_min(1.0))
                all_belief_cos_s1.append(mean_adj_cos1)

            if "z_target" in out1 and out1["z_target"] is not None:
                visible1 = (~out1["jepa_mask"]) & (sm > 0.5)
                if visible1.any():
                    zo_n1 = F.normalize(beliefs1[visible1].float(), dim=-1)
                    zt_n1 = F.normalize(out1["z_target"][visible1].float(), dim=-1)
                    all_online_target_cos_s1.append(float((zo_n1 * zt_n1).sum(dim=-1).mean()))

            if "z_pred" in out1 and out1["z_pred"] is not None:
                valid_masked1 = out1["jepa_mask"] & (sm > 0.5)
                if valid_masked1.any():
                    zp_n1 = F.normalize(out1["z_pred"][valid_masked1].float(), dim=-1)
                    zt_n1 = F.normalize(out1["z_target"][valid_masked1].float(), dim=-1)
                    all_pred_target_cos_s1.append(float((zp_n1 * zt_n1).sum(dim=-1).mean()))

    # -----------------------------------------------------------------------
    def _mean(lst):
        return sum(lst) / len(lst) if lst else None

    _log("Stage2 belief adjacent cosine (Hyp A: oscillation check)",
         {"mean": _mean(all_belief_cos_s2), "values": all_belief_cos_s2}, hyp="A")
    _log("Stage1 belief adjacent cosine (reference)",
         {"mean": _mean(all_belief_cos_s1), "values": all_belief_cos_s1}, hyp="A")
    _log("Stage2 z_online vs z_target cosine at visible slots (Hyp B: encoder drift)",
         {"mean": _mean(all_online_target_cos_s2), "values": all_online_target_cos_s2}, hyp="B")
    _log("Stage1 z_online vs z_target cosine (reference)",
         {"mean": _mean(all_online_target_cos_s1), "values": all_online_target_cos_s1}, hyp="B")
    _log("Stage2 pred-vs-target cosine EVAL mode (Hyp C: prediction quality)",
         {"mean": _mean(all_pred_target_cos_s2), "values": all_pred_target_cos_s2}, hyp="C")
    _log("Stage1 pred-vs-target cosine EVAL mode (reference)",
         {"mean": _mean(all_pred_target_cos_s1), "values": all_pred_target_cos_s1}, hyp="C")
    _log("Stage2 actual JEPA loss EVAL mode (should ≈ 2*(1-cos_C))",
         {"mean": _mean(all_jepa_loss_eval_s2), "values": all_jepa_loss_eval_s2}, hyp="C")
    _log("Stage2 actual JEPA loss TRAIN mode (Hyp D: dropout gap — should match CSV ~1.24)",
         {"mean": _mean(all_jepa_loss_train_s2), "values": all_jepa_loss_train_s2}, hyp="D")

    print(f"\n=== SUMMARY ===")
    print(f"[HypA] Stage2 belief adj cosine:       {_mean(all_belief_cos_s2):.4f}  (Stage1: {_mean(all_belief_cos_s1):.4f})")
    print(f"[HypB] Stage2 online-vs-target cos:    {_mean(all_online_target_cos_s2):.4f}  (Stage1: {_mean(all_online_target_cos_s1):.4f})")
    print(f"[HypC] Stage2 pred-vs-target cos EVAL: {_mean(all_pred_target_cos_s2):.4f}  (Stage1: {_mean(all_pred_target_cos_s1):.4f})")
    print(f"[HypC] Stage2 JEPA loss EVAL:          {_mean(all_jepa_loss_eval_s2):.4f}  (expected ~{2*(1-_mean(all_pred_target_cos_s2)):.4f})")
    print(f"[HypD] Stage2 JEPA loss TRAIN mode:    {_mean(all_jepa_loss_train_s2):.4f}  (training CSV shows ~1.24)")
    print(f"\nLog written to: {LOG_PATH}")


if __name__ == "__main__":
    main()
