"""Train the Belief World Model with PyTorch Lightning.

Supports two data modes:
1) offline pre-extracted features (`--features-dir`)
2) online extraction from HuggingFace LibriSpeech (`--online-features`)

Usage (overfit on smoke data):
    conda activate phn
    python train_wm_belief.py \
        --features-dir artifacts/wm_features_smoke \
        --metadata-dir artifacts/metadata_smoke \
        --output-dir runs/wm_belief_overfit \
        --evidence-type logits \
        --epochs 200 --batch-size 4 --lr 3e-4
"""

import argparse
import json
import math
from pathlib import Path

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from torch.optim import AdamW
from torch.utils.data import DataLoader

from wm_common import Vocabulary, write_json
from wm_core import (
    BeliefWMCollator,
    BeliefWMConfig,
    BeliefWMDataset,
    BeliefWorldModel,
    compute_ctc_loss,
    evaluate_belief_wm,
)
from wm_comparison import convergence_loss
from wm_jepa import check_collapse, compute_jepa_loss, compute_sigreg_loss, compute_vicreg_loss
from wm_online_data import OnlineLibriSpeechWMDataset
from wm_teacher import load_teacher_phone_cache


def _maybe_teacher_cache(args) -> dict | None:
    if not args.teacher_cache:
        return None
    return load_teacher_phone_cache(args.teacher_cache)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Belief World Model.")
    p.add_argument("--features-dir", default=None)
    p.add_argument("--metadata-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--online-features", action="store_true")
    p.add_argument("--hf-dataset-name", type=str, default="openslr/librispeech_asr")
    p.add_argument("--hf-dataset-config", type=str, default="all")
    p.add_argument("--hf-train-split", type=str, default="train.clean.360")
    p.add_argument("--hf-val-split", type=str, default="validation.clean")
    p.add_argument("--huper-repo", type=str, default="huper29/huper_recognizer")
    p.add_argument("--feature-device", type=str, default="cuda")
    p.add_argument(
        "--teacher-cache",
        nargs="*",
        default=None,
        help="JSONL files from wm_cache_teacher_phones.py (segment_id, teacher_phones); "
        "merge multiple (e.g. train + validation) for lookup by segment_id.",
    )

    p.add_argument("--evidence-type", choices=["logits", "hidden"], default="logits")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument(
        "--pooling-type",
        choices=["mean", "attention", "energy", "boundary_attention", "enhanced_boundary"],
        default="mean",
    )
    p.add_argument("--boundary-attn-heads", type=int, default=4,
                   help="num_heads for BoundaryAwareCrossAttnPool (pooling_type=boundary_attention).")
    # --- Word Distortion Module (HuperJEPA v2) ---
    p.add_argument("--use-distortion", action="store_true",
                   help="Enable WordDistortionModule (v2). Adds distortion loss to Stage 2.")
    p.add_argument("--distortion-loss-weight", type=float, default=0.5,
                   help="Weight for distortion alignment loss (default 0.5).")
    p.add_argument("--distortion-init-threshold", type=float, default=0.3,
                   help="Initial learnable EMIT threshold (default 0.3).")
    p.add_argument("--upsample-factor", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--min-lr-ratio", type=float, default=0.01,
                   help="Minimum LR as fraction of peak LR (cosine floor).")
    p.add_argument("--max-grad-norm", type=float, default=5.0)
    p.add_argument("--accumulate-grad-batches", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-examples", type=int, default=None)
    p.add_argument("--max-val-examples", type=int, default=None)

    p.add_argument("--frame-phone-weight", type=float, default=1.0)
    p.add_argument("--evidence-phone-weight", type=float, default=0.5)
    p.add_argument("--canonical-weight", type=float, default=0.5)
    p.add_argument("--future-weight", type=float, default=0.3)
    p.add_argument("--recon-weight", type=float, default=0.3)
    p.add_argument("--diversity-weight", type=float, default=0.1,
                   help="Weight for belief temporal diversity loss (penalizes static beliefs).")
    p.add_argument("--diversity-hinge", type=float, default=0.8,
                   help="Cosine similarity threshold above which diversity penalty activates.")

    # --- Identity / Prosody / Uncertainty / Mismatch ---
    p.add_argument("--use-identity", action="store_true")
    p.add_argument("--identity-dim", type=int, default=128)
    p.add_argument("--use-prosody", action="store_true")
    p.add_argument("--prosody-dim", type=int, default=64)
    p.add_argument("--use-uncertainty", action="store_true")
    p.add_argument("--uncertainty-dim", type=int, default=32)
    p.add_argument("--use-mismatch", action="store_true")
    p.add_argument("--mismatch-dim", type=int, default=64)
    p.add_argument("--belief-grad-scale", type=float, default=0.1,
                   help="Gradient scale for belief_frames in frame_phone_head (0=detach, 1=full, 0.1=recommended).")
    p.add_argument("--frame-phone-dropout", type=float, default=0.1,
                   help="Dropout in frame_phone_head to reduce overfitting.")

    # --- JEPA / Plan C ---
    p.add_argument("--belief-type", choices=["gru", "jepa", "comparison"], default="gru")
    p.add_argument("--jepa-encoder-layers", type=int, default=6)
    p.add_argument("--jepa-encoder-heads", type=int, default=8)
    p.add_argument("--jepa-encoder-ff-dim", type=int, default=1024)
    p.add_argument("--jepa-encoder-conv-kernel", type=int, default=15)
    p.add_argument("--jepa-daam-num-gaussians", type=int, default=4)
    p.add_argument("--jepa-daam-alpha-init", type=float, default=0.05)
    p.add_argument("--jepa-predictor-layers", type=int, default=2)
    p.add_argument("--jepa-predictor-heads", type=int, default=8)
    p.add_argument("--jepa-prior-layers", type=int, default=2)
    p.add_argument("--jepa-prior-heads", type=int, default=8)
    p.add_argument("--jepa-ema-tau", type=float, default=0.996)
    p.add_argument(
        "--jepa-ema-tau-end", type=float, default=None,
        help="End value for EMA tau cosine schedule (e.g. 0.9999, like Stage 1). "
             "None = keep tau fixed at --jepa-ema-tau throughout Stage 2.",
    )
    p.add_argument(
        "--jepa-predictor-lr-mult", type=float, default=1.0,
        help="LR multiplier for JEPA predictor parameters relative to the rest of "
             "the model (e.g. 2.0, matching Stage 1 default). 1.0 = no difference.",
    )
    p.add_argument("--jepa-mask-ratio", type=float, default=0.5)
    p.add_argument("--jepa-mask-min-span", type=int, default=1)
    p.add_argument("--jepa-mask-max-span", type=int, default=None)
    p.add_argument("--jepa-aux-weight", type=float, default=0.1,
                   help="Weight for JEPA auxiliary loss during Stage 2.")
    p.add_argument("--vicreg-weight", type=float, default=0.0,
                   help="Weight for VICReg variance+covariance regularization on online encoder "
                        "output (prevents representation collapse; 0.01 recommended for Stage 2).")
    p.add_argument("--vicreg-var-gamma", type=float, default=1.0,
                   help="Target std per dimension for VICReg variance hinge (default 1.0).")
    p.add_argument("--sigreg-weight", type=float, default=0.0,
                   help="Weight for SIGReg (Sketched-Isotropic-Gaussian Regularizer) on online "
                        "encoder beliefs. Alternative to VICReg; 0.05 recommended.")
    p.add_argument("--sigreg-projections", type=int, default=64,
                   help="Number of random projection directions for SIGReg (default 64).")
    p.add_argument("--canonical-head-dropout", type=float, default=0.0,
                   help="Input dropout rate for the canonical CTC head (default 0.0).")
    p.add_argument("--stage1-checkpoint", type=str, default=None,
                   help="Path to Stage 1 JEPA checkpoint to initialise from.")

    # --- Comparison Refinement (v3) ---
    p.add_argument("--num-refinements", type=int, default=2,
                   help="Number of comparison-refinement iterations (v3).")
    p.add_argument("--refinement-heads", type=int, default=4,
                   help="Attention heads per refinement Conformer block (v3).")
    p.add_argument("--refinement-ff-dim", type=int, default=512,
                   help="Feed-forward dim in refinement Conformer block (v3).")
    p.add_argument("--refinement-conv-kernel", type=int, default=15,
                   help="Convolution kernel size in refinement Conformer block (v3).")
    p.add_argument("--convergence-loss-weight", type=float, default=0.2,
                   help="Weight for convergence loss (encourages error to decrease across iterations).")

    # --- TTS ---
    p.add_argument("--use-tts", action="store_true", help="Enable TTS flow-matching decoder.")
    p.add_argument("--tts-weight", type=float, default=1.0)
    p.add_argument("--tts-dur-weight", type=float, default=0.2)
    p.add_argument("--tts-decoder-dim", type=int, default=512)
    p.add_argument("--tts-decoder-layers", type=int, default=6)
    p.add_argument(
        "--mel-stats-path",
        type=str,
        default=None,
        help="Path to mel_stats.pt for mel normalization (required for TTS).",
    )
    p.add_argument(
        "--tts-finetune-encoder",
        action="store_true",
        help="Allow TTS loss gradients to flow into BWM encoder (no detach).",
    )

    p.add_argument("--devices", type=int, default=None)
    p.add_argument("--precision", type=str, default="32-true")
    p.add_argument("--eval-every-epochs", type=int, default=5)
    p.add_argument("--log-every-steps", type=int, default=10)
    p.add_argument("--patience", type=int, default=0,
                   help="Early stopping patience in epochs (0 = disabled).")
    p.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a best.pt / last.pt checkpoint. Loads model weights only; "
             "optimizer and LR schedule restart from step 0.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class BeliefWMDataModule(pl.LightningDataModule):
    def __init__(self, args, phone_vocab, text_vocab, teacher_cache=None):
        super().__init__()
        self.args = args
        self.phone_vocab = phone_vocab
        self.text_vocab = text_vocab
        self.teacher_cache = teacher_cache

    def train_dataloader(self):
        if self.args.online_features:
            ds = OnlineLibriSpeechWMDataset(
                hf_dataset_name=self.args.hf_dataset_name,
                hf_dataset_config=self.args.hf_dataset_config,
                hf_split=self.args.hf_train_split,
                phone_vocab=self.phone_vocab,
                text_vocab=self.text_vocab,
                evidence_type=self.args.evidence_type,
                huper_repo=self.args.huper_repo,
                feature_device=self.args.feature_device,
                max_examples=self.args.max_train_examples,
                teacher_cache=self.teacher_cache,
                extract_mel=getattr(self.args, "use_tts", False),
            )
        else:
            ds = BeliefWMDataset(
                self.args.features_dir,
                "train",
                self.args.metadata_dir,
                self.phone_vocab,
                self.text_vocab,
                evidence_type=self.args.evidence_type,
                max_examples=self.args.max_train_examples,
                teacher_cache=self.teacher_cache,
            )
        return DataLoader(
            ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=BeliefWMCollator(),
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.args.online_features:
            ds = OnlineLibriSpeechWMDataset(
                hf_dataset_name=self.args.hf_dataset_name,
                hf_dataset_config=self.args.hf_dataset_config,
                hf_split=self.args.hf_val_split,
                phone_vocab=self.phone_vocab,
                text_vocab=self.text_vocab,
                evidence_type=self.args.evidence_type,
                huper_repo=self.args.huper_repo,
                feature_device=self.args.feature_device,
                max_examples=self.args.max_val_examples,
                teacher_cache=self.teacher_cache,
                extract_mel=getattr(self.args, "use_tts", False),
            )
        else:
            ds = BeliefWMDataset(
                self.args.features_dir,
                "validation",
                self.args.metadata_dir,
                self.phone_vocab,
                self.text_vocab,
                evidence_type=self.args.evidence_type,
                max_examples=self.args.max_val_examples,
                teacher_cache=self.teacher_cache,
            )
        return DataLoader(
            ds,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=BeliefWMCollator(),
            pin_memory=True,
        )


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------


class BeliefWMLitModule(pl.LightningModule):
    def __init__(self, config: BeliefWMConfig, phone_vocab: Vocabulary, args, tts_decoder=None):
        super().__init__()
        self.save_hyperparameters(ignore=["phone_vocab", "tts_decoder"])
        self.model = BeliefWorldModel(config)
        self.phone_vocab = phone_vocab
        self.args = args
        self.tts_decoder = tts_decoder
        self._use_jepa = config.belief_type == "jepa"
        self._use_comparison = config.belief_type == "comparison"
        if self._use_jepa:
            self.automatic_optimization = False

    def _model_forward(self, batch):
        return self.model(
            batch["evidence"],
            batch["boundaries"],
            batch["slot_mask"],
            batch["num_frames"],
            frame_mask=batch.get("frame_mask"),
        )

    def _compute_losses(self, batch, outputs):
        ev = batch["evidence"]
        nf = batch["num_frames"].to(ev.device)
        ns = batch["num_syllables"].to(ev.device)
        sm = outputs["slot_mask"]
        up_lengths = ns * self.model.config.upsample_factor

        frame_phone_loss = compute_ctc_loss(
            outputs["frame_phone_logits"],
            nf,
            batch["teacher_ids"],
            self.phone_vocab.blank_id,
        )
        evidence_phone_loss = compute_ctc_loss(
            outputs["evidence_phone_logits"],
            up_lengths,
            batch["teacher_ids"],
            self.phone_vocab.blank_id,
        )
        canonical_loss = compute_ctc_loss(
            outputs["canonical_logits"],
            up_lengths,
            batch["canonical_ids"],
            self.phone_vocab.blank_id,
        )

        future_loss = ev.new_zeros(())
        H = outputs["future_pred"].shape[-1]
        if outputs["future_pred"].shape[1] > 1:
            fm = sm[:, 1:].unsqueeze(-1)
            future_loss = (
                F.mse_loss(
                    outputs["future_pred"][:, :-1] * fm,
                    outputs["slots"][:, 1:].detach() * fm,
                    reduction="sum",
                )
                / (fm.sum().clamp_min(1.0) * H)
            )

        rm = sm.unsqueeze(-1)
        recon_loss = (
            F.mse_loss(
                outputs["evidence_recon"] * rm,
                outputs["slots"].detach() * rm,
                reduction="sum",
            )
            / (rm.sum().clamp_min(1.0) * H)
        )

        total = (
            self.args.frame_phone_weight * frame_phone_loss
            + self.args.evidence_phone_weight * evidence_phone_loss
            + self.args.canonical_weight * canonical_loss
            + self.args.future_weight * future_loss
            + self.args.recon_weight * recon_loss
        )
        loss_dict = {
            "loss": total,
            "frame_phone_loss": frame_phone_loss,
            "evidence_phone_loss": evidence_phone_loss,
            "canonical_loss": canonical_loss,
            "future_loss": future_loss,
            "recon_loss": recon_loss,
        }

        # --- TTS loss ---
        if self.tts_decoder is not None and "mel_target" in batch:
            identity = outputs.get("identity")
            if identity is None:
                identity = torch.zeros(
                    ev.shape[0], self.model.config.identity_dim, device=ev.device
                )
            priors = outputs["priors"]
            finetune = getattr(self.args, "tts_finetune_encoder", False)
            tts_beliefs = outputs["beliefs"] if finetune else outputs["beliefs"].detach()
            tts_priors = priors if finetune else priors.detach()
            tts_identity = identity if finetune else identity.detach()
            mel_target = batch["mel_target"].to(ev.device)
            mel_lengths = batch["mel_lengths"].to(ev.device)
            tts_losses = self.tts_decoder.compute_loss(
                beliefs=tts_beliefs,
                priors=tts_priors,
                identity=tts_identity,
                slot_mask=sm,
                mel_target=mel_target,
                mel_lengths=mel_lengths,
                boundaries=batch["boundaries"].to(ev.device),
                num_frames=nf,
            )
            loss_dict["tts_flow_loss"] = tts_losses["tts_flow_loss"]
            loss_dict["tts_dur_loss"] = tts_losses["tts_dur_loss"]
            loss_dict["loss"] = (
                loss_dict["loss"]
                + self.args.tts_weight * tts_losses["tts_flow_loss"]
                + self.args.tts_dur_weight * tts_losses["tts_dur_loss"]
            )

        # --- Belief temporal diversity loss ---
        # Always compute adjacent cosine for real-time oscillation monitoring,
        # regardless of whether diversity_weight > 0.
        div_weight = getattr(self.args, "diversity_weight", 0.0)
        if outputs["beliefs"].shape[1] > 1:
            beliefs = outputs["beliefs"]
            b_prev = F.normalize(beliefs[:, :-1], dim=-1)
            b_next = F.normalize(beliefs[:, 1:], dim=-1)
            adj_mask = sm[:, 1:]
            cos_sim = (b_prev * b_next).sum(dim=-1)  # (B, K-1)
            # Log mean adjacent cosine so oscillation (cos → -1) is visible in
            # training metrics without waiting for the full PER evaluation.
            mean_cos = (cos_sim * adj_mask).sum() / adj_mask.sum().clamp_min(1.0)
            loss_dict["belief_mean_cosine"] = mean_cos
            if div_weight > 0:
                hinge = getattr(self.args, "diversity_hinge", 0.8)
                # FIX: use abs(cos_sim) to penalise *both* over-similar beliefs
                # (cos > +hinge) AND oscillating anti-correlated beliefs
                # (cos < -hinge).  The original one-sided relu(cos - hinge)
                # created a zero-loss loophole at cos ≈ -1, causing the model
                # to learn pathologically alternating adjacent belief vectors.
                diversity_loss = (
                    F.relu(cos_sim.abs() - hinge) * adj_mask
                ).sum() / adj_mask.sum().clamp_min(1.0)
                loss_dict["diversity_loss"] = diversity_loss
                loss_dict["loss"] = loss_dict["loss"] + div_weight * diversity_loss

        # --- JEPA auxiliary loss (Stage 2) ---
        jepa_weight = getattr(self.args, "jepa_aux_weight", 0.0)
        if self._use_jepa and jepa_weight > 0 and "z_pred" in outputs:
            jepa_loss = compute_jepa_loss(
                outputs["z_pred"], outputs["z_target"],
                outputs["jepa_mask"], outputs["slot_mask"],
            )
            loss_dict["jepa_loss"] = jepa_loss
            loss_dict["loss"] = loss_dict["loss"] + jepa_weight * jepa_loss

        # --- VICReg regularization (Stage 2) ---
        # VICReg: variance + covariance regularization to prevent collapse.
        vicreg_weight = getattr(self.args, "vicreg_weight", 0.0)
        if self._use_jepa and vicreg_weight > 0:
            vicreg_gamma = getattr(self.args, "vicreg_var_gamma", 1.0)
            vicreg_target = (
                outputs["z_pred"] if "z_pred" in outputs else outputs.get("beliefs")
            )
            if vicreg_target is not None:
                var_loss, cov_loss = compute_vicreg_loss(
                    vicreg_target, outputs["slot_mask"], gamma=vicreg_gamma
                )
                vicreg_loss = var_loss + cov_loss
                loss_dict["vicreg_loss"] = vicreg_loss
                loss_dict["loss"] = loss_dict["loss"] + vicreg_weight * vicreg_loss

        # SIGReg (Maes et al., LeWM 2026): alternative to VICReg via Epps-Pulley test.
        # Active for both JEPA and Comparison modes.
        sigreg_weight = getattr(self.args, "sigreg_weight", 0.0)
        if (self._use_jepa or self._use_comparison) and sigreg_weight > 0:
            sigreg_n = getattr(self.args, "sigreg_projections", 64)
            sigreg_target = (
                outputs["z_pred"] if "z_pred" in outputs else outputs.get("beliefs")
            )
            if sigreg_target is not None:
                sigreg_loss = compute_sigreg_loss(
                    sigreg_target, outputs["slot_mask"], n_projections=sigreg_n
                )
                loss_dict["sigreg_loss"] = sigreg_loss
                loss_dict["loss"] = loss_dict["loss"] + sigreg_weight * sigreg_loss

        # --- Convergence loss (HuperJEPA v3) ---
        conv_weight = getattr(self.args, "convergence_loss_weight", 0.0)
        if self._use_comparison and conv_weight > 0 and "comparison_errors" in outputs:
            conv_loss = convergence_loss(outputs["comparison_errors"], sm)
            loss_dict["convergence_loss"] = conv_loss
            loss_dict["loss"] = loss_dict["loss"] + conv_weight * conv_loss

        # --- Word Distortion loss (HuperJEPA v2) ---
        dist_weight = getattr(self.args, "distortion_loss_weight", 0.0)
        if (
            dist_weight > 0
            and self.model.distortion_module is not None
            and "distortions" in outputs
        ):
            B, K = sm.shape
            # Build proportional oracle_emit from word count in text — Tier-3 heuristic.
            # This avoids a circular dependency on CTC alignment (Stage 2 labels).
            texts = batch.get("texts", [])
            oracle_emit = sm.new_zeros(B, K)  # float zeros
            for b_idx, txt in enumerate(texts):
                n_words = len(txt.strip().split()) if isinstance(txt, str) else 0
                n_slots = int(sm[b_idx].sum().item())
                if n_words == 0 or n_slots == 0:
                    continue
                step = n_slots / n_words
                last_slot = -2  # allow emit at slot 0
                for w in range(n_words):
                    raw = step * (w + 1) - 1
                    cand = int(round(raw))
                    cand = max(cand, last_slot + 2)   # min_gap=2
                    cand = min(cand, n_slots - 1)
                    oracle_emit[b_idx, cand] = 1.0
                    last_slot = cand

            distortion_loss = self.model.distortion_module.distortion_loss(
                outputs["distortions"], oracle_emit, sm
            )
            loss_dict["distortion_loss"] = distortion_loss
            loss_dict["loss"] = loss_dict["loss"] + dist_weight * distortion_loss

        return loss_dict


    def training_step(self, batch, batch_idx):
        if self._use_jepa:
            return self._training_step_jepa(batch, batch_idx)
        outputs = self._model_forward(batch)
        losses = self._compute_losses(batch, outputs)
        for k, v in losses.items():
            self.log(f"train_{k}", v, on_step=True, on_epoch=True, prog_bar=(k == "loss"), sync_dist=True)
        return losses["loss"]

    def _training_step_jepa(self, batch, batch_idx):
        """Manual optimisation step that adds EMA update for JEPA."""
        opt = self.optimizers()
        sch = self.lr_schedulers()

        outputs = self._model_forward(batch)
        losses = self._compute_losses(batch, outputs)

        self.manual_backward(losses["loss"])
        if self.args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.args.max_grad_norm
            )
        opt.step()
        opt.zero_grad()
        sch.step()

        # EMA tau schedule: cosine anneal from jepa_ema_tau → jepa_ema_tau_end,
        # matching Stage 1 behaviour.  A rising tau progressively stabilises the
        # target encoder (longer half-life), making JEPA prediction easier as
        # training matures — the key fix for the Stage-2 JEPA-loss plateau.
        tau_end = getattr(self.args, "jepa_ema_tau_end", None)
        if tau_end is not None:
            total = self.trainer.estimated_stepping_batches
            tau_s = self.args.jepa_ema_tau
            progress = min(1.0, self.global_step / max(total, 1))
            tau = tau_s + (tau_end - tau_s) * (1.0 - math.cos(math.pi * progress)) / 2.0
            self.model.update_ema(tau=tau)
            self.log("ema_tau", tau, on_step=True, sync_dist=True)
        else:
            self.model.update_ema()

        for k, v in losses.items():
            self.log(f"train_{k}", v, on_step=True, on_epoch=True,
                     prog_bar=(k == "loss"), sync_dist=True)
        if "z_pred" in outputs:
            std_val = check_collapse(outputs["z_pred"])
            self.log("predictor_std", std_val, on_step=True, sync_dist=True)


    def validation_step(self, batch, batch_idx):
        outputs = self._model_forward(batch)
        losses = self._compute_losses(batch, outputs)
        for k, v in losses.items():
            self.log(f"val_{k}", v, on_epoch=True, sync_dist=True, prog_bar=(k == "loss"))

    def configure_optimizers(self):
        import math as _math

        betas = (0.8, 0.99) if self._use_jepa else (0.9, 0.999)
        predictor_lr_mult = getattr(self.args, "jepa_predictor_lr_mult", 1.0)

        # When using JEPA with predictor_lr_mult > 1, give the predictor a
        # separate, higher-LR param group (matching Stage 1 design).
        if self._use_jepa and predictor_lr_mult != 1.0:
            predictor_params_set = set(
                self.model.jepa_predictor.parameters()
            )
            main_params = [
                p for p in self.model.parameters()
                if p.requires_grad and p not in predictor_params_set
            ]
            pred_params = [
                p for p in predictor_params_set if p.requires_grad
            ]
            if self.tts_decoder is not None:
                main_params += list(self.tts_decoder.parameters())
            param_groups = [
                {"params": main_params},
                {"params": pred_params, "lr": self.args.lr * predictor_lr_mult},
            ]
        else:
            params = [p for p in self.model.parameters() if p.requires_grad]
            if self.tts_decoder is not None:
                params += list(self.tts_decoder.parameters())
            param_groups = [{"params": params}]

        opt = AdamW(
            param_groups,
            lr=self.args.lr,
            betas=betas,
            weight_decay=self.args.weight_decay,
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup = min(self.args.warmup_steps, max(1, total_steps // 10))
        min_lr_ratio = getattr(self.args, "min_lr_ratio", 0.01)

        def lr_lambda(step):
            if step < warmup:
                return float(step) / max(1.0, float(warmup))
            progress = (step - warmup) / max(1.0, total_steps - warmup)
            cosine = 0.5 * (1.0 + _math.cos(_math.pi * progress))
            return max(min_lr_ratio, min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

        # One lambda per param group (predictor group inherits scaled base lr)
        n_groups = len(opt.param_groups)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, [lr_lambda] * n_groups)
        return [opt], [{"scheduler": sched, "interval": "step", "frequency": 1}]


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class FullEvalCallback(Callback):
    """Run detailed PER evaluation every N epochs."""

    def __init__(self, val_loader, phone_vocab, every_n_epochs, output_dir,
                 patience: int = 0):
        self.val_loader = val_loader
        self.phone_vocab = phone_vocab
        self.every = every_n_epochs
        self.output_dir = output_dir
        self.best_canonical_per = float("inf")
        self.patience = patience
        self._epochs_without_improve = 0
        self.history: list[dict] = []

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.every != 0 and epoch != trainer.max_epochs:
            return

        if trainer.is_global_zero:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            base_ckpt = {
                "model_state_dict": pl_module.model.state_dict(),
                "config": pl_module.model.config,
                "epoch": epoch,
            }
            if pl_module.tts_decoder is not None:
                base_ckpt["tts_decoder_state_dict"] = pl_module.tts_decoder.state_dict()
                base_ckpt["tts_config"] = pl_module.tts_decoder.cfg

            torch.save({**base_ckpt, "history": self.history},
                       self.output_dir / "last.pt")
            print(f"[epoch {epoch}] Saved last.pt", flush=True)

            metrics = evaluate_belief_wm(
                pl_module.model,
                self.val_loader,
                self.phone_vocab,
                pl_module.device,
            )
            record = {"epoch": epoch, **metrics}
            self.history.append(record)
            print(json.dumps(record, ensure_ascii=True), flush=True)

            base_ckpt["metrics"] = metrics
            torch.save({**base_ckpt, "history": self.history},
                       self.output_dir / "last.pt")

            if metrics["canonical_per"] < self.best_canonical_per:
                self.best_canonical_per = metrics["canonical_per"]
                self._epochs_without_improve = 0
                torch.save(base_ckpt, self.output_dir / "best.pt")
            else:
                self._epochs_without_improve += self.every
                if self.patience > 0 and self._epochs_without_improve >= self.patience:
                    print(
                        f"[epoch {epoch}] Early stopping: no improvement for "
                        f"{self._epochs_without_improve} epochs (patience={self.patience})",
                        flush=True,
                    )
                    trainer.should_stop = True

        if trainer.world_size > 1:
            trainer.strategy.barrier("eval_checkpoint")

    def on_fit_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            path = self.output_dir / "last.pt"
            final_ckpt = {
                "model_state_dict": pl_module.model.state_dict(),
                "config": pl_module.model.config,
                "epoch": trainer.current_epoch + 1,
                "history": self.history,
            }
            if pl_module.tts_decoder is not None:
                final_ckpt["tts_decoder_state_dict"] = pl_module.tts_decoder.state_dict()
                final_ckpt["tts_config"] = pl_module.tts_decoder.cfg
            torch.save(final_ckpt, path)
            write_json(self.output_dir / "eval_history.json", self.history)


class JSONLogCallback(Callback):
    def __init__(self, every_n_steps: int = 50):
        self.every = every_n_steps
        self._last = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step < 1 or step == self._last or step % self.every != 0:
            return
        if not trainer.is_global_zero:
            return
        self._last = step
        m = trainer.callback_metrics
        log_dict = {
            "step": step,
            "epoch": trainer.current_epoch,
            "loss": float(m.get("train_loss_step", 0)),
            "frame_phone": float(m.get("train_frame_phone_loss_step", 0)),
            "canonical": float(m.get("train_canonical_loss_step", 0)),
            "future": float(m.get("train_future_loss_step", 0)),
            "recon": float(m.get("train_recon_loss_step", 0)),
            "lr": trainer.optimizers[0].param_groups[0]["lr"],
        }
        if "train_diversity_loss_step" in m:
            log_dict["diversity"] = float(m["train_diversity_loss_step"])
        if "train_tts_flow_loss_step" in m:
            log_dict["tts_flow"] = float(m["train_tts_flow_loss_step"])
            log_dict["tts_dur"] = float(m.get("train_tts_dur_loss_step", 0))
        if "train_jepa_loss_step" in m:
            log_dict["jepa"] = float(m["train_jepa_loss_step"])
        if "predictor_std" in m:
            log_dict["pred_std"] = float(m["predictor_std"])
        print(json.dumps(log_dict), flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if not args.online_features and not args.features_dir:
        raise ValueError("--features-dir is required unless --online-features is set.")
    if args.online_features and args.num_workers > 0:
        print(
            "Warning: online feature extraction works best with --num-workers 0.",
            flush=True,
        )
    pl.seed_everything(args.seed, workers=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "train_args.json", vars(args))

    phone_vocab = Vocabulary.from_file(Path(args.metadata_dir) / "phone_vocab.json")
    text_vocab = Vocabulary.from_file(Path(args.metadata_dir) / "text_vocab.json")

    evidence_dim = 46 if args.evidence_type == "logits" else 1024
    config = BeliefWMConfig(
        evidence_dim=evidence_dim,
        hidden_dim=args.hidden_dim,
        phone_vocab_size=len(phone_vocab.tokens),
        belief_type=args.belief_type,
        pooling_type=args.pooling_type,
        upsample_factor=args.upsample_factor,
        dropout=args.dropout,
        use_identity=args.use_identity,
        identity_dim=args.identity_dim,
        use_prosody=args.use_prosody,
        prosody_dim=args.prosody_dim,
        use_uncertainty=args.use_uncertainty,
        uncertainty_dim=args.uncertainty_dim,
        use_mismatch=args.use_mismatch,
        mismatch_dim=args.mismatch_dim,
        detach_belief_for_frame_phone=False,
        belief_grad_scale=args.belief_grad_scale,
        frame_phone_dropout=args.frame_phone_dropout,
        canonical_head_dropout=getattr(args, "canonical_head_dropout", 0.0),
        jepa_encoder_layers=args.jepa_encoder_layers,
        jepa_encoder_heads=args.jepa_encoder_heads,
        jepa_encoder_ff_dim=args.jepa_encoder_ff_dim,
        jepa_encoder_conv_kernel=args.jepa_encoder_conv_kernel,
        jepa_daam_num_gaussians=args.jepa_daam_num_gaussians,
        jepa_daam_alpha_init=args.jepa_daam_alpha_init,
        jepa_predictor_layers=args.jepa_predictor_layers,
        jepa_predictor_heads=args.jepa_predictor_heads,
        jepa_prior_layers=args.jepa_prior_layers,
        jepa_prior_heads=args.jepa_prior_heads,
        jepa_ema_tau=args.jepa_ema_tau,
        jepa_mask_ratio=args.jepa_mask_ratio,
        jepa_mask_min_span=args.jepa_mask_min_span,
        jepa_mask_max_span=args.jepa_mask_max_span,
        boundary_attn_heads=args.boundary_attn_heads,
        use_distortion=args.use_distortion,
        distortion_loss_weight=args.distortion_loss_weight,
        distortion_init_threshold=args.distortion_init_threshold,
        num_refinements=args.num_refinements,
        refinement_heads=args.refinement_heads,
        refinement_ff_dim=args.refinement_ff_dim,
        refinement_conv_kernel=args.refinement_conv_kernel,
        convergence_loss_weight=args.convergence_loss_weight,
    )

    tts_decoder = None
    if args.use_tts:
        from wm_tts import FlowMatchingTTSDecoder, TTSConfig

        tts_cfg = TTSConfig(
            cond_dim=args.hidden_dim,
            identity_dim=args.identity_dim if args.use_identity else 128,
            decoder_dim=args.tts_decoder_dim,
            decoder_layers=args.tts_decoder_layers,
        )
        tts_decoder = FlowMatchingTTSDecoder(tts_cfg)
        if args.mel_stats_path:
            tts_decoder.mel_normalizer.load_stats(args.mel_stats_path)
            print(f"Loaded mel stats from {args.mel_stats_path}", flush=True)
        else:
            print(
                "Warning: --mel-stats-path not provided.  Mel normalization disabled, "
                "which may hurt TTS quality.",
                flush=True,
            )

    dm = BeliefWMDataModule(args, phone_vocab, text_vocab, _maybe_teacher_cache(args))
    lit = BeliefWMLitModule(config, phone_vocab, args, tts_decoder=tts_decoder)

    # --- Stage 1 JEPA checkpoint → initialise encoder weights ---
    if getattr(args, "stage1_checkpoint", None) and args.belief_type == "jepa":
        s1_ckpt = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=False)
        s1_state = s1_ckpt["model_state_dict"]
        current = lit.model.state_dict()
        loaded, skipped = [], []
        for name, param in s1_state.items():
            if name in current and current[name].shape == param.shape:
                current[name].copy_(param)
                loaded.append(name)
            else:
                skipped.append(name)
        lit.model.load_state_dict(current)
        print(
            f"Loaded Stage 1 JEPA weights from {args.stage1_checkpoint}: "
            f"{len(loaded)} params loaded, {len(skipped)} re-initialised.",
            flush=True,
        )
        if skipped:
            print(f"  Skipped: {skipped[:10]}{'...' if len(skipped) > 10 else ''}", flush=True)

    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        saved = ckpt["model_state_dict"]
        current = lit.model.state_dict()
        loaded, skipped = [], []
        for name, param in saved.items():
            if name in current and current[name].shape == param.shape:
                current[name].copy_(param)
                loaded.append(name)
            else:
                skipped.append(name)
        new_params = [n for n in current if n not in saved]
        lit.model.load_state_dict(current)
        if skipped or new_params:
            print(
                f"Partial resume: loaded {len(loaded)} params, "
                f"skipped {len(skipped)} (shape mismatch), "
                f"{len(new_params)} new params randomly initialised.",
                flush=True,
            )
            if skipped:
                print(f"  Skipped: {skipped}", flush=True)
            if new_params:
                print(f"  New: {new_params}", flush=True)
        else:
            print(f"Full resume: all {len(loaded)} params loaded.", flush=True)

        if tts_decoder is not None and "tts_decoder_state_dict" in ckpt:
            try:
                tts_decoder.load_state_dict(ckpt["tts_decoder_state_dict"])
                print(f"Restored TTS decoder weights from {args.resume_from}", flush=True)
            except RuntimeError as e:
                print(
                    f"TTS decoder architecture changed — training from scratch. ({e})",
                    flush=True,
                )

        resumed_epoch = ckpt.get("epoch", "?")
        print(
            f"Resumed model weights from {args.resume_from} "
            f"(saved at epoch {resumed_epoch}). "
            f"Optimizer and LR schedule restart from step 0.",
            flush=True,
        )

    val_loader = dm.val_dataloader()

    num_devices = args.devices or max(1, torch.cuda.device_count())
    strategy = "auto"
    if num_devices > 1:
        from lightning.pytorch.strategies import DDPStrategy

        strategy = DDPStrategy(find_unused_parameters=True)

    callbacks = [
        FullEvalCallback(val_loader, phone_vocab, args.eval_every_epochs, output_dir,
                         patience=getattr(args, "patience", 0)),
        JSONLogCallback(every_n_steps=args.log_every_steps),
        LearningRateMonitor(logging_interval="step"),
    ]

    csv_logger = CSVLogger(save_dir=str(output_dir), name="csv_logs")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=num_devices,
        strategy=strategy,
        gradient_clip_val=0 if args.belief_type == "jepa" else args.max_grad_norm,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=csv_logger,
        log_every_n_steps=args.log_every_steps,
        enable_progress_bar=True,
        default_root_dir=str(output_dir),
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        precision=args.precision,
        check_val_every_n_epoch=args.eval_every_epochs,
    )

    trainer.fit(lit, datamodule=dm)
    print(f"Training complete → {output_dir}")


if __name__ == "__main__":
    main()
