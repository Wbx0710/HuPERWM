"""Train the Belief World Model (Stage 2) with PyTorch Lightning.

Usage:
    python train_world_model.py \\
        --features-dir /data/wm_features \\
        --metadata-dir /data/metadata \\
        --output-dir runs/world_model_v3 \\
        --evidence-type hidden \\
        --epochs 150

Multi-GPU (DDP):
    torchrun --nproc_per_node=4 train_world_model.py ...
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

from huperwm.data.vocab import Vocabulary, write_json
from huperwm.data.world_model import (
    BeliefWMCollator,
    BeliefWMDataset,
    compute_ctc_loss,
    evaluate_belief_wm,
)
from huperwm.data.online import OnlineLibriSpeechWMDataset
from huperwm.model.conformer import compute_sigreg_loss
from huperwm.model.encoder import convergence_loss
from huperwm.model.world_model import BeliefWorldModel, WorldModelConfig
from huperwm.teacher import load_teacher_phone_cache


def _maybe_teacher_cache(args) -> dict | None:
    if not args.teacher_cache:
        return None
    return load_teacher_phone_cache(args.teacher_cache)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Belief World Model (v3).")
    # Data
    p.add_argument("--features-dir", default=None)
    p.add_argument("--metadata-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--online-features", action="store_true")
    p.add_argument("--hf-dataset-name", default="openslr/librispeech_asr")
    p.add_argument("--hf-dataset-config", default="all")
    p.add_argument("--hf-train-split", default="train.clean.360")
    p.add_argument("--hf-val-split", default="validation.clean")
    p.add_argument("--huper-repo", default="huper29/huper_recognizer")
    p.add_argument("--feature-device", default="cuda")
    p.add_argument("--teacher-cache", nargs="*", default=None)
    p.add_argument("--evidence-type", choices=["logits", "hidden"], default="hidden",
                   help="'hidden' = 1024-dim HuPER last hidden state (recommended). "
                        "'logits' = 46-dim CTC logits (legacy).")

    # Model
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--upsample-factor", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--boundary-attn-heads", type=int, default=4)
    # Prior encoder
    p.add_argument("--prior-layers", type=int, default=3)
    p.add_argument("--prior-heads", type=int, default=8)
    p.add_argument("--prior-ff-dim", type=int, default=1024)
    p.add_argument("--prior-conv-kernel", type=int, default=15)
    # Comparison refinement
    p.add_argument("--num-refinements", type=int, default=2)
    p.add_argument("--refinement-heads", type=int, default=4)
    p.add_argument("--refinement-ff-dim", type=int, default=512)
    p.add_argument("--refinement-conv-kernel", type=int, default=15)
    # Frame-phone head
    p.add_argument("--belief-grad-scale", type=float, default=0.1)
    p.add_argument("--frame-phone-dropout", type=float, default=0.1)
    p.add_argument("--canonical-head-dropout", type=float, default=0.0)

    # Loss weights
    p.add_argument("--frame-phone-weight", type=float, default=1.0)
    p.add_argument("--evidence-phone-weight", type=float, default=0.5)
    p.add_argument("--canonical-weight", type=float, default=0.5)
    p.add_argument("--future-weight", type=float, default=0.3)
    p.add_argument("--recon-weight", type=float, default=0.3)
    p.add_argument("--convergence-loss-weight", type=float, default=0.2)
    p.add_argument("--sigreg-weight", type=float, default=0.0,
                   help="SIGReg anti-collapse regularization weight (0.05 recommended).")
    p.add_argument("--sigreg-projections", type=int, default=64)
    p.add_argument("--diversity-weight", type=float, default=0.0)
    p.add_argument("--diversity-hinge", type=float, default=0.8)

    # Training
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--min-lr-ratio", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=5.0)
    p.add_argument("--accumulate-grad-batches", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-examples", type=int, default=None)
    p.add_argument("--max-val-examples", type=int, default=None)
    p.add_argument("--devices", type=int, default=None)
    p.add_argument("--precision", type=str, default="32-true")
    p.add_argument("--eval-every-epochs", type=int, default=5)
    p.add_argument("--log-every-steps", type=int, default=10)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--resume-from", type=str, default=None,
                   help="Path to a best.pt / last.pt checkpoint (weights only; optimizer restarts).")
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

    def _make_dataset(self, split: str, max_examples):
        if self.args.online_features:
            return OnlineLibriSpeechWMDataset(
                hf_dataset_name=self.args.hf_dataset_name,
                hf_dataset_config=self.args.hf_dataset_config,
                hf_split=self.args.hf_train_split if split == "train" else self.args.hf_val_split,
                phone_vocab=self.phone_vocab,
                text_vocab=self.text_vocab,
                evidence_type=self.args.evidence_type,
                huper_repo=self.args.huper_repo,
                feature_device=self.args.feature_device,
                max_examples=max_examples,
                teacher_cache=self.teacher_cache,
            )
        return BeliefWMDataset(
            self.args.features_dir,
            split,
            self.args.metadata_dir,
            self.phone_vocab,
            self.text_vocab,
            evidence_type=self.args.evidence_type,
            max_examples=max_examples,
            teacher_cache=self.teacher_cache,
        )

    def train_dataloader(self):
        return DataLoader(
            self._make_dataset("train", self.args.max_train_examples),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=BeliefWMCollator(),
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self._make_dataset("validation", self.args.max_val_examples),
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=BeliefWMCollator(),
            pin_memory=True,
        )


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------


class WorldModelLitModule(pl.LightningModule):
    def __init__(self, config: WorldModelConfig, phone_vocab: Vocabulary, args):
        super().__init__()
        self.save_hyperparameters(ignore=["phone_vocab"])
        self.model = BeliefWorldModel(config)
        self.phone_vocab = phone_vocab
        self.args = args

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
        H = outputs["future_pred"].shape[-1]

        frame_phone_loss = compute_ctc_loss(
            outputs["frame_phone_logits"], nf, batch["teacher_ids"], self.phone_vocab.blank_id,
        )
        evidence_phone_loss = compute_ctc_loss(
            outputs["evidence_phone_logits"], up_lengths, batch["teacher_ids"], self.phone_vocab.blank_id,
        )
        canonical_loss = compute_ctc_loss(
            outputs["canonical_logits"], up_lengths, batch["canonical_ids"], self.phone_vocab.blank_id,
        )

        future_loss = ev.new_zeros(())
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

        # Belief temporal diversity loss.
        if outputs["beliefs"].shape[1] > 1:
            beliefs = outputs["beliefs"]
            b_prev = F.normalize(beliefs[:, :-1], dim=-1)
            b_next = F.normalize(beliefs[:, 1:], dim=-1)
            adj_mask = sm[:, 1:]
            cos_sim = (b_prev * b_next).sum(dim=-1)
            mean_cos = (cos_sim * adj_mask).sum() / adj_mask.sum().clamp_min(1.0)
            loss_dict["belief_mean_cosine"] = mean_cos
            div_weight = getattr(self.args, "diversity_weight", 0.0)
            if div_weight > 0:
                hinge = getattr(self.args, "diversity_hinge", 0.8)
                diversity_loss = (
                    F.relu(cos_sim.abs() - hinge) * adj_mask
                ).sum() / adj_mask.sum().clamp_min(1.0)
                loss_dict["diversity_loss"] = diversity_loss
                loss_dict["loss"] = loss_dict["loss"] + div_weight * diversity_loss

        # SIGReg anti-collapse regularization.
        sigreg_weight = getattr(self.args, "sigreg_weight", 0.0)
        if sigreg_weight > 0:
            sigreg_n = getattr(self.args, "sigreg_projections", 64)
            sigreg_loss = compute_sigreg_loss(outputs["beliefs"], sm, n_projections=sigreg_n)
            loss_dict["sigreg_loss"] = sigreg_loss
            loss_dict["loss"] = loss_dict["loss"] + sigreg_weight * sigreg_loss

        # Convergence loss: error should decrease across refinement iterations.
        conv_weight = getattr(self.args, "convergence_loss_weight", 0.0)
        if conv_weight > 0 and "comparison_errors" in outputs:
            conv_loss = convergence_loss(outputs["comparison_errors"], sm)
            loss_dict["convergence_loss"] = conv_loss
            loss_dict["loss"] = loss_dict["loss"] + conv_weight * conv_loss

        return loss_dict

    def training_step(self, batch, batch_idx):
        outputs = self._model_forward(batch)
        losses = self._compute_losses(batch, outputs)
        for k, v in losses.items():
            self.log(f"train_{k}", v, on_step=True, on_epoch=True,
                     prog_bar=(k == "loss"), sync_dist=True)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self._model_forward(batch)
        losses = self._compute_losses(batch, outputs)
        for k, v in losses.items():
            self.log(f"val_{k}", v, on_epoch=True, sync_dist=True, prog_bar=(k == "loss"))

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = AdamW(params, lr=self.args.lr, betas=(0.9, 0.999), weight_decay=self.args.weight_decay)
        total_steps = self.trainer.estimated_stepping_batches
        warmup = min(self.args.warmup_steps, max(1, total_steps // 10))
        min_lr_ratio = getattr(self.args, "min_lr_ratio", 0.01)

        def lr_lambda(step):
            if step < warmup:
                return float(step) / max(1.0, float(warmup))
            progress = (step - warmup) / max(1.0, total_steps - warmup)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr_ratio, min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return [opt], [{"scheduler": sched, "interval": "step", "frequency": 1}]


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class FullEvalCallback(Callback):
    def __init__(self, val_loader, phone_vocab, every_n_epochs, output_dir, patience=0):
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
        if not trainer.is_global_zero:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        base_ckpt = {
            "model_state_dict": pl_module.model.state_dict(),
            "config": pl_module.model.config,
            "epoch": epoch,
        }
        torch.save({**base_ckpt, "history": self.history}, self.output_dir / "last.pt")

        metrics = evaluate_belief_wm(pl_module.model, self.val_loader, self.phone_vocab, pl_module.device)
        record = {"epoch": epoch, **metrics}
        self.history.append(record)
        print(json.dumps(record, ensure_ascii=True), flush=True)

        base_ckpt["metrics"] = metrics
        torch.save({**base_ckpt, "history": self.history}, self.output_dir / "last.pt")

        if metrics["canonical_per"] < self.best_canonical_per:
            self.best_canonical_per = metrics["canonical_per"]
            self._epochs_without_improve = 0
            torch.save(base_ckpt, self.output_dir / "best.pt")
        else:
            self._epochs_without_improve += self.every
            if self.patience > 0 and self._epochs_without_improve >= self.patience:
                print(f"[epoch {epoch}] Early stopping after {self._epochs_without_improve} epochs.", flush=True)
                trainer.should_stop = True

        if trainer.world_size > 1:
            trainer.strategy.barrier("eval_checkpoint")

    def on_fit_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            final_ckpt = {
                "model_state_dict": pl_module.model.state_dict(),
                "config": pl_module.model.config,
                "epoch": trainer.current_epoch + 1,
                "history": self.history,
            }
            torch.save(final_ckpt, self.output_dir / "last.pt")
            write_json(self.output_dir / "eval_history.json", self.history)


class JSONLogCallback(Callback):
    def __init__(self, every_n_steps=50):
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
        for extra in ("sigreg_loss", "convergence_loss", "diversity_loss"):
            key = f"train_{extra}_step"
            if key in m:
                log_dict[extra] = float(m[key])
        print(json.dumps(log_dict), flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if not args.online_features and not args.features_dir:
        raise ValueError("--features-dir is required unless --online-features is set.")
    if args.online_features and args.num_workers > 0:
        print("Warning: online feature extraction works best with --num-workers 0.", flush=True)

    pl.seed_everything(args.seed, workers=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "train_args.json", vars(args))

    phone_vocab = Vocabulary.from_file(Path(args.metadata_dir) / "phone_vocab.json")
    text_vocab = Vocabulary.from_file(Path(args.metadata_dir) / "text_vocab.json")

    evidence_dim = 46 if args.evidence_type == "logits" else 1024
    config = WorldModelConfig(
        evidence_dim=evidence_dim,
        hidden_dim=args.hidden_dim,
        phone_vocab_size=len(phone_vocab.tokens),
        upsample_factor=args.upsample_factor,
        dropout=args.dropout,
        boundary_attn_heads=args.boundary_attn_heads,
        prior_layers=args.prior_layers,
        prior_heads=args.prior_heads,
        prior_ff_dim=args.prior_ff_dim,
        prior_conv_kernel=args.prior_conv_kernel,
        num_refinements=args.num_refinements,
        refinement_heads=args.refinement_heads,
        refinement_ff_dim=args.refinement_ff_dim,
        refinement_conv_kernel=args.refinement_conv_kernel,
        convergence_loss_weight=args.convergence_loss_weight,
        belief_grad_scale=args.belief_grad_scale,
        frame_phone_dropout=args.frame_phone_dropout,
        canonical_head_dropout=args.canonical_head_dropout,
    )

    dm = BeliefWMDataModule(args, phone_vocab, text_vocab, _maybe_teacher_cache(args))
    lit = WorldModelLitModule(config, phone_vocab, args)

    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        lit.model.load_state_dict(ckpt["model_state_dict"])
        print(f"Resumed from {args.resume_from}", flush=True)

    val_loader = dm.val_dataloader()
    num_devices = args.devices or max(1, torch.cuda.device_count())
    strategy = "auto"
    if num_devices > 1:
        from lightning.pytorch.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=False)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=num_devices,
        strategy=strategy,
        gradient_clip_val=args.max_grad_norm,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[
            FullEvalCallback(val_loader, phone_vocab, args.eval_every_epochs, output_dir,
                             patience=args.patience),
            JSONLogCallback(every_n_steps=args.log_every_steps),
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=CSVLogger(save_dir=str(output_dir), name="csv_logs"),
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
