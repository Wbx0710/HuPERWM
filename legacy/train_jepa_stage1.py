"""Stage 1: Self-supervised JEPA pretraining on syllable slots.

Trains the online encoder (Conformer+DAAM) and predictor using masked
slot prediction in latent space.  No CTC labels or TTS targets are
required — only HuPER evidence and Sylber boundaries.

The EMA target encoder provides stable prediction targets and is
updated after each optimiser step.

Usage:
    conda activate phn
    python train_jepa_stage1.py \
        --features-dir artifacts/wm_features_librispeech \
        --metadata-dir artifacts/metadata_librispeech \
        --output-dir runs/jepa_stage1 \
        --evidence-type logits \
        --epochs 50 --batch-size 64 --lr 1.5e-4
"""

from __future__ import annotations

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
)
from wm_jepa import check_collapse, compute_jepa_loss, compute_vicreg_loss
from wm_online_data import OnlineLibriSpeechWMDataset
from wm_teacher import load_teacher_phone_cache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1: JEPA self-supervised pretraining.")
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
        "--teacher-cache", nargs="*", default=None,
        help="Optional teacher cache for dataset compatibility.",
    )

    p.add_argument("--evidence-type", choices=["logits", "hidden"], default="logits")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--pooling-type", choices=["mean", "attention"], default="mean")
    p.add_argument("--dropout", type=float, default=0.1)

    # JEPA architecture
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
    p.add_argument("--jepa-ema-tau", type=float, default=0.996,
                   help="EMA momentum start value (following I-JEPA, >= 0.996).")
    p.add_argument("--jepa-ema-tau-end", type=float, default=0.9999,
                   help="EMA momentum end value (cosine schedule from tau to tau-end).")
    p.add_argument("--jepa-mask-ratio", type=float, default=0.5)
    p.add_argument("--jepa-mask-min-span", type=int, default=1)
    p.add_argument("--jepa-mask-max-span", type=int, default=None)

    # VICReg regularization (prevents collapse after L2-normalized loss)
    p.add_argument("--vicreg-weight", type=float, default=0.01,
                   help="Weight for VICReg (variance+covariance) regularization.")
    p.add_argument("--vicreg-var-gamma", type=float, default=1.0,
                   help="Target per-dimension std for VICReg variance hinge.")
    p.add_argument("--vicreg-warmup-steps", type=int, default=500,
                   help="Linear warmup steps for VICReg (avoid disrupting early prediction learning).")

    # Predictor learning rate
    p.add_argument("--predictor-lr-mult", type=float, default=2.0,
                   help="LR multiplier for predictor (faster adaptation to moving target).")

    # Training
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=500)
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
    p.add_argument("--collapse-threshold", type=float, default=0.01)
    return p.parse_args()


# ---------------------------------------------------------------------------
# DataModule (reuses existing dataset infrastructure)
# ---------------------------------------------------------------------------


class Stage1DataModule(pl.LightningDataModule):
    def __init__(self, args, phone_vocab, text_vocab, teacher_cache=None):
        super().__init__()
        self.args = args
        self.phone_vocab = phone_vocab
        self.text_vocab = text_vocab
        self.teacher_cache = teacher_cache

    def _make_ds(self, split_name, hf_split, max_examples):
        if self.args.online_features:
            return OnlineLibriSpeechWMDataset(
                hf_dataset_name=self.args.hf_dataset_name,
                hf_dataset_config=self.args.hf_dataset_config,
                hf_split=hf_split,
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
            split_name,
            self.args.metadata_dir,
            self.phone_vocab,
            self.text_vocab,
            evidence_type=self.args.evidence_type,
            max_examples=max_examples,
            teacher_cache=self.teacher_cache,
        )

    def train_dataloader(self):
        ds = self._make_ds("train", self.args.hf_train_split, self.args.max_train_examples)
        return DataLoader(
            ds, batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.num_workers, collate_fn=BeliefWMCollator(),
            pin_memory=True,
        )

    def val_dataloader(self):
        ds = self._make_ds("validation", self.args.hf_val_split, self.args.max_val_examples)
        return DataLoader(
            ds, batch_size=self.args.eval_batch_size, shuffle=False,
            num_workers=self.args.num_workers, collate_fn=BeliefWMCollator(),
            pin_memory=True,
        )


# ---------------------------------------------------------------------------
# Lightning Module for Stage 1
# ---------------------------------------------------------------------------


class JEPAStage1LitModule(pl.LightningModule):
    def __init__(self, config: BeliefWMConfig, args):
        super().__init__()
        self.save_hyperparameters(ignore=[])
        self.model = BeliefWorldModel(config)
        self.args = args
        self.collapse_threshold = args.collapse_threshold
        self.automatic_optimization = False

    # ---- schedules ----------------------------------------------------------

    def _get_ema_tau(self) -> float:
        """Cosine EMA momentum schedule: tau_start → tau_end over full training.

        Following I-JEPA (Assran et al., 2023), tau starts high (e.g. 0.996)
        and increases to tau_end (e.g. 0.9999) via cosine annealing.
        No warmup-from-zero — that makes early targets trivially easy and
        causes a regime shift once tau reaches operational range.
        """
        total = self.trainer.estimated_stepping_batches
        tau_s = self.args.jepa_ema_tau
        tau_e = self.args.jepa_ema_tau_end

        progress = self.global_step / max(total, 1)
        progress = min(progress, 1.0)
        return tau_s + (tau_e - tau_s) * (1.0 - math.cos(math.pi * progress)) / 2.0

    def _get_vicreg_weight(self) -> float:
        """Linear warmup for VICReg to avoid disrupting early prediction."""
        warmup = self.args.vicreg_warmup_steps
        if warmup > 0 and self.global_step < warmup:
            return self.args.vicreg_weight * (self.global_step / warmup)
        return self.args.vicreg_weight

    # ---- forward / loss -----------------------------------------------------

    def _forward_jepa(self, batch):
        return self.model(
            batch["evidence"],
            batch["boundaries"],
            batch["slot_mask"],
            batch["num_frames"],
            frame_mask=batch.get("frame_mask"),
            compute_jepa_loss=True,
            jepa_only=True,
        )

    def _jepa_loss(self, outputs):
        """Compute JEPA prediction loss + scheduled VICReg regularization.

        Returns (total_loss, jepa_mse, vicreg_loss) for logging.
        """
        zero = outputs["beliefs"].new_zeros((), requires_grad=True)
        if "z_pred" not in outputs or "z_target" not in outputs:
            return zero, zero, zero

        jepa_mse = compute_jepa_loss(
            outputs["z_pred"],
            outputs["z_target"],
            outputs["jepa_mask"],
            outputs["slot_mask"],
        )

        var_loss, cov_loss = compute_vicreg_loss(
            outputs["z_pred"],
            outputs["slot_mask"],
            gamma=self.args.vicreg_var_gamma,
        )
        vicreg = var_loss + cov_loss
        w_vicreg = self._get_vicreg_weight()
        total = jepa_mse + w_vicreg * vicreg
        return total, jepa_mse, vicreg

    # ---- training / validation ----------------------------------------------

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()

        outputs = self._forward_jepa(batch)
        loss, jepa_mse, vicreg = self._jepa_loss(outputs)

        self.manual_backward(loss)

        if self.args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.max_grad_norm
            )
        opt.step()
        opt.zero_grad()
        sch.step()

        ema_tau = self._get_ema_tau()
        self.model.update_ema(tau=ema_tau)

        self.log("train_jepa_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log("train_jepa_mse", jepa_mse, on_step=True, sync_dist=True)
        self.log("train_vicreg", vicreg, on_step=True, sync_dist=True)
        self.log("ema_tau", ema_tau, on_step=True, sync_dist=True)

        if "z_pred" in outputs:
            std_val = check_collapse(outputs["z_pred"], self.collapse_threshold)
            self.log("predictor_std", std_val, on_step=True, sync_dist=True)
            if std_val < self.collapse_threshold:
                self.print(
                    f"[WARNING step={self.global_step}] "
                    f"Possible collapse: predictor_std={std_val:.4f}"
                )

    def validation_step(self, batch, batch_idx):
        outputs = self._forward_jepa(batch)
        loss, jepa_mse, vicreg = self._jepa_loss(outputs)
        self.log("val_jepa_loss", loss, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_jepa_mse", jepa_mse, on_epoch=True, sync_dist=True)
        self.log("val_vicreg", vicreg, on_epoch=True, sync_dist=True)

        if "z_pred" in outputs:
            std_val = check_collapse(outputs["z_pred"])
            self.log("val_predictor_std", std_val, on_epoch=True, sync_dist=True)

    # ---- optimizer with separate predictor LR -------------------------------

    def configure_optimizers(self):
        predictor_params = list(self.model.jepa_predictor.parameters())
        predictor_ids = {id(p) for p in predictor_params}
        other_params = [
            p for p in self.model.parameters()
            if p.requires_grad and id(p) not in predictor_ids
        ]
        opt = AdamW(
            [
                {"params": other_params},
                {"params": predictor_params, "lr": self.args.lr * self.args.predictor_lr_mult},
            ],
            lr=self.args.lr,
            betas=(0.9, 0.95),
            weight_decay=self.args.weight_decay,
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup = min(self.args.warmup_steps, max(1, total_steps // 10))
        min_lr_ratio = 0.05

        def lr_lambda(step):
            if step < warmup:
                return float(step) / max(1.0, float(warmup))
            progress = (step - warmup) / max(1.0, total_steps - warmup)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr_ratio, min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

        sched = torch.optim.lr_scheduler.LambdaLR(opt, [lr_lambda, lr_lambda])
        return [opt], [{"scheduler": sched, "interval": "step", "frequency": 1}]


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class Stage1CheckpointCallback(Callback):
    def __init__(self, output_dir: Path, every_n_epochs: int = 5):
        self.output_dir = output_dir
        self.every = every_n_epochs
        self.best_val_mse = float("inf")
        self.history: list[dict] = []

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.every != 0 and epoch != trainer.max_epochs:
            return
        if not trainer.is_global_zero:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        m = trainer.callback_metrics
        val_mse = float(m.get("val_jepa_mse", float("inf")))
        val_loss = float(m.get("val_jepa_loss", float("inf")))
        val_std = float(m.get("val_predictor_std", 0.0))
        val_vicreg = float(m.get("val_vicreg", 0.0))

        record = {
            "epoch": epoch,
            "val_jepa_mse": val_mse,
            "val_jepa_loss": val_loss,
            "val_predictor_std": val_std,
            "val_vicreg": val_vicreg,
        }
        self.history.append(record)
        print(json.dumps(record, ensure_ascii=True), flush=True)

        ckpt = {
            "model_state_dict": pl_module.model.state_dict(),
            "config": pl_module.model.config,
            "epoch": epoch,
            "history": self.history,
        }
        torch.save(ckpt, self.output_dir / "last_stage1.pt")

        if val_mse < self.best_val_mse:
            self.best_val_mse = val_mse
            torch.save(ckpt, self.output_dir / "best_stage1.pt")
            print(f"  [epoch {epoch}] New best val_jepa_mse={val_mse:.6f}", flush=True)

    def on_fit_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            write_json(self.output_dir / "stage1_history.json", self.history)


class Stage1LogCallback(Callback):
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
            "jepa_loss": float(m.get("train_jepa_loss_step", 0)),
            "jepa_mse": float(m.get("train_jepa_mse", 0)),
            "vicreg": float(m.get("train_vicreg", 0)),
            "predictor_std": float(m.get("predictor_std", 0)),
            "ema_tau": float(m.get("ema_tau", 0)),
            "lr": trainer.optimizers[0].param_groups[0]["lr"],
            "pred_lr": trainer.optimizers[0].param_groups[1]["lr"],
        }
        print(json.dumps(log_dict), flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if not args.online_features and not args.features_dir:
        raise ValueError("--features-dir is required unless --online-features is set.")
    pl.seed_everything(args.seed, workers=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage1_args.json", vars(args))

    phone_vocab = Vocabulary.from_file(Path(args.metadata_dir) / "phone_vocab.json")
    text_vocab = Vocabulary.from_file(Path(args.metadata_dir) / "text_vocab.json")

    evidence_dim = 46 if args.evidence_type == "logits" else 1024
    config = BeliefWMConfig(
        evidence_dim=evidence_dim,
        hidden_dim=args.hidden_dim,
        phone_vocab_size=len(phone_vocab.tokens),
        belief_type="jepa",
        pooling_type=args.pooling_type,
        dropout=args.dropout,
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
    )

    teacher_cache = None
    if args.teacher_cache:
        teacher_cache = load_teacher_phone_cache(args.teacher_cache)

    dm = Stage1DataModule(args, phone_vocab, text_vocab, teacher_cache)
    lit = JEPAStage1LitModule(config, args)

    num_devices = args.devices or max(1, torch.cuda.device_count())
    strategy = "auto"
    if num_devices > 1:
        from lightning.pytorch.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=True)

    callbacks = [
        Stage1CheckpointCallback(output_dir, args.eval_every_epochs),
        Stage1LogCallback(every_n_steps=args.log_every_steps),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=num_devices,
        strategy=strategy,
        gradient_clip_val=0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
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
    print(f"Stage 1 training complete → {output_dir}")


if __name__ == "__main__":
    main()
