"""Evaluate a trained Belief World Model checkpoint and produce figures.

Usage:
    conda activate phn
    python eval_wm_belief.py \
        --features-dir artifacts/wm_features_xs \
        --metadata-dir artifacts/metadata_xs \
        --checkpoint runs/wm_belief/best.pt \
        --output-dir runs/wm_belief/eval
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from wm_common import Vocabulary, write_json, write_jsonl
from wm_core import (
    BeliefWMCollator,
    BeliefWMConfig,
    BeliefWMDataset,
    BeliefWorldModel,
    evaluate_belief_wm,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Belief World Model.")
    p.add_argument("--features-dir", required=True)
    p.add_argument("--metadata-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--evidence-type", choices=["logits", "hidden"], default="logits")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def render_metric_bar(path: Path, metrics: dict) -> None:
    items = [
        ("Canonical PER", metrics.get("canonical_per", math.nan)),
        ("Teacher PER", metrics.get("teacher_per", math.nan)),
        ("Future MSE", metrics.get("future_mse", math.nan)),
        ("Recon MSE", metrics.get("recon_mse", math.nan)),
        ("Belief Cos", metrics.get("belief_evolution_cosine", math.nan)),
    ]
    items = [(n, v) for n, v in items if not math.isnan(v)]
    if not items:
        return
    names, vals = zip(*items)

    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    colours = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    bars = ax.bar(names, vals, color=colours[: len(vals)], width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Value")
    ax.set_title("Belief World Model — Evaluation Metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def render_history(path: Path, history: list[dict]) -> None:
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    canon = [h.get("canonical_per", math.nan) for h in history]
    teacher = [h.get("teacher_per", math.nan) for h in history]

    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)

    axes[0].plot(epochs, canon, "o-", color="#4C78A8", linewidth=2, label="Canonical PER")
    axes[0].plot(epochs, teacher, "s--", color="#F58518", linewidth=1.5, label="Teacher PER")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Phone Error Rate")
    axes[0].set_title("a  PER over training", loc="left")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    future = [h.get("future_mse", math.nan) for h in history]
    recon = [h.get("recon_mse", math.nan) for h in history]
    axes[1].plot(epochs, future, "o-", color="#54A24B", linewidth=2, label="Future MSE")
    axes[1].plot(epochs, recon, "s--", color="#E45756", linewidth=1.5, label="Recon MSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("b  Prediction losses", loc="left")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Belief World Model Training Curves", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_csv(path: Path, metrics: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config: BeliefWMConfig = ckpt["config"]

    phone_vocab = Vocabulary.from_file(Path(args.metadata_dir) / "phone_vocab.json")
    text_vocab = Vocabulary.from_file(Path(args.metadata_dir) / "text_vocab.json")

    ds = BeliefWMDataset(
        args.features_dir,
        args.split,
        args.metadata_dir,
        phone_vocab,
        text_vocab,
        evidence_type=args.evidence_type,
        max_examples=args.max_examples,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=BeliefWMCollator(),
        pin_memory=True,
    )

    model = BeliefWorldModel(config)
    state = ckpt["model_state_dict"]
    current = model.state_dict()
    loaded = 0
    for name, param in state.items():
        if name in current and current[name].shape == param.shape:
            current[name].copy_(param)
            loaded += 1
    model.load_state_dict(current)
    if loaded < len(state):
        print(f"Partial load: {loaded}/{len(state)} params matched "
              f"({len(current) - loaded} re-initialised)")
    model.to(device)

    belief_type = getattr(config, "belief_type", "gru")
    print(f"Model belief_type: {belief_type}")

    metrics = evaluate_belief_wm(model, loader, phone_vocab, device)
    full_metrics = {
        "split": args.split, "checkpoint": args.checkpoint,
        "belief_type": belief_type, **metrics,
    }

    write_json(output_dir / "metrics.json", full_metrics)
    save_csv(output_dir / "metrics.csv", full_metrics)
    render_metric_bar(output_dir / "metrics_bar.png", metrics)
    print(json.dumps(full_metrics, indent=2, ensure_ascii=True))

    history = ckpt.get("history", [])
    if history:
        render_history(output_dir / "training_curves.png", history)

    print(f"Results → {output_dir}")


if __name__ == "__main__":
    main()
