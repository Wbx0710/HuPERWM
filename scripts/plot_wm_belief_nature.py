#!/usr/bin/env python3
"""Figures from eval_history.json using a Nature-*inspired* look.

We follow Nature-*like* **colour palette** and **layout** (clean spines, muted
colours, legible type, legend away from data). We do **not** aim for a strict
journal submission template (exact mm widths etc.); defaults here are a
reasonable single-column *starting point* (~89 mm) at 300 dpi.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def nature_rc():
    # Single-column-ish width (~89 mm) + Nature-like palette; adjust freely.
    mm = 1 / 25.4
    col_w = 89 * mm
    mpl.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return col_w


def load_history(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def plot_figure1(history: list[dict], out_dir: Path, col_w: float) -> None:
    """Two panels: canonical PER; future & recon MSE."""
    epochs = np.array([h["epoch"] for h in history], dtype=float)
    canon = np.array([h["canonical_per"] for h in history])
    future = np.array([h["future_mse"] for h in history])
    recon = np.array([h["recon_mse"] for h in history])

    # Nature-ish palette (colorblind-conscious, muted)
    c1, c2, c3 = "#0072B2", "#D55E00", "#009E73"

    fig_h = col_w * 0.95
    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(col_w, fig_h),
        sharex=True,
        # Leave right margin for legend (Nature single-column still tolerates this).
        gridspec_kw={"hspace": 0.12, "left": 0.18, "right": 0.83, "top": 0.94, "bottom": 0.14},
    )

    ax0.plot(epochs, canon, "o-", color=c1, markersize=3, clip_on=False)
    ax0.set_ylabel("Canonical PER")
    ax0.text(
        -0.22,
        1.02,
        "a",
        transform=ax0.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
    )

    ax1.plot(epochs, future, "s-", color=c2, markersize=2.8, label="Future slot MSE", clip_on=False)
    ax1.plot(epochs, recon, "^-", color=c3, markersize=2.8, label="Evidence recon. MSE", clip_on=False)
    ax1.set_xlabel("Training epoch")
    ax1.set_ylabel("MSE")
    # Put legend outside axes to avoid overlapping curves.
    ax1.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        handletextpad=0.4,
    )
    ax1.text(
        -0.22,
        1.02,
        "b",
        transform=ax1.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
    )

    for ax in (ax0, ax1):
        ax.minorticks_off()

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "fig1_training_curves"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_figure2(history: list[dict], out_dir: Path, col_w: float) -> None:
    """Normalized improvement (epoch 5 → last) bar-style summary."""
    first, last = history[0], history[-1]
    labels = ["Canonical\nPER", "Future\nMSE", "Recon\nMSE"]
    v0 = np.array(
        [first["canonical_per"], first["future_mse"], first["recon_mse"]],
        dtype=float,
    )
    v1 = np.array(
        [last["canonical_per"], last["future_mse"], last["recon_mse"]],
        dtype=float,
    )
    # Lower is better for all three → show fractional change from baseline
    rel = v1 / np.maximum(v0, 1e-12)

    fig_w = col_w * 1.05
    fig_h = col_w * 0.55
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, np.ones_like(rel), width=w, label="Epoch {}".format(int(first["epoch"])), color="#E0E0E0", edgecolor="#333333", linewidth=0.4)
    ax.bar(x + w / 2, rel, width=w, label="Epoch {}".format(int(last["epoch"])), color="#0072B2", edgecolor="#333333", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Relative value\n(÷ epoch {})".format(int(first["epoch"])))
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.text(-0.15, 1.05, "a", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
    ax.set_ylim(0, max(1.05, float(rel.max()) * 1.08))

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "fig2_relative_metrics"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--eval-history",
        type=Path,
        default=Path("runs/wm_belief_librispeech/eval_history.json"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/wm_belief_librispeech/figures_nature"),
    )
    args = p.parse_args()

    col_w = nature_rc()
    hist = load_history(args.eval_history)
    plot_figure1(hist, args.output_dir, col_w)
    plot_figure2(hist, args.output_dir, col_w)
    print("Wrote:", args.output_dir)


if __name__ == "__main__":
    main()
