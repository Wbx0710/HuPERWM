#!/usr/bin/env python3
"""Plot training / validation loss curves from Lightning CSVLogger output.

Expects ``metrics.csv`` under ``<run_dir>/csv_logs/version_*/`` (written when
training with ``train_wm_belief.py`` after CSVLogger was enabled).

Usage::

    conda activate phn
    python scripts/plot_wm_belief_losses.py --run-dir runs/wm_belief_librispeech
    python scripts/plot_wm_belief_losses.py \\
        --metrics-csv runs/wm_belief_librispeech/csv_logs/version_0/metrics.csv \\
        --eval-history runs/wm_belief_librispeech/eval_history.json \\
        --output runs/wm_belief_librispeech/figures_loss/curves.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Belief WM loss curves from Lightning CSV.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--run-dir",
        type=str,
        help="Training output dir; uses latest csv_logs/version_*/metrics.csv",
    )
    src.add_argument("--metrics-csv", type=str, help="Path to Lightning metrics.csv")
    p.add_argument(
        "--eval-history",
        type=str,
        default=None,
        help="Optional eval_history.json (epoch PER / MSE from FullEvalCallback).",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (.png or .pdf). Default: <run-dir>/figures_loss/loss_curves.png",
    )
    p.add_argument("--show", action="store_true", help="Open an interactive window after saving.")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def _find_latest_metrics_csv(run_dir: Path) -> Path:
    base = run_dir / "csv_logs"
    if not base.is_dir():
        raise FileNotFoundError(
            f"No {base}/ — train with current train_wm_belief.py (CSVLogger) "
            "or pass --metrics-csv explicitly."
        )
    versions = sorted(
        [p for p in base.iterdir() if p.is_dir() and p.name.startswith("version_")],
        key=lambda p: int(p.name.split("_", 1)[1]),
    )
    if not versions:
        raise FileNotFoundError(f"No version_* under {base}")
    csv_path = versions[-1] / "metrics.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path}")
    return csv_path


def _float_cell(x: str | None) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _int_cell(x: str | None) -> int | None:
    v = _float_cell(x)
    if v is None:
        return None
    return int(v)


def load_metrics_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return fieldnames, rows


def series_step(rows: list[dict], col_y: str, col_x: str = "step") -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for r in rows:
        x = _int_cell(r.get(col_x))
        y = _float_cell(r.get(col_y))
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def series_epoch(rows: list[dict], col_y: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for r in rows:
        x = _float_cell(r.get("epoch"))
        y = _float_cell(r.get(col_y))
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def pick_train_step_columns(fieldnames: list[str]) -> list[str]:
    out = []
    for c in fieldnames:
        if c in ("step", "epoch"):
            continue
        if c.startswith("train_") and c.endswith("_step") and "loss" in c:
            out.append(c)
    return sorted(out)


def pick_val_epoch_columns(fieldnames: list[str]) -> list[str]:
    out = []
    for c in fieldnames:
        if c.startswith("val_") and "loss" in c:
            if c.endswith("_epoch") or c.endswith("_step"):
                if c.endswith("_epoch"):
                    out.append(c)
            else:
                out.append(c)
    # Prefer *_epoch for validation aggregates
    preferred = [c for c in out if c.endswith("_epoch")]
    return sorted(preferred if preferred else out)


def load_eval_history(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        csv_path = _find_latest_metrics_csv(run_dir)
        default_out = run_dir / "figures_loss" / "loss_curves.png"
    else:
        csv_path = Path(args.metrics_csv).resolve()
        run_dir = csv_path.parent.parent.parent
        default_out = run_dir / "figures_loss" / "loss_curves.png"

    out_path = Path(args.output) if args.output else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames, rows = load_metrics_rows(csv_path)
    if not rows:
        raise SystemExit(f"No rows in {csv_path}")

    train_cols = pick_train_step_columns(fieldnames)
    val_cols = pick_val_epoch_columns(fieldnames)

    n_panels = 1 + (1 if val_cols else 0) + (1 if args.eval_history else 0)
    n_panels = max(n_panels, 1)

    fig_h = 2.8 * n_panels + 0.8
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(8.0, fig_h),
        squeeze=False,
    )
    ax_flat = axes.ravel()

    ai = 0
    if train_cols:
        ax = ax_flat[ai]
        for c in train_cols:
            xs, ys = series_step(rows, c)
            if xs.size == 0:
                continue
            order = np.argsort(xs)
            short = c.replace("train_", "").replace("_step", "")
            ax.plot(xs[order], ys[order], label=short, alpha=0.85, linewidth=1.0)
        ax.set_xlabel("global step")
        ax.set_ylabel("loss")
        ax.set_title("Training (per-step)")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.25)
        ai += 1
    else:
        ax_flat[ai].text(0.5, 0.5, "No train_*_step loss columns", ha="center", va="center")
        ax_flat[ai].set_axis_off()
        ai += 1

    if val_cols and ai < len(ax_flat):
        ax = ax_flat[ai]
        for c in val_cols:
            xs, ys = series_epoch(rows, c)
            if xs.size == 0:
                continue
            order = np.argsort(xs)
            short = c.replace("val_", "").replace("_epoch", "").replace("_step", "")
            ax.plot(xs[order], ys[order], "o-", label=short, markersize=3, alpha=0.85)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("Validation (Lightning epoch logs)")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.25)
        ai += 1

    if args.eval_history and ai < len(ax_flat):
        hist_path = Path(args.eval_history).resolve()
        hist = load_eval_history(hist_path)
        epochs = np.array([h["epoch"] for h in hist], dtype=float)
        ax = ax_flat[ai]
        if "canonical_per" in hist[0]:
            ax.plot(epochs, [h["canonical_per"] for h in hist], "o-", color="C0", label="canonical PER")
        if "teacher_per" in hist[0]:
            ax.plot(epochs, [h["teacher_per"] for h in hist], "s-", color="C1", label="teacher PER")
        ax.set_xlabel("epoch")
        ax.set_ylabel("PER")
        ax.set_title(f"Eval callback ({hist_path.name})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.25)
        ai += 1

    fig.suptitle(f"Source: {csv_path}", fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {out_path}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
