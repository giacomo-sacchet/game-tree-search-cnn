# =============================================================================
# plot_training.py
# Local script: generates the OthelloCNN training figures for the thesis
# from the two CSV files written by 5_OthelloCNN.py:
#
#   training_log.csv     one row per epoch (loss, MAE, correlation metrics, LR)
#   val_predictions.csv  one row per validation position (game_id,
#                        n_moves_played, y_true, y_pred) from the final model
#
# Usage:
#   python plot_training.py                         (CSV files in current folder)
#   python plot_training.py --log path/training_log.csv \
#                           --preds path/val_predictions.csv \
#                           --out figures/cnn_training
#
# Output (PNG, 300 dpi, ready for \includegraphics; no in-figure titles,
# the LaTeX caption plays that role):
#   fig_A_loss.png         MSE train/valid per epoch, LR-drop markers
#   fig_B_mae.png          MAE train/valid per epoch, LR-drop markers
#   fig_C_scatter.png      predicted vs true margin (hexbin density, y = x)
#   fig_D_mae_by_move.png  validation MAE vs number of moves played
#   fig_E_correlation.png  Pearson, Spearman, R^2 and sign accuracy per epoch
#
# Requirements: pandas, numpy, matplotlib
# =============================================================================

import argparse
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "results", "cnn_training")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # no display needed
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

# ---------------------------------------------------------------------------
# Style: validated colour-blind-safe pair (blue/orange) + distinct line styles,
# so train/valid are never distinguished by colour alone (print-safe).
# ---------------------------------------------------------------------------
C_TRAIN = "#2a78d6"     # blue
C_VALID = "#eb6834"     # orange
C_AUX1  = "#1baf7a"     # aqua
C_AUX2  = "#4a3aa7"     # violet
C_GRID  = "#d9d9d9"
C_INK   = "#333333"
C_MUTED = "#777777"

# Single-hue sequential ramp (light -> dark blue) for the density plot
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.family":       "serif",       # matches the LaTeX body text
    "font.size":         10,
    "axes.edgecolor":    C_MUTED,
    "axes.labelcolor":   C_INK,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        C_GRID,
    "grid.linewidth":    0.6,
    "xtick.color":       C_MUTED,
    "ytick.color":       C_MUTED,
    "legend.frameon":    False,
    "lines.linewidth":   2.0,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lr_drop_epochs(log):
    """Epochs at which the learning rate changes w.r.t. the previous epoch."""
    lr = log["lr"].to_numpy()
    ep = log["epoch"].to_numpy()
    return [int(ep[i]) for i in range(1, len(lr)) if not np.isclose(lr[i], lr[i - 1])]


def _mark_lr_drops(ax, log):
    """Draw a thin vertical line + label at every learning-rate change."""
    for e in _lr_drop_epochs(log):
        new_lr = log.loc[log["epoch"] == e, "lr"].iloc[0]
        ax.axvline(e - 0.5, color=C_MUTED, linewidth=0.8, linestyle=":")
        ax.annotate(f"LR → {new_lr:g}", xy=(e - 0.5, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(3, -12), textcoords="offset points",
                    fontsize=8, color=C_MUTED)


def _train_valid_plot(log, col_train, col_valid, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(log["epoch"], log[col_train], color=C_TRAIN, linestyle="--",
            label="Training")
    ax.plot(log["epoch"], log[col_valid], color=C_VALID, linestyle="-",
            label="Validation")
    _mark_lr_drops(ax, log)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xlim(log["epoch"].min() - 0.5, log["epoch"].max() + 0.5)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_loss(log, out):
    _train_valid_plot(log, "loss/train", "loss/valid",
                      "MSE (discs²)", "Training and validation loss (MSE)",
                      os.path.join(out, "fig_A_loss.png"))


def fig_mae(log, out):
    _train_valid_plot(log, "mae/train", "mae/valid",
                      "MAE (discs)", "Training and validation mean absolute error",
                      os.path.join(out, "fig_B_mae.png"))


def fig_scatter(preds, out):
    y, p = preds["y_true"].to_numpy(), preds["y_pred"].to_numpy()
    r = np.corrcoef(p, y)[0, 1]

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    hb = ax.hexbin(y, p, gridsize=65, extent=(-64, 64, -64, 64),
                   cmap=SEQ_BLUE, norm=LogNorm(), mincnt=1, linewidths=0)
    ax.plot([-64, 64], [-64, 64], color=C_VALID, linewidth=1.2,
            linestyle="--", label="y = x")
    ax.set_xlim(-64, 64)
    ax.set_ylim(-64, 64)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("True final margin (Black − White discs)")
    ax.set_ylabel("Predicted margin")
    ax.text(0.03, 0.97, f"Pearson r = {r:.3f}\nn = {len(y):,}",
            transform=ax.transAxes, va="top", fontsize=9, color=C_INK)
    ax.legend(loc="lower right")
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Positions per cell (log scale)", color=C_INK)
    path = os.path.join(out, "fig_C_scatter.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


def fig_mae_by_move(preds, out, min_count=500):
    df = preds.assign(abs_err=(preds["y_pred"] - preds["y_true"]).abs())
    g = (df.groupby("n_moves_played")["abs_err"]
           .agg(["mean", "count"]).reset_index())
    g = g[g["count"] >= min_count]           # drop sparse tail (very long games)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(g["n_moves_played"], g["mean"], color=C_VALID, marker="o",
            markersize=3.5, linewidth=1.8)
    ax.set_xlabel("Moves already played in the position")
    ax.set_ylabel("Validation MAE (discs)")
    ax.set_ylim(bottom=0)
    path = os.path.join(out, "fig_D_mae_by_move.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


def fig_correlation(log, out):
    series = [
        ("pearson/valid",  "Pearson r",     C_TRAIN, "-"),
        ("spearman/valid", "Spearman ρ",    C_VALID, "--"),
        ("r2/valid",       "R²",            C_AUX1,  "-."),
        ("sign_acc/valid", "Sign accuracy", C_AUX2,  ":"),
    ]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for col, label, color, ls in series:
        ax.plot(log["epoch"], log[col], color=color, linestyle=ls, label=label)
    _mark_lr_drops(ax, log)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_ylim(0, 1)
    ax.set_xlim(log["epoch"].min() - 0.5, log["epoch"].max() + 0.5)
    ax.legend(loc="lower right", ncol=2)
    path = os.path.join(out, "fig_E_correlation.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


def print_summary(log, preds):
    """Final-epoch numbers to copy into the thesis table."""
    last = log.iloc[-1]
    print("\nFinal epoch summary (for the thesis table):")
    for col in ["epoch", "lr", "loss/train", "loss/valid", "mae/train", "mae/valid",
                "pearson/valid", "spearman/valid", "r2/valid",
                "sign_acc/valid", "pred_std/valid"]:
        print(f"  {col:<16} {last[col]:.4f}")
    if preds is not None:
        print(f"  {'val positions':<16} {len(preds):,}")
        print(f"  {'val games':<16} {preds['game_id'].nunique():,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="OthelloCNN training figures")
    ap.add_argument("--log",   default=os.path.join(TRAIN_DIR, "training_log.csv"))
    ap.add_argument("--preds", default=os.path.join(TRAIN_DIR, "val_predictions.csv"))
    ap.add_argument("--out",   default=os.path.join(TRAIN_DIR, "figures"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    log = pd.read_csv(args.log)
    print(f"Loaded {len(log)} epochs from {args.log}")
    fig_loss(log, args.out)
    fig_mae(log, args.out)
    fig_correlation(log, args.out)

    preds = None
    if os.path.isfile(args.preds):
        preds = pd.read_csv(args.preds)
        print(f"Loaded {len(preds):,} validation predictions from {args.preds}")
        fig_scatter(preds, args.out)
        fig_mae_by_move(preds, args.out)
    else:
        print(f"[skip] {args.preds} not found — figures C and D not generated")

    print_summary(log, preds)