"""
src/utils/visualize.py
Visualization helpers:
  - overlay segmentation mask on image
  - draw colour swatches for regions
  - save side-by-side comparison grid
  - plot training curves from CSV log
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless – works in Colab & server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from config import LAPA_CLASS_NAMES, LAPA_NUM_CLASSES

# ──────────────────────────────────────────────
# Per-class colour palette (RGB)
# ──────────────────────────────────────────────
CLASS_COLORS = np.array([
    [0,   0,   0  ],   # 0  background
    [255, 200, 150],   # 1  skin
    [100, 60,  20 ],   # 2  left eyebrow
    [100, 60,  20 ],   # 3  right eyebrow
    [0,   120, 200],   # 4  left eye
    [0,   120, 200],   # 5  right eye
    [220, 130, 100],   # 6  nose
    [200, 80,  80 ],   # 7  upper lip
    [180, 50,  50 ],   # 8  inner mouth
    [200, 80,  80 ],   # 9  lower lip
    [80,  50,  20 ],   # 10 hair
], dtype=np.uint8)


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """
    Convert (H, W) class-index mask → (H, W, 3) uint8 RGB colour image.
    """
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx in range(LAPA_NUM_CLASSES):
        color_img[mask == cls_idx] = CLASS_COLORS[cls_idx]
    return color_img


def overlay_mask(img_rgb: np.ndarray,
                 mask: np.ndarray,
                 alpha: float = 0.45) -> np.ndarray:
    """
    Blend colour mask onto original image.
    img_rgb : (H, W, 3) uint8
    mask    : (H, W)    int
    Returns  : (H, W, 3) uint8 blended image
    """
    color_mask = mask_to_color(mask)
    # Resize mask to image size if needed
    if color_mask.shape[:2] != img_rgb.shape[:2]:
        color_mask = cv2.resize(color_mask,
                                (img_rgb.shape[1], img_rgb.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(img_rgb, 1 - alpha, color_mask, alpha, 0)


def draw_region_swatches(dominant_colors: dict,
                          swatch_size: int = 60,
                          padding: int = 8) -> np.ndarray:
    """
    Create a row of labelled colour swatches.
    dominant_colors: {region_name: {"hex": "#RRGGBB", "rgb": [R,G,B]}}
    Returns: (H, W, 3) uint8 RGB image
    """
    n = len(dominant_colors)
    if n == 0:
        return np.zeros((swatch_size + 30, 1, 3), dtype=np.uint8)

    cell_w  = swatch_size + padding * 2
    total_w = cell_w * n
    total_h = swatch_size + 32   # room for label

    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 30  # dark bg

    for i, (name, info) in enumerate(dominant_colors.items()):
        rgb = info["rgb"] if isinstance(info, dict) else [128, 128, 128]
        x0  = i * cell_w + padding
        y0  = padding

        # Draw swatch
        canvas[y0:y0 + swatch_size, x0:x0 + swatch_size] = rgb[::-1]  # RGB→BGR? no stay RGB

        # Label
        label = name[:6]
        cv2.putText(canvas, label,
                    (x0, y0 + swatch_size + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (200, 200, 200), 1, cv2.LINE_AA)

    return canvas


def save_comparison_grid(img_rgb: np.ndarray,
                          pred_mask: np.ndarray,
                          gt_mask: np.ndarray | None,
                          dominant_colors: dict,
                          season: str,
                          out_path: str):
    """
    Save a comparison figure:
      Col 1: original image
      Col 2: predicted mask overlay
      Col 3: ground-truth mask overlay (optional)
      Bottom: colour swatch row + season label
    """
    cols = 3 if gt_mask is not None else 2
    fig  = plt.figure(figsize=(cols * 4 + 2, 6), facecolor="#0d0d0f")
    gs   = GridSpec(2, cols, figure=fig,
                    height_ratios=[5, 1], hspace=0.08, wspace=0.04)

    def _ax(r, c):
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")
        return ax

    # Original
    ax0 = _ax(0, 0)
    ax0.imshow(img_rgb)
    ax0.set_title("Input", color="white", fontsize=9, pad=4)

    # Predicted overlay
    ax1 = _ax(0, 1)
    ax1.imshow(overlay_mask(img_rgb, pred_mask))
    ax1.set_title("Predicted", color="white", fontsize=9, pad=4)

    # GT overlay
    if gt_mask is not None:
        ax2 = _ax(0, 2)
        ax2.imshow(overlay_mask(img_rgb, gt_mask))
        ax2.set_title("Ground Truth", color="white", fontsize=9, pad=4)

    # Bottom: legend + season
    ax_bot = fig.add_subplot(gs[1, :])
    ax_bot.axis("off")

    # Build colour patches for legend
    patches = [mpatches.Patch(color=CLASS_COLORS[i] / 255,
                               label=LAPA_CLASS_NAMES[i])
               for i in range(LAPA_NUM_CLASSES)]
    ax_bot.legend(handles=patches, ncol=6,
                  loc="center", fontsize=6.5,
                  labelcolor="white",
                  facecolor="#151518", edgecolor="#2a2a30",
                  framealpha=0.9)

    # Season label
    season_colors = {
        "Spring": "#ffcba4", "Summer": "#b0c4de",
        "Autumn": "#c47c3b", "Winter": "#7b9fb8",
    }
    sc = season_colors.get(season, "#c9a96e")
    fig.text(0.5, 0.97,
             f"Season: {season}",
             ha="center", va="top",
             color=sc, fontsize=14,
             fontfamily="serif", fontstyle="italic")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_training_curves(log_csv: str, out_path: str):
    """Read a training log CSV and save a loss/mIoU plot."""
    import pandas as pd

    df = pd.read_csv(log_csv)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4),
                              facecolor="#0d0d0f")
    for ax in axes:
        ax.set_facecolor("#151518")
        ax.tick_params(colors="grey")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a30")

    axes[0].plot(df["epoch"], df["train_loss"],
                 color="#c9a96e", linewidth=1.5, label="Train")
    axes[0].plot(df["epoch"], df["val_loss"],
                 color="#8fbcbb", linewidth=1.5, label="Val",
                 linestyle="--")
    axes[0].set_title("Loss", color="white")
    axes[0].legend(facecolor="#151518", labelcolor="white")
    axes[0].set_xlabel("Epoch", color="grey")

    if "val_mIoU" in df.columns:
        axes[1].plot(df["epoch"], df["val_mIoU"],
                     color="#a3be8c", linewidth=1.8)
        axes[1].set_title("Val mIoU", color="white")
        axes[1].set_xlabel("Epoch", color="grey")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Training curve saved → {out_path}")
