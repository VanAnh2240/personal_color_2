# File: classification/visualizer.py
"""
Visualisation utilities for Seasonal Colour Analysis results.

Changes v3:
  - draw_dominants_strip: mỗi swatch hiển thị màu + hex code + label rõ ràng,
    có viền tách biệt, trông giống color palette chuyên nghiệp.
  - draw_result_overlay: banner dùng gradient 2 màu từ palette thay vì 1 màu.
  - save_result_figure: layout gọn hơn, dominant strip cao hơn để đọc được.
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .classifier import ClassificationResult
    from .palettes import SeasonPalette

RGB = Tuple[int, int, int]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _brightness(rgb: RGB) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def _to_bgr(rgb: RGB) -> Tuple[int, int, int]:
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def _hex(rgb: RGB) -> str:
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def _text_color(rgb: RGB) -> Tuple[int, int, int]:
    """White text on dark bg, black on light bg."""
    return (255, 255, 255) if _brightness(rgb) < 140 else (20, 20, 20)


# ─────────────────────────────────────────────────────────────────────────────
# Dominant colour strip  (skin | hair | lips | eyes)
# ─────────────────────────────────────────────────────────────────────────────

def draw_dominants_strip(
    dominants: Dict[str, Optional[RGB]],
    total_width: int = 320,
    height: int = 90,
) -> np.ndarray:
    """
    Horizontal strip — 4 swatches side by side.
    Mỗi swatch gồm:
      • Màu fill
      • Hex code (ở giữa trên)
      • Tên region (label bar tối ở dưới)
    """
    order  = ["skin", "hair", "lips", "eyes"]
    n      = len(order)
    sw_w   = total_width // n
    actual_w = sw_w * n          # tránh 1px gap do floor division
    strip  = np.zeros((height, actual_w, 3), dtype=np.uint8)
    label_h = 22                  # chiều cao label bar cuối mỗi swatch

    for i, key in enumerate(order):
        rgb = dominants.get(key)
        x0, x1 = i * sw_w, (i + 1) * sw_w

        if rgb is None:
            strip[:, x0:x1] = (45, 45, 45)
            cv2.putText(strip, "N/A",
                        (x0 + sw_w // 2 - 12, height // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (120, 120, 120), 1, cv2.LINE_AA)
        else:
            bgr = _to_bgr(rgb)
            strip[:height - label_h, x0:x1] = bgr

        # ── dark label bar at bottom ──────────────────────────────────────────
        strip[height - label_h:height, x0:x1] = (25, 25, 25)
        (lw, lh), _ = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        lx = x0 + (sw_w - lw) // 2
        ly = height - label_h + lh + 3
        cv2.putText(strip, key, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (210, 210, 210), 1, cv2.LINE_AA)

        # ── vertical separator ────────────────────────────────────────────────
        if i > 0:
            strip[:, x0:x0 + 1] = (15, 15, 15)

    return strip


# ─────────────────────────────────────────────────────────────────────────────
# Season palette strip
# ─────────────────────────────────────────────────────────────────────────────

def draw_palette_strip(
    season,          # SeasonPalette
    total_width: int = 320,
    height: int = 50,
) -> np.ndarray:
    """Horizontal strip of all season palette colours."""
    colors = season.colors
    n      = len(colors)
    sw_w   = total_width // n
    actual_w = sw_w * n
    strip  = np.zeros((height, actual_w, 3), dtype=np.uint8)

    for i, rgb in enumerate(colors):
        x0, x1 = i * sw_w, (i + 1) * sw_w
        strip[:, x0:x1] = _to_bgr(rgb)
        if i > 0:
            strip[:, x0:x0 + 1] = (15, 15, 15)

    return strip


# ─────────────────────────────────────────────────────────────────────────────
# Full result overlay on face image
# ─────────────────────────────────────────────────────────────────────────────

def draw_result_overlay(
    face_bgr: np.ndarray,
    result,              # ClassificationResult
    target_height: int = 400,
) -> np.ndarray:
    img = face_bgr.copy()
    h, w = img.shape[:2]
    scale = target_height / h
    img = cv2.resize(img, (int(w * scale), target_height))

    season  = result.season
    metrics = result.metrics

    # ── banner: gradient từ palette[0] → palette[-1] ─────────────────────────
    banner_h = 58
    banner   = np.zeros((banner_h, img.shape[1], 3), dtype=np.uint8)
    c0 = np.array(_to_bgr(season.colors[0]),  dtype=np.float32)
    c1 = np.array(_to_bgr(season.colors[-1]), dtype=np.float32)
    for x in range(img.shape[1]):
        t = x / max(img.shape[1] - 1, 1)
        banner[:, x] = (c0 * (1 - t) + c1 * t).astype(np.uint8)

    # semi-dark overlay để text dễ đọc
    overlay = np.zeros_like(banner)
    banner  = cv2.addWeighted(banner, 0.6, overlay, 0.4, 0)

    img = np.vstack([banner, img])

    # ── season name ───────────────────────────────────────────────────────────
    cv2.putText(img, season.name.upper(),
                (12, 40), cv2.FONT_HERSHEY_DUPLEX, 1.15,
                (255, 255, 255), 2, cv2.LINE_AA)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# Full summary figure
# ─────────────────────────────────────────────────────────────────────────────

def _label_bar(text: str, width: int, height: int = 26) -> np.ndarray:
    bar = np.full((height, width, 3), 30, dtype=np.uint8)
    cv2.putText(bar, text, (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (180, 180, 180), 1, cv2.LINE_AA)
    return bar


def save_result_figure(
    face_bgr: np.ndarray,
    result,              
    output_path: str,
    face_height: int = 380,
) -> None:
   
    annotated = draw_result_overlay(face_bgr, result, target_height=face_height)
    W = annotated.shape[1]

    dom_strip = draw_dominants_strip(result.dominants, total_width=W, height=90)
    pal_strip = draw_palette_strip(result.season,      total_width=W, height=52)

    def _fit_w(img, target):
        if img.shape[1] == target:
            return img
        if img.shape[1] < target:
            pad = np.zeros((img.shape[0], target - img.shape[1], 3), dtype=np.uint8)
            return np.hstack([img, pad])
        return img[:, :target]

    dom_strip = _fit_w(dom_strip, W)
    pal_strip = _fit_w(pal_strip, W)

    figure = np.vstack([
        annotated,
        _label_bar("Dominant colours", W),
        dom_strip,
        _label_bar(f"Season palette : {result.season.name}", W),
        pal_strip,
    ])

    cv2.imwrite(output_path, figure)
    print(f"[visualizer] Saved result figure → {output_path}")