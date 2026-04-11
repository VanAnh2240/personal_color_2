"""
classification/color_utils.py  (fixed v3)

Fixes vs v2:
  - classify_hue: nhận skin_rgb (không phải lips_rgb) — paper DSCAS tính
    subtone trên skin undertone. Anchor peach/purple giữ nguyên.
  - classify_chroma threshold default: 127 → 60  (HSV S của da người thường
    30-120; 127 khiến hầu hết kết quả ra "muted", bias Autumn/Summer).
  - classify_value: skin được weight ×2 vì chiếm diện tích lớn nhất.
  - classify_contrast threshold default: 127 → 65  (hair vs eyes chênh 127/255
    chỉ xảy ra ở tóc đen tuyền + mắt rất sáng; 65 phân loại thực tế hơn).
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import List, Tuple, Optional

RGB    = Tuple[int, int, int]
BGRArr = np.ndarray


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Colour-space helpers
# ─────────────────────────────────────────────────────────────────────────────

def rgb_to_lab(rgb: RGB) -> np.ndarray:
    bgr_pixel = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
    return cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2Lab)[0, 0].astype(np.float32)


def lab_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(float) - b.astype(float)))


def _hsv_value_255(rgb: RGB) -> float:
    bgr = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0, 2])


def _hsv_saturation_255(rgb: RGB) -> float:
    bgr = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0, 1])


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Pure-numpy k-means
# ─────────────────────────────────────────────────────────────────────────────

def _kmeans_numpy(pixels: np.ndarray, k: int,
                  max_iter: int = 50, random_state: int = 42) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    N   = len(pixels)

    # k-means++ init
    centers = [pixels[rng.integers(N)].copy()]
    for _ in range(1, k):
        c_arr   = np.array(centers, dtype=np.float32)
        sq_dist = np.sum((pixels[:, None] - c_arr[None]) ** 2, axis=2)
        min_d   = sq_dist.min(axis=1)
        total   = min_d.sum()
        centers.append(
            pixels[rng.integers(N)].copy() if total == 0
            else pixels[rng.choice(N, p=min_d / total)].copy()
        )
    centers = np.array(centers, dtype=np.float32)

    for _ in range(max_iter):
        labels = np.sum((pixels[:, None] - centers[None]) ** 2, axis=2).argmin(axis=1)
        new_c  = np.array([
            pixels[labels == j].mean(axis=0) if (labels == j).any() else centers[j]
            for j in range(k)
        ], dtype=np.float32)
        if np.allclose(centers, new_c, atol=0.5):
            break
        centers = new_c
    return centers


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Dominant-colour extraction
# ─────────────────────────────────────────────────────────────────────────────

def _brightness(rgb: RGB) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def extract_dominant_color(
    bgr_image: BGRArr,
    mask: np.ndarray,
    k: int = 3,
    min_brightness: float = 15.0,
    max_brightness: float = 240.0,
    prefer_bright: bool = True,
    random_state: int = 42,
) -> Optional[RGB]:
    region_pixels = bgr_image[mask == 255]
    if len(region_pixels) < k:
        return None

    rgb_pixels = region_pixels[:, ::-1].astype(np.float32)
    centers    = _kmeans_numpy(rgb_pixels, k=k, random_state=random_state)
    candidates: List[RGB] = [tuple(int(c) for c in ctr) for ctr in centers]

    valid = [c for c in candidates if min_brightness <= _brightness(c) <= max_brightness]
    if not valid:
        valid = candidates

    best_rgb, best_score = None, float("inf")
    for candidate in valid:
        diff   = rgb_pixels - np.array(candidate, dtype=np.float32)
        rmse   = float(np.sqrt(np.mean(diff ** 2)))
        bright = _brightness(candidate)
        weight = (1.0 + (1.0 - bright / 255.0) * 0.5 if prefer_bright
                  else 1.0 + (bright / 255.0) * 0.5)
        score  = rmse * weight
        if score < best_score:
            best_score = score
            best_rgb   = candidate
    return best_rgb


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Munsell metrics
# ─────────────────────────────────────────────────────────────────────────────

# Anchors: peach (warm undertone) vs purple (cool undertone)
_PEACH_LAB  = rgb_to_lab((255, 230, 182))
_PURPLE_LAB = rgb_to_lab((210, 120, 180))


def classify_hue(skin_rgb: RGB) -> str:
    """
    S (Subtone) metric — warm vs cool.

    FIX v3: nhận skin_rgb thay vì lips_rgb.
    Paper DSCAS tính subtone trên skin undertone (peach = warm, purple = cool).
    Lips dễ bị ảnh hưởng bởi màu máu / son, không đại diện undertone tốt.

    Returns "warm" | "cool".
    """
    skin_lab = rgb_to_lab(skin_rgb)
    d_peach  = lab_distance(skin_lab, _PEACH_LAB)
    d_purple = lab_distance(skin_lab, _PURPLE_LAB)
    return "warm" if d_peach <= d_purple else "cool"


def classify_chroma(skin_rgb: RGB, threshold: float = 60.0) -> str:
    """
    I (Intensity / chroma) metric — bright vs muted.

    HSV Saturation (0-255) của skin.
    Returns "bright" | "muted".

    FIX v3: threshold default 127 → 60.
    Da người thường có HSV S trong khoảng 30-120; threshold 127 khiến
    hầu hết kết quả ra "muted" (bias nặng về Autumn/Summer).
    60 phân chia thực tế hơn: da rạng rỡ/sáng khoẻ ≥60, da trung tính/xỉn <60.
    """
    sat = _hsv_saturation_255(skin_rgb)
    return "bright" if sat >= threshold else "muted"


def classify_value(
    skin_rgb: RGB,
    hair_rgb: Optional[RGB],
    eyes_rgb: RGB,
    threshold: float = 127.0,
) -> str:
    # skin ×2 weight, hair ×1 , eyes ×1
    vals = [_hsv_value_255(skin_rgb), _hsv_value_255(skin_rgb), _hsv_value_255(eyes_rgb)]
    if hair_rgb is not None:
        vals.append(_hsv_value_255(hair_rgb))
    return "light" if np.mean(vals) >= threshold else "dark"


def classify_contrast(
    hair_rgb: Optional[RGB],
    eyes_rgb: RGB,
    threshold: float = 65.0,
) -> Optional[str]:
    """
    C (Contrast) metric — high vs low.

    Absolute HSV Value difference giữa hair và eyes (0-255).
    Returns "high" | "low" | None nếu bald (hair_rgb is None).

    FIX v3: threshold default 127 → 65.
    Chênh lệch 127/255 chỉ xảy ra khi tóc đen tuyền + mắt rất sáng (hoặc ngược lại).
    65 ≈ 25% thang HSV V — phân biệt được tóc nâu tối vs mắt nâu nhạt,
    tóc vàng vs mắt xanh lá, v.v.
    """
    if hair_rgb is None:
        return None
    diff = abs(_hsv_value_255(hair_rgb) - _hsv_value_255(eyes_rgb))
    return "high" if diff >= threshold else "low"


