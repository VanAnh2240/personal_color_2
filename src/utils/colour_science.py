"""
src/utils/colour_science.py
Advanced colour science helpers:
  - sRGB → CIE L*a*b* → Munsell (via lookup table approach)
  - undertone analysis (warm / cool / neutral)
  - seasonal contrast scoring
  - colour harmony suggestions based on season
"""

import math
import colorsys
import numpy as np


# ──────────────────────────────────────────────
# sRGB ↔ linear RGB ↔ XYZ ↔ L*a*b*
# ──────────────────────────────────────────────
def srgb_to_linear(c: float) -> float:
    """Single channel, 0–1 range."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def rgb_to_xyz(r: int, g: int, b: int) -> tuple:
    """(R,G,B) uint8 → CIE XYZ (D65)"""
    r_, g_, b_ = [srgb_to_linear(x / 255.0) for x in (r, g, b)]
    X = 0.4124564 * r_ + 0.3575761 * g_ + 0.1804375 * b_
    Y = 0.2126729 * r_ + 0.7151522 * g_ + 0.0721750 * b_
    Z = 0.0193339 * r_ + 0.1191920 * g_ + 0.9503041 * b_
    return X, Y, Z


def xyz_to_lab(X: float, Y: float, Z: float) -> tuple:
    """CIE XYZ → L*a*b* (D65 white point)"""
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx, fy, fz = f(X / Xn), f(Y / Yn), f(Z / Zn)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return L, a, b


def rgb_to_lab(r: int, g: int, b: int) -> tuple:
    """Convenience: (R,G,B) uint8 → (L*, a*, b*)"""
    return xyz_to_lab(*rgb_to_xyz(r, g, b))


def lab_chroma(a: float, b: float) -> float:
    return math.sqrt(a ** 2 + b ** 2)


def lab_hue_angle(a: float, b: float) -> float:
    """Hue angle in degrees (0–360)."""
    h = math.degrees(math.atan2(b, a))
    return h % 360


# ──────────────────────────────────────────────
# Approximate Munsell HVC from Lab
# ──────────────────────────────────────────────
def lab_to_munsell_approx(L: float, a: float, b: float) -> dict:
    """
    Approximate Munsell notation from CIE L*a*b*.
    This is an approximation; a full lookup table (Munsell Renotation Data)
    would be needed for exact values.

    Returns
    -------
    {
      "hue_angle": float,   0–360 degrees
      "hue_name":  str,     Munsell hue letter (5R, 10YR, …)
      "value":     float,   0–10
      "chroma":    float,   0–∞ (typically 0–18 in practice)
    }
    """
    value  = L / 10.0                   # Munsell value ≈ L*/10
    chroma = lab_chroma(a, b) / 5.0     # rough scale factor
    hue_angle = lab_hue_angle(a, b)

    # Map hue angle → approximate Munsell hue name
    HUE_NAMES = [
        (  0, 22.5, "5R"),
        ( 22.5, 45, "10R"),
        ( 45, 67.5, "5YR"),
        ( 67.5, 90, "10YR"),
        ( 90, 112.5, "5Y"),
        (112.5, 135, "10Y"),
        (135, 157.5, "5GY"),
        (157.5, 180, "10GY"),
        (180, 202.5, "5G"),
        (202.5, 225, "10G"),
        (225, 247.5, "5BG"),
        (247.5, 270, "10BG"),
        (270, 292.5, "5B"),
        (292.5, 315, "5PB"),
        (315, 337.5, "5P"),
        (337.5, 360, "5RP"),
    ]
    hue_name = "N"   # neutral fallback
    for lo, hi, name in HUE_NAMES:
        if lo <= hue_angle < hi:
            hue_name = name
            break

    return {
        "hue_angle": hue_angle,
        "hue_name":  hue_name,
        "value":     round(max(0.0, min(10.0, value)), 2),
        "chroma":    round(max(0.0, chroma), 2),
    }


# ──────────────────────────────────────────────
# Undertone detection
# ──────────────────────────────────────────────
def detect_undertone(skin_rgb: list) -> str:
    """
    Classify skin undertone as 'warm', 'cool', or 'neutral'.
    Uses the ratio of red-yellow energy vs blue-pink energy in Lab space.
    """
    r, g, b = skin_rgb
    L, a, lab_b = rgb_to_lab(r, g, b)
    hue = lab_hue_angle(a, lab_b)

    # Warm tones: reddish-yellow hues (reds, oranges, yellows: 0–90°)
    # Cool tones: pinkish-blue hues (magentas, blues: 270–360° and 315–360°)
    # Neutral   : in-between

    if 315 <= hue or hue < 45:
        return "cool"   # magenta/pink/red families
    elif 45 <= hue < 90:
        return "warm"   # yellow/orange/yellow-red
    elif 90 <= hue < 180:
        return "neutral"
    elif 180 <= hue < 270:
        return "cool"   # blue-green range
    return "neutral"


# ──────────────────────────────────────────────
# Seasonal contrast score
# ──────────────────────────────────────────────
def seasonal_contrast(skin_rgb: list, hair_rgb: list) -> dict:
    """
    Measure the contrast between skin and hair colours.
    Returns a dict with 'score' (0–100) and 'level' label.
    """
    Ls, _, _ = rgb_to_lab(*skin_rgb)
    Lh, _, _ = rgb_to_lab(*hair_rgb)
    delta_L  = abs(Ls - Lh)

    score = min(100.0, delta_L)
    if delta_L >= 60:
        level = "high"     # Winter
    elif delta_L >= 30:
        level = "medium"   # Spring / Autumn
    else:
        level = "low"      # Summer

    return {"score": round(score, 1), "level": level}


# ──────────────────────────────────────────────
# Colour harmony suggestions
# ──────────────────────────────────────────────
SEASON_HARMONIES = {
    "Spring": {
        "analogous": ["#FFCBA4", "#FFD699", "#FF9E7A"],
        "accent":    ["#7ECBA1", "#6BBFDF"],
        "neutral":   ["#F5EBD5", "#C8A882"],
        "tip": "Use warm neutrals as base; coral and peach are your power colours.",
    },
    "Summer": {
        "analogous": ["#B0C4DE", "#C8D8E8", "#E6E6FA"],
        "accent":    ["#BC8F9F", "#9DB4C0"],
        "neutral":   ["#F0EFF4", "#A9A9B0"],
        "tip": "Dusty pastels and grayed tones. Avoid anything too sharp or warm.",
    },
    "Autumn": {
        "analogous": ["#C47C3B", "#8B4513", "#A0522D"],
        "accent":    ["#6B8E23", "#B8860B"],
        "neutral":   ["#D2B48C", "#8B7355"],
        "tip": "Earth tones and muted oranges. Avoid cold blues and bright pinks.",
    },
    "Winter": {
        "analogous": ["#000080", "#DC143C", "#7B68EE"],
        "accent":    ["#FFFFFF", "#00CED1"],
        "neutral":   ["#2F2F2F", "#C0C0C0"],
        "tip": "Bold, saturated, high-contrast. Pure white and black are your best neutrals.",
    },
}


def get_colour_harmony(season: str) -> dict:
    """Return colour harmony palette and styling tip for the given season."""
    return SEASON_HARMONIES.get(season, {})


# ──────────────────────────────────────────────
# Delta-E (CIE76) colour distance
# ──────────────────────────────────────────────
def delta_e_76(rgb1: list, rgb2: list) -> float:
    """Perceptual colour distance between two RGB colours."""
    L1, a1, b1 = rgb_to_lab(*rgb1)
    L2, a2, b2 = rgb_to_lab(*rgb2)
    return math.sqrt((L1-L2)**2 + (a1-a2)**2 + (b1-b2)**2)
