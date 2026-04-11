# File: classification/palettes.py
"""
Seasonal Color Analysis — 4-season palette definitions.

Source: mrcmich/deep-seasonal-color-analysis-system
CSV files encode metrics as SIVC (Subtone, Intensity, Value, Contrast):
  Autumn : 1000  → warm, muted (low intensity), dark (low value), low contrast
  Spring : 1111  → warm, bright (high intensity), light (high value), high contrast
  Summer : 0010  → cool, muted (low intensity), light (high value), low contrast
  Winter : 0101  → cool, bright (high intensity), dark (low value), high contrast

metric_vector in this code uses canonical SIVC order:
  S (Subtone)  : warm=1, cool=0
  I (Intensity): bright=1, muted=0
  V (Value)    : light=1, dark=0
  C (Contrast) : high=1,  low=0

FIX (v2): metric_vector now follows true SIVC order, matching the CSV header
values exactly. Previously the code swapped I and V (used SVIC), which caused
Summer and Winter to get incorrect binary vectors and mis-classification.

  Season  | Paper SIVC | Old (SVIC) | Fixed (SIVC)
  --------|------------|------------|-------------
  Spring  |  1 1 1 1   |  1 1 1 1   |  1 1 1 1   ← unchanged (all-1s, order irrelevant)
  Summer  |  0 0 1 0   |  0 1 0 0   |  0 0 1 0   ← FIXED
  Autumn  |  1 0 0 0   |  1 0 0 0   |  1 0 0 0   ← unchanged (all-0s except S)
  Winter  |  0 1 0 1   |  0 0 1 1   |  0 1 0 1   ← FIXED
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

RGB = Tuple[int, int, int]


@dataclass
class SeasonPalette:
    name: str
    hue: str         # 'warm' | 'cool'
    chroma: str      # 'bright' | 'muted'   (Intensity in Munsell / SIVC)
    value: str       # 'light' | 'dark'
    contrast: str    # 'high' | 'low'
    colors: List[RGB] = field(default_factory=list)

    @property
    def metric_vector(self) -> Tuple[int, int, int, int]:
        """
        Binary encoding in canonical SIVC order:
          S (Subtone / hue)    : warm=1, cool=0
          I (Intensity / chroma): bright=1, muted=0
          V (Value)            : light=1, dark=0
          C (Contrast)         : high=1,  low=0

        This matches the CSV header values exactly.
        """
        return (
            1 if self.hue == "warm" else 0,      # S
            1 if self.chroma == "bright" else 0,  # I
            1 if self.value == "light" else 0,    # V
            1 if self.contrast == "high" else 0,  # C
        )


# ─────────────────────────────────────────────────────────────────────────────
# SPRING  —  SIVC: 1111  →  warm | bright | light | high contrast
# ─────────────────────────────────────────────────────────────────────────────
SPRING = SeasonPalette(
    name="Spring",
    hue="warm",
    chroma="bright",
    value="light",
    contrast="high",
    colors=[
        ( 28,  46, 112),   # bright blue
        ( 60, 137, 188),   # clear sky blue
        (126, 174,  53),   # warm yellow-green
        (115, 189, 168),   # warm teal
        (101,  61, 126),   # warm violet
        (246,  68,  44),   # vivid warm red
        (245,  89,  72),   # coral red
        (251, 134,  48),   # warm orange
        (253, 230,  55),   # warm yellow
        (230, 174,  91),   # warm gold
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# SUMMER  —  SIVC: 0010  →  cool | muted | light | low contrast
# ─────────────────────────────────────────────────────────────────────────────
SUMMER = SeasonPalette(
    name="Summer",
    hue="cool",
    chroma="muted",
    value="light",
    contrast="low",
    colors=[
        (163, 206, 222),   # powder blue
        (153, 141, 179),   # soft lavender
        ( 77,  77, 120),   # muted blue-violet
        ( 55,  51,  87),   # dark muted violet
        ( 29, 163, 148),   # cool teal
        (151, 142, 137),   # muted warm grey
        (124, 116, 169),   # dusty purple
        (185,  68, 137),   # cool berry
        (202,  61,  79),   # cool rose-red
        (241, 175, 193),   # soft blush pink
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# AUTUMN  —  SIVC: 1000  →  warm | muted | dark | low contrast
# ─────────────────────────────────────────────────────────────────────────────
AUTUMN = SeasonPalette(
    name="Autumn",
    hue="warm",
    chroma="muted",
    value="dark",
    contrast="low",
    colors=[
        ( 59,  68,  52),   # dark olive green
        (142, 115,  61),   # warm golden brown
        ( 69,  69,  70),   # warm dark grey
        ( 63,  46,  50),   # dark warm maroon
        (111,  55,  48),   # brick red
        (248,  90,  89),   # muted warm red
        (208,  58,  69),   # warm red
        (220, 101,  78),   # terracotta
        (249, 194,  91),   # golden mustard
        (168, 114,  65),   # caramel brown
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# WINTER  —  SIVC: 0101  →  cool | bright | dark | high contrast
# ─────────────────────────────────────────────────────────────────────────────
WINTER = SeasonPalette(
    name="Winter",
    hue="cool",
    chroma="bright",
    value="dark",
    contrast="high",
    colors=[
        (255, 246, 107),   # icy yellow
        ( 26,  21,  22),   # near black
        ( 47,  27,  76),   # deep violet
        ( 55, 118, 179),   # cool royal blue
        (254, 249, 237),   # icy white
        ( 18,  15,   6),   # true black
        (214,  50,  49),   # vivid cool red
        (158,  23,  76),   # deep cool magenta
        (220,  49,  95),   # vivid cool pink
        ( 39, 100,  14),   # vivid dark green
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# Master list + lookup
# ─────────────────────────────────────────────────────────────────────────────

ALL_SEASONS: List[SeasonPalette] = [SPRING, SUMMER, AUTUMN, WINTER]

SEASON_MAP = {s.name.lower(): s for s in ALL_SEASONS}