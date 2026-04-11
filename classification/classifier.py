"""
classification/classifier.py  (fixed v3)

Fixes vs v2:
  - classify_hue nhận skin_rgb (không phải lips_rgb) — theo paper DSCAS.
  - Default thresholds cập nhật theo color_utils v3:
      skin_chroma_thresh : 60.0  (was 127)
      contrast_thresh    : 65.0  (was 127)
      value_thresh       : 127.0 (giữ nguyên)
  - Guard None cho từng metric độc lập (giữ từ v2).
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .palettes import ALL_SEASONS, SeasonPalette
from .color_utils import (
    extract_dominant_color,
    classify_hue,
    classify_chroma,
    classify_value,
    classify_contrast,
)

LABEL_SKIN        = 1
LABEL_LEFT_EYE    = 4
LABEL_RIGHT_EYE   = 5
LABEL_NOSE        = 6
LABEL_UPPER_LIP   = 7
LABEL_INNER_MOUTH = 8
LABEL_LOWER_LIP   = 9
LABEL_HAIR        = 10


@dataclass
class ClassificationResult:
    season: SeasonPalette
    dominants: Dict[str, Optional[Tuple[int, int, int]]]
    metrics: Dict[str, Optional[str]]
    user_vector: Tuple[int, ...]
    hamming_scores: Dict[str, float]
    is_bald: bool = False


class PaletteClassifier:

    def __init__(
        self,
        k_clusters: int = 3,
        skin_chroma_thresh: float = 60.0,    # FIX v3: was 127
        value_thresh:       float = 127.0,
        contrast_thresh:    float = 65.0,    # FIX v3: was 127
        hair_label:  int   = LABEL_HAIR,
        lips_label:  tuple = (LABEL_UPPER_LIP, LABEL_LOWER_LIP),
        skin_label:  int   = LABEL_SKIN,
        eye_labels:  tuple = (LABEL_LEFT_EYE, LABEL_RIGHT_EYE),
    ):
        self.k                  = k_clusters
        self.skin_chroma_thresh = skin_chroma_thresh
        self.value_thresh       = value_thresh
        self.contrast_thresh    = contrast_thresh
        self.hair_label         = hair_label
        self.lips_label         = lips_label
        self.skin_label         = skin_label
        self.eye_labels         = eye_labels

    def classify(self, face_bgr: np.ndarray, seg_mask: np.ndarray) -> ClassificationResult:
        t0  = time.time()
        seg = seg_mask.astype(np.int32)

        skin_mask = self._build_mask(seg, [self.skin_label])
        hair_mask = self._build_mask(seg, [self.hair_label])
        lips_mask = self._build_mask(seg, list(self.lips_label))
        eyes_mask = self._build_mask(seg, list(self.eye_labels))

        for name, m in [("skin", skin_mask), ("hair", hair_mask),
                        ("lips", lips_mask), ("eyes", eyes_mask)]:
            print(f"    {name} pixels: {int((m > 0).sum())}")

        is_bald = int((hair_mask > 0).sum()) < 500

        print(">>> [Classifier] Extracting dominant colors...")
        skin_rgb = self._fast_dominant(face_bgr, skin_mask, "skin", prefer_bright=True)
        hair_rgb = (None if is_bald
                    else self._fast_dominant(face_bgr, hair_mask, "hair", prefer_bright=True))
        lips_rgb = self._fast_dominant(face_bgr, lips_mask, "lips", prefer_bright=True)
        eyes_rgb = self._fast_dominant(face_bgr, eyes_mask, "eyes", prefer_bright=False)

        dominants = {"skin": skin_rgb, "hair": hair_rgb,
                     "lips": lips_rgb, "eyes": eyes_rgb}
        print(f"    skin={skin_rgb}  hair={hair_rgb}  lips={lips_rgb}  eyes={eyes_rgb}")

        print(">>> [Classifier] Computing metrics...")

        print("\n========== RAW METRICS DEBUG ==========")
        if skin_rgb is not None:
            sat = classify_chroma.__globals__["_hsv_saturation_255"](skin_rgb)
            val_skin = classify_chroma.__globals__["_hsv_value_255"](skin_rgb)
            print(f"SKIN RGB: {skin_rgb}")
            print(f"  sat: {sat:.2f}")
            print(f"  val_skin: {val_skin:.2f}")

        if hair_rgb is not None:
            val_hair = classify_chroma.__globals__["_hsv_value_255"](hair_rgb)
            print(f"HAIR RGB: {hair_rgb}")
            print(f"  val_hair: {val_hair:.2f}")

        if eyes_rgb is not None:
            val_eyes = classify_chroma.__globals__["_hsv_value_255"](eyes_rgb)
            print(f"EYES RGB: {eyes_rgb}")
            print(f"  val_eyes: {val_eyes:.2f}")

        if hair_rgb is not None and eyes_rgb is not None:
            contrast_raw = abs(val_hair - val_eyes)
            print(f"CONTRAST RAW: {contrast_raw:.2f}")

        print("======================================\n")

        # FIX v3: hue dùng skin_rgb (không phải lips_rgb)
        hue = classify_hue(skin_rgb) if skin_rgb is not None else None

        chroma = (classify_chroma(skin_rgb, threshold=self.skin_chroma_thresh)
                  if skin_rgb is not None else None)

        value = (classify_value(skin_rgb, hair_rgb, eyes_rgb, threshold=self.value_thresh)
                 if (skin_rgb is not None and eyes_rgb is not None) else None)

        contrast = (classify_contrast(hair_rgb, eyes_rgb, threshold=self.contrast_thresh)
                    if eyes_rgb is not None else None)

        metrics = {"hue": hue, "chroma": chroma, "value": value, "contrast": contrast}
        print(f"    hue={hue}  chroma={chroma}  value={value}  contrast={contrast}")

        user_vec = self._build_user_vector(hue, chroma, value, contrast)
        season, hamming_scores = self._match_season(hue, user_vec, contrast is None)

        print(f">>> [Classifier] DONE | {time.time()-t0:.2f}s")
        print(f"    user_vec(SIVC)={user_vec}  → {season.name}  scores={hamming_scores}")

        return ClassificationResult(
            season=season, dominants=dominants, metrics=metrics,
            user_vector=user_vec, hamming_scores=hamming_scores, is_bald=is_bald,
        )

    def _fast_dominant(self, face_bgr, mask, name, prefer_bright, max_pixels=1000):
        n = int((mask > 0).sum())
        if n == 0:
            print(f"    WARNING {name}: no pixels")
            return None
        sampled = mask.copy()
        if n > max_pixels:
            ys, xs = np.where(mask > 0)
            chosen = np.random.choice(len(ys), max_pixels, replace=False)
            sampled[:] = 0
            sampled[ys[chosen], xs[chosen]] = 255
        return extract_dominant_color(face_bgr, sampled, k=self.k, prefer_bright=prefer_bright)

    @staticmethod
    def _build_mask(seg, labels):
        mask = np.zeros(seg.shape, dtype=np.uint8)
        for lbl in labels:
            mask[seg == lbl] = 255
        return mask

    @staticmethod
    def _build_user_vector(hue, chroma, value, contrast) -> tuple:
        """SIVC binary encoding. -1 = unknown."""
        def enc(m, pos): return -1 if m is None else (1 if m == pos else 0)
        return (
            enc(hue,      "warm"),    # S
            enc(chroma,   "bright"),  # I
            enc(value,    "light"),   # V
            enc(contrast, "high"),    # C
        )

    @staticmethod
    def _hamming(user_vec, season_vec, ignore_contrast) -> float:
        dist = 0
        for i, (u, s) in enumerate(zip(user_vec, season_vec)):
            if i == 3 and ignore_contrast: continue
            if u == -1: continue
            if u != s:  dist += 1
        return float(dist)

    def _match_season(self, hue, user_vec, ignore_contrast):
        _COOL = {"cool", "cold"}
        if hue == "warm":
            candidates = [s for s in ALL_SEASONS if s.hue == "warm"]
        elif hue in _COOL:
            candidates = [s for s in ALL_SEASONS if s.hue == "cool"]
        else:
            candidates = list(ALL_SEASONS)

        if not candidates:
            candidates = list(ALL_SEASONS)

        scores = {s.name: self._hamming(user_vec, s.metric_vector, ignore_contrast)
                  for s in candidates}
        best   = min(scores, key=scores.__getitem__)
        return next(s for s in ALL_SEASONS if s.name == best), scores
    
