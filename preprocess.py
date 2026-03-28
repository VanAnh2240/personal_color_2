"""
preprocess.py
Step 2 & 3 of the Personal Colour Pipeline:
  Step 2: K-Means dominant colour extraction per facial region
  Step 3: Munsell-based season classification

Also exposes PersonalColorPipeline — an end-to-end object
that wraps a trained segmentation model.
"""

import os
import sys
import colorsys
import warnings

import cv2
import numpy as np
from sklearn.cluster import KMeans
import skimage.color as skc   # pip install scikit-image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    IMG_SIZE, PIGMENT_REGIONS,
    KMEANS_CLUSTERS, KMEANS_MAX_ITER, KMEANS_N_INIT,
    KMEANS_COLOR_SPACE,
    WARM_HUE_MIN, WARM_HUE_MAX,
    SEASON_RULES,
)


# ──────────────────────────────────────────────
# Colour-space helpers
# ──────────────────────────────────────────────
def rgb_to_lab(rgb_array: np.ndarray) -> np.ndarray:
    """(N, 3) uint8 → (N, 3) float LAB."""
    img = rgb_array.reshape(-1, 1, 3).astype(np.float32) / 255.0
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    return lab.reshape(-1, 3)


def rgb_to_hsv(rgb_array: np.ndarray) -> np.ndarray:
    """(N, 3) uint8 → (N, 3) float HSV."""
    img = rgb_array.reshape(-1, 1, 3).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return hsv.reshape(-1, 3)


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


# ──────────────────────────────────────────────
# Step 2 — K-Means dominant colour extraction
# ──────────────────────────────────────────────
def extract_dominant_color(mask_region: np.ndarray,
                            full_img_rgb: np.ndarray,
                            n_clusters: int = KMEANS_CLUSTERS) -> dict:
    """
    Parameters
    ----------
    mask_region  : (H, W) bool / binary mask for one region
    full_img_rgb : (H, W, 3) uint8 RGB image
    n_clusters   : K for K-Means

    Returns
    -------
    dict with keys: 'hex', 'rgb', 'lab', 'density'
    """
    pixels = full_img_rgb[mask_region > 0]   # (N, 3) uint8
    if len(pixels) < n_clusters:
        # Fallback: mean colour
        mean_rgb = pixels.mean(axis=0).astype(np.uint8) if len(pixels) > 0 \
                   else np.array([128, 128, 128], dtype=np.uint8)
        return {
            "hex":     rgb_to_hex(*mean_rgb),
            "rgb":     mean_rgb.tolist(),
            "lab":     rgb_to_lab(mean_rgb.reshape(1, 3))[0].tolist(),
            "density": 1.0,
        }

    # Choose colour space for clustering
    if KMEANS_COLOR_SPACE == "LAB":
        cluster_input = rgb_to_lab(pixels)
    elif KMEANS_COLOR_SPACE == "HSV":
        cluster_input = rgb_to_hsv(pixels)
    else:
        cluster_input = pixels.astype(np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KMeans(n_clusters=n_clusters,
                    max_iter=KMEANS_MAX_ITER,
                    n_init=KMEANS_N_INIT,
                    random_state=42)
        km.fit(cluster_input)

    # Dominant cluster = largest
    counts     = np.bincount(km.labels_)
    dom_idx    = int(counts.argmax())
    density    = counts[dom_idx] / counts.sum()

    # Map cluster centre back to RGB
    if KMEANS_COLOR_SPACE == "LAB":
        centre_lab = km.cluster_centers_[dom_idx]
        centre_rgb_f = cv2.cvtColor(
            centre_lab.reshape(1, 1, 3).astype(np.float32),
            cv2.COLOR_LAB2RGB
        )[0, 0] * 255
        centre_rgb = np.clip(centre_rgb_f, 0, 255).astype(np.uint8)
    elif KMEANS_COLOR_SPACE == "HSV":
        centre_hsv = km.cluster_centers_[dom_idx]
        centre_rgb_f = cv2.cvtColor(
            centre_hsv.reshape(1, 1, 3).astype(np.float32),
            cv2.COLOR_HSV2RGB
        )[0, 0] * 255
        centre_rgb = np.clip(centre_rgb_f, 0, 255).astype(np.uint8)
    else:
        centre_rgb = np.clip(km.cluster_centers_[dom_idx], 0, 255).astype(np.uint8)

    return {
        "hex":     rgb_to_hex(*centre_rgb),
        "rgb":     centre_rgb.tolist(),
        "lab":     rgb_to_lab(centre_rgb.reshape(1, 3))[0].tolist(),
        "density": float(density),
    }


# ──────────────────────────────────────────────
# Step 3 — Munsell colour parameters
# ──────────────────────────────────────────────
def rgb_to_munsell_approx(rgb: list) -> dict:
    """
    Approximate Munsell HVC from RGB.
    Uses:
      hue    → from HSV hue (0–360 degrees)
      value  → from HSV value scaled to 0–10
      chroma → approximated from HSV saturation × value × scale factor
    """
    r, g, b = [x / 255.0 for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    hue_deg  = h * 360.0          # 0–360
    value    = v * 10.0           # 0–10
    chroma   = s * v * 20.0       # rough Munsell chroma proxy

    return {
        "hue":    hue_deg,
        "value":  value,
        "chroma": chroma,
    }


def classify_season(munsell: dict) -> str:
    """
    Map Munsell HVC to one of four fashion seasons.

    Warm tone: hue_deg in [WARM_HUE_MIN, WARM_HUE_MAX] (reds/yellows)
    Cool tone: otherwise
    """
    hue    = munsell["hue"]
    value  = munsell["value"]
    chroma = munsell["chroma"]
    warm   = WARM_HUE_MIN <= hue <= WARM_HUE_MAX

    scores = {}

    # Spring: warm + bright + moderate-high chroma
    if warm and value >= SEASON_RULES["Spring"]["value_min"] \
             and chroma >= SEASON_RULES["Spring"]["chroma_min"]:
        scores["Spring"] = (value + chroma)

    # Summer: cool + medium-light + low chroma (muted)
    if (not warm) and value >= SEASON_RULES["Summer"]["value_min"] \
                   and chroma <= SEASON_RULES["Summer"]["chroma_max"]:
        scores["Summer"] = value - chroma

    # Autumn: warm + medium-dark + muted
    if warm and value <= SEASON_RULES["Autumn"].get("value_max", 10) \
             and chroma <= SEASON_RULES["Autumn"]["chroma_max"]:
        scores["Autumn"] = (10 - value) + chroma

    # Winter: cool + high chroma (vivid)
    if (not warm) and chroma >= SEASON_RULES["Winter"]["chroma_min"]:
        scores["Winter"] = chroma + (10 - value)

    if not scores:
        # Fallback: nearest by value/chroma
        if warm:
            season = "Spring" if value > 5 else "Autumn"
        else:
            season = "Winter" if chroma > 5 else "Summer"
        return season

    return max(scores, key=scores.get)


# ──────────────────────────────────────────────
# Full pipeline wrapper
# ──────────────────────────────────────────────
class PersonalColorPipeline:
    """
    End-to-end: portrait image → season classification.

    Parameters
    ----------
    seg_model : torch.nn.Module   — trained DeepLabV3 or ClipUNet
    device    : torch.device
    """

    def __init__(self, seg_model, device):
        import torch
        self.model  = seg_model.eval()
        self.device = device
        self.torch  = torch

    def _segment(self, img_rgb: np.ndarray) -> np.ndarray:
        """Returns (H, W) class-index mask."""
        from src.dataset import get_val_transforms
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        tf  = get_val_transforms(IMG_SIZE)
        aug = tf(image=img_rgb)["image"].unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            logits = self.model(aug)               # (1, C, H, W)
        pred = logits.argmax(dim=1).squeeze(0)     # (H, W)
        mask = pred.cpu().numpy().astype(np.uint8)
        # Resize back to original image size
        return mask

    def run(self, img_path: str) -> dict:
        """
        Parameters
        ----------
        img_path : str  — path to portrait image

        Returns
        -------
        dict:
          season         : str
          dominant_colors: {region: hex_str}
          munsell        : {hue, value, chroma}
          raw_colors     : full detail per region
        """
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot open: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ── Step 1: Segmentation ──
        orig_h, orig_w = img_rgb.shape[:2]
        resized = cv2.resize(img_rgb, (IMG_SIZE[1], IMG_SIZE[0]))
        mask    = self._segment(resized)             # (H, W) on IMG_SIZE

        # ── Step 2: K-Means per region ──
        region_colors = {}
        for region_name, class_idx in PIGMENT_REGIONS.items():
            region_mask = (mask == class_idx).astype(np.uint8)
            if region_mask.sum() == 0:
                continue
            color_info = extract_dominant_color(region_mask, resized)
            region_colors[region_name] = color_info

        if not region_colors:
            return {
                "season":          "Unknown",
                "dominant_colors": {},
                "munsell":         {"hue": 0, "value": 5, "chroma": 0},
                "raw_colors":      {},
            }

        # Average Munsell across regions (skin + hair weighted more)
        weight_map   = {"skin": 3, "hair": 2, "left_eye": 1, "right_eye": 1, "nose": 1}
        total_w      = 0
        avg_rgb      = np.zeros(3, dtype=np.float64)
        for rname, cinfo in region_colors.items():
            w = weight_map.get(rname, 1)
            avg_rgb += np.array(cinfo["rgb"]) * w
            total_w += w
        avg_rgb = (avg_rgb / total_w).astype(int).tolist()

        # ── Step 3: Munsell → Season ──
        munsell = rgb_to_munsell_approx(avg_rgb)
        season  = classify_season(munsell)

        # ── Bonus: undertone + contrast ──
        try:
            from src.utils.colour_science import (
                detect_undertone, seasonal_contrast, get_colour_harmony
            )
            skin_rgb  = region_colors.get("skin",  {}).get("rgb", avg_rgb)
            hair_rgb  = region_colors.get("hair",  {}).get("rgb", avg_rgb)
            undertone = detect_undertone(skin_rgb)
            contrast  = seasonal_contrast(skin_rgb, hair_rgb)
            harmony   = get_colour_harmony(season)
        except Exception:
            undertone, contrast, harmony = "neutral", {}, {}

        return {
            "season":          season,
            "dominant_colors": {r: c["hex"] for r, c in region_colors.items()},
            "munsell":         munsell,
            "raw_colors":      region_colors,
            "avg_rgb":         avg_rgb,
            "undertone":       undertone,
            "contrast":        contrast,
            "harmony":         harmony,
        }


# ──────────────────────────────────────────────
# CLI preprocessing utility (resize + normalise images)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess LaPa images")
    parser.add_argument("--input",  required=True, help="Raw data directory")
    parser.add_argument("--output", required=True, help="Processed directory")
    parser.add_argument("--size",   nargs=2, type=int,
                        default=list(IMG_SIZE), help="H W")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    for fname in sorted(os.listdir(args.input)):
        fpath = os.path.join(args.input, fname)
        img   = cv2.imread(fpath)
        if img is None:
            continue
        resized = cv2.resize(img, (args.size[1], args.size[0]))
        out_path = os.path.join(args.output, fname)
        cv2.imwrite(out_path, resized)
    print(f"Done. Processed images → {args.output}")
