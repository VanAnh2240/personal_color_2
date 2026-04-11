"""
seg_visualizer.py
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Optional, Tuple

DEFAULT_LABEL_MAP: Dict[int, Tuple[str, Tuple[int, int, int]]] = {
    0:  ("background",    (0,   0,   0)),
    1:  ("skin",          (0,   153, 255)),
    2:  ("left eyebrow",  (102, 255, 153)),
    3:  ("right eyebrow", (0,   240, 153)),
    4:  ("left eye",      (255, 255, 102)),
    5:  ("right eye",     (255, 225, 204)),
    6:  ("nose",          (255, 153, 0)),
    7:  ("upper lip",     (255, 102, 255)),
    8:  ("inner mouth",   (102, 0,   51)),
    9:  ("lower lip",     (255, 204, 255)),
    10: ("hair",          (255, 0,   102)),
}


def _get_label_map(
    base_map: Dict[int, Tuple[str, Tuple[int, int, int]]],
    hair_label: Optional[int] = None,
) -> Dict[int, Tuple[str, Tuple[int, int, int]]]:
    lmap = dict(base_map)
    if hair_label is not None and hair_label != 10:
        hair_entry = lmap.pop(10, ("hair", (255, 0, 102)))
        lmap[hair_label] = (hair_entry[0], hair_entry[1])
    return lmap


def lmap_key(lmap: dict, name_fragment: str) -> Optional[int]:
    """Tìm label index đầu tiên có name chứa name_fragment."""
    for idx, (name, _) in lmap.items():
        if name_fragment.lower() == name.lower():
            return idx
    return None


def _build_colour_mask(
    seg: np.ndarray,
    label_map: Dict[int, Tuple[str, Tuple[int, int, int]]],
) -> np.ndarray:
    colour_mask = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for label_idx, (_, rgb) in label_map.items():
        colour_mask[seg == label_idx] = [rgb[2], rgb[1], rgb[0]]  # RGB→BGR
    return colour_mask


def draw_seg_overlay(
    face_bgr: np.ndarray,
    seg_mask: np.ndarray,
    alpha: float = 0.55,
    label_map: Optional[Dict] = None,
    hair_label: Optional[int] = None,
) -> np.ndarray:
    lmap = _get_label_map(label_map or DEFAULT_LABEL_MAP)
    colour_mask = _build_colour_mask(seg_mask.astype(np.int32), lmap)
    if colour_mask.shape[:2] != face_bgr.shape[:2]:
        colour_mask = cv2.resize(colour_mask,
                                 (face_bgr.shape[1], face_bgr.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(face_bgr, 1 - alpha, colour_mask, alpha, 0)


def draw_seg_legend(
    label_map: Dict[int, Tuple[str, Tuple[int, int, int]]],
    seg_mask: np.ndarray,
    width: int = 200,
    row_h: int = 28,
    font_scale: float = 0.48,
) -> np.ndarray:
    present = set(np.unique(seg_mask.astype(np.int32)))
    rows = [(idx, name, rgb)
            for idx, (name, rgb) in sorted(label_map.items())
            if idx in present]

    h = max(row_h * len(rows), 1)
    legend = np.full((h, width, 3), 30, dtype=np.uint8)

    for i, (idx, name, rgb) in enumerate(rows):
        y0, y1 = i * row_h, (i + 1) * row_h
        swatch_w = width // 4
        legend[y0:y1, 0:swatch_w] = [rgb[2], rgb[1], rgb[0]]
        bri = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        txt_col = (0, 0, 0) if bri > 128 else (255, 255, 255)
        cv2.putText(legend, f"{idx}: {name}",
                    (swatch_w + 6, y0 + row_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (210, 210, 210), 1, cv2.LINE_AA)
        cv2.putText(legend, str(idx),
                    (4, y0 + row_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    txt_col, 1, cv2.LINE_AA)
    return legend


def draw_region_panels(
    face_bgr: np.ndarray,
    seg_mask: np.ndarray,
    regions: Dict[str, int],
    panel_size: int = 160,
) -> np.ndarray:
    panels = []
    seg = seg_mask.astype(np.int32)

    for region_name, label_idx in regions.items():
        panel = np.full((panel_size, panel_size, 3), 40, dtype=np.uint8)
        mask  = (seg == label_idx).astype(np.uint8) * 255

        if mask.sum() > 0:
            ys, xs = np.where(mask > 0)
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
            pad = 8
            y0 = max(0, y0 - pad);  y1 = min(face_bgr.shape[0], y1 + pad)
            x0 = max(0, x0 - pad);  x1 = min(face_bgr.shape[1], x1 + pad)

            crop      = face_bgr[y0:y1, x0:x1].copy()
            crop_mask = mask[y0:y1, x0:x1]
            crop[crop_mask == 0] = (40, 40, 40)

            # Preserve aspect ratio
            h_crop, w_crop = crop.shape[:2]
            scale  = min(panel_size / max(h_crop, 1), panel_size / max(w_crop, 1))
            new_h  = max(1, int(h_crop * scale))
            new_w  = max(1, int(w_crop * scale))
            resized = cv2.resize(crop, (new_w, new_h))

            # Centre in panel
            y_off = (panel_size - new_h) // 2
            x_off = (panel_size - new_w) // 2
            panel[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        else:
            cv2.putText(panel, "N/A",
                        (panel_size // 2 - 15, panel_size // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)

        # Label bar at bottom of each panel
        cv2.rectangle(panel, (0, panel_size - 22),
                      (panel_size, panel_size), (20, 20, 20), -1)
        cv2.putText(panel, region_name,
                    (4, panel_size - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (220, 220, 220), 1, cv2.LINE_AA)

        panels.append(panel)   # ← append mọi lúc, không chỉ khi N/A

    return (np.hstack(panels) if panels
            else np.zeros((panel_size, panel_size, 3), dtype=np.uint8))


def save_seg_figure(
    face_bgr: np.ndarray,
    seg_mask: np.ndarray,
    output_path: str,
    label_map: Optional[Dict] = None,
    alpha: float = 0.55,
    target_h: int = 420,
) -> None:
    lmap = _get_label_map(label_map or DEFAULT_LABEL_MAP)

    h0, w0 = face_bgr.shape[:2]
    scale   = target_h / h0
    W       = int(w0 * scale)
    orig    = cv2.resize(face_bgr, (W, target_h))
    overlay = draw_seg_overlay(face_bgr, seg_mask, alpha=alpha, label_map=lmap)
    overlay = cv2.resize(overlay, (W, target_h))

    legend = draw_seg_legend(lmap, seg_mask, width=180, row_h=30)
    leg_h  = legend.shape[0]
    if leg_h < target_h:
        pad    = np.full((target_h - leg_h, 180, 3), 30, dtype=np.uint8)
        legend = np.vstack([legend, pad])
    else:
        legend = legend[:target_h]

    top_row = np.hstack([orig, overlay, legend])
    total_w = top_row.shape[1]

    def label_bar(text, width, height=26):
        bar = np.full((height, width, 3), 25, dtype=np.uint8)
        cv2.putText(bar, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (180, 180, 180), 1, cv2.LINE_AA)
        return bar

    regions_to_show = {}
    for display_name, search_str in [
        ("skin",       "skin"),
        ("hair",       "hair"),
        ("lips",       "upper lip"),
        ("left eye",   "left eye"),
        ("right eye",  "right eye"),
        ("nose",       "nose"),
    ]:
        idx = lmap_key(lmap, search_str)
        if idx is not None:
            regions_to_show[display_name] = idx

    panel_size   = total_w // max(len(regions_to_show), 1)
    region_strip = draw_region_panels(face_bgr, seg_mask,
                                      regions_to_show, panel_size=panel_size)
    region_strip = cv2.resize(region_strip, (total_w, panel_size))

    figure = np.vstack([
        label_bar("Segmentation overlay  (DeepLab)", total_w),
        top_row,
        label_bar("Facial regions", total_w),
        region_strip,
    ])
    cv2.imwrite(output_path, figure)
    print(f"[seg_visualizer] Saved → {output_path}")


def show_seg_window(
    face_bgr: np.ndarray,
    seg_mask: np.ndarray,
    label_map: Optional[Dict] = None,
    hair_label: Optional[int] = None,
    alpha: float = 0.55,
    window_name: str = "Segmentation",
) -> None:
    lmap    = _get_label_map(label_map or DEFAULT_LABEL_MAP)
    overlay = draw_seg_overlay(face_bgr, seg_mask, alpha=alpha, label_map=lmap)
    combined = np.hstack([
        cv2.resize(face_bgr, (400, 400)),
        cv2.resize(overlay,  (400, 400)),
    ])
    cv2.imshow(window_name, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()