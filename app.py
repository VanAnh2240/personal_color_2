"""
app.py  (fixed v2)
==================
End-to-end CLI: face image → DeepLab segmentation → seg visualisation → palette classification.

Usage
-----
    # Full pipeline, save both outputs
    python app.py --img portrait.jpg --checkpoint checkpoints/system_1_deeplabv3.pth \\
        --hair_label 10 --save palette_result.png --save_seg seg_result.png

    # Show seg overlay in a window (no --save_seg → opens cv2 window)
    python app.py --img portrait.jpg --checkpoint checkpoints/system_1_deeplabv3.pth \\
        --hair_label 10

    # Skip model inference, load pre-saved mask .npy
    python app.py --img portrait.jpg --seg_npy mask.npy \\
        --hair_label 10 --save result.png
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from classification import PaletteClassifier
from classification.visualizer import save_result_figure
from seg_visualizer import save_seg_figure, show_seg_window
from config import RESULT_IMG

DEFAULT_CHECKPOINT = "checkpoints/system_1_deeplabv3.pth"


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, num_classes: int = 11, device: str = "cpu"):
    from src.models.system_1_deeplabv3 import DeepLabV3
    model = DeepLabV3(num_classes=num_classes)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing / inference
# ─────────────────────────────────────────────────────────────────────────────

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess(bgr_img: np.ndarray, input_size: int = 473) -> torch.Tensor:
    rgb     = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size, input_size))
    tensor  = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    tensor  = (tensor - _MEAN) / _STD
    return tensor.unsqueeze(0)


@torch.no_grad()
def segment(model, bgr_img: np.ndarray, device: str = "cpu") -> np.ndarray:
    orig_h, orig_w = bgr_img.shape[:2]
    tensor = preprocess(bgr_img).to(device)
    output = model(tensor)
    logits = output["out"] if isinstance(output, dict) else output
    logits = F.interpolate(logits, size=(orig_h, orig_w),
                           mode="bilinear", align_corners=False)
    return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Personal Colour Analysis + Seg Viz")
    parser.add_argument("--img",         required=True)
    parser.add_argument("--checkpoint",  default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seg_npy",     default=None,
                        help="Pre-saved segmentation .npy mask — skips model inference")
    # FIX: default=None so flags are truly optional (was "palette_result.png" / "seg_result.png")
    parser.add_argument("--save",        default=f"{RESULT_IMG}/palette.png", help="Save palette result PNG")
    parser.add_argument("--save_seg",    default=f"{RESULT_IMG}/seg.png", help="Save segmentation figure PNG")
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--num_classes", default=11,    type=int)
    parser.add_argument("--input_size",  default=473,   type=int)
    parser.add_argument("--chroma_thresh",   default=127.0, type=float)
    parser.add_argument("--value_thresh",    default=127.0, type=float)
    parser.add_argument("--contrast_thresh", default=127.0, type=float)
    parser.add_argument("--hair_label",      default=10,    type=int)
    args = parser.parse_args()

    t_start = time.time()

    # STEP 1 — load image
    print(">>> [STEP 1] Loading image...")
    bgr = cv2.imread(args.img)
    if bgr is None:
        sys.exit(f"[ERROR] Cannot read image: {args.img}")
    print(f"    Shape: {bgr.shape} | {time.time()-t_start:.2f}s")

    # STEP 2 — segmentation
    if args.seg_npy:
        print(f">>> [STEP 2] Loading pre-saved mask: {args.seg_npy}")
        seg_mask = np.load(args.seg_npy)
    else:
        if not args.checkpoint:
            sys.exit("[ERROR] Provide --checkpoint or --seg_npy")
        print(">>> [STEP 2] Loading model...")
        t1 = time.time()
        model = load_model(args.checkpoint, args.num_classes, args.device)
        print(f"    Model loaded | {time.time()-t1:.2f}s")

        print(">>> [STEP 3] Running segmentation...")
        t2 = time.time()
        seg_mask = segment(model, bgr, device=args.device)
        print(f"    Seg done | {time.time()-t2:.2f}s")

    print(f"    Unique labels: {np.unique(seg_mask)}")

    # STEP 3 — segmentation visualisation
    print(">>> [STEP 4] Segmentation visualisation...")
    if args.save_seg:
        save_seg_figure(bgr, seg_mask, output_path=args.save_seg, alpha=0.55)
        print(f"    Saved → {args.save_seg}")
    else:
        show_seg_window(bgr, seg_mask)

    # STEP 4 — palette classification
    print(">>> [STEP 5] Palette classification...")
    clf = PaletteClassifier(
        skin_chroma_thresh=args.chroma_thresh,
        value_thresh=args.value_thresh,
        contrast_thresh=args.contrast_thresh,
        hair_label=args.hair_label,
    )
    result = clf.classify(bgr, seg_mask)

    print("\n" + "=" * 52)
    print(f"  Season      : {result.season.name}")
    print(f"  Metrics     : {result.metrics}")
    print(f"  User vector : {result.user_vector}")
    print(f"  Hamming dist: {result.hamming_scores}")
    print(f"  Is bald     : {result.is_bald}")
    print("  Dominants   :")
    for region, rgb in result.dominants.items():
        print(f"    {region:6s} → {rgb}")
    print("=" * 52 + "\n")

    # STEP 6 — palette result figure
    print(">>> [STEP 6] Palette result figure...")
    from classification.visualizer import draw_result_overlay, draw_palette_strip

    annotated = draw_result_overlay(bgr, result)
    pal_strip = draw_palette_strip(result.season)

    # fix width mismatch
    pal_strip = cv2.resize(pal_strip, (annotated.shape[1], 50))

    out = np.vstack([annotated, pal_strip])

    if args.save:
        save_result_figure(bgr, result, args.save)
        print(f"    Saved → {args.save}")
    else:
        cv2.imwrite("palette_result.png", out)
        print("    Saved → palette_result.png")

    print(f">>> TOTAL TIME: {time.time()-t_start:.2f}s")


if __name__ == "__main__":
    main()