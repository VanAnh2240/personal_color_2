"""
evaluate.py
Evaluation script for LaPa face-parsing models.

Modes
─────
  seg   → run segmentation metrics on the TEST split (has labels)
           LaPa val split has NO labels, so cannot be evaluated here.
  full  → run the full personal-colour pipeline on portrait images
           (segmentation → K-Means → Munsell season classification)
  vis   → generate visual comparisons: original | pred mask | (gt mask)
           works on test split (gt available) or val split (pred only)

Usage
─────
  python evaluate.py --model clipunet --mode seg
  python evaluate.py --model clipunet --mode vis --split test
  python evaluate.py --model clipunet --mode vis --split val
  python evaluate.py --model clipunet --mode full --img portrait.jpg
  python evaluate.py --model clipunet --mode full --img_dir /path/to/imgs
"""

import argparse, os, sys, csv
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (CKPT_DEEPLAB, CKPT_CLIPUNET, RESULT_DIR,
                    ACTIVE_MODEL, LAPA_CLASS_NAMES, BATCH_SIZE, NUM_WORKERS)
from src.dataset import (
    _collect_labeled, _collect_images,
    LapaSegDataset, LapaInferenceDataset,
    get_val_transforms, load_single_image,
    dataset_summary,
)
from src.metrics import SegMetrics
from preprocess  import PersonalColorPipeline
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────
def build_model(model_name: str, device: torch.device):
    if model_name == "deeplab":
        from src.models.system_1_deeplabv3 import DeepLabV3
        model = DeepLabV3(pretrained=False)
        ckpt  = CKPT_DEEPLAB
    else:
        from src.models.system_2_clipunet import ClipUNet
        model = ClipUNet(freeze_clip=False)
        ckpt  = CKPT_CLIPUNET

    if not os.path.exists(ckpt):
        print(f"WARNING: checkpoint not found at {ckpt}.")
        print("Running with random weights (for debugging only).")
    else:
        state = torch.load(ckpt, map_location=device)
        # Support both plain state-dict and wrapped checkpoints
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {ckpt}")

    return model.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1: Segmentation metrics on TEST split
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_segmentation(model_name: str, device: torch.device):
    """
    Compute mIoU / pixel accuracy / per-class IoU on the test split.
    LaPa test split has labels; val split does NOT → only test is used.
    """
    model = build_model(model_name, device)

    te_imgs, te_lbls = _collect_labeled("test")
    test_ds = LapaSegDataset(te_imgs, te_lbls, get_val_transforms())
    test_dl = DataLoader(test_ds, BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

    metrics = SegMetrics()
    for imgs, masks in test_dl:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs).argmax(dim=1)
        metrics.update(preds, masks)

    results = metrics.compute()

    print(f"\n{'═'*55}")
    print(f"  {model_name.upper()} — Test split segmentation metrics")
    print(f"{'═'*55}")
    metrics.print_results()

    # Save to CSV
    out_path = os.path.join(RESULT_DIR, f"{model_name}_test_results.csv")
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["mIoU",      f"{results['mIoU']:.4f}"])
        w.writerow(["pixel_acc", f"{results['pixel_acc']:.4f}"])
        w.writerow(["mean_acc",  f"{results['mean_acc']:.4f}"])
        for cls, val in results["per_class_iou"].items():
            w.writerow([f"iou_{cls}", f"{val:.4f}"])

    print(f"\nSaved → {out_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2: Visual comparison grid
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_visual(model_name: str, device: torch.device,
                    split: str = "test", n_samples: int = 20):
    """
    Save side-by-side visualisations:
      test split → original | pred overlay | GT overlay
      val  split → original | pred overlay  (no GT)
    """
    from src.utils import save_comparison_grid

    model   = build_model(model_name, device)
    vis_dir = os.path.join(RESULT_DIR, f"{model_name}_vis_{split}")
    os.makedirs(vis_dir, exist_ok=True)

    has_labels = (split == "test") or os.path.isdir(
        os.path.join(__import__('config').RAW_DIR, split, "labels"))

    if has_labels:
        imgs_p, lbls_p = _collect_labeled(split)
        imgs_p = imgs_p[:n_samples]; lbls_p = lbls_p[:n_samples]
    else:
        imgs_p = _collect_images(split)[:n_samples]
        lbls_p = [None] * len(imgs_p)

    tf = get_val_transforms()
    for img_path, lbl_path in zip(imgs_p, lbls_p):
        # Load original (un-normalised) for display
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Model inference
        tensor = tf(image=img_rgb)["image"].unsqueeze(0).to(device)
        pred   = model(tensor).argmax(1).squeeze(0).cpu().numpy()

        # GT mask (resized to model output size)
        gt = None
        if lbl_path:
            gt_raw = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt_raw, (pred.shape[1], pred.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

        img_disp = cv2.resize(img_rgb, (pred.shape[1], pred.shape[0]))
        out_name  = os.path.splitext(os.path.basename(img_path))[0] + "_vis.png"
        out_path  = os.path.join(vis_dir, out_name)
        save_comparison_grid(img_disp, pred, gt,
                             dominant_colors={}, season="",
                             out_path=out_path)

    print(f"\n{len(imgs_p)} visualisations saved → {vis_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 3: Full personal-colour pipeline
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_pipeline(model_name: str, device: torch.device,
                      img_dir: str = None, single_img: str = None):
    """Run end-to-end: image → season classification."""
    model    = build_model(model_name, device)
    pipeline = PersonalColorPipeline(model, device)

    if single_img:
        img_paths = [single_img]
    elif img_dir:
        exts = (".jpg", ".jpeg", ".png")
        img_paths = sorted(
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith(exts))
    else:
        raise ValueError("Provide --img or --img_dir")

    out_path = os.path.join(RESULT_DIR, f"{model_name}_pipeline_results.csv")
    os.makedirs(RESULT_DIR, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "season", "undertone",
                    "skin_hex", "hair_hex", "eye_hex", "nose_hex",
                    "hue_angle", "value", "chroma",
                    "contrast_score", "contrast_level"])

        for img_path in img_paths:
            try:
                r = pipeline.run(img_path)
                dc = r["dominant_colors"]
                mu = r["munsell"]
                ct = r.get("contrast", {})
                w.writerow([
                    os.path.basename(img_path),
                    r["season"],
                    r.get("undertone", "?"),
                    dc.get("skin",      "N/A"),
                    dc.get("hair",      "N/A"),
                    dc.get("left_eye",  "N/A"),
                    dc.get("nose",      "N/A"),
                    f"{mu.get('hue_angle', mu.get('hue', 0)):.2f}",
                    f"{mu['value']:.2f}",
                    f"{mu['chroma']:.2f}",
                    ct.get("score", ""),
                    ct.get("level", ""),
                ])
                print(f"  {os.path.basename(img_path):30s}"
                      f"  {r['season']:8s}"
                      f"  undertone={r.get('undertone','?')}"
                      f"  skin={dc.get('skin','?')}")
            except Exception as exc:
                print(f"  ERROR {img_path}: {exc}")

    print(f"\nPipeline results → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default=ACTIVE_MODEL,
                   choices=["deeplab", "clipunet"])
    p.add_argument("--mode",   default="seg",
                   choices=["seg", "vis", "full"],
                   help=("seg=segmentation metrics on TEST split | "
                         "vis=visual masks | "
                         "full=personal-colour pipeline"))
    p.add_argument("--split",  default="test",
                   choices=["test", "val"],
                   help="Which split to visualise (vis mode only)")
    p.add_argument("--n",      default=20, type=int,
                   help="Number of samples for vis mode")
    p.add_argument("--img_dir", default=None)
    p.add_argument("--img",     default=None)
    p.add_argument("--device",  default="auto")
    args = p.parse_args()

    # Dataset structure overview
    dataset_summary()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device : {device}")
    print(f"Model  : {args.model}")
    print(f"Mode   : {args.mode}\n")

    if args.mode == "seg":
        evaluate_segmentation(args.model, device)
    elif args.mode == "vis":
        evaluate_visual(args.model, device,
                        split=args.split, n_samples=args.n)
    else:
        evaluate_pipeline(args.model, device,
                          img_dir=args.img_dir, single_img=args.img)


if __name__ == "__main__":
    main()
