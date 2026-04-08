# File train.py

"""
train.py  –  Training script for DeepLabV3 / ClipUNet on LaPa.

LaPa split reality
──────────────────
  train/  images + labels  → supervised training
  val/    images ONLY      → no labels → inference / visual preview only
  test/   images + labels  → final evaluation  (run evaluate.py separately)

Strategy
────────
  --kfold  (recommended): K-Fold CV over train split only → proper held-out
           mIoU per fold, then save per-fold checkpoints.
  default : carve 10 % of train as a labeled monitor set; track monitor mIoU
            for early stopping / checkpoint selection.
  --resume : resume from a saved checkpoint (optimizer + scheduler state).

Usage
─────
  python train.py --model clipunet --epochs 30 --kfold
  python train.py --model deeplab  --epochs 50
  python train.py --model clipunet --resume checkpoints/system_2_clipunet.pth
"""

import argparse, os, sys, csv, time, random, json
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ACTIVE_MODEL, NUM_EPOCHS, LR, WEIGHT_DECAY,
    SCHEDULER, SEED, K_FOLDS,
    CKPT_DEEPLAB, CKPT_CLIPUNET, RESULT_DIR,
    BATCH_SIZE, NUM_WORKERS,
)
from src.dataset import (
    _collect_labeled, LapaSegDataset,
    get_train_transforms, get_val_transforms,
    get_kfold_dataloaders,
)
from src.metrics import SegMetrics, ComboLoss
from src.utils   import save_checkpoint, load_checkpoint, save_best, log_epoch
from torch.utils.data import DataLoader


# ── Reproducibility ──────────────────────────────────────────────────────────
def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── Model factory ─────────────────────────────────────────────────────────────
def build_model(name):
    if name == "deeplab":
        from src.models.system_1_deeplabv3 import DeepLabV3
        return DeepLabV3(pretrained=True), CKPT_DEEPLAB
    from src.models.system_2_clipunet import ClipUNet
    return ClipUNet(freeze_clip=True), CKPT_CLIPUNET


def build_scheduler(opt, epochs):
    if SCHEDULER == "cosine":
        return CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    return StepLR(opt, step_size=max(1, epochs // 3), gamma=0.1)


# ── Epoch helpers ─────────────────────────────────────────────────────────────
def train_one(model, loader, opt, crit, device, scaler=None):
    model.train()
    total = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        opt.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                loss = crit(model(imgs), masks)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss = crit(model(imgs), masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def eval_labeled(model, loader, crit, device, metrics):
    """Evaluation on a labeled split → (loss, mIoU, full_results_dict)."""
    model.eval(); metrics.reset(); total = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        total += crit(logits, masks).item()
        metrics.update(logits.argmax(1), masks)
    res = metrics.compute()
    return total / max(len(loader), 1), res["mIoU"], res


# ── Standard training (train split with 10 % labeled monitor) ─────────────────
def run_standard(model_name, epochs, device, resume=None):
    from sklearn.model_selection import train_test_split

    model, ckpt_path = build_model(model_name)
    model = model.to(device)

    # 90 % for training, 10 % as labeled monitor (val labels don't exist)
    all_imgs, all_lbls = _collect_labeled("train")
    tr_imgs, mo_imgs, tr_lbls, mo_lbls = train_test_split(
        all_imgs, all_lbls, test_size=0.10, random_state=SEED, shuffle=True
    )
    print(f"\n  Train={len(tr_imgs)}  Monitor={len(mo_imgs)}  "
          f"(10 % labeled hold-out — LaPa val has no labels)\n")

    train_ds   = LapaSegDataset(tr_imgs, tr_lbls, get_train_transforms())
    monitor_ds = LapaSegDataset(mo_imgs, mo_lbls, get_val_transforms())
    train_dl   = DataLoader(train_ds,   BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            drop_last=True)
    monitor_dl = DataLoader(monitor_ds, BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    crit     = ComboLoss()
    opt      = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)
    sched    = build_scheduler(opt, epochs)
    scaler   = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    metrics  = SegMetrics()
    start_ep = 1
    best_miou = 0.0

    if resume and os.path.exists(resume):
        state     = load_checkpoint(resume, model, opt, sched, device)
        start_ep  = state.get("epoch", 0) + 1
        best_miou = state.get("best_miou", 0.0)

    log_path = os.path.join(RESULT_DIR, f"{model_name}_train_log.csv")
    mode = "a" if (resume and os.path.exists(log_path)) else "w"

    with open(log_path, mode, newline="") as f:
        w = csv.writer(f)
        if mode == "w":
            w.writerow(["epoch", "train_loss", "monitor_loss",
                        "monitor_mIoU", "lr"])

        for ep in range(start_ep, epochs + 1):
            t0      = time.time()
            tr_loss = train_one(model, train_dl, opt, crit, device, scaler)
            mo_loss, mo_miou, _ = eval_labeled(
                model, monitor_dl, crit, device, metrics)
            lr = opt.param_groups[0]["lr"]
            sched.step()

            log_epoch(model_name, ep, epochs,
                      tr_loss, mo_loss, mo_miou, lr,
                      time.time() - t0, best_miou)
            w.writerow([ep, f"{tr_loss:.5f}", f"{mo_loss:.5f}",
                        f"{mo_miou:.5f}", f"{lr:.2e}"])

            best_miou = save_best(
                model, ckpt_path, mo_miou, best_miou,
                extra={"epoch": ep, "best_miou": mo_miou,
                       "optimizer": opt.state_dict(),
                       "scheduler": sched.state_dict()})

            if ep % 10 == 0:
                save_checkpoint(
                    {"epoch": ep, "model": model.state_dict(),
                     "optimizer": opt.state_dict(), "best_miou": best_miou},
                    ckpt_path.replace(".pth", f"_ep{ep:03d}.pth"))

    print(f"\nTraining complete.")
    print(f"  Best monitor mIoU : {best_miou:.4f}")
    print(f"  Log               → {log_path}")
    print(f"  Best checkpoint   → {ckpt_path}")
    print(f"\n  → Run  python evaluate.py --model {model_name} --mode seg"
          f"  for test-set mIoU.\n")

    try:
        from src.utils import plot_training_curves
        plot_training_curves(
            log_path,
            os.path.join(RESULT_DIR, f"{model_name}_curves.png"))
    except Exception:
        pass


# ── K-Fold cross-validation (on labeled train split only) ────────────────────
def run_kfold(model_name, epochs, device):
    """
    3-Fold CV as reported in the paper.
    Each fold uses a disjoint 1/3 of the labeled train split as held-out val.
    LaPa val (no labels) is NOT included in the fold pool.
    """
    fold_mious = []
    log_path   = os.path.join(RESULT_DIR, f"{model_name}_kfold_log.csv")

    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "epoch", "train_loss", "val_loss", "val_mIoU"])

        for fold, tr_dl, va_dl in get_kfold_dataloaders(k=K_FOLDS):
            print(f"\n{'═'*54}")
            print(f"  FOLD {fold + 1} / {K_FOLDS}")
            print(f"{'═'*54}")

            model, base_ckpt = build_model(model_name)
            model   = model.to(device)
            crit    = ComboLoss()
            opt     = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR, weight_decay=WEIGHT_DECAY)
            sched   = build_scheduler(opt, epochs)
            scaler  = (torch.cuda.amp.GradScaler()
                       if device.type == "cuda" else None)
            metrics = SegMetrics()
            best_m  = 0.0
            f_ckpt  = base_ckpt.replace(".pth", f"_fold{fold+1}.pth")

            for ep in range(1, epochs + 1):
                t0      = time.time()
                tr_loss = train_one(model, tr_dl, opt, crit, device, scaler)
                va_loss, va_miou, _ = eval_labeled(
                    model, va_dl, crit, device, metrics)
                lr = opt.param_groups[0]["lr"]
                sched.step()

                log_epoch(model_name, ep, epochs,
                          tr_loss, va_loss, va_miou, lr,
                          time.time() - t0, best_m)
                w.writerow([fold + 1, ep, f"{tr_loss:.5f}",
                            f"{va_loss:.5f}", f"{va_miou:.5f}"])
                best_m = save_best(model, f_ckpt, va_miou, best_m,
                                   extra={"epoch": ep, "fold": fold + 1})

            fold_mious.append(best_m)
            print(f"\n  Fold {fold+1} best mIoU = {best_m:.4f}")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    avg = float(np.mean(fold_mious))
    std = float(np.std(fold_mious))
    print(f"\n{'─'*54}")
    print(f"  K-Fold mIoUs : {[f'{v:.3f}' for v in fold_mious]}")
    print(f"  Mean mIoU    : {avg:.4f} ± {std:.4f}")
    print(f"{'─'*54}\n")

    summary = {
        "model":        model_name,
        "k_folds":      K_FOLDS,
        "fold_mious":   fold_mious,
        "average_mIoU": avg,
        "std_mIoU":     std,
    }
    summary_path = os.path.join(
        RESULT_DIR, f"{model_name}_kfold_summary.json")
    with open(summary_path, "w") as jf:
        json.dump(summary, jf, indent=2)
    print(f"Summary → {summary_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Train DeepLabV3 or ClipUNet on LaPa dataset")
    p.add_argument("--model",  default=ACTIVE_MODEL,
                   choices=["deeplab", "clipunet"])
    p.add_argument("--epochs", default=NUM_EPOCHS, type=int)
    p.add_argument("--kfold",  action="store_true",
                   help="K-Fold CV on labeled train split (recommended for mIoU tracking)")
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--device", default="auto",
                   help="'auto' | 'cuda' | 'cpu' | 'mps'")
    args = p.parse_args()

    seed_everything()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n{'━'*54}")
    print(f"  Model   : {args.model}")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Mode    : {'K-Fold CV' if args.kfold else 'Standard'}")
    if args.resume:
        print(f"  Resume  : {args.resume}")
    print(f"{'━'*54}\n")

    if args.kfold:
        run_kfold(args.model, args.epochs, device)
    else:
        run_standard(args.model, args.epochs, device, resume=args.resume)


if __name__ == "__main__":
    main()
