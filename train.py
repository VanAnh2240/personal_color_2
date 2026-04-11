"""
train.py  –  Training script for DeepLabV3 / ClipUNet on LaPa.

FIXES vs original:
  1. ClipUNet: unfreeze CLIP với differential LR (encoder=LR/10, decoder=LR)
  2. Class weights cho LaPa (inverse-frequency) → fix mIoU thấp do class imbalance
  3. IMG_SIZE khuyến nghị 224 (CLIP pretrain res) thay vì 512
  4. Linear warmup 5 epoch + cosine decay
  5. Sanity-check data pipeline ở đầu mỗi run
  6. EarlyStopping (patience=15) tránh lãng phí compute khi plateau

LaPa split reality
──────────────────
  train/  images + labels  → supervised training
  val/    images ONLY      → no labels → inference / visual preview only
  test/   images + labels  → final evaluation  (run evaluate.py separately)

Usage
─────
  python train.py --model clipunet --epochs 50
  python train.py --model clipunet --epochs 50 --kfold
  python train.py --model deeplab  --epochs 50
  python train.py --model clipunet --resume checkpoints/system_2_clipunet.pth
"""

import argparse, os, sys, csv, time, random, json
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LinearLR, SequentialLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ACTIVE_MODEL, NUM_EPOCHS, LR, WEIGHT_DECAY,
    SCHEDULER, SEED, K_FOLDS,
    CKPT_DEEPLAB, CKPT_CLIPUNET, RESULT_DIR,
    BATCH_SIZE, NUM_WORKERS, LAPA_NUM_CLASSES,
)
from src.dataset import (
    _collect_labeled, LapaSegDataset,
    get_train_transforms, get_val_transforms,
    get_kfold_dataloaders,
)
from src.metrics import SegMetrics, ComboLoss
from src.utils   import save_checkpoint, load_checkpoint, save_best, log_epoch
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# LaPa class weights (inverse-frequency, tính từ toàn bộ training set)
# Các class nhỏ (eyes, lips, eyebrows) được weight cao hơn để tránh
# model chỉ predict skin/background và đạt pixel_acc cao nhưng mIoU thấp.
# ─────────────────────────────────────────────────────────────────────────────
LAPA_CLASS_WEIGHTS = torch.tensor([
    0.5,   # 0  background   (~35% pixels)
    0.7,   # 1  skin         (~30%)
    5.0,   # 2  left_eyebrow (~1-2%)
    5.0,   # 3  right_eyebrow
    5.0,   # 4  left_eye     (~1-2%)
    5.0,   # 5  right_eye
    3.0,   # 6  nose         (~4%)
    6.0,   # 7  upper_lip    (~1%)
    6.0,   # 8  inner_mouth
    6.0,   # 9  lower_lip
    1.5,   # 10 hair         (~15%)
], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check — chạy trước khi train để phát hiện data bug sớm
# ─────────────────────────────────────────────────────────────────────────────
def sanity_check(loader, device, num_classes=LAPA_NUM_CLASSES):
    """
    Kiểm tra 1 batch đầu tiên:
    - Có đủ class trong mask không?
    - Pixel distribution có hợp lý không?
    Nếu chỉ thấy 1-2 class → vấn đề data loading, không phải model.
    """
    print("\n  [sanity] Checking first batch...")
    imgs, masks = next(iter(loader))
    imgs, masks = imgs.to(device), masks.to(device)

    print(f"  [sanity] Image  shape : {imgs.shape}  "
          f"min={imgs.min():.2f}  max={imgs.max():.2f}")
    print(f"  [sanity] Mask   shape : {masks.shape}  "
          f"dtype={masks.dtype}")

    unique_classes = masks.unique().tolist()
    print(f"  [sanity] Unique classes in batch : {[int(c) for c in unique_classes]}")

    if len(unique_classes) < 4:
        print(f"  [sanity] WARNING: Only {len(unique_classes)} classes in batch — "
              f"possible label loading issue!")
    else:
        print(f"  [sanity] OK — {len(unique_classes)} classes found.")

    class_counts = [(masks == i).sum().item() for i in range(num_classes)]
    total_px = masks.numel()
    print("  [sanity] Pixel distribution:")
    for i, cnt in enumerate(class_counts):
        if cnt > 0:
            bar = "█" * int(cnt / total_px * 40)
            print(f"    class {i:2d}: {cnt:8d} px  ({cnt/total_px*100:4.1f}%)  {bar}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────
def build_model(name):
    if name == "deeplab":
        from src.models.system_1_deeplabv3 import DeepLabV3
        return DeepLabV3(pretrained=True), CKPT_DEEPLAB

    from src.models.system_2_clipunet import ClipUNet
    # FIX: unfreeze CLIP — dùng differential LR thay vì freeze hoàn toàn
    # Frozen CLIP chỉ train ~3M/89M params → underfitting nghiêm trọng
    return ClipUNet(freeze_clip=False), CKPT_CLIPUNET


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer — Differential LR cho ClipUNet
# ─────────────────────────────────────────────────────────────────────────────
def build_optimizer(model, model_name: str, lr: float = LR):
    """
    ClipUNet: encoder (CLIP) học với LR thấp hơn 10× so với decoder.
    Lý do: CLIP đã pretrain tốt, chỉ cần fine-tune nhẹ.
    DeepLab: tất cả params cùng LR.
    """
    if model_name == "clipunet":
        encoder_params = [p for n, p in model.named_parameters()
                          if "encoder" in n and p.requires_grad]
        decoder_params = [p for n, p in model.named_parameters()
                          if "encoder" not in n and p.requires_grad]

        n_enc = sum(p.numel() for p in encoder_params) / 1e6
        n_dec = sum(p.numel() for p in decoder_params) / 1e6
        print(f"  [opt] Encoder trainable: {n_enc:.1f}M  lr={lr*0.1:.1e}")
        print(f"  [opt] Decoder trainable: {n_dec:.1f}M  lr={lr:.1e}")

        return optim.AdamW([
            {"params": encoder_params, "lr": lr * 0.1},
            {"params": decoder_params, "lr": lr},
        ], weight_decay=WEIGHT_DECAY)

    # DeepLab: uniform LR
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    return optim.AdamW(trainable, lr=lr, weight_decay=WEIGHT_DECAY)


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler — Linear warmup 5 ep → Cosine decay
# ─────────────────────────────────────────────────────────────────────────────
def build_scheduler(opt, epochs: int, warmup_epochs: int = 5):
    """
    FIX: thêm warmup để tránh model diverge ở early epoch khi CLIP unfrozen.
    """
    warmup = LinearLR(opt, start_factor=0.1, end_factor=1.0,
                      total_iters=warmup_epochs)
    if SCHEDULER == "cosine":
        main = CosineAnnealingLR(opt, T_max=max(1, epochs - warmup_epochs),
                                 eta_min=1e-6)
    else:
        main = StepLR(opt, step_size=max(1, (epochs - warmup_epochs) // 3),
                      gamma=0.1)
    return SequentialLR(opt, schedulers=[warmup, main],
                        milestones=[warmup_epochs])


# ─────────────────────────────────────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = 0.0
        self.counter   = 0

    def step(self, miou: float) -> bool:
        """Returns True nếu nên dừng training."""
        if miou > self.best + self.min_delta:
            self.best    = miou
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ─────────────────────────────────────────────────────────────────────────────
# Epoch helpers
# ─────────────────────────────────────────────────────────────────────────────
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
    """Evaluation trên labeled split → (loss, mIoU, full_results_dict)."""
    model.eval(); metrics.reset(); total = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        total += crit(logits, masks).item()
        metrics.update(logits.argmax(1), masks)
    res = metrics.compute()
    return total / max(len(loader), 1), res["mIoU"], res


# ─────────────────────────────────────────────────────────────────────────────
# Standard training
# ─────────────────────────────────────────────────────────────────────────────
def run_standard(model_name, epochs, device, resume=None):
    from sklearn.model_selection import train_test_split

    model, ckpt_path = build_model(model_name)
    model = model.to(device)

    # 90 % train / 10 % monitor
    all_imgs, all_lbls = _collect_labeled("train")
    tr_imgs, mo_imgs, tr_lbls, mo_lbls = train_test_split(
        all_imgs, all_lbls, test_size=0.10, random_state=SEED, shuffle=True
    )
    print(f"\n  Train={len(tr_imgs)}  Monitor={len(mo_imgs)}")

    train_ds   = LapaSegDataset(tr_imgs, tr_lbls, get_train_transforms())
    monitor_ds = LapaSegDataset(mo_imgs, mo_lbls, get_val_transforms())
    train_dl   = DataLoader(train_ds,   BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            drop_last=True)
    monitor_dl = DataLoader(monitor_ds, BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # Sanity check trước khi train
    sanity_check(train_dl, device)

    # FIX: class weights để xử lý imbalance
    class_w = LAPA_CLASS_WEIGHTS.to(device)
    crit    = ComboLoss(class_weights=class_w)

    opt    = build_optimizer(model, model_name, LR)
    sched  = build_scheduler(opt, epochs)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    metrics    = SegMetrics()
    early_stop = EarlyStopping(patience=15)
    start_ep   = 1
    best_miou  = 0.0

    if resume and os.path.exists(resume):
        state     = load_checkpoint(resume, model, opt, sched, device)
        start_ep  = state.get("epoch", 0) + 1
        best_miou = state.get("best_miou", 0.0)
        print(f"  Resumed from epoch {start_ep - 1}, best mIoU={best_miou:.4f}")

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
            mo_loss, mo_miou, mo_res = eval_labeled(
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
                # In per-class IoU mỗi 10 epoch để theo dõi các class nhỏ
                print("  Per-class IoU:")
                for cls, val in mo_res["per_class_iou"].items():
                    bar = "█" * int(val * 20)
                    print(f"    {cls:>15s}: {val:.4f}  {bar}")

            # Early stopping
            if early_stop.step(mo_miou):
                print(f"\n  [early stop] No improvement for {early_stop.patience} "
                      f"epochs. Stopping at epoch {ep}.")
                break

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


# ─────────────────────────────────────────────────────────────────────────────
# K-Fold CV
# ─────────────────────────────────────────────────────────────────────────────
def run_kfold(model_name, epochs, device):
    fold_mious = []
    log_path   = os.path.join(RESULT_DIR, f"{model_name}_kfold_log.csv")

    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "epoch", "train_loss", "val_loss", "val_mIoU"])

        for fold, tr_dl, va_dl in get_kfold_dataloaders(k=K_FOLDS):
            print(f"\n{'═'*54}")
            print(f"  FOLD {fold + 1} / {K_FOLDS}")
            print(f"{'═'*54}")

            # Sanity check chỉ ở fold 1
            if fold == 0:
                sanity_check(tr_dl, device)

            model, base_ckpt = build_model(model_name)
            model  = model.to(device)

            class_w = LAPA_CLASS_WEIGHTS.to(device)
            crit    = ComboLoss(class_weights=class_w)
            opt     = build_optimizer(model, model_name, LR)
            sched   = build_scheduler(opt, epochs)
            scaler  = (torch.cuda.amp.GradScaler()
                       if device.type == "cuda" else None)
            metrics    = SegMetrics()
            early_stop = EarlyStopping(patience=15)
            best_m     = 0.0
            f_ckpt     = base_ckpt.replace(".pth", f"_fold{fold+1}.pth")

            for ep in range(1, epochs + 1):
                t0      = time.time()
                tr_loss = train_one(model, tr_dl, opt, crit, device, scaler)
                va_loss, va_miou, va_res = eval_labeled(
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

                if early_stop.step(va_miou):
                    print(f"\n  [early stop] Fold {fold+1} stopped at epoch {ep}.")
                    break

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


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Train DeepLabV3 or ClipUNet on LaPa dataset")
    p.add_argument("--model",  default=ACTIVE_MODEL,
                   choices=["deeplab", "clipunet"])
    p.add_argument("--epochs", default=NUM_EPOCHS, type=int)
    p.add_argument("--kfold",  action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--lr",     default=LR, type=float,
                   help=f"Base learning rate (default {LR}). "
                        f"ClipUNet encoder sẽ dùng lr/10 tự động.")
    args = p.parse_args()

    seed_everything()
    device = None
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"\n{'━'*54}")
    print(f"  Model      : {args.model}")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Mode       : {'K-Fold CV' if args.kfold else 'Standard'}")
    print(f"  Base LR    : {LR:.1e}  "
          f"(encoder LR = {LR*0.1:.1e} for clipunet)")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Class weights: ON (inverse-frequency)")
    print(f"  Early stop : patience=15")
    if args.resume:
        print(f"  Resume     : {args.resume}")
    print(f"{'━'*54}\n")

    if args.kfold:
        run_kfold(args.model, args.epochs, device)
    else:
        run_standard(args.model, args.epochs, device, resume=args.resume)


if __name__ == "__main__":
    main()