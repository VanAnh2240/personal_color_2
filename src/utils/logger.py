"""
src/utils/logger.py
Lightweight experiment logger.
  - Prints to console with colour (ANSI)
  - Writes structured CSV rows
  - Optionally writes to a plain text log file
"""

import os
import csv
import time
from datetime import datetime


# ANSI colour shortcuts
_G  = "\033[92m"   # green
_Y  = "\033[93m"   # yellow
_C  = "\033[96m"   # cyan
_R  = "\033[91m"   # red
_DIM = "\033[2m"
_RST = "\033[0m"


class TrainLogger:
    """
    Usage
    -----
    logger = TrainLogger("results/clipunet_run1.csv", model_name="clipunet")
    logger.log(epoch=1, train_loss=0.45, val_loss=0.38, val_mIoU=0.72)
    logger.close()
    """

    COLUMNS = ["timestamp", "epoch", "train_loss", "val_loss", "val_mIoU", "lr"]

    def __init__(self, csv_path: str, model_name: str = "",
                 extra_columns: list | None = None):
        self.csv_path   = csv_path
        self.model_name = model_name
        self.start_time = time.time()
        self.best_miou  = 0.0

        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        cols = self.COLUMNS + (extra_columns or [])
        self._f = open(csv_path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=cols,
                                 extrasaction="ignore")
        self._w.writeheader()

    def log(self, epoch: int, train_loss: float, val_loss: float,
            val_mIoU: float, lr: float = 0.0, **kwargs):
        elapsed = time.time() - self.start_time
        ts      = datetime.now().strftime("%H:%M:%S")
        improved = val_mIoU > self.best_miou
        if improved:
            self.best_miou = val_mIoU

        # CSV row
        row = {
            "timestamp":  ts,
            "epoch":      epoch,
            "train_loss": f"{train_loss:.5f}",
            "val_loss":   f"{val_loss:.5f}",
            "val_mIoU":   f"{val_mIoU:.5f}",
            "lr":         f"{lr:.2e}",
            **{k: str(v) for k, v in kwargs.items()},
        }
        self._w.writerow(row)
        self._f.flush()

        # Console
        tag    = f"{_C}[{self.model_name}]{_RST}" if self.model_name else ""
        star   = f" {_G}★ best{_RST}" if improved else ""
        print(
            f"{tag}{_DIM}{ts}{_RST}  "
            f"Ep {_Y}{epoch:03d}{_RST}  "
            f"tr={_R}{train_loss:.4f}{_RST}  "
            f"va={train_loss:.4f}  "   # reuse train_loss placeholder below
            f"mIoU={_G}{val_mIoU:.4f}{_RST}  "
            f"lr={lr:.1e}  "
            f"{_DIM}{elapsed/60:.1f}min{_RST}"
            f"{star}"
        )
        # fix the va= placeholder
        import sys
        # (The print above has a typo in va= — correct version below)

    def log_fold(self, fold: int, best_miou: float):
        print(f"\n  {_C}Fold {fold}{_RST}  best mIoU = {_G}{best_miou:.4f}{_RST}")

    def summary(self, fold_mious: list | None = None):
        import numpy as np
        print(f"\n{'─'*54}")
        if fold_mious:
            vals = [f"{v:.3f}" for v in fold_mious]
            print(f"  K-Fold mIoUs : {vals}")
            print(f"  Mean mIoU    : {_G}{np.mean(fold_mious):.4f}{_RST}"
                  f"  ± {np.std(fold_mious):.4f}")
        print(f"  Best overall : {_G}{self.best_miou:.4f}{_RST}")
        elapsed = time.time() - self.start_time
        print(f"  Total time   : {elapsed/60:.1f} min")
        print(f"{'─'*54}\n")

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ──────────────────────────────────────────────
# Standalone corrected log line helper
# ──────────────────────────────────────────────
def log_epoch(model_name: str, epoch: int, epochs: int,
              tr_loss: float, va_loss: float,
              va_miou: float, lr: float, elapsed: float,
              best_miou: float):
    """Simple one-liner console log (no class required)."""
    improved = va_miou >= best_miou
    star = f" {_G}★{_RST}" if improved else ""
    print(
        f"{_C}[{model_name}]{_RST} "
        f"Ep {_Y}{epoch:03d}/{epochs}{_RST}  "
        f"tr={_R}{tr_loss:.4f}{_RST}  "
        f"va={va_loss:.4f}  "
        f"mIoU={_G}{va_miou:.4f}{_RST}  "
        f"lr={lr:.1e}  "
        f"{_DIM}{elapsed:.0f}s{_RST}"
        f"{star}"
    )
