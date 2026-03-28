"""
src/utils/checkpoint.py
Unified checkpoint save / load / resume helpers.

Saves not just model weights but also optimizer state, epoch, best mIoU
so training can be resumed exactly after a Colab disconnection.
"""

import os
import torch


def save_checkpoint(state: dict, path: str, verbose: bool = True):
    """
    state = {
        "epoch":      int,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict(),   # optional
        "best_miou":  float,
        "config":     dict,                     # optional
    }
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)
    if verbose:
        print(f"  [ckpt] Saved  → {path}  "
              f"(epoch {state.get('epoch','?')}, "
              f"mIoU {state.get('best_miou', 0):.4f})")


def load_checkpoint(path: str, model, optimizer=None,
                    scheduler=None, device=None):
    """
    Loads checkpoint into model (and optionally optimizer/scheduler).
    Returns the full state dict so callers can read 'epoch', 'best_miou' etc.
    """
    if device is None:
        device = torch.device("cpu")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(path, map_location=device)

    # Support plain state-dict checkpoints (only model weights)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
        if optimizer and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler and "scheduler" in state:
            try:
                scheduler.load_state_dict(state["scheduler"])
            except Exception:
                pass
        print(f"  [ckpt] Loaded ← {path}  "
              f"(epoch {state.get('epoch','?')}, "
              f"mIoU {state.get('best_miou', 0):.4f})")
    else:
        # Legacy: raw state dict
        model.load_state_dict(state)
        print(f"  [ckpt] Loaded ← {path}  (weights only)")
        state = {"epoch": 0, "best_miou": 0.0}

    return state


def save_best(model, path: str, current_miou: float,
              best_miou: float, extra: dict | None = None) -> float:
    """
    Save checkpoint only if current_miou > best_miou.
    Returns the new best mIoU.
    """
    if current_miou > best_miou:
        payload = {"model": model.state_dict(),
                   "best_miou": current_miou}
        if extra:
            payload.update(extra)
        save_checkpoint(payload, path)
        return current_miou
    return best_miou


def list_checkpoints(ckpt_dir: str) -> list[str]:
    """Return sorted list of .pth files in checkpoint directory."""
    if not os.path.isdir(ckpt_dir):
        return []
    return sorted(
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.endswith(".pth")
    )
