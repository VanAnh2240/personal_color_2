# """
# prepare_dataset.py
# ──────────────────
# LaPa dataset — raw download structure vs project structure.

# RAW DOWNLOAD (what you get after unzipping LaPa):
#   data/raw/
#     test/
#       images/     *.jpg   (~18 000 images  — the large labeled split)
#       labels/     *.png   (segmentation masks 0-10)
#       landmarks/  *.txt   (106-point landmarks)
#     val/
#       images/     *.jpg   (~2 000 images)
#       landmarks/  *.txt   (NO labels)

# PROJECT STRUCTURE (what the training code expects):
#   data/raw/
#     train/          ← created from 90 % of test/
#       images/
#       labels/
#       landmarks/
#     val/            ← kept as-is  (images + landmarks, NO labels)
#       images/
#       landmarks/
#     test/           ← created from 10 % of test/  (held-out eval)
#       images/
#       labels/
#       landmarks/

# This script performs that reorganisation in-place (no file copies —
# uses symlinks on Linux / Colab, or hard-copies on Windows).

# Usage (Colab):
#   !python prepare_dataset.py
#   !python prepare_dataset.py --raw_dir data/raw --train_ratio 0.9 --seed 42
#   !python prepare_dataset.py --copy     # force file copy instead of symlinks
# """

# import argparse
# import os
# import sys
# import random
# import shutil

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# # ─────────────────────────────────────────────────────────────────────────────
# # Helpers
# # ─────────────────────────────────────────────────────────────────────────────
# def _link_or_copy(src: str, dst: str, use_copy: bool):
#     """Create a symlink (fast) or copy (safe) from src → dst."""
#     if os.path.exists(dst) or os.path.islink(dst):
#         return
#     if use_copy:
#         shutil.copy2(src, dst)
#     else:
#         os.symlink(os.path.abspath(src), dst)


# def _make_split_dirs(base: str, split: str, has_labels: bool = True):
#     os.makedirs(os.path.join(base, split, "images"),    exist_ok=True)
#     os.makedirs(os.path.join(base, split, "landmarks"), exist_ok=True)
#     if has_labels:
#         os.makedirs(os.path.join(base, split, "labels"), exist_ok=True)


# def _collect_stems(img_dir: str) -> list:
#     """Return sorted list of filename stems (without extension) in img_dir."""
#     return sorted(
#         os.path.splitext(f)[0]
#         for f in os.listdir(img_dir)
#         if f.lower().endswith((".jpg", ".jpeg", ".png"))
#     )


# # ─────────────────────────────────────────────────────────────────────────────
# # Main reorganisation logic
# # ─────────────────────────────────────────────────────────────────────────────
# def prepare(raw_dir: str, train_ratio: float = 0.9,
#             seed: int = 42, use_copy: bool = False):
#     """
#     Reorganise LaPa from the raw download layout into train/val/test.

#     Parameters
#     ----------
#     raw_dir      : path that contains  test/  and  val/  after unzipping
#     train_ratio  : fraction of test/ images used for train (rest → test)
#     seed         : random seed for reproducible split
#     use_copy     : True = copy files; False = symlink (faster, saves disk)
#     """
#     # ── Validate raw download ──────────────────────────────────────────────
#     src_test_img = os.path.join(raw_dir, "test", "images")
#     src_test_lbl = os.path.join(raw_dir, "test", "labels")
#     src_test_lmk = os.path.join(raw_dir, "test", "landmarks")
#     src_val_img  = os.path.join(raw_dir, "val",  "images")
#     src_val_lmk  = os.path.join(raw_dir, "val",  "landmarks")

#     missing = [p for p in [src_test_img, src_test_lbl, src_val_img]
#                if not os.path.isdir(p)]
#     if missing:
#         print("ERROR: The following expected directories were not found:")
#         for p in missing:
#             print(f"  {p}")
#         print("\nMake sure you have unzipped LaPa into data/raw/ so that")
#         print("data/raw/test/images/ and data/raw/val/images/ exist.")
#         sys.exit(1)

#     # ── Collect all labeled stems from test/ ──────────────────────────────
#     all_stems = _collect_stems(src_test_img)
#     n_total   = len(all_stems)
#     if n_total == 0:
#         print("ERROR: No images found in", src_test_img); sys.exit(1)

#     # Shuffle deterministically
#     rng = random.Random(seed)
#     rng.shuffle(all_stems)

#     n_train  = int(n_total * train_ratio)
#     train_stems = all_stems[:n_train]
#     eval_stems  = all_stems[n_train:]

#     print(f"\nLaPa raw layout detected:")
#     print(f"  test/ (labeled) : {n_total:6d} images")
#     print(f"  val/  (no labels): {len(_collect_stems(src_val_img)):6d} images")
#     print(f"\nSplit plan (seed={seed}, train_ratio={train_ratio}):")
#     print(f"  → train/  : {len(train_stems):6d} images  (with labels)")
#     print(f"  → test/   : {len(eval_stems):6d} images  (with labels, eval only)")
#     print(f"  → val/    : kept as-is   (no labels, visual inspection only)")
#     print(f"\nMethod: {'file copy' if use_copy else 'symlink (fast)'}")

#     # ── Check if already done ─────────────────────────────────────────────
#     train_img_dir = os.path.join(raw_dir, "train", "images")
#     if os.path.isdir(train_img_dir):
#         n_existing = len(os.listdir(train_img_dir))
#         if n_existing == len(train_stems):
#             print(f"\n✓ train/ already exists with {n_existing} images. "
#                   f"Skipping (use --force to redo).")
#             _print_summary(raw_dir)
#             return
#         print(f"\n  train/ exists but has {n_existing} images "
#               f"(expected {len(train_stems)}). Rebuilding...")
#         shutil.rmtree(os.path.join(raw_dir, "train"))

#     # ── Backup original test/ before overwriting ──────────────────────────
#     bak_dir = os.path.join(raw_dir, "_test_original")
#     if not os.path.exists(bak_dir):
#         print(f"\n  Backing up original test/ → _test_original/ ...")
#         shutil.copytree(os.path.join(raw_dir, "test"), bak_dir)
#     else:
#         print(f"  Backup already exists at _test_original/")

#     # ── Build train/ ──────────────────────────────────────────────────────
#     print("\n  Building train/ ...")
#     _make_split_dirs(raw_dir, "train", has_labels=True)
#     _populate_split(bak_dir, raw_dir, "train", train_stems, use_copy)

#     # ── Rebuild test/ (eval subset) ────────────────────────────────────────
#     print("  Rebuilding test/ (eval subset) ...")
#     shutil.rmtree(os.path.join(raw_dir, "test"))
#     _make_split_dirs(raw_dir, "test", has_labels=True)
#     _populate_split(bak_dir, raw_dir, "test", eval_stems, use_copy)

#     # ── val/ stays untouched ──────────────────────────────────────────────
#     print("  val/ kept as-is (no labels).")

#     _print_summary(raw_dir)
#     print("\n✅ Dataset ready.  Run:  python train.py --model clipunet --kfold\n")


# def _populate_split(src_base: str, dst_base: str,
#                     split: str, stems: list, use_copy: bool):
#     """Link/copy images, labels, landmarks for the given stems."""
#     dst_img = os.path.join(dst_base, split, "images")
#     dst_lbl = os.path.join(dst_base, split, "labels")
#     dst_lmk = os.path.join(dst_base, split, "landmarks")

#     src_img = os.path.join(src_base, "images")
#     src_lbl = os.path.join(src_base, "labels")
#     src_lmk = os.path.join(src_base, "landmarks")

#     missing_img = missing_lbl = missing_lmk = 0

#     for stem in stems:
#         # Image (.jpg)
#         for ext in (".jpg", ".jpeg", ".png"):
#             src = os.path.join(src_img, stem + ext)
#             if os.path.exists(src):
#                 _link_or_copy(src, os.path.join(dst_img, stem + ext), use_copy)
#                 break
#         else:
#             missing_img += 1

#         # Label (.png)  — only for labeled splits
#         if dst_lbl:
#             src = os.path.join(src_lbl, stem + ".png")
#             if os.path.exists(src):
#                 _link_or_copy(src, os.path.join(dst_lbl, stem + ".png"), use_copy)
#             else:
#                 missing_lbl += 1

#         # Landmark (.txt)
#         src = os.path.join(src_lmk, stem + ".txt")
#         if os.path.exists(src):
#             _link_or_copy(src, os.path.join(dst_lmk, stem + ".txt"), use_copy)
#         else:
#             missing_lmk += 1

#     if missing_img:
#         print(f"    WARNING: {missing_img} stems had no matching image in {src_img}")
#     if missing_lbl:
#         print(f"    WARNING: {missing_lbl} stems had no matching label in {src_lbl}")


# def _print_summary(raw_dir: str):
#     print(f"\nFinal structure in {raw_dir}:")
#     print(f"{'split':8s}  {'images':>8s}  {'labels':>8s}  {'landmarks':>10s}")
#     print("─" * 44)
#     for split in ("train", "val", "test"):
#         def _count(subdir):
#             d = os.path.join(raw_dir, split, subdir)
#             return len(os.listdir(d)) if os.path.isdir(d) else 0
#         n_i = _count("images")
#         n_l = _count("labels")
#         n_m = _count("landmarks")
#         lbl_str = str(n_l) if n_l else "—  (none)"
#         print(f"{split:8s}  {n_i:>8d}  {lbl_str:>8s}  {n_m:>10d}")


# # ─────────────────────────────────────────────────────────────────────────────
# # Entry point
# # ─────────────────────────────────────────────────────────────────────────────
# def main():
#     p = argparse.ArgumentParser(
#         description="Reorganise raw LaPa download into train/val/test splits"
#     )
#     p.add_argument("--raw_dir",     default="data/raw",
#                    help="Directory containing test/ and val/ after unzipping")
#     p.add_argument("--train_ratio", default=0.9, type=float,
#                    help="Fraction of test/ (labeled) to use as train (default 0.9)")
#     p.add_argument("--seed",        default=42, type=int)
#     p.add_argument("--copy",        action="store_true",
#                    help="Copy files instead of symlinking (slower but Windows-safe)")
#     p.add_argument("--force",       action="store_true",
#                    help="Delete and redo even if train/ already exists")
#     args = p.parse_args()

#     raw_dir = os.path.abspath(args.raw_dir)
#     if args.force:
#         train_dir = os.path.join(raw_dir, "train")
#         if os.path.isdir(train_dir):
#             shutil.rmtree(train_dir)
#             print("Removed existing train/ (--force)")

#     prepare(raw_dir, train_ratio=args.train_ratio,
#             seed=args.seed, use_copy=args.copy)


# if __name__ == "__main__":
#     main()
