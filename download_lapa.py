"""
download_lapa.py
Helper to download and organise the LaPa face-parsing dataset.

LaPa is hosted on Google Drive (official JDAI-CV release).
This script uses gdown to fetch the zip archives, unzip them,
and reorganise them into the layout expected by src/dataset.py:

  data/raw/
    train/images/*.jpg
    train/labels/*.png
    val/images/*.jpg
    val/labels/*.png
    test/images/*.jpg
    test/labels/*.png

Usage (Colab):
  !python download_lapa.py
  !python download_lapa.py --dest /content/drive/MyDrive/personal_color/data/raw
"""

import os
import sys
import argparse
import zipfile
import shutil

try:
    import gdown
except ImportError:
    print("gdown not found — installing...")
    os.system(f"{sys.executable} -m pip install -q gdown")
    import gdown


# ──────────────────────────────────────────────
# Official LaPa Google Drive IDs
# Source: https://github.com/JDAI-CV/lapa-dataset
# ──────────────────────────────────────────────
GDRIVE_FILES = {
    "train": "1Epi7t9wkRsB_KGYaJqVNvqt3XpRE2PpW",
    "val":   "1P8A63QnGcUfQvdFd7-hKLBQlf2Ks6oJ-",
    "test":  "1OqkJEk3PVBVETuqtqKYxI2RSgBp3YZUU",
}


def download_and_extract(split: str, dest_root: str, force: bool = False):
    img_dir = os.path.join(dest_root, split, "images")
    lbl_dir = os.path.join(dest_root, split, "labels")

    if os.path.exists(img_dir) and len(os.listdir(img_dir)) > 10 and not force:
        print(f"  {split}: already present, skipping.")
        return

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    zip_path = os.path.join(dest_root, f"lapa_{split}.zip")
    file_id  = GDRIVE_FILES[split]

    print(f"  Downloading {split}...")
    gdown.download(id=file_id, output=zip_path, quiet=False)

    print(f"  Extracting {split}...")
    tmp_dir = os.path.join(dest_root, f"_tmp_{split}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)

    # Reorganise: find images/*.jpg and labels/*.png anywhere in tmp_dir
    for root, dirs, files in os.walk(tmp_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            if fname.lower().endswith((".jpg", ".jpeg")):
                shutil.move(fpath, os.path.join(img_dir, fname))
            elif fname.lower().endswith(".png"):
                shutil.move(fpath, os.path.join(lbl_dir, fname))

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.remove(zip_path)

    n_imgs = len(os.listdir(img_dir))
    n_lbls = len(os.listdir(lbl_dir))
    print(f"  {split}: {n_imgs} images, {n_lbls} labels")


def verify_structure(dest_root: str):
    print("\nVerification:")
    ok = True
    for split in ("train", "val", "test"):
        img_dir = os.path.join(dest_root, split, "images")
        lbl_dir = os.path.join(dest_root, split, "labels")
        n_i = len(os.listdir(img_dir)) if os.path.isdir(img_dir) else 0
        n_l = len(os.listdir(lbl_dir)) if os.path.isdir(lbl_dir) else 0
        status = "✓" if n_i > 0 and n_l == n_i else "✗"
        print(f"  {status} {split:6s}: {n_i} images / {n_l} labels")
        if n_i == 0:
            ok = False
    return ok


def main():
    p = argparse.ArgumentParser(
        description="Download and organise the LaPa dataset"
    )
    p.add_argument("--dest",   default="data/raw",
                   help="Destination root for raw data")
    p.add_argument("--splits", nargs="+",
                   default=["train", "val", "test"],
                   choices=["train", "val", "test"])
    p.add_argument("--force",  action="store_true",
                   help="Re-download even if files exist")
    args = p.parse_args()

    dest = os.path.abspath(args.dest)
    os.makedirs(dest, exist_ok=True)
    print(f"Destination: {dest}\n")

    for split in args.splits:
        download_and_extract(split, dest, force=args.force)

    all_ok = verify_structure(dest)
    if all_ok:
        print("\n✅ Dataset ready. Run  python train.py  to start training.")
    else:
        print("\n⚠️  Some splits are missing. Check the download or "
              "visit https://github.com/JDAI-CV/lapa-dataset manually.")


if __name__ == "__main__":
    main()
