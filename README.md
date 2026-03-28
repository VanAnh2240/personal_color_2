# 🎨 Personal Colour Analysis System

AI-powered facial segmentation → dominant colour extraction → Munsell season classification.

## Architecture

```
Portrait Image
     │
     ▼
┌─────────────────────────────────┐
│  Step 1: Face Segmentation      │
│  DeepLabV3  (ResNet50 + ASPP)   │
│  ClipUNet   (CLIP ViT-B/16)     │
│  → 11-class pixel mask          │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Step 2: K-Means Colour         │
│  Regions: skin, hair, eyes, nose│
│  K=5 clusters per region        │
│  → dominant hex colour per zone │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Step 3: Munsell → Season       │
│  Hue · Value · Chroma           │
│  → Spring / Summer / Autumn / Winter
└─────────────────────────────────┘
```

## Project Structure

```
personal_color/
├── checkpoints/
│   ├── system_1_deeplabv3.pth
│   └── system_2_clipunet.pth
├── data/
│   ├── raw/          ← LaPa dataset here
│   └── processed/
├── results/          ← CSV logs and evaluation outputs
├── src/
│   ├── models/
│   │   ├── system_1_deeplabv3.py
│   │   └── system_2_clipunet.py
│   ├── dataset.py
│   └── metrics.py
├── app/
│   ├── app.py        ← Flask API
│   └── app.html      ← Web UI
├── config.py
├── train.py
├── evaluate.py
├── preprocess.py
├── requirements.txt
└── colab_runner.ipynb
```

## Quick Start (Google Colab)

1. Open `colab_runner.ipynb` in Google Colab
2. Set runtime to **GPU (T4)**
3. Run cells in order

## Dataset — LaPa (Actual Structure)

```
data/raw/
  train/
    images/     *.jpg    ← ~18,000 training images
    labels/     *.png    ← segmentation masks (0-10 class index)
    landmarks/  *.txt    ← 106-point landmarks
  val/
    images/     *.jpg    ← ~2,100 validation images
    landmarks/  *.txt    ← landmarks only
    ⚠️  NO labels/ folder — val split has no segmentation annotations
  test/
    images/     *.jpg    ← ~2,000 test images
    labels/     *.png    ← segmentation masks (used for mIoU evaluation)
    landmarks/  *.txt    ← landmarks
```

**Key consequence for training:**
- `val` cannot be used for mIoU monitoring (no labels)
- K-Fold CV runs **only on the labeled `train` split**
- Standard training carves a **10 % labeled hold-out** from `train` as monitor set
- Final mIoU is always reported on `test`

Download: https://github.com/JDAI-CV/lapa-dataset

## Training

```bash
# Standard training
python train.py --model deeplab  --epochs 50
python train.py --model clipunet --epochs 30

# 3-Fold cross-validation
python train.py --model clipunet --epochs 30 --kfold
```

## Evaluation

```bash
# Segmentation metrics on test split
python evaluate.py --model clipunet --mode seg

# Full pipeline on a folder of portraits
python evaluate.py --model clipunet --mode full --img_dir /path/to/portraits

# Single image
python evaluate.py --model clipunet --mode full --img portrait.jpg
```

## Web Demo

```bash
cd app
python app.py          # → http://localhost:5000
```

## Cross-validation Results (paper)

| Model      | Fold 1 mIoU | Fold 2 mIoU | Fold 3 mIoU | Avg mIoU | Avg Loss |
|------------|-------------|-------------|-------------|----------|----------|
| DeepLabV3  | 0.798       | 0.812       | 0.824       | 0.811    | 0.217    |
| **ClipUNet** | **0.848** | **0.855**   | **0.861**   | **0.854**| **0.144**|

## Season Classification

| Season  | Tone | Brightness | Saturation |
|---------|------|------------|------------|
| Spring  | Warm | Bright     | Clear      |
| Summer  | Cool | Medium     | Muted      |
| Autumn  | Warm | Medium-Dark| Muted      |
| Winter  | Cool | High contrast | Vivid   |
