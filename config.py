"""
config.py - Central configuration for Personal Color Segmentation System
Covers: paths, model hyperparameters, K-Means, Munsell classification
"""

import os

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
RAW_DIR    = os.path.join(DATA_DIR, "raw")
PROC_DIR   = os.path.join(DATA_DIR, "processed")
CKPT_DIR   = os.path.join(BASE_DIR, "checkpoints")
RESULT_DIR = os.path.join(BASE_DIR, "results")
SRC_DIR    = os.path.join(BASE_DIR, "src")

os.makedirs(RAW_DIR,    exist_ok=True)
os.makedirs(PROC_DIR,   exist_ok=True)
os.makedirs(CKPT_DIR,   exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Dataset  (LaPa)
# ─────────────────────────────────────────────
LAPA_NUM_CLASSES = 11          # 0-background … 10-cloth
LAPA_CLASS_NAMES = [
    "background", "skin", "left_eyebrow", "right_eyebrow",
    "left_eye",   "right_eye", "nose", "upper_lip",
    "inner_mouth","lower_lip",  "hair"
]
# Indices of regions used for pigment extraction
# skin=1, left_eye=4, right_eye=5, nose=6, hair=10
PIGMENT_REGIONS = {
    "skin":  1,
    "left_eye":  4,
    "right_eye": 5,
    "nose":  6,
    "hair":  10,
}

IMG_SIZE   = (512, 512)   # H x W fed to the model
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
BATCH_SIZE    = 8
NUM_EPOCHS    = 50
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
SCHEDULER     = "cosine"      # "cosine" | "step"
NUM_WORKERS   = 4
SEED          = 42
K_FOLDS       = 3             # cross-validation folds

# ─────────────────────────────────────────────
# Model selection
# ─────────────────────────────────────────────
# "deeplab" | "clipunet"
ACTIVE_MODEL  = "clipunet"

# DeepLabV3 specific
DEEPLAB_BACKBONE     = "resnet50"
DEEPLAB_OUTPUT_STRIDE = 16

# ClipUNet specific
CLIP_MODEL_NAME  = "ViT-B/16"
CLIP_EMBED_DIM   = 512
UNET_CHANNELS    = [256, 128, 64, 32]  # decoder channels per up-block

# ─────────────────────────────────────────────
# K-Means color extraction
# ─────────────────────────────────────────────
KMEANS_CLUSTERS    = 5        # clusters per region
KMEANS_MAX_ITER    = 300
KMEANS_N_INIT      = 10
KMEANS_COLOR_SPACE = "LAB"    # "RGB" | "LAB" | "HSV"

# ─────────────────────────────────────────────
# Munsell / Season classification
# ─────────────────────────────────────────────
# Season decision thresholds (hue angle in degrees on the colour wheel,
# value=lightness 0-10, chroma=saturation 0-∞)
SEASON_RULES = {
    "Spring":  {"warm": True,  "value_min": 6.0, "chroma_min": 4.0},
    "Summer":  {"warm": False, "value_min": 5.0, "chroma_max": 6.0},
    "Autumn":  {"warm": True,  "value_max": 6.5, "chroma_max": 6.5},
    "Winter":  {"warm": False, "value_min": 4.0, "chroma_min": 6.0},
}
# Warm hue angle range (degrees, on the RYB/Munsell wheel)
WARM_HUE_MIN = 0    # reds through yellows
WARM_HUE_MAX = 90

# ─────────────────────────────────────────────
# Checkpoint names (mirrors folder layout)
# ─────────────────────────────────────────────
CKPT_DEEPLAB  = os.path.join(CKPT_DIR, "system_1_deeplabv3.pth")
CKPT_CLIPUNET = os.path.join(CKPT_DIR, "system_2_clipunet.pth")
