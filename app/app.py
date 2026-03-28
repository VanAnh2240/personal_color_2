"""
app/app.py
Flask backend — serves the Personal Colour Analysis web demo.

Endpoints:
  GET  /            → app.html
  POST /analyze     → JSON result (season, colours, Munsell)
  GET  /health      → {"status": "ok"}

Run locally:
  cd app
  python app.py

Google Colab (with ngrok):
  !pip install flask pyngrok
  from pyngrok import ngrok
  public_url = ngrok.connect(5000)
  print(public_url)
  !python app/app.py &
"""

import io
import os
import sys
import base64
import traceback

import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Allow root-level imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CKPT_DEEPLAB, CKPT_CLIPUNET, ACTIVE_MODEL,
    IMG_SIZE, PIGMENT_REGIONS,
)
from preprocess import PersonalColorPipeline

# ──────────────────────────────────────────────
# App init
# ──────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=APP_DIR, static_url_path="")
CORS(app)

# ──────────────────────────────────────────────
# Model loading (lazy, cached)
# ──────────────────────────────────────────────
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = os.environ.get("MODEL", ACTIVE_MODEL)

    if model_name == "deeplab":
        from src.models.system_1_deeplabv3 import DeepLabV3
        model = DeepLabV3(pretrained=False)
        ckpt  = CKPT_DEEPLAB
    else:
        from src.models.system_2_clipunet import ClipUNet
        model = ClipUNet(freeze_clip=False)
        ckpt  = CKPT_CLIPUNET

    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        print(f"[app] Loaded checkpoint: {ckpt}")
    else:
        print(f"[app] WARNING: No checkpoint at {ckpt}. "
              f"Running with random weights (for demo only).")

    _pipeline = PersonalColorPipeline(model.to(device), device)
    return _pipeline


# ──────────────────────────────────────────────
# Season metadata
# ──────────────────────────────────────────────
SEASON_META = {
    "Spring": {
        "emoji":       "🌸",
        "description": "Warm, bright, and clear tones. "
                       "You suit fresh, lively colours.",
        "palette":     ["#FFCBA4", "#FFD700", "#FF8C69", "#90EE90", "#FFA07A"],
        "avoid":       ["#696969", "#2F4F4F", "#00008B"],
    },
    "Summer": {
        "emoji":       "🌊",
        "description": "Cool, soft, and muted tones. "
                       "You suit dusty pastels and grayed hues.",
        "palette":     ["#B0C4DE", "#D8BFD8", "#E6E6FA", "#AFEEEE", "#C0C0C0"],
        "avoid":       ["#FF4500", "#FFD700", "#FF8C00"],
    },
    "Autumn": {
        "emoji":       "🍂",
        "description": "Warm, earthy, and muted tones. "
                       "You suit rich, natural colours.",
        "palette":     ["#8B4513", "#D2691E", "#DAA520", "#6B8E23", "#B8860B"],
        "avoid":       ["#FF69B4", "#00CED1", "#E0E0E0"],
    },
    "Winter": {
        "emoji":       "❄️",
        "description": "Cool, vivid, and high-contrast tones. "
                       "You suit bold, saturated colours.",
        "palette":     ["#000080", "#DC143C", "#FFFFFF", "#9400D3", "#00CED1"],
        "avoid":       ["#DAA520", "#D2691E", "#F5DEB3"],
    },
}


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(APP_DIR, "app.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accepts:
      multipart/form-data  with field 'image' (file upload)
      OR
      JSON {"image_b64": "<base64-encoded image>"}

    Returns JSON:
    {
      "season":    "Spring",
      "emoji":     "🌸",
      "description": "...",
      "dominant_colors": {"skin": "#FFDAB9", ...},
      "munsell":   {"hue": 45.2, "value": 7.1, "chroma": 3.8},
      "palette":   [...],
      "avoid":     [...],
      "preview_b64": "<base64 annotated image>"
    }
    """
    try:
        # ── 1. Decode image ──
        if "image" in request.files:
            file_bytes = request.files["image"].read()
        elif request.is_json and "image_b64" in request.json:
            file_bytes = base64.b64decode(request.json["image_b64"])
        else:
            return jsonify({"error": "No image provided"}), 400

        nparr   = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({"error": "Cannot decode image"}), 400

        # Save temp file for pipeline
        tmp_path = "/tmp/_personal_color_upload.jpg"
        cv2.imwrite(tmp_path, img_bgr)

        # ── 2. Run pipeline ──
        pipe   = get_pipeline()
        result = pipe.run(tmp_path)

        season = result["season"]
        meta   = SEASON_META.get(season, {})

        # ── 3. Annotated preview ──
        preview_b64 = _build_preview(img_bgr, result)

        return jsonify({
            "season":          season,
            "emoji":           meta.get("emoji", ""),
            "description":     meta.get("description", ""),
            "dominant_colors": result["dominant_colors"],
            "munsell":         result["munsell"],
            "palette":         meta.get("palette", []),
            "avoid":           meta.get("avoid", []),
            "undertone":       result.get("undertone", "neutral"),
            "contrast":        result.get("contrast", {}),
            "harmony":         result.get("harmony", {}),
            "preview_b64":     preview_b64,
        })

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


def _build_preview(img_bgr: np.ndarray, result: dict) -> str:
    """Resize image, overlay season label, return base64 JPEG."""
    thumb = cv2.resize(img_bgr, (300, 300))

    season = result.get("season", "?")
    meta   = SEASON_META.get(season, {})
    label  = f"{meta.get('emoji','')}{season}"

    # Semi-transparent banner
    overlay = thumb.copy()
    cv2.rectangle(overlay, (0, 260), (300, 300), (0, 0, 0), -1)
    thumb = cv2.addWeighted(overlay, 0.55, thumb, 0.45, 0)
    cv2.putText(thumb, label, (8, 290),
                cv2.FONT_HERSHEY_DUPLEX, 0.85,
                (255, 255, 255), 2, cv2.LINE_AA)

    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "0") == "1"
    print(f"Starting Personal Colour API on port {port} ...")
    # Pre-load model
    get_pipeline()
    app.run(host="0.0.0.0", port=port, debug=debug)
