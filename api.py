"""
api.py — Personal Colour Analysis API
Run: uvicorn api:app --host 0.0.0.0 --port 8000
"""

import io, base64, tempfile, time, os
import cv2, numpy as np, torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Colour Analysis API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model_cache = {}
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ── helpers ──────────────────────────────────────────────────────────────────

def get_model(ckpt="checkpoints/system_1_deeplabv3.pth", num_classes=11, device="cpu"):
    if ckpt not in _model_cache:
        from src.models.system_1_deeplabv3 import DeepLabV3
        m = DeepLabV3(num_classes=num_classes)
        sd = torch.load(ckpt, map_location=device)
        m.load_state_dict(sd.get("model", sd))
        _model_cache[ckpt] = m.to(device).eval()
    return _model_cache[ckpt]


def decode_img(data: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Cannot decode image")
    return bgr


@torch.no_grad()
def run_seg(model, bgr, device="cpu") -> np.ndarray:
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(bgr, (473, 473)), cv2.COLOR_BGR2RGB)
    t = ((torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0 - _MEAN) / _STD).unsqueeze(0).to(device)
    out = model(t)
    logits = out["out"] if isinstance(out, dict) else out
    return F.interpolate(logits, (h, w), mode="bilinear", align_corners=False)\
             .argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)


def to_b64(bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", bgr)
    return base64.b64encode(buf).decode()


def result_image_b64(bgr, result) -> str:
    from classification.visualizer import draw_result_overlay, draw_dominants_strip, draw_palette_strip
    ann = draw_result_overlay(bgr, result, target_height=380)
    W = ann.shape[1]
    dom = cv2.resize(draw_dominants_strip(result.dominants, W // 4), (W, W // 4))
    sw  = max(30, W // len(result.season.colors))
    pal = cv2.resize(draw_palette_strip(result.season, sw), (W, sw))

    def bar(txt):
        b = np.full((28, W, 3), 40, np.uint8)
        cv2.putText(b, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return b

    return to_b64(np.vstack([ann, bar("Dominant colours"), dom, bar(f"Palette: {result.season.name}"), pal]))


def seg_image_b64(bgr, seg_mask) -> str:
    tmp = tempfile.mktemp(suffix=".png")
    from seg_visualizer import save_seg_figure
    save_seg_figure(bgr, seg_mask, output_path=tmp)
    img = cv2.imread(tmp); os.unlink(tmp)
    return to_b64(img)


def serialize(result) -> dict:
    s = result.season
    return {
        "season": {
            "name": s.name, "hue": s.hue, "chroma": s.chroma,
            "value": s.value, "contrast": s.contrast,
            "palette_colors": [{"r": r, "g": g, "b": b} for r, g, b in s.colors],
        },
        "metrics": result.metrics,
        "user_vector_SIVC": list(result.user_vector),
        "hamming_scores": result.hamming_scores,
        "dominant_colors": {
            k: ({"r": v[0], "g": v[1], "b": v[2]} if v else None)
            for k, v in result.dominants.items()
        },
        "is_bald": result.is_bald,
    }


def classify(bgr, seg_mask, hair_label=10, chroma=127., value=127., contrast=127.):
    from classification import PaletteClassifier
    return PaletteClassifier(
        skin_chroma_thresh=chroma, value_thresh=value,
        contrast_thresh=contrast, hair_label=hair_label,
    ).classify(bgr, seg_mask)


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/seasons")
def seasons():
    from classification.palettes import ALL_SEASONS
    return {"seasons": [
        {"name": s.name, "hue": s.hue, "chroma": s.chroma, "value": s.value,
         "contrast": s.contrast, 
         "metric_vector_SIVC": list(s.metric_vector),
         "palette_colors": [{"r": r, "g": g, "b": b} for r, g, b in s.colors]}
        for s in ALL_SEASONS
    ]}


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    checkpoint: str   = Form("checkpoints/system_1_deeplabv3.pth"),
    device: str       = Form("cpu"),
    num_classes: int  = Form(11),
    hair_label: int   = Form(10),
    chroma_thresh: float   = Form(127.),
    value_thresh: float    = Form(127.),
    contrast_thresh: float = Form(127.),
    return_result_image: bool = Form(True),
    return_seg_image: bool    = Form(False),
):
    t0 = time.time()
    try:
        bgr = decode_img(await image.read())
    except ValueError as e:
        raise HTTPException(422, str(e))

    try:
        seg = run_seg(get_model(checkpoint, num_classes, device), bgr, device)
    except Exception as e:
        raise HTTPException(500, f"Segmentation error: {e}")

    try:
        result = classify(bgr, seg, hair_label, chroma_thresh, value_thresh, contrast_thresh)
    except Exception as e:
        raise HTTPException(500, f"Classification error: {e}")

    data = serialize(result)
    if return_result_image:
        data["result_image_b64"] = result_image_b64(bgr, result)
    if return_seg_image:
        data["seg_image_b64"] = seg_image_b64(bgr, seg)
    data["elapsed_seconds"] = round(time.time() - t0, 3)
    return JSONResponse(data)

