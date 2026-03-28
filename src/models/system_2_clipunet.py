"""
src/models/system_2_clipunet.py
ClipUNet: CLIP ViT-B/16 encoder  +  UNet-style multi-level decoder
for LaPa 11-class face parsing.

Architecture:
  Encoder : CLIP ViT-B/16 — patch embeddings + Transformer blocks
            → produces HLF (deep semantic) + 3 levels of LLF (spatial detail)
  Decoder : 3 UpBlocks, each using Skip Connections (Concatenation)
            to fuse HLF context with LLF spatial cues
  Head    : 1×1 conv → num_classes logits (same H×W as input)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import clip
except ImportError:
    raise ImportError(
        "OpenAI CLIP not installed. Run:  pip install git+https://github.com/openai/CLIP.git"
    )

from config import LAPA_NUM_CLASSES, CLIP_MODEL_NAME, CLIP_EMBED_DIM, UNET_CHANNELS


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, pad=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    """Upsample × 2 → cat skip → 2 × ConvBnRelu"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear",
                                align_corners=False)
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor,
                skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if spatial dims mismatch (edge case with odd sizes)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ──────────────────────────────────────────────
# CLIP feature extractor (frozen by default)
# ──────────────────────────────────────────────
class ClipEncoder(nn.Module):
    """
    Wraps CLIP ViT-B/16 and exposes intermediate patch-token maps
    as multi-scale spatial feature pyramids.

    For ViT-B/16 with 512×512 input:
      patch_size = 16  →  grid = 32×32 = 1024 tokens
    We split the 12 transformer layers into 3 groups
    to produce LLF at 3 resolutions.
    """

    N_LAYERS = 12
    SPLIT    = [4, 8, 12]   # layer indices for LLF extraction

    def __init__(self, model_name: str = CLIP_MODEL_NAME,
                 freeze: bool = True):
        super().__init__()
        clip_model, _ = clip.load(model_name, device="cpu",
                                  jit=False)
        self.visual = clip_model.visual          # VisionTransformer
        self.visual = self.visual.float()        # ensure fp32
        if freeze:
            for p in self.visual.parameters():
                p.requires_grad_(False)

        self.patch_size  = self.visual.conv1.kernel_size[0]   # 16
        self.embed_dim   = CLIP_EMBED_DIM                     # 512

    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        hlf  : (B, D, H/16, W/16)  — deep semantic feature map
        lf1  : (B, D, H/16, W/16)  — after layer block 4
        lf2  : (B, D, H/16, W/16)  — after layer block 8
        lf3  : (B, D, H/16, W/16)  — after layer block 12 (= hlf)
        """
        vit = self.visual
        B, C, H, W = x.shape
        ph = H // self.patch_size
        pw = W // self.patch_size

        # 1. Patch embedding
        x_tok = vit.conv1(x)                        # (B, D, ph, pw)
        x_tok = x_tok.reshape(B, self.embed_dim, -1).permute(0, 2, 1)  # (B, N, D)
        cls   = vit.class_embedding.unsqueeze(0).expand(B, -1, -1)     # (B, 1, D)
        x_tok = torch.cat([cls, x_tok], dim=1)      # (B, N+1, D)
        x_tok = x_tok + vit.positional_embedding.to(x_tok.dtype)
        x_tok = vit.ln_pre(x_tok)

        # 2. Transformer — tap at layers 4, 8, 12
        intermediates = []
        for i, blk in enumerate(vit.transformer.resblocks):
            x_tok = blk(x_tok)
            if (i + 1) in self.SPLIT:
                # Remove CLS token and reshape to spatial map
                feat = x_tok[:, 1:, :].permute(0, 2, 1)  # (B, D, N)
                feat = feat.reshape(B, self.embed_dim, ph, pw)
                intermediates.append(feat)

        lf1, lf2, hlf = intermediates
        return hlf, lf1, lf2, hlf   # lf3 == hlf for this 3-split scheme


# ──────────────────────────────────────────────
# Projection bridge (adapt CLIP dim → UNet dim)
# ──────────────────────────────────────────────
class ProjectionBridge(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = ConvBnRelu(in_ch, out_ch, k=1, pad=0)

    def forward(self, x):
        return self.proj(x)


# ──────────────────────────────────────────────
# Full ClipUNet
# ──────────────────────────────────────────────
class ClipUNet(nn.Module):
    """
    Input  : (B, 3, H, W)   — normalised RGB  (H, W multiples of 16)
    Output : (B, num_classes, H, W)  — logits

    Decoder channels: UNET_CHANNELS = [256, 128, 64, 32]
    Three UpBlocks bringing feature map from (H/16, W/16) → (H, W).
    """

    def __init__(self, num_classes: int = LAPA_NUM_CLASSES,
                 freeze_clip: bool = True):
        super().__init__()
        D = CLIP_EMBED_DIM              # 512
        C = UNET_CHANNELS               # [256, 128, 64, 32]

        self.encoder = ClipEncoder(freeze=freeze_clip)

        # Project each scale down to decoder width
        self.bridge_hlf = ProjectionBridge(D, C[0])
        self.bridge_lf2 = ProjectionBridge(D, C[1])
        self.bridge_lf1 = ProjectionBridge(D, C[2])

        # Three upsampling stages (each ×2 spatial)
        # Stage 1: (H/16) → (H/8)   ; skip = lf2 projected
        self.up1 = UpBlock(C[0], C[1], C[1])
        # Stage 2: (H/8)  → (H/4)   ; skip = lf1 projected
        self.up2 = UpBlock(C[1], C[2], C[2])
        # Stage 3: (H/4)  → (H/2)   ; no skip (use zeros or a learned map)
        self.up3 = UpBlock(C[2], C[2], C[3])

        # Final 2× upsample + head  (H/2) → (H)
        self.final_up   = nn.Upsample(scale_factor=2, mode="bilinear",
                                      align_corners=False)
        self.seg_head   = nn.Conv2d(C[3], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]

        # ── Encode ──
        hlf, lf1, lf2, _ = self.encoder(x)   # all (B, 512, H/16, W/16)

        # ── Project ──
        p_hlf = self.bridge_hlf(hlf)   # (B, 256, H/16, W/16)
        p_lf2 = self.bridge_lf2(lf2)   # (B, 128, H/16, W/16)
        p_lf1 = self.bridge_lf1(lf1)   # (B,  64, H/16, W/16)

        # ── Decode (UNet skip connections) ──
        d1 = self.up1(p_hlf, p_lf2)    # (B, 128, H/8,  W/8)
        d2 = self.up2(d1,    p_lf1)    # (B,  64, H/4,  W/4)

        # Dummy skip for stage 3 (no more LLF available)
        dummy = torch.zeros(d2.shape[0], d2.shape[1],
                            d2.shape[2] * 2, d2.shape[3] * 2,
                            device=d2.device)
        d3 = self.up3(d2, dummy[:, :, :d2.shape[2]*2, :d2.shape[3]*2])  # (B, 32, H/2, W/2)

        # ── Final head ──
        out = self.final_up(d3)                         # (B, 32, H, W)
        out = self.seg_head(out)                        # (B, C, H, W)

        # Safety: ensure exact input resolution
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W),
                                mode="bilinear", align_corners=False)
        return out


# ──────────────────────────────────────────────
# Sanity check
# ──────────────────────────────────────────────
if __name__ == "__main__":
    model = ClipUNet(num_classes=LAPA_NUM_CLASSES, freeze_clip=True)
    dummy = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(dummy)
    print(f"ClipUNet output shape: {out.shape}")   # (1, 11, 512, 512)
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad) / 1e6
    total     = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Trainable / Total params: {trainable:.1f}M / {total:.1f}M")
