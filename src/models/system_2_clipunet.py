"""
src/models/system_2_clipunet.py
ClipUNet: CLIP ViT-B/16 encoder + UNet-style multi-level decoder
for LaPa 11-class face parsing.

Architecture:
  Encoder : CLIP ViT-B/16
            - Patch Embedding → Transformer Blocks (12 layers)
            - Taps at layer 4  → LLF_1  (shallow, detail-rich)
            - Taps at layer 8  → LLF_2  (mid-level)
            - Taps at layer 12 → HLF    (deep, semantic-rich)
  Decoder : UNet-style
            - HLF is the bottleneck input
            - UpBlock1: upsample(HLF)  + skip(LLF_2) → D1
            - UpBlock2: upsample(D1)   + skip(LLF_1) → D2
            - UpBlock3: upsample(D2)   + skip(LLF_0) → D3  [extra shallow skip]
            - Final ×2 upsample + 1×1 conv → Prediction Mask (H×W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import clip
except ImportError:
    raise ImportError(
        "OpenAI CLIP not installed.\n"
        "Run: pip install git+https://github.com/openai/CLIP.git"
    )

from config import LAPA_NUM_CLASSES, CLIP_MODEL_NAME, UNET_CHANNELS


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Sequential):
    """Conv2d → BN → ReLU"""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, pad: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    """
    UNet decoder block:
        x    : (B, in_ch,   H,   W)   — coarse feature from previous stage
        skip : (B, skip_ch, H*2, W*2) — fine feature from encoder skip connection
        out  : (B, out_ch,  H*2, W*2)

    Steps:
        1. Bilinear upsample x → match skip spatial size
        2. Concatenate [x_up, skip]
        3. Two ConvBnRelu layers to fuse and refine
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.fuse = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch,          out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Guard against off-by-one from integer division
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class ProjectionBridge(nn.Module):
    """1×1 ConvBnRelu: adapt CLIP embedding dim → decoder channel width."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = ConvBnRelu(in_ch, out_ch, k=1, pad=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# CLIP ViT-B/16 Encoder
# ─────────────────────────────────────────────────────────────────────────────

class ClipEncoder(nn.Module):
    """
    Frozen CLIP ViT-B/16 used as a multi-scale feature extractor.

    Tap points (for 512×512 input, patch_size=16 → grid 32×32):
        After layer  3 → LLF_0  (B, 768, 32, 32)  very shallow / edge detail
        After layer  6 → LLF_1  (B, 768, 32, 32)  low-level structure
        After layer  9 → LLF_2  (B, 768, 32, 32)  mid-level semantics
        After layer 12 → HLF    (B, 768, 32, 32)  deep semantics (bottleneck)

    All feature maps share the same spatial resolution (H/16 × W/16) because
    ViT has no spatial downsampling inside. The UNet decoder recovers resolution
    by successive ×2 upsampling in 4 stages.
    """

    # Layer indices (1-based) at which we tap intermediate features
    TAPS = [3, 6, 9, 12]

    def __init__(self, model_name: str = CLIP_MODEL_NAME, freeze: bool = True):
        super().__init__()
        clip_model, _ = clip.load(model_name, device="cpu", jit=False)
        self.visual = clip_model.visual.float()

        if freeze:
            for p in self.visual.parameters():
                p.requires_grad_(False)

        self.patch_size = self.visual.conv1.kernel_size[0]   # 16
        self.embed_dim  = self.visual.transformer.width       # 768

    # ── positional embedding interpolation ───────────────────────────────────
    def _resize_pos_embed(self, posemb: torch.Tensor,
                          n_patches: int) -> torch.Tensor:
        """
        Interpolate patch position embeddings to match a new spatial grid.

        Parameters
        ----------
        posemb   : (1, 1 + n_orig, D)  — CLS token + patch tokens
        n_patches: int                  — desired number of patch tokens

        Returns
        -------
        (1, 1 + n_patches, D)
        """
        D      = posemb.size(2)
        cls    = posemb[:, :1, :]        # (1, 1, D)
        grid   = posemb[:, 1:, :]        # (1, n_orig, D)
        n_orig = grid.size(1)

        if n_orig == n_patches:
            return posemb                # nothing to do

        gs_old = int(round(n_orig    ** 0.5))
        gs_new = int(round(n_patches ** 0.5))

        assert gs_old ** 2 == n_orig, (
            f"Original positional grid is not square: {n_orig} tokens "
            f"(gs_old={gs_old}, gs_old²={gs_old**2})"
        )
        assert gs_new ** 2 == n_patches, (
            f"Target patch count {n_patches} is not a perfect square. "
            f"Ensure H and W are both divisible by patch_size={self.patch_size}."
        )

        # (1, n_orig, D) → (1, D, gs_old, gs_old) → interpolate → (1, n_new, D)
        grid = (grid
                .reshape(1, gs_old, gs_old, D)
                .permute(0, 3, 1, 2))                              # (1, D, gs, gs)
        grid = F.interpolate(grid, size=(gs_new, gs_new),
                             mode="bilinear", align_corners=False)
        grid = (grid
                .permute(0, 2, 3, 1)
                .reshape(1, gs_new * gs_new, D))                   # (1, n_new, D)

        return torch.cat([cls, grid], dim=1)                        # (1, 1+n_new, D)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, 3, H, W)

        Returns  (all share spatial size H/16 × W/16)
        -------
        hlf  : (B, D, H/16, W/16)  after layer 12  — deep semantic (bottleneck)
        lf0  : (B, D, H/16, W/16)  after layer  3  — very shallow
        lf1  : (B, D, H/16, W/16)  after layer  6  — low-level
        lf2  : (B, D, H/16, W/16)  after layer  9  — mid-level
        """
        vit = self.visual
        B, _, H, W = x.shape
        ph, pw   = H // self.patch_size, W // self.patch_size
        n_patches = ph * pw

        # ── Patch embedding ──────────────────────────────────────────────────
        tok = vit.conv1(x)                                         # (B, D, ph, pw)
        tok = tok.reshape(B, self.embed_dim, -1).permute(0, 2, 1) # (B, N, D)

        cls = vit.class_embedding.unsqueeze(0).expand(B, -1, -1)  # (B, 1, D)
        tok = torch.cat([cls, tok], dim=1)                         # (B, N+1, D)

        # ── Positional embedding (interpolate if input size ≠ 224×224) ──────
        posemb = vit.positional_embedding.unsqueeze(0)             # (1, 1+n_orig, D)
        posemb = self._resize_pos_embed(posemb, n_patches)         # (1, 1+n_new,  D)
        tok    = tok + posemb.to(tok.dtype)
        tok    = vit.ln_pre(tok)

        # ── Transformer: collect features at tap layers ───────────────────────
        feats = {}
        for i, blk in enumerate(vit.transformer.resblocks):
            tok = blk(tok)
            layer = i + 1
            if layer in self.TAPS:
                # Drop CLS, reshape to spatial map
                f = tok[:, 1:, :].permute(0, 2, 1)                # (B, D, N)
                feats[layer] = f.reshape(B, self.embed_dim, ph, pw)

        lf0 = feats[3]   # very shallow
        lf1 = feats[6]   # low-level
        lf2 = feats[9]   # mid-level
        hlf = feats[12]  # deep semantic (bottleneck)

        return hlf, lf0, lf1, lf2


# ─────────────────────────────────────────────────────────────────────────────
# ClipUNet
# ─────────────────────────────────────────────────────────────────────────────

class ClipUNet(nn.Module):
    """
    Full segmentation network.

    Encoder output spatial size : (H/16) × (W/16)   [e.g. 32×32 for 512×512]

    Decoder resolution recovery:
        UpBlock1 : /16  →  /8    using LLF_2 skip
        UpBlock2 : /8   →  /4    using LLF_1 skip
        UpBlock3 : /4   →  /2    using LLF_0 skip
        FinalUp  : /2   →  /1    (plain bilinear, no skip)
        SegHead  : 1×1 conv → num_classes

    UNET_CHANNELS = [C0, C1, C2, C3]  e.g. [256, 128, 64, 32]
    """

    def __init__(self, num_classes: int = LAPA_NUM_CLASSES,
                 freeze_clip: bool = True):
        super().__init__()
        C = UNET_CHANNELS   # [256, 128, 64, 32]

        self.encoder = ClipEncoder(freeze=freeze_clip)
        D = self.encoder.embed_dim                  # 768

        # Project CLIP features (768-D) → decoder widths
        self.proj_hlf = ProjectionBridge(D, C[0])   # bottleneck
        self.proj_lf2 = ProjectionBridge(D, C[0])   # skip for UpBlock1
        self.proj_lf1 = ProjectionBridge(D, C[1])   # skip for UpBlock2
        self.proj_lf0 = ProjectionBridge(D, C[2])   # skip for UpBlock3

        # Decoder
        #   UpBlock(in_ch, skip_ch, out_ch)
        self.up1 = UpBlock(C[0], C[0], C[1])   # /16 → /8
        self.up2 = UpBlock(C[1], C[1], C[2])   # /8  → /4
        self.up3 = UpBlock(C[2], C[2], C[3])   # /4  → /2

        # Final upsample × 2 then classify
        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear",
                                    align_corners=False)
        self.seg_head = nn.Conv2d(C[3], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]

        # ── Encoder ──────────────────────────────────────────────────────────
        # hlf, lf0, lf1, lf2 all have spatial size (H/16, W/16)
        hlf, lf0, lf1, lf2 = self.encoder(x)

        # ── Project to decoder widths ─────────────────────────────────────────
        p_hlf = self.proj_hlf(hlf)   # (B, C0, H/16, W/16)  ← bottleneck
        p_lf2 = self.proj_lf2(lf2)   # (B, C0, H/16, W/16)  ← skip 1
        p_lf1 = self.proj_lf1(lf1)   # (B, C1, H/8,  W/8 )  — spatial same /16
        p_lf0 = self.proj_lf0(lf0)   # (B, C2, H/4,  W/4 )  — spatial same /16

        # ── Decoder: UNet-style upsampling with skip connections ──────────────
        #
        #   All encoder features share the same spatial size (H/16).
        #   The skip tensors are spatially upsampled by UpBlock's internal ×2
        #   upsample step to produce the next resolution level.
        #
        d1 = self.up1(p_hlf, p_lf2)  # upsample /16→/8,  cat p_lf2  → (B,C1,H/8, W/8)
        d2 = self.up2(d1,    p_lf1)  # upsample /8 →/4,  cat p_lf1  → (B,C2,H/4, W/4)
        d3 = self.up3(d2,    p_lf0)  # upsample /4 →/2,  cat p_lf0  → (B,C3,H/2, W/2)

        # ── Final head ────────────────────────────────────────────────────────
        out = self.final_up(d3)       # (B, C3, H, W)
        out = self.seg_head(out)      # (B, num_classes, H, W)

        # Safety: force exact output size (handles edge rounding)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W),
                                mode="bilinear", align_corners=False)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = ClipUNet(num_classes=LAPA_NUM_CLASSES, freeze_clip=True)
    dummy = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(dummy)
    print(f"Output shape : {out.shape}")    # (1, 11, 512, 512)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    total     = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Trainable    : {trainable:.1f}M / {total:.1f}M total")