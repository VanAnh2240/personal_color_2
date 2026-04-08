# File src/models/system_1_deeplabv3.py

"""
src/models/system_1_deeplabv3.py
DeepLabV3 with ResNet-50 backbone + custom 11-class head for LaPa face parsing.

Architecture:
  Encoder  : ResNet-50 (pretrained ImageNet, atrous conv in layer3/4)
  ASPP     : 5 parallel branches (1×1, 3 dilated 3×3, global avg pool)
  Decoder  : ASPP output + low-level features → bilinear upsample → conv → mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from config import LAPA_NUM_CLASSES


# ──────────────────────────────────────────────
# ASPP Module
# ──────────────────────────────────────────────
class ASPPConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        x = self.conv(self.pool(x))
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """
    5 parallel branches:
      branch 0 : 1×1 conv
      branch 1-3: dilated 3×3 conv (rates 6, 12, 18)
      branch 4 : global avg pool
    """
    def __init__(self, in_ch: int = 2048, out_ch: int = 256,
                 rates=(6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ),
            ASPPConv(in_ch, out_ch, rates[0]),
            ASPPConv(in_ch, out_ch, rates[1]),
            ASPPConv(in_ch, out_ch, rates[2]),
            ASPPPooling(in_ch, out_ch),
        ])
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([b(x) for b in self.branches], dim=1)
        return self.project(out)


# ──────────────────────────────────────────────
# Decoder
# ──────────────────────────────────────────────
class DeepLabDecoder(nn.Module):
    def __init__(self, low_ch: int = 256, aspp_ch: int = 256,
                 num_classes: int = LAPA_NUM_CLASSES):
        super().__init__()
        # Reduce low-level feature channels
        self.low_conv = nn.Sequential(
            nn.Conv2d(low_ch, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(aspp_ch + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, aspp_feat: torch.Tensor,
                low_feat: torch.Tensor,
                target_size) -> torch.Tensor:
        low = self.low_conv(low_feat)
        aspp_up = F.interpolate(aspp_feat, size=low.shape[-2:],
                                mode="bilinear", align_corners=False)
        x = torch.cat([aspp_up, low], dim=1)
        x = self.classifier(x)
        return F.interpolate(x, size=target_size,
                             mode="bilinear", align_corners=False)


# ──────────────────────────────────────────────
# Full DeepLabV3+ model
# ──────────────────────────────────────────────
class DeepLabV3(nn.Module):
    """
    DeepLabV3 (DeepLab v3+) for face parsing.
    Input  : (B, 3, H, W)  — normalised RGB
    Output : (B, num_classes, H, W)  — logits
    """
    def __init__(self, num_classes: int = LAPA_NUM_CLASSES,
                 pretrained: bool = True):
        super().__init__()

        # ── Encoder ──
        backbone = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Make layer3 & layer4 use atrous convolution (output_stride=16)
        self._make_atrous(backbone.layer3, stride=1, dilation=2)
        self._make_atrous(backbone.layer4, stride=1, dilation=4)

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1   # low-level features (256 ch)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4   # high-level features (2048 ch)

        self.aspp    = ASPP(in_ch=2048, out_ch=256)
        self.decoder = DeepLabDecoder(low_ch=256, aspp_ch=256,
                                      num_classes=num_classes)

    @staticmethod
    def _make_atrous(layer: nn.Module, stride: int, dilation: int):
        """Replace strides with atrous (dilated) convolutions."""
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                if m.stride == (2, 2):
                    m.stride = (stride, stride)
                # Chỉ áp dụng dilation cho conv 3×3, KHÔNG phải 1×1
                if m.kernel_size == (3, 3):
                    m.dilation = (dilation, dilation)
                    m.padding  = (dilation, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        x = self.layer0(x)
        low = self.layer1(x)    # 1/4 resolution, 256 ch
        x   = self.layer2(low)
        x   = self.layer3(x)
        x   = self.layer4(x)
        x   = self.aspp(x)
        return self.decoder(x, low, (h, w))


# ──────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────
class SegmentationLoss(nn.Module):
    """Cross-entropy + optional OHEM."""
    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets)


# ──────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────
if __name__ == "__main__":
    model = DeepLabV3(num_classes=LAPA_NUM_CLASSES, pretrained=False)
    dummy = torch.randn(2, 3, 512, 512)
    out   = model(dummy)
    print(f"DeepLabV3 output shape: {out.shape}")  # (2, 11, 512, 512)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {total:.1f}M")
