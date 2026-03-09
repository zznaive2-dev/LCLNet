from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18

from model.ffdnet_network import FFDNet


class SE(nn.Module):
    def __init__(self, ch: int, r: int = 16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // r, ch, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.avg(x))
        return x * w


def replace_bn_with_gn(module: nn.Module, num_groups: int = 32) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            setattr(
                module,
                name,
                nn.GroupNorm(
                    num_groups=min(int(num_groups), int(num_channels)),
                    num_channels=int(num_channels),
                ),
            )
        else:
            replace_bn_with_gn(child, num_groups=num_groups)


class ResNetEmbed(nn.Module):
    """
    train_img.py's lightweight embedding model:
    - ResNet18 backbone with BN->GN replacement
    - Only uses the first block of each stage
    - SE after each stage
    - Outputs L2-normalized feature vector
    """

    def __init__(self, out_dim: int = 256):
        super().__init__()
        backbone = resnet18(weights=None)
        replace_bn_with_gn(backbone, num_groups=32)

        self.stem = nn.Sequential(
            backbone.conv1,
            nn.GroupNorm(num_groups=32, num_channels=64),
            backbone.relu,
            backbone.maxpool,
        )
        self.stage1 = nn.Sequential(backbone.layer1[0], SE(64))
        self.stage2 = nn.Sequential(backbone.layer2[0], SE(128))
        self.stage3 = nn.Sequential(backbone.layer3[0], SE(256))
        self.stage4 = nn.Sequential(backbone.layer4[0], SE(512))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, int(out_dim), bias=False)
        self.ln = nn.LayerNorm(int(out_dim), elementwise_affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect grayscale [B,1,H,W] and repeat to 3 channels.
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        x = self.ln(self.fc(x))
        return F.normalize(x, dim=1)


class AmsMarginProduct(nn.Module):
    """
    Additive margin softmax (ArcFace-like) head used in train_img.py.
    For training, it needs labels to apply margin; for inference, use cosine * s.
    """

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.30, easy_margin: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(int(out_features), int(in_features)))
        nn.init.xavier_uniform_(self.weight)
        self.s = float(s)
        self.m = float(m)
        self.easy_margin = bool(easy_margin)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - cosine.pow(2)).clamp(0, 1)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.s

    @torch.no_grad()
    def inference_logits(self, x: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        return cosine * self.s


class LCLNet(nn.Module):
    """
    End-to-end module for train_img.py-style training:
      x -> FFDNet denoise -> ResNetEmbed -> AmsMarginProduct
    """

    def __init__(
        self,
        num_classes: int,
        out_dim: int = 128,
        sigma: float = 12 / 255.0,
        den_in_nc: int = 1,
        den_out_nc: int = 1,
        den_nc: int = 64,
        den_nb: int = 15,
        den_act_mode: str = "R",
        ams_s: float = 26.0,
        ams_m: float = 0.30,
    ) -> None:
        super().__init__()
        self.sigma_value = float(sigma)
        self.den = FFDNet(den_in_nc, den_out_nc, den_nc, den_nb, den_act_mode)
        self.emb = ResNetEmbed(out_dim=int(out_dim))
        self.ams = AmsMarginProduct(int(out_dim), int(num_classes), s=ams_s, m=ams_m)

    def make_sigma(self, x: torch.Tensor) -> torch.Tensor:
        b = int(x.shape[0])
        return x.new_full((b, 1, 1, 1), self.sigma_value)

    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        return self.den(x, self.make_sigma(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)

    def forward(
        self,
        a: torch.Tensor,
        p: Optional[torch.Tensor] = None,
        n: Optional[torch.Tensor] = None,
        m: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        denoise_no_grad: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        ctx = torch.no_grad() if denoise_no_grad else nullcontext()
        with ctx:
            ra = self.denoise(a)
            rp = self.denoise(p) if p is not None else None
            rn = self.denoise(n) if n is not None else None
            rm = self.denoise(m) if m is not None else None

        ea = self.embed(ra)
        ep = self.embed(rp) if rp is not None else None
        en = self.embed(rn) if rn is not None else None
        em = self.embed(rm) if rm is not None else None

        logits = self.ams(ea, label) if label is not None else None
        return ea, ep, en, em, logits

