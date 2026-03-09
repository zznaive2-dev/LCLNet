from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch import nn


class HierQuadLoss(nn.Module):
    def __init__(self, m1: float, m2: float, lambda_m: float):
        super().__init__()
        self.m1 = float(m1)
        self.m2 = float(m2)
        self.lm = float(lambda_m)

    def forward(self, ea: torch.Tensor, ep: torch.Tensor, em: torch.Tensor, en: torch.Tensor) -> torch.Tensor:
        dap = F.pairwise_distance(ea, ep, 2)
        dam = F.pairwise_distance(ea, em, 2)
        dan = F.pairwise_distance(ea, en, 2)
        return dap.mean() + self.lm * F.relu(self.m1 - dam).mean() + F.relu(self.m2 - dan).mean()


class HardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.m = float(margin)

    def forward(self, feat: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(feat, feat)
        mask = label[:, None] == label[None, :]
        hard_p = (dist * mask.float()).max(1).values
        hard_n = (dist + 1e5 * mask.float()).min(1).values
        return F.relu(hard_p - hard_n + self.m).mean()

