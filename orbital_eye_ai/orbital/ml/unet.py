"""U-Net for retinal layer segmentation of OCT B-scans (PyTorch).

Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015.
Input : (N, 1, H, W) grayscale B-scan, H and W divisible by 2**depth.
Output: (N, num_classes, H, W) logits. With the Duke DME labelling used in
training/dataset.py, num_classes = 9 (above ILM, 7 retinal layers, below BM).
"""
from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 9, base: int = 32, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.depth = depth
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.downs = nn.ModuleList()
        prev = in_channels
        for c in chs[:-1]:
            self.downs.append(DoubleConv(prev, c))
            prev = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(chs[-2], chs[-1], dropout)
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for c in reversed(chs[:-1]):
            self.ups.append(nn.ConvTranspose2d(c * 2, c, 2, stride=2))
            self.up_convs.append(DoubleConv(c * 2, c))
        self.head = nn.Conv2d(chs[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            x = conv(torch.cat([skip, x], dim=1))
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
