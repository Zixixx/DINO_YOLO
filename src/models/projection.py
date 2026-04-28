from __future__ import annotations

import torch
from torch import nn


class ProjectionHead(nn.Module):
    """投影头模块, 用于对齐YOLO与DINO的维度"""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int | None = None) -> None:
        super().__init__()
        # 1x1 卷积只改变通道维度，保留 YOLO 特征的空间位置结构。
        hidden_channels = hidden_channels or min(max(in_channels, out_channels // 2), out_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
