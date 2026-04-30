from __future__ import annotations

import torch
import torch.nn.functional as F


def spatial_cosine_distillation_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """计算逐位置余弦蒸馏损失，让学生特征在语义方向上靠近教师特征。"""

    if student.shape[-2:] != teacher.shape[-2:]:
        # YOLO 和 DINO 的特征网格大小通常不同，先把学生特征对齐到教师网格。
        student = F.interpolate(student, size=teacher.shape[-2:], mode="bilinear", align_corners=False)
    # 归一化后只比较方向相似度，弱化不同模型特征幅值差异。
    student = F.normalize(student, dim=1)
    teacher = F.normalize(teacher.detach(), dim=1)
    return 1.0 - (student * teacher).sum(dim=1).mean()
