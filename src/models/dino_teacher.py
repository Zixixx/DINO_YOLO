from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class DinoTeacher(nn.Module):
    """冻结的 DINO 教师模型，负责输出高维语义空间特征。"""

    def __init__(
        self,
        arch: str,
        weights_path: str | Path,
        image_size: int,
        patch_size: int,
        repo_or_dir: str,
        precision: str = "fp32",
        source: str = "torchhub",
        trust_repo: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.precision = precision.lower()
        self.dtype = self._precision_to_dtype(self.precision)
        self.model = self._load_model(arch, repo_or_dir, source, trust_repo, weights_path)
        self.model.to(dtype=self.dtype)
        # 教师模型只提供软目标，不参与训练更新。
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def _precision_to_dtype(self, precision: str) -> torch.dtype:
        if precision == "fp32":
            return torch.float32
        if precision == "fp16":
            return torch.float16
        if precision == "bf16":
            return torch.bfloat16
        raise ValueError(f"Unsupported DINO precision: {precision}. Choose fp32, fp16, or bf16.")

    def _load_model(
        self,
        arch: str,
        repo_or_dir: str,
        source: str,
        trust_repo: bool,
        weights_path: str | Path,
    ) -> nn.Module:
        if source not in {"local", "github"}:
            raise ValueError(f"Unsupported DINO source: {source}. Choose local or github.")
        repo_or_dir = str(Path(repo_or_dir).resolve()) if source == "local" else repo_or_dir
        weights_path = str(Path(weights_path).resolve())
        model = torch.hub.load(repo_or_dir, arch, source=source, trust_repo=trust_repo, pretrained=False)
        self._load_local_weights(model, weights_path)
        return model

    def _load_local_weights(self, model: nn.Module, weights_path: str | Path) -> None:
        checkpoint = torch.load(Path(weights_path), map_location="cpu")
        # 兼容常见 checkpoint 格式：state_dict、teacher/model/state_dict。
        if isinstance(checkpoint, dict):
            state_dict = (
                checkpoint.get("teacher")
                or checkpoint.get("model")
                or checkpoint.get("state_dict")
                or checkpoint
            )
        else:
            state_dict = checkpoint

        cleaned = {}
        for key, value in state_dict.items():
            # 去掉分布式训练或封装模型常见的前缀，提高本地权重加载兼容性。
            key = key.removeprefix("module.").removeprefix("backbone.").removeprefix("model.")
            cleaned[key] = value
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"[DINO] Missing keys while loading local weights: {len(missing)}")
        if unexpected:
            print(f"[DINO] Unexpected keys while loading local weights: {len(unexpected)}")

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # DINO 对输入尺寸更敏感，统一 resize 到配置中的教师尺寸。
        images = F.interpolate(
            images.to(dtype=self.dtype),
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
        )
        output = self._forward_features(images)
        return self._tokens_to_spatial(output)

    def _forward_features(self, images: torch.Tensor) -> Any:
        # DINO 模型通常暴露 forward_features；没有时退回普通 forward。
        if hasattr(self.model, "forward_features"):
            return self.model.forward_features(images)
        return self.model(images)

    def _tokens_to_spatial(self, output: Any) -> torch.Tensor:
        # 不同 DINO 实现的特征字段名可能不同，这里按常见字段依次尝试。
        if isinstance(output, dict):
            for key in ("x_norm_patchtokens", "patch_tokens", "tokens"):
                if key in output:
                    output = output[key]
                    break
            else:
                output = next(v for v in output.values() if torch.is_tensor(v))

        if output.ndim == 4:
            return output

        if output.ndim != 3:
            raise RuntimeError(f"Unsupported DINO feature shape: {tuple(output.shape)}")

        tokens = output
        grid_size = self.image_size // self.patch_size
        num_patches = grid_size * grid_size
        if tokens.shape[1] > num_patches:
            # DINO 输出可能包含 CLS/register tokens，保留最后的 patch tokens 用于空间蒸馏。
            tokens = tokens[:, -num_patches:, :]
        if tokens.shape[1] != num_patches:
            # 如果输入尺寸或实现导致 patch 数变化，则尝试从 token 数反推方形网格。
            grid_size = int(tokens.shape[1] ** 0.5)
            if grid_size * grid_size != tokens.shape[1]:
                raise RuntimeError(f"Cannot reshape DINO tokens: {tuple(tokens.shape)}")
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], grid_size, grid_size)
