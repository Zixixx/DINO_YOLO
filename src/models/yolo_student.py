from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from ultralytics import YOLO


class YoloFeatureStudent(nn.Module):
    """YOLO 学生模型封装，用 forward hook 捕获中间层特征。"""

    def __init__(self, weights_path: str | Path, feature_layer: int = -2, freeze_detection_head: bool = True) -> None:
        super().__init__()
        # 本地 YOLO 权重构建。
        yolo = YOLO(str(Path(weights_path).resolve()))
        self.__dict__["yolo"] = yolo
        self.model = yolo.model
        self.feature_layer = feature_layer
        self._features: torch.Tensor | None = None
        self._hook = self._register_feature_hook(feature_layer)
        if freeze_detection_head:
            self._freeze_detection_head()

    def _register_feature_hook(self, layer_index: int):
        layers = self._layers()
        layer = layers[layer_index]

        def hook(_module, _inputs, output):
            # 某些 YOLO 模块可能返回 list/tuple，这里取第一个张量作为蒸馏特征。
            if isinstance(output, (list, tuple)):
                output = output[0]
            self._features = output

        return layer.register_forward_hook(hook)

    def _layers(self) -> nn.ModuleList:
        layers = getattr(self.model, "model", None)
        if isinstance(layers, nn.ModuleList):
            return layers
        if isinstance(layers, nn.Sequential):
            return nn.ModuleList(list(layers.children()))
        candidates: list[tuple[str, nn.Module]] = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.ModuleList, nn.Sequential)) and len(module) > 0:
                candidates.append((name, module))
        if candidates:
            name, module = max(candidates, key=lambda item: len(item[1]))
            print(f"[YOLO] Using layer container '{name}' with {len(module)} layers for feature hooks.")
            return nn.ModuleList(list(module.children())) if isinstance(module, nn.Sequential) else module

        child_summary = ", ".join(f"{name}:{type(module).__name__}" for name, module in list(self.model.named_children())[:20])
        raise RuntimeError(
            "Unsupported YOLO model layout: could not find a ModuleList/Sequential layer stack. "
            f"Top-level children: {child_summary}"
        )

    def _freeze_detection_head(self) -> None:
        layers = self._layers()
        if layers:
            # 蒸馏阶段主要塑造 backbone/neck 表示，默认不更新最后的检测头。
            for param in layers[-1].parameters():
                param.requires_grad_(False)

    def train(self, mode: bool = True):
        self.training = mode
        self.model.train(mode)
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # 每次前向前清空缓存，确保返回的是当前 batch 的中间特征。
        self._features = None
        _ = self.model(images)
        if self._features is None:
            raise RuntimeError(f"No YOLO features captured from layer {self.feature_layer}")
        return self._features

    def remove_hook(self) -> None:
        # 训练结束后移除 hook，避免长时间进程中重复注册或引用泄漏。
        self._hook.remove()
