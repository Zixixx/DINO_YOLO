from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class UnlabeledImageDataset(Dataset):
    """无标注图像数据集，只返回训练图像和文件路径，不读取任何标签。"""

    def __init__(self, image_dir: str | Path, image_size: int) -> None:
        self.image_dir = Path(image_dir).resolve()
        # 递归收集图片，便于用户按子目录组织无标注数据。
        self.paths = sorted(
            p for p in self.image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"在目录中未找到图像： {self.image_dir}")

        # 蒸馏阶段要求同一 batch 尺寸一致，这里统一 resize 后转 tensor。
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        # 强制转 RGB，避免灰度图或带 alpha 通道的图片破坏模型输入维度。
        image = Image.open(path).convert("RGB")
        return self.transform(image), str(path)
