from __future__ import annotations

import sys
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "default.yaml"

# 允许脚本执行时找到 src 下的工具函数。
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, require_file, resolve_path


def main() -> None:
    config = load_config(CONFIG_PATH)
    project_root = Path(config["_project_root"])
    paths = config["paths"]
    finetune_cfg = config["finetune"]

    yolo_weights = require_file(resolve_path(project_root, paths["yolo_weights"]), "YOLO local weights")
    data_yaml = require_file(resolve_path(project_root, paths["labeled_data_yaml"]), "YOLO labeled dataset yaml")
    output_dir = resolve_path(project_root, paths["output_dir"])
    checkpoint_path = require_file(output_dir / "pretrain" / "best.pt", "Distillation checkpoint")

    # 先从本地 YOLO 权重构建完整检测器，再覆盖蒸馏阶段更新过的学生参数。
    model = YOLO(str(yolo_weights))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    yolo_key = "yolo_ema_state_dict" if "yolo_ema_state_dict" in checkpoint else "yolo_state_dict"
    missing, unexpected = model.model.load_state_dict(checkpoint[yolo_key], strict=False)
    print(f"Loaded distilled YOLO weights from {checkpoint_path} ({yolo_key})")
    print(f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    # 微调阶段回到标准 Ultralytics 训练流程，使用标注框监督优化检测头和 backbone。
    model.train(
        data=str(data_yaml),
        project=str(output_dir / "finetune"),
        name="dino_yolo",
        epochs=int(finetune_cfg["epochs"]),
        imgsz=int(finetune_cfg["image_size"]),
        batch=int(finetune_cfg["batch_size"]),
        amp=bool(finetune_cfg.get("amp", False)),
        device=finetune_cfg["device"],
        workers=int(finetune_cfg["workers"]),
        optimizer=finetune_cfg["optimizer"],
        lr0=float(finetune_cfg["lr0"]),
        lrf=float(finetune_cfg["lrf"]),
        momentum=float(finetune_cfg["momentum"]),
        weight_decay=float(finetune_cfg["weight_decay"]),
        plots=bool(finetune_cfg.get("plots", False)),
        pretrained=False,
        exist_ok=True,
        val=True,
        save=True,
    )

    print(f"Fine-tuning complete. Best model: {output_dir / 'finetune' / 'dino_yolo' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
