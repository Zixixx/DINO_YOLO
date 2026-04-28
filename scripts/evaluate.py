from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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
    eval_cfg = config["evaluate"]

    output_dir = resolve_path(project_root, paths["output_dir"])
    # 默认评估微调阶段保存的 best.pt。
    weights = require_file(
        output_dir / "finetune" / "dino_yolo" / "weights" / "best.pt",
        "Fine-tuned detector weights",
    )
    data_yaml = require_file(resolve_path(project_root, paths["labeled_data_yaml"]), "YOLO labeled dataset yaml")

    # Ultralytics 的 val 会根据 dataset.yaml 自动加载验证集并计算检测指标。
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=int(eval_cfg["image_size"]),
        device=eval_cfg["device"],
    )

    print("Evaluation results")
    print(f"mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision:    {np.mean(metrics.box.p):.4f}")
    print(f"Recall:       {np.mean(metrics.box.r):.4f}")


if __name__ == "__main__":
    main()
