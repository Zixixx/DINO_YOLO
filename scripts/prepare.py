from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "default.yaml"

# 允许脚本执行时找到 src 下的工具函数。
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, resolve_path


def exists(path: Path) -> str:
    # 统一输出检查状态，方便快速定位缺失的数据、权重或本地代码仓库。
    return "OK" if path.exists() else "MISSING"


def main() -> None:
    config = load_config(CONFIG_PATH)
    project_root = Path(config["_project_root"])
    paths = config["paths"]
    dino_repo = resolve_path(project_root, config["models"]["dino"]["repo_or_dir"])

    # 这些路径是训练前的硬性依赖；缺任何一个都会导致后续脚本无法完整运行。
    checks = {
        "unlabeled_images": resolve_path(project_root, paths["unlabeled_images"]),
        "labeled_data_yaml": resolve_path(project_root, paths["labeled_data_yaml"]),
        "dino_weights": resolve_path(project_root, paths["dino_weights"]),
        "dino_repo": dino_repo,
        "yolo_weights": resolve_path(project_root, paths["yolo_weights"]),
    }
    for name, path in checks.items():
        print(f"{name:18s} {exists(path):8s} {path}")

    # 这里只检查包是否可导入，不触发模型加载。
    for package in ("torch", "torchvision", "ultralytics", "yaml", "tqdm", "tensorboard"):
        print(f"{package:18s} {'OK' if importlib.util.find_spec(package) else 'MISSING'}")


if __name__ == "__main__":
    main()
