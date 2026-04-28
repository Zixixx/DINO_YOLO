from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # 记录项目根目录，后续所有相对路径都以根目录为基准解析。
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parents[1])
    return config


def resolve_path(project_root: str | Path, path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    # 配置文件里保持短路径，运行时统一转换成绝对路径。
    return (Path(project_root) / path).resolve()


def require_file(path: str | Path, description: str) -> Path:
    path = Path(path).resolve()
    if not path.is_file():
        # 主动报错比后续模型加载时抛出错误更容易排查。
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def require_dir(path: str | Path, description: str) -> Path:
    path = Path(path).resolve()
    if not path.is_dir():
        # 对数据目录和本地仓库目录做显式检查。
        raise FileNotFoundError(f"{description} not found: {path}")
    return path
