from __future__ import annotations

from copy import deepcopy
import sys
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "default.yaml"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 允许脚本执行时找到 src 下的工具函数。
sys.path.insert(0, str(ROOT))

from src.data.unlabeled import UnlabeledImageDataset
from src.models.dino_teacher import DinoTeacher
from src.models.losses import spatial_cosine_distillation_loss
from src.models.projection import ProjectionHead
from src.models.yolo_student import YoloFeatureStudent
from src.utils.config import load_config, require_dir, require_file, resolve_path
from src.utils.seed import set_seed


def build_projection(
    student: YoloFeatureStudent,
    teacher: DinoTeacher,
    image_size: int,
    device: torch.device,
    out_dim: int,
) -> ProjectionHead:
    # 用一张 dummy 图跑一次前向，自动探测 YOLO 特征通道数，避免手写通道配置。
    student.eval()
    teacher.eval()
    dummy = torch.zeros(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        yolo_features = student(dummy)
        _ = teacher(dummy)
    projection = ProjectionHead(yolo_features.shape[1], out_dim).to(device)
    student.train()
    return projection


def save_checkpoint(
    path: Path,
    student: YoloFeatureStudent,
    projection: ProjectionHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict,
    loss: float,
    yolo_ema_state_dict: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # checkpoint 同时保存 YOLO、投影头和优化器，便于继续训练或进入微调阶段。
    checkpoint = {
        "epoch": epoch,
        "loss": loss,
        "config": config,
        "yolo_state_dict": student.model.state_dict(),
        "projection_state_dict": projection.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if yolo_ema_state_dict is not None:
        checkpoint["yolo_ema_state_dict"] = yolo_ema_state_dict
    torch.save(checkpoint, path)


def build_optimizer(params: list[torch.nn.Parameter], train_cfg: dict) -> torch.optim.Optimizer:
    optimizer_name = str(train_cfg.get("optimizer", "AdamW")).lower()
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])

    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(train_cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {train_cfg.get('optimizer')}. choose AdamW, Adam, or SGD.")


class ModelEMA:
    """对YOLO学生权重采用EMA。"""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.ema = deepcopy(model).eval()
        self.decay = decay
        for param in self.ema.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        model_state = model.state_dict()
        for key, ema_value in self.ema.state_dict().items():
            model_value = model_state[key].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(model_value.to(dtype=ema_value.dtype), alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)

    def state_dict(self) -> dict:
        return self.ema.state_dict()


def ema_state_dict(ema: ModelEMA | None) -> dict | None:
    return None if ema is None else ema.state_dict()


def main() -> None:
    config = load_config(CONFIG_PATH)
    project_root = Path(config["_project_root"])
    set_seed(int(config["seed"]))

    paths = config["paths"]
    model_cfg = config["models"]
    train_cfg = config["distill"]

    # 预训练必须使用本地数据和本地权重，缺失时立即报错。
    unlabeled_dir = require_dir(resolve_path(project_root, paths["unlabeled_images"]), "Unlabeled image directory")
    dino_weights = require_file(resolve_path(project_root, paths["dino_weights"]), "DINO local weights")
    yolo_weights = require_file(resolve_path(project_root, paths["yolo_weights"]), "YOLO local weights")
    output_dir = resolve_path(project_root, paths["output_dir"])
    ckpt_dir = output_dir / "distill"
    writer = SummaryWriter(str(output_dir / "logs" / "distill"))

    device = torch.device(DEVICE)
    # 无标注数据只需要图像本身；路径会随 batch 返回，便于后续排查坏图或异常样本。
    dataset = UnlabeledImageDataset(unlabeled_dir, int(train_cfg["image_size"]))
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    dino_cfg = model_cfg["dino"]
    dino_precision = str(dino_cfg.get("precision", "fp32")).lower()
    if device.type != "cuda" and dino_precision != "fp32":
        print(f"DINO precision {dino_precision} requires CUDA for this training script; falling back to fp32.")
        dino_precision = "fp32"
    # 教师模型冻结，只提供稳定的语义软目标，不参与梯度更新。
    teacher = DinoTeacher(
        arch=dino_cfg["arch"],
        weights_path=dino_weights,
        image_size=int(dino_cfg["image_size"]),
        patch_size=int(dino_cfg["patch_size"]),
        repo_or_dir=str(resolve_path(project_root, dino_cfg["repo_or_dir"])),
        precision=dino_precision,
        source=dino_cfg["source"],
        trust_repo=bool(dino_cfg["trust_repo"]),
    ).to(device)

    yolo_cfg = model_cfg["yolo"]
    # 学生模型通过 forward hook 暴露中间特征，用于和 DINO patch 特征对齐。
    student = YoloFeatureStudent(
        weights_path=yolo_weights,
        feature_layer=int(yolo_cfg["feature_layer"]),
        freeze_detection_head=bool(yolo_cfg["freeze_detection_head"]),
    ).to(device)
    projection = build_projection(
        student,
        teacher,
        int(train_cfg["image_size"]),
        device,
        int(dino_cfg["embed_dim"]),
    )

    # 只优化学生模型中 requires_grad=True 的参数以及投影头参数。
    trainable_params = [p for p in student.parameters() if p.requires_grad] + list(projection.parameters())
    optimizer = build_optimizer(trainable_params, train_cfg)
    # AMP 仅在 CUDA 上启用，CPU 环境会自动退化为普通精度。
    amp_enabled = bool(train_cfg["amp"]) and device.type == "cuda"
    scaler = GradScaler(device.type, enabled=amp_enabled)
    accumulate_steps = max(int(train_cfg.get("accumulate_steps", 1)), 1)
    student_ema = ModelEMA(student.model, float(train_cfg.get("ema_decay", 0.999))) if bool(train_cfg.get("use_ema", False)) else None

    best_loss = float("inf")
    global_step = 0
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        running_loss = 0.0
        progress = tqdm(loader, desc=f"distill epoch {epoch}", ncols=100)
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, (images, _paths) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)

            # DINO 是冻结教师，不需要保存计算图，能显著降低显存占用。
            with torch.no_grad():
                teacher_features = teacher(images)

            with autocast(device_type=device.type, enabled=amp_enabled):
                student_features = student(images)
                projected = projection(student_features)
                # 核心蒸馏目标：让投影后的 YOLO 空间特征逐位置逼近 DINO 语义特征。
                loss = float(train_cfg["cosine_loss_weight"]) * spatial_cosine_distillation_loss(projected, teacher_features)
                loss_for_backward = loss / accumulate_steps

            scaler.scale(loss_for_backward).backward()
            should_step = batch_idx % accumulate_steps == 0 or batch_idx == len(loader)
            if should_step:
                if float(train_cfg["grad_clip_norm"]) > 0:
                    # 梯度裁剪。
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, float(train_cfg["grad_clip_norm"]))
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_was_updated = scaler.get_scale() >= old_scale
                if student_ema is not None and optimizer_was_updated:
                    student_ema.update(student.model)
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if global_step % int(train_cfg["log_interval"]) == 0:
                writer.add_scalar("loss/cosine_distill", loss.item(), global_step)

        epoch_loss = running_loss / max(len(loader), 1)
        writer.add_scalar("loss/epoch", epoch_loss, epoch)
        # last 用于断点观察，best 用于后续微调，epoch_xxx 用于阶段性回溯。
        save_checkpoint(ckpt_dir / "last.pt", student, projection, optimizer, epoch, config, epoch_loss, ema_state_dict(student_ema))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(ckpt_dir / "best.pt", student, projection, optimizer, epoch, config, epoch_loss, ema_state_dict(student_ema))
        if epoch % int(train_cfg["save_interval"]) == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", student, projection, optimizer, epoch, config, epoch_loss, ema_state_dict(student_ema))

    writer.close()
    student.remove_hook()
    print(f"distill complete. Best checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
