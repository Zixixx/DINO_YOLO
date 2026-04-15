# step1
from pathlib import Path
from ultralytics import YOLO
import lightly_train
# 1. 初始化一个未经训练的 YOLO 模型
model = YOLO("yolov8n.yaml")  # 从随机权重开始

# 2. 配置 DINOv3 教师模型 
teacher_config = {
    "teacher": "dinov3/vitl16",           # 指定教师模型架构
    #"teacher_weights": str(Path("").absolute()) # 若使用本地权重，指定路径
}

# 3. 执行蒸馏预训练
lightly_train.pretrain(
    out="out/pretrain",               # 输出目录
    data="datasets/unlabeled/images",     # 无标注图像路径
    model=model,                          # YOLO 学生模型
    method="distillation",                # 使用知识蒸馏方法
    method_args=teacher_config,           # 传入教师模型配置
    epochs=100,                           # 预训练轮数，可根据效果调整
    batch_size=16,                        # 根据你的GPU内存调整
    overwrite=True,                       # 是否覆盖输出目录
)
