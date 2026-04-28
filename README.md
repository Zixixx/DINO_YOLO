# 基于DINO与YOLO的自监督目标检测方案

这是一个自监督目标检测工程，目标是用大量无标注图像降低目标检测任务的标注成本。项目采用冻结的 DINO 教师模型为 YOLO 学生模型提供逐位置语义监督，再将蒸馏后的 YOLO 用于有监督微调和检测评估。

## 核心方法

- 教师-学生蒸馏框架：使用本地 DINO 权重作为教师模型，使用本地 YOLO 作为学生模型。
- 特征空间对齐：通过投影头把 YOLO 中间特征映射到 DINO 的高维语义空间。
- 蒸馏损失约束：使用逐位置余弦相似度损失，让 YOLO 特征逼近 DINO 特征。
- 预训练阶段优化：冻结 DINO 教师，只优化 YOLO 学生中可训练参数和投影头。
- 微调与评估阶段：丢弃 DINO 教师和投影头，只加载蒸馏后的 YOLO 权重，在标注数据上微调并输出 mAP、Precision、Recall。

## 目录结构

```text
DINO_YOLO/
  configs/
    default.yaml              # 全局配置
  data/
    unlabeled/images/         # 无标注图像
    labeled/
      dataset.yaml            # YOLO 数据集配置
      images/train/
      images/val/
      images/test/
      labels/train/
      labels/val/
      labels/test/
  model/                      # 放置本地 DINO 权重、DINO 仓库和 YOLO 权重
  scripts/
    prepare.py
    pretrain.py
    finetune.py
    evaluate.py
  src/
    data/
    models/
    utils/
```

## 数据集要求

无标注图像放在：

```text
data/unlabeled/images/
```

标注数据按 YOLO 格式放在：

```text
data/labeled/images/train/
data/labeled/images/val/
data/labeled/images/test/
data/labeled/labels/train/
data/labeled/labels/val/
data/labeled/labels/test/
```

同时需要修改：

```text
data/labeled/dataset.yaml
```

示例：

```yaml
train: images/train
val: images/val
test: images/test

nc: 3
names:
  0: cat
  1: dog
  2: rabbit
```

## 环境依赖
请先安装Anaconda/Miniconda，推荐Python 3.12+。

```bash
conda create -n dino_yolo python=3.12
conda activate dino_yolo
pip install -r requirements.txt
```

`prepare.py` 会检查数据目录、本地 DINO 权重、本地 YOLO 权重、本地 DINO 仓库和关键 Python 包是否存在。

## 运行流程

1. 数据检查：

```bash
python scripts/prepare.py
```

脚本会检查数据目录、本地 DINO 权重、本地 YOLO 权重、本地 DINO 仓库和关键 Python 包是否存在。

2. 自监督蒸馏预训练：

```bash
python scripts/pretrain.py
```

输出：

```text
outputs/pretrain/best.pt
outputs/pretrain/last.pt
```

3. 标注数据微调：

```bash
python scripts/finetune.py
```

输出：

```text
outputs/finetune/dino_yolo/weights/best.pt
```

4. 评估检测效果：

```bash
python scripts/evaluate.py
```

## 致谢
- [DINOv3](https://github.com/facebookresearch/dinov3)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
