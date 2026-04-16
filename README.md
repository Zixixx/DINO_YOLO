# DINO_YOLO: 基于DINO的自监督蒸馏YOLO目标检测

## 项目简介
本项目结合了DINOv3自监督视觉特征学习与YOLO目标检测，通过知识蒸馏实现无标签数据的自监督预训练，并在少量标注数据上微调，提升目标检测性能，适用于标注数据稀缺、需利用大量无标签图片的场景。

## 主要流程
1. **自监督预训练**（scripts/pretrain.py）
   - 使用DINOv3作为教师模型，YOLO作为学生模型，在无标签图片上进行知识蒸馏式自监督训练。
   - 输出预训练YOLO权重。
2. **微调**（scripts/finetune.py）
   - 加载自监督预训练权重，在标注数据集上微调YOLO模型。
3. **评估**（scripts/evaluate.py）
   - 使用微调后的模型在测试集上评估，输出mAP、Precision、Recall等指标。

## 目录结构
```
DINO_YOLO/
├── datasets/
│   ├── labeled/         # 标注数据集
│   │   ├── dataset.yaml
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   │   └── test/
│   │   └── labels/
│   │       ├── train/
│   │       └── val/
│   │       └── test/
│   └── unlabeled/      # 无标签图片
│       └── images/
├── scripts/
│   ├── pretrain.py     # 自监督预训练脚本
│   ├── finetune.py     # 微调脚本
│   └── evaluate.py     # 评估脚本
├── requirement.txt     # 依赖包
└── ...
```

## 数据集说明
- 标注数据集配置见 `datasets/labeled/dataset.yaml`。
- 有标签图片放于 `datasets/labeled/images/`。
- 无标签图片放于 `datasets/unlabeled/images/`。

## 环境依赖
请先安装Anaconda/Miniconda，推荐Python 3.9+。

```bash
conda create -n dino_yolo python=3.9
conda activate dino_yolo
pip install -r requirement.txt
```

## 快速开始
### 1. 自监督预训练
```bash
python scripts/pretrain.py
```

### 2. 微调
```bash
python scripts/finetune.py
```

### 3. 评估
```bash
python scripts/evaluate.py
```

## 主要依赖
- torch, torchvision
- ultralytics (YOLO)
- lightly-train
- opencv-python, pillow, albumentations
- numpy, pandas, matplotlib, seaborn

## 致谢
- [DINOv3](https://github.com/facebookresearch/dinov3)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Lightly](https://github.com/lightly-ai/lightly)

