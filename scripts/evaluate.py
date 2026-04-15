# step3
from pathlib import Path
from ultralytics import YOLO
import numpy as np

# 配置路径
MODEL_PATH = "out/finetune/dino_yolo/weights/best.pt"                  # 模型路径
DATA_PATH = "datasets/dataset.yaml"                                    # 数据集配置

if __name__ == "__main__":
    model_path = Path(MODEL_PATH).absolute()
    data_path = Path(DATA_PATH).absolute()
    
    print(f"模型: {model_path}")
    print(f"数据: {data_path}")
    
    # 检查文件
    if not model_path.exists():
        print(f"模型不存在: {model_path}")
        exit(1)
    
    # 加载并评估
    model = YOLO(str(model_path))
    metrics = model.val(data=str(data_path), imgsz=640, device=0)
    
    # 输出结果
    print("评估结果")
    print(f"mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95:  {metrics.box.map:.4f}")
    print(f"Precision:     {np.mean(metrics.box.p):.4f}")
    print(f"Recall:        {np.mean(metrics.box.r):.4f}")