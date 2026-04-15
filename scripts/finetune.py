# step2
from pathlib import Path
from ultralytics import YOLO


if __name__ == "__main__":
    # 1. 加载预训练好的模型（蒸馏后的 YOLO backbone）
    model_path = Path("out/pretrain/exported_models/exported_last.pt").absolute()
    data_path = Path("datasets/dataset.yaml").absolute()
    
    print("自监督 YOLO 微调训练")
    print(f"预训练模型: {model_path}")
    print(f"数据集配置: {data_path}")
    
    # 检查文件是否存在
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"数据集配置不存在: {data_path}")
    
    # 加载模型
    model = YOLO(str(model_path))
    
    # 2. 在标注数据上进行微调
    results = model.train(
        data=str(data_path),
        project=str(Path("out/finetune").absolute()),  # 绝对路径
        name="dino_yolo",         # 实验名称
        epochs=100,                   # 增加训练轮数
        imgsz=640,                    # 图像尺寸
        batch=4,                      # 批次大小（GTX 1050 Ti 4GB）
        device=0,                     # GPU 设备
        workers=4,                    # 数据加载进程数
        
        # 优化器配置
        optimizer="SGD",              # 改用 SGD，对小数据集更稳定
        lr0=0.01,                     # 初始学习率
        lrf=0.01,                     # 最终学习率因子
        momentum=0.937,               # 动量
        weight_decay=0.0005,          # 权重衰减
        
        # 预热配置
        warmup_epochs=3,              # 预热轮数
        warmup_momentum=0.8,          # 预热动量
        
        # 损失权重
        box=7.5,                      # 框损失权重
        cls=0.5,                      # 分类损失权重
        dfl=1.5,                      # DFL 损失权重
        
        # 保存配置
        save=True,                    # 保存模型
        save_period=10,               # 每 10 轮保存一次
        exist_ok=True,                # 覆盖已有目录
        
        # 其他配置
        pretrained=True,              # 使用预训练权重
        verbose=True,                 # 详细输出
        amp=False,                    # 禁用 AMP 避免检查
        
        # 数据增强
        hsv_h=0.015,                  # HSV 色调增强
        hsv_s=0.7,                    # HSV 饱和度增强
        hsv_v=0.4,                    # HSV 明度增强
        degrees=0.0,                  # 旋转角度
        translate=0.1,                # 平移
        scale=0.5,                    # 缩放
        shear=0.0,                    # 剪切
        perspective=0.0,              # 透视
        flipud=0.0,                   # 上下翻转
        fliplr=0.5,                   # 左右翻转
        mosaic=1.0,                   # Mosaic 增强
        mixup=0.0,                    # MixUp 增强
        copy_paste=0.0,               # Copy-Paste 增强
        
        # 验证设置
        val=True,                     # 训练过程中验证
        val_period=5,                 # 每 5 轮验证一次
    )
    
    print("微调训练完成！")
    print(f"模型保存在: {Path('out/finetune/dino_yolo_ssl/weights').absolute()}")