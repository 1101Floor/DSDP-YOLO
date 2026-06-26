from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 指定改好的蛇形卷积配置文件
    model_config = 'ultralytics/cfg/models/v8/yolov8m-HDSB.yaml'

    # 2. 初始化模型
    model = YOLO(model_config)

    # 3. 严格对齐参数启动训练 (Experiment B: HDSB)
    model.train(
        data='datasets/LADOS/data.yaml', # 严格对齐路径
        epochs=300,
        batch=2,                         # 严格对齐批次
        imgsz=640,
        workers=0,                       # 严格对齐线程
        lr0=0.01,
        device=0,
        project='runs/dir/detect',       # 严格对齐保存目录
        name='Exp_B_Baseline_HDSB',      # 实验B的名称
        optimizer='SGD',                 # 严格对齐优化器
        amp=True                         # 严格对齐混合精度
    )