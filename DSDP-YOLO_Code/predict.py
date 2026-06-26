from ultralytics import YOLO

# 1. 加载模型权重
model = YOLO('runs/dir/detect/Exp_C_YOLOv8m_DPMA/weights/best.pt')

# 2. 执行推理
results = model.predict(source='data/test.jpg', save=True, conf=0.5)

print("Inference completed. Results saved.")