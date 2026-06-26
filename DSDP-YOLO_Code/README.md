DSDP-YOLO (Improved YOLOv8)
本仓库是针对海洋油污/舰船检测任务对 YOLOv8 进行的改进版本。通过引入注意力机制、蛇形卷积及优化损失函数，显著提升了模型在复杂背景下的特征提取与目标定位能力。
目录结构
Plaintext
DSDP-YOLO_Code/
├── cfg/                # 模型 YAML 配置文件
├── image/              # 实验可视化脚本与结果
├── models/             # 核心模型定义与改进模块
│   ├── block.py        # 包含改进模块: HDSB, DPMA
│   ├── head.py         # 模型检测头定义
│   ├── tasks.py        # 模型解析与任务分发逻辑
│   └── __init__.py
├── predict.py          # 推理脚本
├── train.py            # 训练脚本
├── requirements.txt    # 项目依赖
└── README.md
改进说明 (Methodology)
本项目在原始 YOLOv8 的基础上进行了以下核心改进：
HDSB (High-order Differential Snake-like Bottleneck):
在 models/block.py 中实现了蛇形卷积结构，增强了模型对细长、不规则油污目标的建模能力。
DPMA (Dual-Path Multi-scale Attention):
引入了通道与空间双路径注意力机制，专注于强化多尺度特征提取，有效抑制背景噪声。
WIoU (Wise-IoU Loss):
引入了 WIoU 损失函数，优化了边界框回归策略，提升了模型对重叠及遮挡目标的定位精度。
快速开始 (Quick Start)
1. 环境安装
```bash
pip install -r requirements.txt
python train.py --cfg cfg/yolov8m-HDSB-DPMA.yaml --data data.yaml --epochs 300
2. 模型训练
使用自定义的配置文件启动训练：
Bash
python train.py --cfg cfg/yolov8m-HDSB-DPMA.yaml --data data.yaml --epochs 300
3. 模型预测
调用训练好的权重进行推理：
Bash
python predict.py --weights path/to/best.pt --source test_image.jpg
实验可视化
位于 image/ 文件夹下的脚本支持以下可视化功能：
Comparison of training convergence curves: 收敛曲线对比。
Grad-CAM heatmap visual comparison: 模型焦点特征热力图对比。
Bar chart of performance comparison: 性能评估柱状图。