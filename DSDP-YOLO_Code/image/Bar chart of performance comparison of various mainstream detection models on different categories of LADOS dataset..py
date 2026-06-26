import os
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 字体与中文乱码修复
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 严格核对的 Oil 类别数据 & 标签
# ==========================================
models = ['YOLOv5s', 'YOLOv8m\n(Baseline)', 'YOLOv8-CBAM', 'YOLOv9c', 'RT-DETR', 'DSDP-YOLO\n(Ours)']
metrics = ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95']

data = {
    'Precision': [0.755, 0.787, 0.765, 0.816, 0.764, 0.804],
    'Recall': [0.612, 0.668, 0.680, 0.645, 0.659, 0.675],
    'mAP@0.5': [0.692, 0.690, 0.748, 0.741, 0.742, 0.758],
    'mAP@0.5:0.95': [0.495, 0.485, 0.520, 0.512, 0.535, 0.552]
}

# ==========================================
# 3. 画布与条形图参数设置
# ==========================================
x = np.arange(len(models))
width = 0.18

fig, ax = plt.subplots(figsize=(14, 7))

# 采用低饱和度的莫兰迪淡色系，高级且不刺眼
colors = ['#AEC7E8', '#FFBB78', '#98DF8A', '#FF9896']

# ==========================================
# 4. 绘制分组柱状图并添加数值标签
# ==========================================
for i, metric in enumerate(metrics):
    offset = (i - 1.5) * width
    rects = ax.bar(x + offset, data[metric], width, label=metric, color=colors[i], edgecolor='black', linewidth=0.8,
                   alpha=1.0)

    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontname='Times New Roman')

# ==========================================
# 5. 坐标轴与图例优化
# ==========================================
ax.set_ylabel('Metrics Value', fontsize=14, fontname='Times New Roman', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=0, ha='center', fontsize=13, fontname='Times New Roman')

# 设置Y轴范围，为上方的图例留出空间
ax.set_ylim(0, 1.0)
# 确保 Y 轴原点只有一个 0，避免重复冗余
yticks = ax.get_yticks()
ax.set_yticks([y for y in yticks if y != 0] + [0])

legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4, fontsize=12, frameon=False)
for text in legend.get_texts():
    text.set_fontname('Times New Roman')

# ==========================================
# 6. 背景网格修饰
# ==========================================
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()

# ==========================================
# 7. 保存到桌面文件夹
# ==========================================
BASE_DIR = r"E:\桌面文件"
output_path = os.path.join(BASE_DIR, 'Fig3_Oil_Performance_Comparison.png')

# 确保文件夹存在，防止报错
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

plt.savefig(output_path, dpi=600, bbox_inches='tight')
plt.show()

print(f"\n大功告成！高清水粉色系柱状图已成功保存到你的桌面：{output_path}")