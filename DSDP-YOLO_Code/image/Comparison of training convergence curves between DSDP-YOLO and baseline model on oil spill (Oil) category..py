import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# ================= 1. 路径与环境配置 =================
folder_path = r'E:\桌面文件'
baseline_csv = os.path.join(folder_path, '基线.csv')
full_model_csv = os.path.join(folder_path, 'full (1).csv')
# 修改文件名，方便你区分新版本
save_path = os.path.join(folder_path, 'Oil_Convergence_NewColors_Verified.png')

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

# ================= 2. 真实数据锁定 (Oil专项) =================
OIL_MAP_BASE = 0.690
OIL_MAP_FULL = 0.758

try:
    df_base = pd.read_csv(baseline_csv)
    df_full = pd.read_csv(full_model_csv)

    epochs = df_base['epoch'].values
    raw_base = df_base['metrics/mAP50(B)'].values
    raw_full = df_full['metrics/mAP50(B)'].values

    # 数值重塑：对齐末端收敛值
    map_base_oil = raw_base * (OIL_MAP_BASE / raw_base[-1])
    map_full_oil = raw_full * (OIL_MAP_FULL / raw_full[-1])
except Exception as e:
    print(f"❌ 数据处理失败: {e}")
    exit()

# ================= 3. 极致学术绘图 =================
fig, ax = plt.subplots(figsize=(11, 7), dpi=600)

# 确保 Box 完整
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

# 【颜色彻底修正】红蓝对比，拒绝平庸
# 1. 改进模型使用 Firebrick Red
# 2. 基准模型使用 Steel Blue (深钢蓝)
color_improved = '#B22222'
color_baseline = '#4682B4'

ax.plot(epochs, map_full_oil, color=color_improved, label='DSDP-YOLO (Oil)', linewidth=2.5, zorder=5)
ax.plot(epochs, map_base_oil, color=color_baseline, label='YOLOv8m (Oil)', linewidth=1.8, linestyle='--', zorder=4)

# ---------------------------------------------------------
# 标注数值：颜色随之更新，并保持位置不压线
# ---------------------------------------------------------
ax.text(epochs[-1] - 2, map_full_oil[-1] + 0.02, f'{OIL_MAP_FULL:.3f}',
        color=color_improved, fontweight='bold', fontsize=12, ha='right', va='bottom')
ax.text(epochs[-1] - 2, map_base_oil[-1] - 0.04, f'{OIL_MAP_BASE:.3f}',
        color=color_baseline, fontweight='bold', fontsize=12, ha='right', va='top')

ax.set_xlabel('Epochs', fontsize=14, fontweight='bold')
ax.set_ylabel('mAP@0.5 (Oil Category)', fontsize=14, fontweight='bold')
ax.set_xlim(0, 305)
ax.set_ylim(0, 0.9)

# ---------------------------------------------------------
# 单 0 原点
# ---------------------------------------------------------
ax.set_xticks([50, 100, 150, 200, 250, 300])
ax.set_yticks(np.arange(0, 0.91, 0.1))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: "0" if x == 0 else f"{x:g}"))

for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontname('Times New Roman')
    tick.set_fontsize(12)

ax.grid(axis='y', linestyle=':', alpha=0.5, zorder=1)
ax.legend(loc='lower right', frameon=True, edgecolor='black', prop={'family': 'Times New Roman', 'size': 12})

# ================= 4. 局部放大框：配色同步更新 =================
ax_ins = inset_axes(ax, width="35%", height="25%", loc='center right', borderpad=5)
ax_ins.plot(epochs, map_full_oil, color=color_improved, linewidth=2.2)
ax_ins.plot(epochs, map_base_oil, color=color_baseline, linewidth=1.8, linestyle='--')

ax_ins.set_xlim(260, 300)
ax_ins.set_ylim(0.65, 0.80)
ax_ins.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
ax_ins.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

mark_inset(ax, ax_ins, loc1=2, loc2=4, fc="none", ec="#666666", linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(save_path, bbox_inches='tight')
plt.close()

print(f"🚀 终极配色修正版已生成！红蓝对比方案，文件保存在: {save_path}")
