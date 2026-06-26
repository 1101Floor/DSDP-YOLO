import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ---------------------------------------------------------
# 1. 文件夹路径与图片名称设置
# ---------------------------------------------------------
BASE_DIR = r"E:\桌面文件"
IMAGE_NAMES = ["111.jpg", "222.jpg", "333.jpg"]

# 2. 基线模型与改进模型权重路径
BASELINE_MODEL_PATH = r"C:\new\runs\dir\detect\train-5\weights\best.pt"
DYSNAKE_MODEL_PATH = r"C:\new\runs\dir\detect\runs\dir\detect\Exp_F_Full_WIoU_Final\weights\best.pt"

# 3. 热力图透明度设置
IMAGE_WEIGHT = 0.65

# 全局学术字体设置
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.family'] = 'serif'


class YOLOv8Target:
    def __init__(self, category_index):
        self.category_index = category_index

    def __call__(self, model_output):
        if isinstance(model_output, (list, tuple)):
            preds = model_output[0]
        else:
            preds = model_output
        return preds[0, 4 + self.category_index, :].max()


def get_heatmap(model, image_path, target_layer_index=-2, target_class_idx=0):
    target_layers = [model.model.model[target_layer_index]]

    rgb_img = cv2.imread(image_path, 1)
    if rgb_img is None:
        raise ValueError(f"读取图片失败，请检查文件是否存在：{image_path}")
    rgb_img = rgb_img[:, :, ::-1]
    rgb_img_float = np.float32(rgb_img) / 255

    cam = GradCAM(model=model.model, target_layers=target_layers)
    targets = [YOLOv8Target(category_index=target_class_idx)]

    input_tensor = torch.from_numpy(rgb_img_float).permute(2, 0, 1).unsqueeze(0).float()
    input_tensor.requires_grad_(True)

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    cam_image = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True, image_weight=IMAGE_WEIGHT)

    return rgb_img_float, cam_image


if __name__ == '__main__':
    print("正在加载模型，请稍候...")
    baseline_model = YOLO(BASELINE_MODEL_PATH)
    dysnake_model = YOLO(DYSNAKE_MODEL_PATH)

    for param in baseline_model.model.parameters():
        param.requires_grad = True
    for param in dysnake_model.model.parameters():
        param.requires_grad = True

    print("模型加载完成，开始生成热力图矩阵...")

    fig, axes = plt.subplots(3, 3, figsize=(13, 12))

    for row_idx, img_name in enumerate(IMAGE_NAMES):
        img_path = os.path.join(BASE_DIR, img_name)
        gt_img_name = img_name.split('.')[0] + "_gt." + img_name.split('.')[1]
        gt_img_path = os.path.join(BASE_DIR, gt_img_name)

        print(f"正在处理第 {row_idx + 1}/3 张图片: {img_name}")

        _, baseline_cam = get_heatmap(baseline_model, img_path, target_layer_index=-2, target_class_idx=0)
        _, dysnake_cam = get_heatmap(dysnake_model, img_path, target_layer_index=-2, target_class_idx=0)

        if os.path.exists(gt_img_path):
            display_img = cv2.imread(gt_img_path, 1)
            display_img = display_img[:, :, ::-1]
            display_img = np.float32(display_img) / 255
        else:
            display_img = cv2.imread(img_path, 1)
            display_img = display_img[:, :, ::-1]
            display_img = np.float32(display_img) / 255

        axes[row_idx, 0].imshow(display_img)
        if row_idx == 0:
            axes[row_idx, 0].set_title('Original Image (GT)', fontname='Times New Roman', fontweight='bold',
                                       fontsize=16)

        axes[row_idx, 1].imshow(baseline_cam)
        if row_idx == 0:
            axes[row_idx, 1].set_title('YOLOv8m Baseline', fontname='Times New Roman', fontweight='bold', fontsize=16)

        axes[row_idx, 2].imshow(dysnake_cam)
        if row_idx == 0:
            axes[row_idx, 2].set_title('DSDP-YOLO (Ours)', fontname='Times New Roman', fontweight='bold',
                                       fontsize=16)

    for row in axes:
        for ax in row:
            ax.axis('off')

    sm = cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])

    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Activation Intensity', fontname='Times New Roman', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)

    for label in cbar.ax.get_yticklabels():
        label.set_family('serif')
        label.set_name('Times New Roman')

    plt.subplots_adjust(left=0.05, right=0.90, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)

    # 修改保存逻辑：将图片直接保存到 BASE_DIR (桌面) 下，并使用标准学术命名
    output_filename = os.path.join(BASE_DIR, 'Fig5_Grad_CAM_Heatmap_Comparison.png')
    plt.savefig(output_filename, bbox_inches='tight', dpi=600)
    plt.show()
    print(f"修改完成，终极版大图已保存为 {output_filename}")