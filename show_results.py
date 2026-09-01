import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import random
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# ==================== 这里改成你自己的路径 ====================
exp_path = "runs/train/exp3"
weights_path = exp_path + "/weights/best.pt"
val_txt_path = "../UATD/val.txt"
save_dir = exp_path + "/results"
conf_thres = 0.25    # 置信度阈值
device = "cuda:0"
# =============================================================

# 加载YOLOv5模型
model = torch.hub.load('.', 'custom', path=weights_path, source='local', device=device)
model.conf = conf_thres

# 类别名
class_names = [
    "ball",
    "circle cage",
    "cube",
    "cylinder",
    "human body",
    "metal bucket",
    "plane",
    "rov",
    "square cage",
    "tyre",
]

# 随机颜色
COLORS = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)

# 读取 val.txt 所有图片
with open(val_txt_path, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

for i in range(0, 5):
    # 随机抽 1 张
    img_path = random.choice(lines)
    img_name = os.path.basename(img_path).replace(".bmp", "")
    save_path = os.path.join(save_dir, f"val_result_{img_name}.jpg")

    # 读取图片
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    img_gt = img.copy()
    img_det = img.copy()

    # ==================== 自动找标签 ====================
    label_path = img_path.replace("/images/", "/labels/").replace(".bmp", ".txt")

    # 画真值框 GT
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            labels = f.readlines()
        for lab in labels:
            cls_id, x, y, bw, bh = map(float, lab.strip().split())
            cls_id = int(cls_id)
            x1 = int((x - bw/2)*w)
            y1 = int((y - bh/2)*h)
            x2 = int((x + bw/2)*w)
            y2 = int((y + bh/2)*h)
            color = COLORS[cls_id].tolist()
            cv2.rectangle(img_gt, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_gt, class_names[cls_id], (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 模型推理
    results = model(img_path)
    df = results.pandas().xyxy[0]

    # 画检测结果
    for _, row in df.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        conf = row['confidence']
        cls = int(row['class'])
        name = row['name']
        color = COLORS[cls].tolist()
        cv2.rectangle(img_det, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_det, f"{name} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 转为RGB
    img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
    img_det = cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB)

    # 画图：左右两张对比图（单张独立）
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_gt)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_det)
    plt.title("Detection Result")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"随机单张结果已保存：\n{save_path}")
