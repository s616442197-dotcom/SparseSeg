import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from utils import compute_metrics

# ------------------------------
# 参数
# ------------------------------
pred_root_list = [30, 50, 80]
pred_root_list = [50]
pred_filename = "volume_mask_pred_4.tiff"

celltype='hela2'
# celltype='jurkat'
# celltype='macrophage'

organelletype='golgi'
organelletype='er'
organelletype='endo'
organelletype='lyso'
organelletype='mito'


threshold = 255 * 0.5

# 👉 你有多少个重复实验
i_list = range(6)

# ------------------------------
# 读取 GT
# ------------------------------
gt_path = f"/mnt/d/vem_data/download/{celltype}_{organelletype}_s3.tif"
gt = tiff.imread(gt_path)
gt = (gt > 0).astype(np.uint8)

print("GT shape:", gt.shape)

# ------------------------------
# 存储 boxplot 数据
# ------------------------------
iou_box = []
iou_star_box = []
labels = []

# ------------------------------
# 主循环
# ------------------------------
for folder in pred_root_list:

    iou_per_folder = []
    iou_star_per_folder = []

    for i in i_list:

        # ---------- MUnet ----------
        pred_path = f'/mnt/d/vem_data/label_{celltype}_{organelletype}_{folder}_{i}/{pred_filename}'

        if not os.path.exists(pred_path):
            print(f"skip: {pred_path}")
            continue

        pred_ori = tiff.imread(pred_path)
        pred = pred_ori[:,:,:,1]
        pred = (pred > threshold).astype(np.uint8)
        iou, _ = compute_metrics(gt, pred)
        iou_per_folder.append(iou)

        # ---------- Stardist ----------
        pred_star_path = f'/mnt/d/vem_data/benchmark/prediction_{celltype}_{organelletype}_{folder}_{i}.tif'

        if not os.path.exists(pred_star_path):
            print(f"skip: {pred_star_path}")
            continue

        pred_star = tiff.imread(pred_star_path)
        pred_star = (pred_star > 0).astype(np.uint8)

        # ---------- metrics ----------

        iou_star, _ = compute_metrics(gt, pred_star)

        iou_star_per_folder.append(iou_star)

    if len(iou_per_folder) > 0:
        iou_box.append(iou_per_folder)
        iou_star_box.append(iou_star_per_folder)
        labels.append(folder)
#%%
# ------------------------------
# 画 boxplot
# ------------------------------
plt.rcParams.update({
    "font.size": 16,
})

plt.figure(figsize=(8,5))

positions = np.arange(len(labels))

# MUnet
plt.boxplot(
    iou_box,
    positions=positions - 0.2,
    widths=0.3,
    patch_artist=True
)

# Stardist
plt.boxplot(
    iou_star_box,
    positions=positions + 0.2,
    widths=0.3,
    patch_artist=True
)

plt.xticks(positions, labels)
plt.xlabel("mask ratio")
plt.ylabel("IoU")
plt.title(f"IoU of {organelletype}")

plt.legend(["MUnet", "Stardist"])
plt.grid(True)

plt.tight_layout()
plt.show()