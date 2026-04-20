import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from utils import compute_metrics

# ------------------------------
# 参数
# ------------------------------
pred_root_list = [30, 50, 80]
pred_root_list = [50,70,95]
pred_filename = "volume_mask_pred_4.tiff"

celltype='hela2'
celltype='jurkat'
celltype='macrophage'

organelletype='golgi'
organelletype='er'
organelletype='endo'
organelletype='lyso'
organelletype='mito'


msk_threshold = 0.9
ratio_threshold = 0.5
# 👉 你有多少个重复实验
i_list = range(10)

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
fpr_box = []
fpr_star_box = []
labels = []

# ------------------------------
# 主循环
# ------------------------------
for folder in pred_root_list:

    iou_per_folder = []
    iou_star_per_folder = []
    fpr_per_folder = []
    fpr_star_per_folder = []

    for i in i_list:
        print(i)
        # ---------- MUnet ----------
        pred_path = f'/mnt/d/vem_data/label_{celltype}_{organelletype}_{folder}_{i}/{pred_filename}'

        if not os.path.exists(pred_path):
            print(f"skip: {pred_path}")
            continue

        pred_ori = tiff.imread(pred_path)
        pred = pred_ori[:,:,:,1]
        thresh_value = np.percentile(pred, 100 * msk_threshold)
        vol010 = (pred >= max(thresh_value, ratio_threshold*255)).astype(np.uint8)

        iou, fpr = compute_metrics(gt, vol010)
        iou_per_folder.append(iou)
        fpr_per_folder.append(fpr)
        # ---------- Stardist ----------
        pred_star_path = f'/mnt/d/vem_data/benchmark/prediction_{celltype}_{organelletype}_{folder}_{i}.tif'

        if not os.path.exists(pred_star_path):
            print(f"skip: {pred_star_path}")
            continue

        pred_star = tiff.imread(pred_star_path)
        pred_star = (pred_star > 0).astype(np.uint8)

        # ---------- metrics ----------

        iou_star, fpr_star = compute_metrics(gt, pred_star)

        iou_star_per_folder.append(iou_star)
        fpr_star_per_folder.append(fpr_star)

    if len(iou_per_folder) > 0:
        iou_box.append(iou_per_folder)
        iou_star_box.append(iou_star_per_folder)
        fpr_box.append(fpr_per_folder)
        fpr_star_box.append(fpr_star_per_folder)
        labels.append(folder)
#%%
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(
    labels,
    iou_box=None,
    iou_star_box=None,
    fpr_box=None,
    fpr_star_box=None,
    organelletype="",
    celltype="",
    mode="both"  # 👉 "iou" / "fpr" / "both"
):
    """
    mode:
        "iou"  -> 只画 IoU
        "fpr"  -> 只画 FPR
        "both" -> 双子图
    """

    # ======================
    # 全局风格
    # ======================
    plt.rcParams.update({
        "font.size": 16,
        "font.family": "DejaVu Sans",
    })

    positions = np.arange(len(labels))

    # ======================
    # 创建画布
    # ======================
    if mode == "both":
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
        axes = axes
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        axes = [ax]

    # ======================
    # 画 IoU
    # ======================
    if mode in ["iou", "both"]:
        ax = axes[0]

        bp1 = ax.boxplot(
            iou_box,
            positions=positions - 0.2,
            widths=0.3,
            patch_artist=True,
            showfliers=False
        )

        bp2 = ax.boxplot(
            iou_star_box,
            positions=positions + 0.2,
            widths=0.3,
            patch_artist=True,
            showfliers=False
        )

        for patch in bp1['boxes']:
            patch.set_facecolor('#1f77b4')
            patch.set_edgecolor('black')

        for patch in bp2['boxes']:
            patch.set_facecolor('#ff7f0e')
            patch.set_edgecolor('black')

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlabel("mask ratio", fontweight='bold')
        ax.set_ylabel("IoU", fontweight='bold')

    # ======================
    # 画 FPR
    # ======================
    if mode == "fpr":
        ax = axes[0]

    if mode in ["fpr", "both"]:
        ax = axes[-1]

        bp1 = ax.boxplot(
            fpr_box,
            positions=positions - 0.2,
            widths=0.3,
            patch_artist=True,
            showfliers=False
        )

        bp2 = ax.boxplot(
            fpr_star_box,
            positions=positions + 0.2,
            widths=0.3,
            patch_artist=True,
            showfliers=False
        )

        for patch in bp1['boxes']:
            patch.set_facecolor('#1f77b4')
            patch.set_edgecolor('black')

        for patch in bp2['boxes']:
            patch.set_facecolor('#ff7f0e')
            patch.set_edgecolor('black')

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlabel("mask ratio", fontweight='bold')
        ax.set_ylabel("FPR", fontweight='bold')

    # ======================
    # 去掉上/右边框
    # ======================
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ======================
    # legend
    # ======================
    legend_elements = [
        Patch(facecolor='#1f77b4', edgecolor='black', label='MUnet'),
        Patch(facecolor='#ff7f0e', edgecolor='black', label='Stardist')
    ]

    fig.legend(
        handles=legend_elements,
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98),
        frameon=False
    )

    # ======================
    # 标题
    # ======================
    fig.suptitle(
        f"Prediction of {organelletype} in {celltype}",
        fontsize=20,
        fontweight='bold',
    )

    # ======================
    # 布局
    # ======================
    plt.subplots_adjust(
        left=0.1,
        right=0.9,
        top=0.8,
        bottom=0.2,
        wspace=0.4
    )

    plt.show()


# plot_metrics(
#     labels,
#     iou_box, iou_star_box,
#     organelletype=organelletype,
#     celltype=celltype,
#     mode="iou"
# )
# plot_metrics(
#     labels,
#     fpr_box=fpr_box,
#     fpr_star_box=fpr_star_box,
#     organelletype=organelletype,
#     celltype=celltype,
#     mode="fpr"
# )
plot_metrics(
    labels,
    iou_box, iou_star_box,
    fpr_box, fpr_star_box,
    organelletype=organelletype,
    celltype=celltype,
    mode="both"
)