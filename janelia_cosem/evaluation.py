import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from utils import compute_metrics
# ------------------------------
# 1. 读取 GT
# ------------------------------



# ------------------------------
# 2. 预测 mask 列表
# ------------------------------
pred_root_list = [30,50,70,80,95]
pred_root_list = [30,50,80]
pred_filename = "volume_mask_pred_5.tiff"

# ------------------------------
# 3. 计算 IoU 和 FPR 的函数
# ------------------------------



# ------------------------------
# 4. 遍历不同 label_x 目录
# ------------------------------
iou_list = []
fpr_list = []
iou_star_list = []
fpr_star_list = []
iou_ilastik_list = []
fpr_ilastik_list = []
labels = []
threshold=255*0.5
celltype='hela2'
celltype='jurkat'
# celltype='macrophage'
organelletype='golgi'
organelletype='er'
organelletype='endo'
organelletype='lyso'

organelletype='mito'

gt_path = f"/mnt/d/vem_data/download/{celltype}_{organelletype}_s3.tif"
gt = tiff.imread(gt_path)

# 保证是 0/1 mask
gt = (gt > 0).astype(np.uint8)

print("GT shape:", gt.shape)

for folder in pred_root_list:
    # pred_path = os.path.join(f'label_{celltype}_{organelletype}_{folder}', pred_filename)
    pred_path = os.path.join(f'/mnt/d/vem_data/label_{celltype}_{organelletype}_{folder}', pred_filename)

    pred_ori = tiff.imread(pred_path)
    pred = pred_ori[:,:,:,1]

    pred_star = tiff.imread(f'/mnt/d/vem_data/benchmark/prediction_{celltype}_{organelletype}_{folder}.tif')
    pred_star = (pred_star > 0).astype(np.uint8)
    # # squeeze [D,H,W,1] → [D,H,W]
    # if pred.ndim == 4 and pred.shape[-1] == 1:
    #     pred = pred[..., 0]

    pred = (pred > threshold).astype(np.uint8)

    print(f"{folder} shape:", pred.shape)


    iou, fpr = compute_metrics(gt, pred)
    iou_star, fpr_star = compute_metrics(gt, pred_star)

    labels.append(folder)
    iou_list.append(iou)
    fpr_list.append(fpr)
    iou_star_list.append(iou_star)
    fpr_star_list.append(fpr_star)

    # # ilastik_path = f"download/pred/exported_data_jurkat_{folder}.tif"
    # ilastik_path = f"download/pred/exported_data_{folder}.tif"
    #
    # pred_ilastik = tiff.imread(ilastik_path)
    # pred_ilastik = pred_ilastik[:, 0].squeeze()
    # pred_ilastik = (pred_ilastik > 0.3 * 65535).astype(np.uint8)
    #
    # iou_ilastik, fpr_ilastik = compute_metrics(gt, pred_ilastik)
    # iou_ilastik_list.append(iou_ilastik)
    # fpr_ilastik_list.append(fpr_ilastik)
#%%
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})
plt.figure(figsize=(7,5))

# IoU
# plt.subplot(1,2,1)
plt.plot(labels, iou_list, marker="o", label="MUnet")
plt.plot(labels, iou_star_list, marker="o", label="Stardist")
# plt.plot(labels, iou_ilastik_list, marker="o", label="Ilastik")

plt.title(f"IoU of {organelletype}")
plt.xlabel("mask ratio")
plt.ylabel("IoU")
plt.grid(True)
plt.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    frameon=False
)# FPR
# plt.subplot(1,2,2)
# plt.plot(labels, fpr_list, marker="o", label="MUnet")
# plt.plot(labels, fpr_star_list, marker="s", label="Stardist")
# plt.title("False Positive Rate Comparison")
# plt.xlabel("Threshold")
# plt.ylabel("FPR")
# plt.grid(True)
# plt.legend()

plt.tight_layout()
# plt.savefig("download/pred/compare_pred_vs_predstar.png", dpi=200)
plt.show()

# #%%
# save_dir = "download/pred"
# os.makedirs(save_dir, exist_ok=True)
#
# # pred_ori shape = [D, H, W, 2]？
# # 你需要的是第 0 个通道
# pred_ori_0 = pred_ori[:, :, :, 0]
#
# # 归一化检查：确保 pred 和 gt 是 0/1 mask
# pred_vis = (pred * 255).astype(np.uint8)
# pred_star_vis = (pred_star * 255).astype(np.uint8)
# gt_vis = (gt * 255).astype(np.uint8)
#
# # 转成 uint8
# pred_ori_vis = pred_ori_0.astype(np.uint8)
#
# # -------------------------
# # 1) pred 版本
# # -------------------------
# stack_pred = np.stack([pred_ori_vis, pred_vis, gt_vis], axis=-1)  # [D,H,W,3]
# tiff.imwrite(
#     os.path.join(save_dir, "compare_pred.tif"),
#     stack_pred,
#     photometric="rgb"
# )
#
# # -------------------------
# # 2) pred_star 版本
# # -------------------------
# stack_pred_star = np.stack([pred_ori_vis, pred_star_vis, gt_vis], axis=-1)
# tiff.imwrite(
#     os.path.join(save_dir, "compare_pred_star.tif"),
#     stack_pred_star,
#     photometric="rgb"
# )
#
# stack_pred_star = np.stack([pred_ori_vis, 255*pred_ilastik, gt_vis], axis=-1)
# tiff.imwrite(
#     os.path.join(save_dir, "compare_pred_ilastik.tif"),
#     stack_pred_star,
#     photometric="rgb"
# )
#
# print("Done! 已保存: compare_pred.tif 和 compare_pred_star.tif")