import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from utils import compute_metrics
from segment_cell import predict_packed,get_or_build_feature_volume
from collections import defaultdict
from utils import process_volume,local_contrast_normalize,filter_connected_regions_shape,intersect_regions
import pickle
# ------------------------------
# 参数
# ------------------------------
pred_root_list = [30, 50, 80]
pred_root_list = [50,70,95]
model_filename = "model_4.pt"
base_dir="/mnt/d/vem_data/download"
# celltype='hela2'
# celltype='jurkat'
# celltype='macrophage'
celltype_list={'hela2','jurkat','macrophage'}

# organelletype='golgi'
# organelletype='er'
# organelletype='endo'
# organelletype='lyso'
organelletype='mito'


msk_threshold = 0.9

i_list = range(10)

# ------------------------------
# 存储 boxplot 数据
# ------------------------------
iou_hela_box = []
iou_jurkat_box = []
iou_macrophage_box = []
fpr_box = []
fpr_star_box = []
labels = []

# ------------------------------
# 主循环
# ------------------------------
for celltype in celltype_list:
    # ======================
    # 1️⃣ 读取或初始化
    # ======================
    save_path = f"{celltype}.pkl"

    if os.path.exists(save_path):
        print(f"✅ Load existing {save_path}")
        with open(save_path, "rb") as f:
            loaded_dict = pickle.load(f)

        # 转成 defaultdict(dict)
        iou_dict = defaultdict(lambda: defaultdict(dict))

        for dt in loaded_dict:
            for fd in loaded_dict[dt]:
                iou_dict[dt][fd] = loaded_dict[dt][fd]

    else:
        print("🆕 Create new iou_dict")
        iou_dict = defaultdict(lambda: defaultdict(dict))

    for datatype in celltype_list:

        raw_name = f'{datatype}_em_s3'
        vol = tiff.imread(os.path.join(base_dir, f"{raw_name}.tif"))
        volume = local_contrast_normalize(vol)

        feature_path = os.path.join("/mnt/c/Users/61644/Downloads", raw_name)
        feature_volume = get_or_build_feature_volume(volume, feature_path)

        gt_path = f"/mnt/d/vem_data/download/{datatype}_{organelletype}_s3.tif"
        gt = tiff.imread(gt_path)
        gt = (gt > 0).astype(np.uint8)

        for folder in pred_root_list:

            for i in i_list:

                # 🔥 最关键：是否已经算过
                if i in iou_dict[datatype][folder]:
                    print(f"⏭️ Skip {datatype}-{folder}-{i}")
                    continue

                print(f"🚀 Running {datatype}-{folder}-{i}")

                model_path = f'/mnt/d/vem_data/label_{celltype}_{organelletype}_{folder}_{i}/{model_filename}'

                pred = predict_packed(
                    model_path,
                    feature_volume,
                )

                thresh_value = np.percentile(pred, 100 * msk_threshold)
                vol010 = (pred >= max(thresh_value, 0.5)).astype(np.uint8)

                iou, _ = compute_metrics(gt, vol010)
                print(iou)
                # ✅ 存成 dict（不会错位）
                iou_dict[datatype][folder][i] = iou

                # 💾 每次保存（防崩）
                with open(save_path, "wb") as f:
                    pickle.dump(dict(iou_dict), f)

                print(f"💾 Saved {datatype}-{folder}-{i}")
    # ======================
    # 最终保存
    # ======================
    with open(save_path, "wb") as f:
        pickle.dump(dict(iou_dict), f)
    print("✅ All done & saved")
#%%
import pickle
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

def plot_iou_grouped(iou_dict,celltype):

    plt.rcParams.update({
        "font.size": 16,
        "font.family": "DejaVu Sans",
    })

    datatypes = list(iou_dict.keys())          # 3种 celltype
    folders = list(next(iter(iou_dict.values())).keys())  # mask ratio

    positions = np.arange(len(folders))

    fig, ax = plt.subplots(figsize=(8,5))

    width = 0.25
    offsets = [-width, 0, width]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # ======================
    # 画每个 datatype
    # ======================
    for idx, datatype in enumerate(datatypes):

        data = [
            list(iou_dict[datatype][f].values())
            for f in folders
        ]

        bp = ax.boxplot(
            data,
            positions=positions + offsets[idx],
            widths=width,
            patch_artist=True,
            showfliers=False
        )

        for patch in bp['boxes']:
            patch.set_facecolor(colors[idx])
            patch.set_edgecolor('black')

    # ======================
    # 坐标轴
    # ======================
    ax.set_xticks(positions)
    ax.set_xticklabels(folders)
    ax.set_xlabel("mask ratio", fontweight='bold')
    ax.set_ylabel("IoU", fontweight='bold')

    # 去边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ======================
    # legend
    # ======================
    legend_elements = [
        Patch(facecolor=color, edgecolor='black', label=dt)
        for color, dt in zip(colors, datatypes)
    ]

    fig.legend(
        handles=legend_elements,
        loc='upper right',
        bbox_to_anchor=(1.00, 0.90),
        frameon=False
    )
    plt.tight_layout()
    plt.subplots_adjust(right=0.8,top=0.85)
    plt.title(f"pretrained on {celltype}",fontweight='bold', fontsize=20)
    plt.show()
celltype='hela2'
# celltype='jurkat'
# celltype='macrophage'
with open(f"{celltype}.pkl", "rb") as f:
    iou_dict = pickle.load(f)
plot_iou_grouped(iou_dict,celltype)