import numpy as np
import tifffile as tiff
from utils import (
    process_volume,
    local_contrast_normalize,
    filter_connected_regions_shape,
    intersect_regions
)
from skimage.transform import downscale_local_mean
import os

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from save_function import save_volume_with_masks_as_rgb_tiff
from MUNET_model import MultiKernelUNet
from prediction_func import infer_volume_edges_whole
from segment_cell import setup_model, dilate_z_binary, ddp_setup, is_main_process
import matplotlib.pyplot as plt
from collections import defaultdict
from utils import compute_metrics

# 结果缓存
results = defaultdict(lambda: defaultdict(dict))
# 结构：results[celltype][ratio] = {"iou": xx, "fpr": xx}


# =========================
# DDP setup（只做一次）
# =========================
rank, world_size, local_rank, device = ddp_setup()
main_proc = is_main_process(rank)

# =========================
# 全局参数
# =========================
thickness = 2
base_folder = "inputdata"
threshold = 0.8
z_threshold = 10

celltypes = ['hela2', 'jurkat', 'macrophage']
organ_type = 'mito'
ratios = ['30', '50', '70', '80', '95']

main_model = 'macrophage'   # 用 hela2 训练好的模型


# =========================
# 主循环
# =========================

gt_path = f"download/{main_model}_{organ_type}_s3.tif"
gt = tiff.imread(gt_path)
gt = (gt > 0).astype(np.uint8)
raw_name = f'{main_model}_em_s3'

for celltype in celltypes:

    for ratio in ratios:

        if main_proc:
            print(f"\n=== Processing {raw_name} ===", flush=True)

        # ---------- load raw volume ----------
        vol0 = tiff.imread(os.path.join(base_folder, f"{raw_name}.tif"))
        volume = local_contrast_normalize(vol0)



        # ---------- load model ----------
        checkpoint_folder = f"label_{celltype}_{organ_type}_{ratio}"

        base_model = setup_model(
            MultiKernelUNet,
            model_args={
                "in_channels": 2 * thickness + 31,
                "out_channels": 2
            },
            checkpoint_folder=checkpoint_folder,
            model_name="model_2.pt",
            device=device,
            rank=rank,
        )

        # DDP wrap
        if world_size > 1:
            model = DDP(
                base_model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
        else:
            model = base_model

        net = model.module if hasattr(model, "module") else model
        net.eval()

        # ---------- inference ----------
        with torch.no_grad():
            edge_vol, edge_line = infer_volume_edges_whole(
                volume,
                net,
                thickness=thickness
            )

        # ---------- save RGB result ----------
        save_path = f"generalization/{main_model}_{celltype}_{organ_type}_{ratio}.tiff"
        save_volume_with_masks_as_rgb_tiff(
            volume,
            edge_vol,
            edge_line,
            save_path
        )

        # ---------- evaluation ----------


        iou, fpr = compute_metrics(gt, (edge_vol > 0.3).astype(np.uint8))
        results[celltype][ratio]["iou"] = iou
        results[celltype][ratio]["fpr"] = fpr
        if main_proc:
            print(
                f"[{celltype} | {ratio}] IoU = {iou:.4f}, FPR = {fpr:.4f}",
                flush=True
            )
#%%
if main_proc:
    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    plt.figure(figsize=(8, 5))

    # x 轴：ratio（转成数值更好看）
    x = [int(r) for r in ratios]

    # 每条线：一个 celltype
    for celltype in celltypes:
        y = [results[celltype][ratio]["iou"] for ratio in ratios]
        plt.plot(
            x, y,
            marker="o",
            linewidth=2,
            label=celltype
        )

    plt.xlabel("Mask ratio (%)")
    plt.ylabel("IoU")
    plt.title(f"Test on {main_model}")
    plt.ylim([0, 0.7])
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False
    )

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"generalization_iou_{main_model}_by_ratio.png", dpi=200)
    plt.show()
