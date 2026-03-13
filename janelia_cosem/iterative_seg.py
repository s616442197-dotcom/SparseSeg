from segment_cell import main
import argparse
from types import SimpleNamespace

# 要调用的目标脚本
z_threshold=10
iou_thresh=0.6
threshold=0.5
patch_scale=140 # 4倍数
area_coef=1.0
edge_coef=0.5
negative_threshold=3
low_weight_coeff=200
sparsity_weight=0.1

# value can be modified according to user's need
raw_name='11442_raw'
mask_name='11442_mito_mask'
patch_scale=80 # 4倍数
sparsity_weight=0.5


# 参数组合列表，每个元素都是一个字典
params_list = [
    {"interation_idx": 0},
    {"interation_idx": 1},
    {"interation_idx": 2},
    {"interation_idx": 3},
    {"interation_idx": 4},
    {"interation_idx": 5},


]
label_stack=[]

for p in params_list:
    print(f"\n=== Running iteration {p['interation_idx']} ===")

    main(
        interation_idx=p["interation_idx"],
        # filer_method=filer_method,
        z_threshold=z_threshold,
        patch_scale=patch_scale,
        raw_name=raw_name,
        mask_name=mask_name,
        # mask_name=params["mask_name"],
        area_coef=area_coef,
        edge_coef=edge_coef,
        iou_thresh=iou_thresh,
        threshold=threshold,
        negative_threshold=negative_threshold,
        low_weight_coeff=low_weight_coeff,
        sparsity_weight=sparsity_weight,
    )
