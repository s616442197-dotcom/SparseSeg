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
low_weight_coeff=40
sparsity_weight=0.1
# filer_method=2

raw_name='10311_raw'
mask_name='10311_mito_mask'
low_weight_coeff=300
threshold=0.8
iou_thresh=0.64
sparsity_weight=1

raw_name='10392_raw'
mask_name='10392_hemozin_mask'
low_weight_coeff=300
threshold=0.8
iou_thresh=0.64
sparsity_weight=1

# raw_name='11416_raw'
# mask_name='11416_mitomask'
# area_coef=1.0
# edge_coef=0.5
# filer_method=2
# patch_scale=40 # 4倍数
# sparsity_weight=0.5


# mask_name='11416_vescilemask'
#
# # mask_name='11416_nucleusmask'
# low_weight_coeff=200
# patch_scale=140 # 4倍数
# z_threshold=40
# threshold=0.8
# iou_thresh=0.8
# sparsity_weight=0.05

#
# raw_name='11420_raw'
# mask_name='11420_heart_mask'
# area_coef=1.0
# edge_coef=0.5
# filer_method=2
# patch_scale=140 # 4倍数
# sparsity_weight=0.5

#
# raw_name='gut_raw'
# mask_name='gut_mask'
# area_coef=1.0
# edge_coef=0.5
# filer_method=2
# patch_scale=100 # 4倍数
# sparsity_weight=0.1
#
# raw_name='10442_raw'
# mask_name='10442_mito_mask'
# area_coef=1.0
# edge_coef=0.5
# filer_method=2
# patch_scale=140 # 4倍数
# sparsity_weight=0.5
#
# raw_name='main_control'
# mask_name='main_control_mitomask'
#
# raw_name='main_patient'
# mask_name='main_patient_mitomask'
# patch_scale=40
# sparsity_weight=5

# raw_name='hela2_em_s3'
# area_coef=1.0
# edge_coef=0.5
# filer_method=2
# patch_scale=140 # 4倍数
# celltype='hela2'
# type='mito'
# type='golgi'
# type='endo'
# type='er'
# type='lyso'
# iou_thresh=0.64
# threshold=0.8

# type='nucleus'
# z_threshold=40
# sparsity_weight=0.1

# mask_name='label_95'
# sparsity_weight=0.3

# raw_name='jurkat_em_s3'
# # area_coef=1.0
# # edge_coef=0.5
# # filer_method=2
# # patch_scale=140 # 4倍数
# celltype='jurkat'
# type='lyso'
# type='golgi'
# type='nucleus'
# z_threshold=40
# sparsity_weight=1.0
# # patch_scale=80 # 4倍数
# type='er'
# sparsity_weight=1.0
# iou_thresh=0.64
# threshold=0.8
#
# raw_name='macrophage_em_s3'
# celltype='macrophage'
# patch_scale=140 # 4倍数
# type='lyso'
# type='golgi'
# type='endo'
# type='er'
# type='nucleus'
# z_threshold=40
#
# # type='mito'
# sparsity_weight=0.1
# low_weight_coeff=20
# iou_thresh=0.64
# threshold=0.8
# 参数组合列表，每个元素都是一个字典
params_list = [
    {"interation_idx": 0},
    {"interation_idx": 1},
    {"interation_idx": 2},
    {"interation_idx": 3},
    {"interation_idx": 4},
    {"interation_idx": 5},

    # {"interation_idx": 0, "filer_method": filer_method, "mask_name": f'label_{celltype}_{type}_80'},
    # {"interation_idx": 1, "filer_method": filer_method, "mask_name": f'label_{celltype}_{type}_80'},
    # {"interation_idx": 2, "filer_method": filer_method, "mask_name": f'label_{celltype}_{type}_80'},
    # # # {"interation_idx": 3, "filer_method": filer_method, "mask_name": 'label_50'},
    # # # {"interation_idx": 4, "filer_method": filer_method, "mask_name": 'label_50'},
    # # # {"interation_idx": 5, "filer_method": filer_method, "mask_name": 'label_50'},
    # {"interation_idx": 0, "filer_method": filer_method, "mask_name": f'label_{celltype}_{type}_70'},
    # {"interation_idx": 1, "filer_method": filer_method, "mask_name": f'label_{celltype}_{type}_70'},
    # {"interation_idx": 2, "filer_method": filer_method, "mask_name": f'label_{celltype}_{type}_70'},
    # # # {"interation_idx": 3, "filer_method": filer_method, "mask_name": 'label_80'},
    # # # {"interation_idx": 4, "filer_method": filer_method, "mask_name": 'label_80'},
    # # # {"interation_idx": 5, "filer_method": filer_method, "mask_name": 'label_80'},
    # {"interation_idx": 0, "filer_method": filer_method, "mask_name": f'label_{celltype}_{type}_50'},
    # {"interation_idx": 1, "filer_method": filer_method, "mask_name": f'label_{celltype}_{type}_50'},
    # {"interation_idx": 2, "filer_method": filer_method, "mask_name": f'label_{celltype}_{type}_50'},
    # # # {"interation_idx": 3, "filer_method": filer_method, "mask_name": 'label_70'},
    # # # {"interation_idx": 4, "filer_method": filer_method, "mask_name": 'label_70'},
    # # # {"interation_idx": 5, "filer_method": filer_method, "mask_name": 'label_70'},
    # {"interation_idx": 0, "filer_method": filer_method, "mask_name": f'label_{celltype}_{type}_95'},
    # {"interation_idx": 1, "filer_method": filer_method, "mask_name": f'label_{celltype}_{type}_95'},
    # {"interation_idx": 2, "filer_method": filer_method, "mask_name": f'label_{celltype}_{type}_95'},
    # # # {"interation_idx": 3, "filer_method": filer_method, "mask_name": 'label_30'},
    # # # {"interation_idx": 4, "filer_method": filer_method, "mask_name": 'label_30'},
    # # # {"interation_idx": 5, "filer_method": filer_method, "mask_name": 'label_30'},
    # {"interation_idx": 0, "filer_method": filer_method, "mask_name": f'label_{celltype}_{type}_30'},
    # {"interation_idx": 1, "filer_method": filer_method, "mask_name": f'label_{celltype}_{type}_30'},
    # {"interation_idx": 2, "filer_method": filer_method, "mask_name": f'label_{celltype}_{type}_30'},

    # {"interation_idx": 1},
    # {"interation_idx": 4},
    # {"model_type": 0, "base_folder": "checkpoints", "input_datatype": 1, "defect_intro": 1},
    # {"model_type": 0, "base_folder": "checkpoints", "input_datatype": 1, "defect_intro": 1},
    # {"model_type": 0, "base_folder": "checkpoints", "input_datatype": 1, "defect_intro": 1},
    # {"model_type": 2, "base_folder": "checkpoints",   "input_datatype": 0, "defect_intro": 1},
    # {"model_type": 2, "base_folder": "checkpoints",   "input_datatype": 1, "defect_intro": 1},
]
label_stack=[]
# 遍历并执行
# for i, params in enumerate(params_list, start=1):
#     print(f"\n=== 第 {i}_{params['interation_idx']} 次调用 ===")
#
#     # 构建命令
#     command = [
#         "python", target_script,
#         "--interation_idx", str(params["interation_idx"]),
#         "--raw_name", str(raw_name),
#         "--mask_name", str(mask_name),
#         # "--mask_name", str(params["mask_name"]),
#         "--area_coef", str(area_coef),
#         "--edge_coef", str(edge_coef),
#         "--iou_thresh", str(iou_thresh),
#         "--filer_method", str(params["filer_method"]),
#         "--z_threshold", str(z_threshold),
#         "--threshold", str(threshold),
#         "--patch_scale", str(patch_scale),
#         "--negative_threshold", str(negative_threshold),
#         "--low_weight_coeff", str(low_weight_coeff),
#         "--sparsity_weight", str(sparsity_weight)
#     ]
#
#     # 调用并输出结果
#     result = subprocess.run(command)
#     print("返回码:", result.returncode)
#     print("标准输出:", result.stdout)
#     print("标准错误:", result.stderr)

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

#     vol00 = tiff.imread("outfig/volume_mask_pred.tiff")
#     label=vol00[:, :, :, 1]
#     label_stack.append(label)
# original_label = vol00[:, :, :, 2]/255
# raw = vol00[:, :, :, 0]/255
# P = np.stack(label_stack, axis=-1)
# P_mean = P.mean(axis=-1)/255   # shape = (D,H,W)
# # 二值化，>0.5 → 1，其他 → 0
# P_bin = (P_mean > 0.6).astype(np.uint8)
# save_volume_with_masks_as_rgb_tiff(raw, P_bin, original_label, "outfig/volume_mask_pred2.tiff")
