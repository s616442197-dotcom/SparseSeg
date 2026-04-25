from segment_cell import main
import argparse
for inter_idx in range(5):
    print(f"\n=== Running iteration {inter_idx} ===")

    main(
        interation_idx=inter_idx,
        z_threshold=1,
        patch_scale=140,
        raw_name='your_raw.tif',
        mask_name='your_positive_mask.tif',
        folder_name="folder_to_store",
        area_coef=1.0,
        edge_coef=1.0,
        iou_thresh=0.6,
        threshold=0.01,
        negative_threshold=3,
        low_weight_coeff=50,
        sparsity_weight=1.0,
    )