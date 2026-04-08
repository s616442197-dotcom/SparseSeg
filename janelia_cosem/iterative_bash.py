# iterative_seg.py
from segment_cell import main
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Iterative segmentation runner"
    )

    # ===== core names =====
    parser.add_argument("--raw_name", type=str, required=True,
                        help="raw volume name, e.g. 11419_raw")
    parser.add_argument("--mask_name", type=str, required=True,
                        help="mask name, e.g. 11419_mito_mask")
    parser.add_argument("--folder_name", type=str, required=True,
                        help="folder name, e.g. 11419_mito_mask")

    # ===== numeric parameters =====
    parser.add_argument("--z_threshold", type=int, default=10)
    parser.add_argument("--iou_thresh", type=float, default=0.6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patch_scale", type=int, default=80)
    parser.add_argument("--area_coef", type=float, default=1.0)
    parser.add_argument("--edge_coef", type=float, default=0.5)
    parser.add_argument("--negative_threshold", type=int, default=3)
    parser.add_argument("--low_weight_coeff", type=float, default=200)
    parser.add_argument("--sparsity_weight", type=float, default=0.5)

    # ===== iteration control =====
    parser.add_argument("--num_iterations", type=int, default=6,
                        help="number of iterative runs")

    return parser.parse_args()


def main_iterative():
    args = parse_args()

    for inter_idx in range(args.num_iterations):
        print(f"\n=== Running iteration {inter_idx} ===")

        main(
            interation_idx=inter_idx,
            z_threshold=args.z_threshold,
            patch_scale=args.patch_scale,
            raw_name=args.raw_name,
            mask_name=args.mask_name,
            folder_name=args.folder_name,
            area_coef=args.area_coef,
            edge_coef=args.edge_coef,
            iou_thresh=args.iou_thresh,
            threshold=args.threshold,
            negative_threshold=args.negative_threshold,
            low_weight_coeff=args.low_weight_coeff,
            sparsity_weight=args.sparsity_weight,
        )


if __name__ == "__main__":
    main_iterative()
