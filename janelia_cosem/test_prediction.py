"""Direct PyCharm example for the complete optimized iterative workflow.

Edit only the four values in ``USER CONFIGURATION``.  ``RAW_NAME`` and
``MASK_NAME`` are basenames without ``.tif``.  The script is intentionally
ordinary Python: running this file directly uses one available GPU, while
``torchrun`` remains optional for multi-GPU execution.
"""

from pathlib import Path

from segment_cell import main


# --------------------------- USER CONFIGURATION ---------------------------
SCRIPT_ROOT = Path(__file__).resolve().parent
BASE_FOLDER = SCRIPT_ROOT / "inputdata"
RAW_NAME = "your_raw"  # requires inputdata/your_raw.tif
MASK_NAME = "your_positive_mask"  # requires inputdata/your_positive_mask.tif
OUTPUT_FOLDER = SCRIPT_ROOT / "outputs" / "your_experiment"
NUM_ITERATIONS = 2
# -------------------------------------------------------------------------


def validate_inputs() -> None:
    required = (
        BASE_FOLDER / f"{RAW_NAME}.tif",
        BASE_FOLDER / f"{MASK_NAME}.tif",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Replace the USER CONFIGURATION basenames or add the required "
            f"TIFF files:\n{formatted}"
        )
    if NUM_ITERATIONS < 1:
        raise ValueError("NUM_ITERATIONS must be at least 1")


def run() -> None:
    validate_inputs()
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    for iteration_index in range(NUM_ITERATIONS):
        print(f"\n=== Running iteration {iteration_index} ===", flush=True)
        main(
            interation_idx=iteration_index,
            filer_method=2,
            z_threshold=10,
            patch_scale=80,
            inference_stride=40,
            raw_name=RAW_NAME,
            mask_name=MASK_NAME,
            folder_name=str(OUTPUT_FOLDER),
            base_folder=str(BASE_FOLDER),
            area_coef=1.0,
            edge_coef=1.0,
            iou_thresh=0.6,
            threshold=0.01,
            negative_threshold=3.0,
            low_weight_coeff=50.0,
            sparsity_weight=1.0,
            repeated_epoch=60,
            batch_size=12,
            num_samples=1000,
            thickness=2,
            kernel_sizes=(3, 5, 7),
            Loss_list=[10.0, 0.1, 0.1, 0.05],
            refinement_profile="safe_abstain",
            area_probability_quantile=99.0,
            pseudo_label_core_quantile=99.9,
            evaluation_probability_quantile=98.95,
        )


if __name__ == "__main__":
    run()
