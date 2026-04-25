#!/bin/bash
set -e
set -o pipefail

DEEPICT_DIR="."
DATA_DIR="/mnt/c/baidunetdiskdownload/empanda/data"

RAW_TIF="${DATA_DIR}/raw.tif"
RAW_MRC="${DATA_DIR}/raw_deepict.mrc"

OUT_ROOT="${DATA_DIR}/deepict_finetune"
TIF_OUT_DIR="${OUT_ROOT}/tif_predictions"

mkdir -p "${OUT_ROOT}"
mkdir -p "${TIF_OUT_DIR}"

ROI_LIST=(1 5 10)
PRED_ROOT_LIST=(100 101 102 103 104)

# =========================
# DeePiCt prediction params
# =========================
PRED_PATCH_SIZE_Y=288
PRED_PATCH_SIZE_X=288
PRED_CROP=48
PRED_THRESHOLD=0.5

# =========================
# 1. raw tif -> mrc
# =========================
python - <<PY
import tifffile as tiff
import mrcfile
import numpy as np
import os

raw_tif = "${RAW_TIF}"
raw_mrc = "${RAW_MRC}"

if not os.path.exists(raw_mrc):
    raw = tiff.imread(raw_tif).astype(np.float32)
    with mrcfile.new(raw_mrc, overwrite=True) as m:
        m.set_data(raw)
    print("saved raw mrc:", raw_mrc, raw.shape)
else:
    with mrcfile.open(raw_mrc, permissive=True) as m:
        raw = m.data
        print("raw mrc exists:", raw_mrc, raw.shape)
PY


# =========================
# 2. batch finetune + prediction
# =========================
for ROI in "${ROI_LIST[@]}"; do
  for ROOT in "${PRED_ROOT_LIST[@]}"; do

    NAME="hela2_mito_${ROOT}_${ROI}"

    LABEL_TIF="${DATA_DIR}/label_hela2_mito_${ROOT}_${ROI}.tif"
    LABEL_MRC="${OUT_ROOT}/${NAME}_label.mrc"

    WORK_DIR="${OUT_ROOT}/${NAME}"
    CONFIG_PATH="${WORK_DIR}/config.yaml"
    DATA_CSV="${WORK_DIR}/data.csv"

    mkdir -p "${WORK_DIR}"

    if [ ! -f "${LABEL_TIF}" ]; then
        echo "[Skip] missing label: ${LABEL_TIF}"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "Start DeePiCt finetune: ${NAME}"
    echo "============================================================"

    # =========================
    # label tif -> mrc
    # =========================
    python - <<PY
import tifffile as tiff
import mrcfile
import numpy as np

label_tif = "${LABEL_TIF}"
label_mrc = "${LABEL_MRC}"

lab = tiff.imread(label_tif)
lab = (lab > 0).astype(np.float32)

with mrcfile.new(label_mrc, overwrite=True) as m:
    m.set_data(lab)

print("saved label mrc:", label_mrc, lab.shape, "nonzero:", int(lab.sum()))
PY

    # =========================
    # create csv + config
    # =========================
    python - <<PY
import csv
import yaml
import math
import os
import mrcfile

orig_config = "${DEEPICT_DIR}/2d_cnn/config.yaml"
config_path = "${CONFIG_PATH}"
data_csv = "${DATA_CSV}"

name = "${NAME}"
raw_mrc = "${RAW_MRC}"
label_mrc = "${LABEL_MRC}"
work_dir = "${WORK_DIR}"

pred_patch_size_y = int("${PRED_PATCH_SIZE_Y}")
pred_patch_size_x = int("${PRED_PATCH_SIZE_X}")
pred_crop = int("${PRED_CROP}")

# -------------------------
# read raw shape
# -------------------------
with mrcfile.open(raw_mrc, permissive=True) as m:
    raw_shape = m.data.shape

# mrcfile normally gives (Z, Y, X)
z, y, x = raw_shape
print("raw shape:", raw_shape)

# -------------------------
# estimate prediction patch_dim
# -------------------------
effective_y = pred_patch_size_y - 2 * pred_crop
effective_x = pred_patch_size_x - 2 * pred_crop

if effective_y <= 0 or effective_x <= 0:
    raise ValueError(
        f"Invalid effective patch size: "
        f"patch_size=({pred_patch_size_y},{pred_patch_size_x}), crop={pred_crop}"
    )

# 对你的 raw shape (200,1500,796):
# y: ceil(1500 / 192) + 1 = 9
# x: ceil(796 / 192) = 5
pred_patch_dim_y = math.ceil(y / effective_y) + 1
pred_patch_dim_x = math.ceil(x / effective_x)

print("prediction effective patch:", (effective_y, effective_x))
print("prediction patch_dim:", [pred_patch_dim_y, pred_patch_dim_x])

# -------------------------
# create data.csv
# -------------------------
header = ["tomo_name", "id", "data", "filtered_data", "labels", "flip_y"]
row = [name, name, raw_mrc, raw_mrc, label_mrc, False]

with open(data_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerow(row)

# -------------------------
# create config.yaml
# -------------------------
with open(orig_config, "r") as f:
    d = yaml.safe_load(f)

# 顶层路径
d["training_data"] = data_csv
d["prediction_data"] = data_csv
d["output_dir"] = work_dir

# DeePiCt 原 config 里面还有 data.output_dir，
# 之前预测输出到 out/，大概率就是这里没改导致的。
if "data" not in d:
    d["data"] = {}

d["data"]["training_data"] = data_csv
d["data"]["prediction_data"] = data_csv
d["data"]["output_dir"] = work_dir

# 保留 train_workdir 的默认结构，但让它进入当前样本目录，避免不同样本中间文件互相污染
d["data"]["train_workdir"] = os.path.join(work_dir, "work")

# -------------------------
# preprocessing
# -------------------------
d["preprocessing"]["filtering"]["active"] = False
d["preprocessing"]["slicing"]["z_cutoff"] = None

# 训练切 patch 也可以适当增大，避免只取到很小区域。
# 如果你担心训练变慢，可以改回 [5, 5]。
d["preprocessing"]["slicing"]["patch_size"] = [
    pred_patch_size_y,
    pred_patch_size_x,
]
d["preprocessing"]["slicing"]["patch_dim"] = [
    pred_patch_dim_y,
    pred_patch_dim_x,
]

# 保留你的二值标签：0=background, 1=target
d["preprocessing"]["remapping"]["active"] = True
d["preprocessing"]["remapping"]["mapping"] = {
    ".": 0,
    0: 0,
    1: 1,
}

# -------------------------
# training
# -------------------------
d["training"]["evaluation"]["active"] = False
d["training"]["production"]["active"] = True

# 每个样本单独保存 model，避免复用 ./model.h5
d["training"]["production"]["model_output"] = os.path.join(work_dir, "model.h5")

# -------------------------
# prediction
# -------------------------
d["prediction"]["active"] = True
d["prediction"]["normalize"] = True
d["prediction"]["compensate_crop"] = True
d["prediction"]["crop"] = pred_crop
d["prediction"]["patch_size"] = [
    pred_patch_size_y,
    pred_patch_size_x,
]
d["prediction"]["patch_dim"] = [
    pred_patch_dim_y,
    pred_patch_dim_x,
]

# 你的 raw z = 200，这里设置为整卷 z 数
d["prediction"]["z_cutoff"] = int(z)

# 如果 DeePiCt 支持 prediction.model，就明确指向当前样本模型
d["prediction"]["model"] = os.path.join(work_dir, "model.h5")

# 后处理不启用，自己在最后转 tif 时 threshold
d["postprocessing"]["active"] = False
d["postprocessing"]["threshold"] = float("${PRED_THRESHOLD}")

with open(config_path, "w") as f:
    yaml.safe_dump(d, f, sort_keys=False)

print("saved csv:", data_csv)
print("saved config:", config_path)
print("model_output:", d["training"]["production"]["model_output"])
print("prediction model:", d["prediction"]["model"])
print("output_dir:", d["output_dir"])
print("data.output_dir:", d["data"]["output_dir"])
PY

    # =========================
    # DeePiCt snakefile 默认读取 meta/metadata.csv
    # =========================
    mkdir -p meta
    cp "${DATA_CSV}" meta/metadata.csv

    echo "Copied metadata to: meta/metadata.csv"
    cat meta/metadata.csv

    # =========================
    # 清理该样本旧中间文件，避免复用错误结果
    # =========================
    rm -rf "${WORK_DIR}/work"
    rm -f "${WORK_DIR}/model.h5"
    rm -f "${WORK_DIR}"/*.mrc
    rm -f "${WORK_DIR}"/*.h5

    # 兼容旧 DeePiCt 仍然写入全局 work/out 的情况
    rm -rf "work/${NAME}"*
    rm -f "work/${NAME}"*.mrc
    rm -f "work/${NAME}"*.h5
    rm -f "out/${NAME}"*.mrc

    # 如果旧 Snakefile 仍然强制使用 ./model.h5，也清掉，避免误用上一个样本模型
    rm -f "./model.h5"

    # =========================
    # run DeePiCt local pipeline
    # =========================
    bash "${DEEPICT_DIR}/2d_cnn/deploy_local.sh" "${CONFIG_PATH}"

    # =========================
    # convert prediction mrc -> tif
    # =========================
    python - <<PY
import os
import glob
import numpy as np
import mrcfile
import tifffile as tiff

name = "${NAME}"
tif_out_dir = "${TIF_OUT_DIR}"
work_dir = "${WORK_DIR}"
threshold = float("${PRED_THRESHOLD}")

os.makedirs(tif_out_dir, exist_ok=True)

# 优先找当前样本目录下的预测结果；
# 同时兼容 DeePiCt 旧逻辑写入 out/ 或 work/ 的情况。
patterns = [
    os.path.join(work_dir, f"*{name}*post*pred*.mrc"),
    os.path.join(work_dir, f"*{name}*pred*.mrc"),
    os.path.join(work_dir, "**", f"*{name}*post*pred*.mrc"),
    os.path.join(work_dir, "**", f"*{name}*pred*.mrc"),

    os.path.join("out", f"*{name}*post*pred*.mrc"),
    os.path.join("out", f"*{name}*pred*.mrc"),

    os.path.join("work", f"*{name}*post*pred*.mrc"),
    os.path.join("work", f"*{name}*pred*.mrc"),
    os.path.join("work", "**", f"*{name}*post*pred*.mrc"),
    os.path.join("work", "**", f"*{name}*pred*.mrc"),
]

matches = []
for p in patterns:
    matches.extend(glob.glob(p, recursive=True))

matches = sorted(set(matches))

if len(matches) == 0:
    print(f"[Warning] No prediction mrc found for {name}")
    print("Searched patterns:")
    for p in patterns:
        print("  ", p)
else:
    print("Matched prediction files:")
    for m in matches:
        print("  ", m)

    # 取最后一个，通常是最新或路径排序靠后的结果
    pred_mrc = matches[-1]
    print("Use prediction mrc:", pred_mrc)

    with mrcfile.open(pred_mrc, permissive=True) as m:
        pred = m.data.copy().astype(np.float32)

    nan_count = int(np.isnan(pred).sum())
    inf_count = int(np.isinf(pred).sum())

    print("prediction shape:", pred.shape)
    print("prediction nan:", nan_count)
    print("prediction inf:", inf_count)
    print("prediction min/max before clean:", np.nanmin(pred), np.nanmax(pred))

    # NaN/Inf 清理，避免 tif 保存和后续指标出问题
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

    print("prediction min/max after clean:", pred.min(), pred.max())

    # 用 0.5 阈值，而不是 >0
    pred_bin = (pred > threshold).astype(np.uint8)

    tif_path = os.path.join(
        tif_out_dir,
        f"deepict_{name}_whole_volume.tif"
    )

    tiff.imwrite(tif_path, pred_bin)

    print("saved tif:", tif_path)
    print("tif shape:", pred_bin.shape)
    print("threshold:", threshold)
    print("nonzero:", int(np.count_nonzero(pred_bin)))
PY

  done
done

echo ""
echo "All DeePiCt finetune jobs finished."
echo "TIF predictions saved in:"
echo "${TIF_OUT_DIR}"