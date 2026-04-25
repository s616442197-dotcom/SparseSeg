#!/bin/bash
set -e
set -o pipefail

# ============================================================
# Conda 环境
# ============================================================
source /home/sbw/miniconda3/etc/profile.d/conda.sh
conda activate sam2_env

# ============================================================
# nnU-Net 路径
# ============================================================
BASE_DIR="/home/sbw/survo2/nnUnet"

export nnUNet_raw="${BASE_DIR}/nnUNet_raw"
export nnUNet_preprocessed="${BASE_DIR}/nnUNet_preprocessed"
export nnUNet_results="${BASE_DIR}/nnUNet_results"

PREDICT_ROOT="${BASE_DIR}/nnUNet_predict"

# ============================================================
# 训练 / 预测参数
# ============================================================
DATASET_START=1
DATASET_END=15
TRAIN_START=7
CONFIG="3d_fullres"
FOLD="all"

# ============================================================
# ResEnc M 设置
# ============================================================
PLANNER="nnUNetPlannerResEncM"
PLANS="nnUNetResEncUNetMPlans"

# ============================================================
# 自定义 trainer: diyepochs
# 使用 nnUNet_extTrainer，不修改 nnU-Net 源码
# ============================================================
TRAINER="nnUNetTrainer_diyepochs"
DIY_NUM_EPOCHS=300

CUSTOM_TRAINER_ROOT="${BASE_DIR}/custom_trainers"
CUSTOM_TRAINER_PKG="${CUSTOM_TRAINER_ROOT}/my_trainers"

mkdir -p "${CUSTOM_TRAINER_PKG}"
touch "${CUSTOM_TRAINER_PKG}/__init__.py"

cat > "${CUSTOM_TRAINER_PKG}/${TRAINER}.py" <<PY
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_diyepochs(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )

        # 自定义训练 epoch 数
        self.num_epochs = ${DIY_NUM_EPOCHS}
PY

export nnUNet_extTrainer="${CUSTOM_TRAINER_ROOT}"
export PYTHONPATH="${CUSTOM_TRAINER_ROOT}:${PYTHONPATH}"

python - <<'PY'
from my_trainers.nnUNetTrainer_diyepochs import nnUNetTrainer_diyepochs
print("Custom trainer import from external path OK:", nnUNetTrainer_diyepochs)
PY

export nnUNet_extTrainer="${CUSTOM_TRAINER_ROOT}"
export PYTHONPATH="${CUSTOM_TRAINER_ROOT}:${PYTHONPATH}"

# ============================================================
# 先检查外部 custom trainer 能不能被 Python 正常 import
# ============================================================
echo "Testing custom trainer from external path..."
python - <<'PY'
from my_trainers.nnUNetTrainer_diyepochs import nnUNetTrainer_diyepochs
print("External custom trainer import OK:", nnUNetTrainer_diyepochs)
PY

# ============================================================
# 兼容旧版 nnU-Net：同时复制 trainer 到 nnU-Net 安装目录
# ============================================================
NNUNET_TRAINER_DIR=$(python - <<'PY'
import nnunetv2
from pathlib import Path
print(Path(nnunetv2.__file__).parent / "training/nnUNetTrainer")
PY
)

echo "Copy custom trainer to nnU-Net trainer dir:"
echo "${NNUNET_TRAINER_DIR}"

cp "${CUSTOM_TRAINER_PKG}/nnUNetTrainer_diyepochs.py" \
   "${NNUNET_TRAINER_DIR}/nnUNetTrainer_diyepochs.py"

# ============================================================
# 再检查从 nnU-Net 内部路径能不能 import
# ============================================================
echo "Testing custom trainer from nnU-Net package..."
python - <<'PY'
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_diyepochs import nnUNetTrainer_diyepochs
print("nnU-Net package custom trainer import OK:", nnUNetTrainer_diyepochs)
PY

# ============================================================
# 检查当前 nnU-Net 是否包含 nnUNet_extTrainer 支持
# 这个只是诊断，不影响后续，因为已经复制进 nnU-Net 安装目录了
# ============================================================
python - <<'PY'
import nnunetv2
from pathlib import Path

root = Path(nnunetv2.__file__).parent
found = False

for p in root.rglob("*.py"):
    try:
        txt = p.read_text(errors="ignore")
    except Exception:
        continue

    if "nnUNet_extTrainer" in txt:
        print("nnUNet_extTrainer support found in:", p)
        found = True
        break

if not found:
    print("[WARNING] This installed nnU-Net may not support nnUNet_extTrainer.")
    print("[WARNING] But the trainer has already been copied into nnU-Net package dir, so training should still work.")
PY

export CUDA_VISIBLE_DEVICES=0
# ============================================================
# 是否清理旧的 preprocessed / results
# 只改 epochs 不需要重新 preprocessing，所以 CLEAN_PREPROCESSED=0 即可
# 第一次用 diyepochs trainer，建议 CLEAN_RESULTS=1
# ============================================================
CLEAN_PREPROCESSED=0
CLEAN_RESULTS=0

# ============================================================
# 是否跳过已经训练完成的数据集
# 1 = 如果 checkpoint_final.pth 已存在，则跳过该 Dataset
# 0 = 不跳过，强制重新训练
# ============================================================
SKIP_FINISHED_TRAINING=1

# ============================================================
# 是否保存预测概率
# 0 = 只保存最终 mask
# 1 = 同时保存 .npz 概率
# ============================================================
SAVE_PROBABILITIES=0

# ============================================================
# 日志目录
# ============================================================
LOG_DIR="${BASE_DIR}/logs_pipeline_resencM_diyepochs"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "nnUNet_raw          = ${nnUNet_raw}"
echo "nnUNet_preprocessed = ${nnUNet_preprocessed}"
echo "nnUNet_results      = ${nnUNet_results}"
echo "PREDICT_ROOT        = ${PREDICT_ROOT}"
echo "CONFIG              = ${CONFIG}"
echo "FOLD                = ${FOLD}"
echo "PLANNER             = ${PLANNER}"
echo "PLANS               = ${PLANS}"
echo "TRAINER             = ${TRAINER}"
echo "DIY_NUM_EPOCHS      = ${DIY_NUM_EPOCHS}"
echo "nnUNet_extTrainer   = ${nnUNet_extTrainer}"
echo "GPU                 = ${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

mkdir -p "${nnUNet_preprocessed}"
mkdir -p "${nnUNet_results}"

# ============================================================
# 0. 检查 raw 里面 Dataset001~015 是否唯一
# ============================================================
echo ""
echo "Checking Dataset folders in nnUNet_raw..."

for DATASET_ID in $(seq ${DATASET_START} ${DATASET_END}); do
    ID_PADDED=$(printf "%03d" ${DATASET_ID})

    COUNT=$(find "${nnUNet_raw}" -maxdepth 1 -type d -name "Dataset${ID_PADDED}_*" | wc -l)

    if [ "${COUNT}" -ne 1 ]; then
        echo "[ERROR] Dataset${ID_PADDED} in nnUNet_raw is not unique."
        echo "Found ${COUNT} folders:"
        find "${nnUNet_raw}" -maxdepth 1 -type d -name "Dataset${ID_PADDED}_*"
        echo ""
        echo "Please remove duplicate Dataset${ID_PADDED}_* folders first."
        exit 1
    fi

    DATASET_NAME=$(find "${nnUNet_raw}" -maxdepth 1 -type d -name "Dataset${ID_PADDED}_*" -printf "%f\n")
    echo "[OK] ${DATASET_ID} -> ${DATASET_NAME}"
done

# ============================================================
# 1. 清理旧 preprocessed / results
# ============================================================
if [ "${CLEAN_PREPROCESSED}" -eq 1 ]; then
    echo ""
    echo "Cleaning old nnUNet_preprocessed Dataset001~015..."
    for DATASET_ID in $(seq ${DATASET_START} ${DATASET_END}); do
        ID_PADDED=$(printf "%03d" ${DATASET_ID})
        rm -rf "${nnUNet_preprocessed}/Dataset${ID_PADDED}_"*
    done
fi

if [ "${CLEAN_RESULTS}" -eq 1 ]; then
    echo ""
    echo "Cleaning old nnUNet_results Dataset001~015..."
    for DATASET_ID in $(seq ${DATASET_START} ${DATASET_END}); do
        ID_PADDED=$(printf "%03d" ${DATASET_ID})
        rm -rf "${nnUNet_results}/Dataset${ID_PADDED}_"*
    done
fi

# ============================================================
# 2. Preprocessing with ResEnc M planner
# ============================================================
echo ""
echo "============================================================"
echo "Start preprocessing Dataset${DATASET_START} ~ Dataset${DATASET_END}"
echo "Planner: ${PLANNER}"
echo "Config : ${CONFIG}"
echo "============================================================"

for DATASET_ID in $(seq ${DATASET_START} ${DATASET_END}); do
    ID_PADDED=$(printf "%03d" ${DATASET_ID})

    PREPROCESSED_DATASET_DIR=$(find "${nnUNet_preprocessed}" -maxdepth 1 -type d -name "Dataset${ID_PADDED}_*" | head -n 1)

    if [ -n "${PREPROCESSED_DATASET_DIR}" ] && \
       [ -f "${PREPROCESSED_DATASET_DIR}/${PLANS}.json" ]; then
        echo ""
        echo "[Skip] ResEnc M preprocessing already exists for Dataset${ID_PADDED}:"
        echo "       ${PREPROCESSED_DATASET_DIR}/${PLANS}.json"
        continue
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "Preprocessing Dataset${ID_PADDED}"
    echo "Planner=${PLANNER}"
    echo "Configuration=${CONFIG}"
    echo "------------------------------------------------------------"

    nnUNetv2_plan_and_preprocess \
        -d ${DATASET_ID} \
        -pl ${PLANNER} \
        -c ${CONFIG} \
        --verify_dataset_integrity \
        2>&1 | tee "${LOG_DIR}/preprocess_Dataset${ID_PADDED}_${PLANS}_${CONFIG}.log"
done

# ============================================================
# 3. Training with ResEnc M plans + diyepochs trainer
# ============================================================
echo ""
echo "============================================================"
echo "Start training Dataset${TRAIN_START} ~ Dataset${DATASET_END}"
echo "Plans  : ${PLANS}"
echo "Trainer: ${TRAINER}"
echo "Epochs : ${DIY_NUM_EPOCHS}"
echo "============================================================"

for DATASET_ID in $(seq ${TRAIN_START} ${DATASET_END}); do
    ID_PADDED=$(printf "%03d" ${DATASET_ID})

    RAW_DATASET_NAME=$(find "${nnUNet_raw}" -maxdepth 1 -type d -name "Dataset${ID_PADDED}_*" -printf "%f\n" | head -n 1)
    RESULT_DIR="${nnUNet_results}/${RAW_DATASET_NAME}/${TRAINER}__${PLANS}__${CONFIG}/fold_${FOLD}"
    FINAL_CKPT="${RESULT_DIR}/checkpoint_final.pth"

    if [ "${SKIP_FINISHED_TRAINING}" -eq 1 ] && [ -f "${FINAL_CKPT}" ]; then
        echo ""
        echo "[Skip] Training already finished for Dataset${ID_PADDED}:"
        echo "       ${FINAL_CKPT}"
        continue
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "Training Dataset${ID_PADDED}"
    echo "CONFIG=${CONFIG}, FOLD=${FOLD}"
    echo "PLANS=${PLANS}"
    echo "TRAINER=${TRAINER}"
    echo "EPOCHS=${DIY_NUM_EPOCHS}"
    echo "------------------------------------------------------------"

    nnUNetv2_train \
        ${DATASET_ID} \
        ${CONFIG} \
        ${FOLD} \
        -p ${PLANS} \
        -tr ${TRAINER} \
        2>&1 | tee "${LOG_DIR}/train_Dataset${ID_PADDED}_${TRAINER}_${PLANS}_${CONFIG}_${FOLD}.log"
done

# ============================================================
# 4. Whole-volume prediction with ResEnc M plans + diyepochs trainer
# ============================================================
echo ""
echo "============================================================"
echo "Start whole-volume prediction Dataset${DATASET_START} ~ Dataset${DATASET_END}"
echo "Plans  : ${PLANS}"
echo "Trainer: ${TRAINER}"
echo "============================================================"

for DATASET_ID in $(seq ${DATASET_START} ${DATASET_END}); do
    ID_PADDED=$(printf "%03d" ${DATASET_ID})

    PRED_DATASET_DIR=$(find "${PREDICT_ROOT}" -maxdepth 1 -type d -name "Dataset${ID_PADDED}_*" | head -n 1)

    if [ -z "${PRED_DATASET_DIR}" ]; then
        echo "[Skip] Cannot find prediction input folder for Dataset${ID_PADDED}"
        continue
    fi

    INPUT_FOLDER="${PRED_DATASET_DIR}/images"
    OUTPUT_FOLDER="${PRED_DATASET_DIR}/pred_${TRAINER}_${PLANS}_${CONFIG}_${FOLD}"

    if [ ! -d "${INPUT_FOLDER}" ]; then
        echo "[Skip] Missing input images folder: ${INPUT_FOLDER}"
        continue
    fi

    mkdir -p "${OUTPUT_FOLDER}"

    echo ""
    echo "------------------------------------------------------------"
    echo "Predicting Dataset${ID_PADDED}"
    echo "Input : ${INPUT_FOLDER}"
    echo "Output: ${OUTPUT_FOLDER}"
    echo "CONFIG=${CONFIG}, FOLD=${FOLD}"
    echo "PLANS=${PLANS}"
    echo "TRAINER=${TRAINER}"
    echo "------------------------------------------------------------"

    if [ "${SAVE_PROBABILITIES}" -eq 1 ]; then
        nnUNetv2_predict \
            -i "${INPUT_FOLDER}" \
            -o "${OUTPUT_FOLDER}" \
            -d ${DATASET_ID} \
            -c ${CONFIG} \
            -f ${FOLD} \
            -p ${PLANS} \
            -tr ${TRAINER} \
            --save_probabilities \
            2>&1 | tee "${LOG_DIR}/predict_Dataset${ID_PADDED}_${TRAINER}_${PLANS}_${CONFIG}_${FOLD}.log"
    else
        nnUNetv2_predict \
            -i "${INPUT_FOLDER}" \
            -o "${OUTPUT_FOLDER}" \
            -d ${DATASET_ID} \
            -c ${CONFIG} \
            -f ${FOLD} \
            -p ${PLANS} \
            -tr ${TRAINER} \
            2>&1 | tee "${LOG_DIR}/predict_Dataset${ID_PADDED}_${TRAINER}_${PLANS}_${CONFIG}_${FOLD}.log"
    fi
done

echo ""
echo "============================================================"
echo "All done!"
echo "Logs are saved in:"
echo "${LOG_DIR}"
echo ""
echo "Prediction results are saved under:"
echo "${PREDICT_ROOT}/DatasetXXX_*/pred_${TRAINER}_${PLANS}_${CONFIG}_${FOLD}"
echo "============================================================"