import os
import numpy as np
import tifffile
from stardist.models import Config3D, StarDist3D
from stardist import fill_label_holes, Rays_GoldenSpiral
from csbdeep.utils import normalize
from stardist.utils import calculate_extents
from scipy.ndimage import label, generate_binary_structure


# =========================
# 0. 参数
# =========================
data_dir = "/mnt/c/baidunetdiskdownload/empanda/data"

raw_path = os.path.join(data_dir, "raw.tif")

roi_num_list = [1, 5, 10]
pred_root_list = [100, 101, 102, 103, 104]

celltype = "hela2"
organelletype = "mito"

basedir = "models"
out_dir = "result"
os.makedirs(basedir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)


# =========================
# 1. 环境设置
# =========================
os.environ["GPU_TOOLS_NO_OPENCL"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


# =========================
# 2. 读取 raw，只读一次
# =========================
print("Loading raw:", raw_path)
X_raw = tifffile.imread(raw_path)

print("raw shape:", X_raw.shape)


# =========================
# 3. 单次训练函数
# =========================
def train_one_stardist(pred_root, roi_num):
    label_path = os.path.join(
        data_dir,
        f"label_hela2_mito_volume_{pred_root}_{roi_num}.tif"
    )

    if not os.path.exists(label_path):
        print(f"[Skip] label not found: {label_path}")
        return

    print("\n================================================")
    print(f"Start StarDist3D: pred_root={pred_root}, roi_num={roi_num}")
    print("Loading label:", label_path)
    print("================================================")

    Y0 = tifffile.imread(label_path)

    assert X_raw.shape == Y0.shape, \
        f"raw 和 label 尺寸不一致: raw={X_raw.shape}, label={Y0.shape}"

    # =========================
    # binary → instance
    # =========================
    vals = np.unique(Y0)
    print("标签值:", vals)

    if len(vals) <= 2:
        print("👉 binary → instance labeling")
        structure = generate_binary_structure(3, 3)
        Y_inst, n_objects = label(Y0 > 0, structure)
        print(f"实例数量: {n_objects}")
    else:
        print("👉 已是 instance label")
        Y_inst = Y0

    debug_path = os.path.join(
        out_dir,
        f"debug_Y_hela2_mito_{pred_root}_{roi_num}.tif"
    )
    tifffile.imwrite(debug_path, Y_inst.astype(np.uint16))

    # =========================
    # StarDist 格式
    # =========================
    X = [X_raw]
    Y = [Y_inst]

    Y = [fill_label_holes(y) for y in Y]
    X = [normalize(x, 1, 99.8, axis=(0, 1, 2)) for x in X]

    # =========================
    # anisotropy
    # =========================
    extents = calculate_extents(Y)
    anisotropy = tuple(np.max(extents) / extents)
    print("anisotropy =", anisotropy)

    # =========================
    # Config
    # =========================
    n_rays = 96
    grid = tuple(1 if a > 1.5 else 2 for a in anisotropy)
    rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)

    conf = Config3D(
        rays=rays,
        grid=grid,
        anisotropy=anisotropy,
        use_gpu=True,
        n_channel_in=1,
        train_patch_size=(4, 80, 80),
        train_batch_size=16,
        train_epochs=50,
        train_learning_rate=3e-4,
        train_foreground_only=0.9,
    )

    print(conf)

    # =========================
    # 自动编号
    # =========================
    model_base = f"stardist3d_hela2_mito_{pred_root}_{roi_num}"

    existing = []
    if os.path.exists(basedir):
        for name in os.listdir(basedir):
            if name.startswith(model_base + "_"):
                try:
                    suffix = int(name.split("_")[-1])
                    existing.append(suffix)
                except Exception:
                    continue

    run_id = 0 if len(existing) == 0 else max(existing) + 1
    model_name = f"{model_base}_{run_id}"

    print(f"👉 当前 run_id = {run_id}")
    print(f"👉 model_name = {model_name}")

    # =========================
    # 初始化 + 训练
    # =========================
    model = StarDist3D(conf, name=model_name, basedir=basedir)

    model.train(
        X,
        Y,
        validation_data=(X, Y)
    )

    # =========================
    # 推理 whole volume
    # =========================
    lbl_pred, details = model.predict_instances(
        normalize(X_raw),
        n_tiles=(8, 8, 8)
    )

    # =========================
    # 保存
    # =========================
    out_path = os.path.join(
        out_dir,
        f"prediction_hela2_mito_{pred_root}_{roi_num}.tif"
    )

    tifffile.imwrite(out_path, lbl_pred.astype(np.uint16))

    print(f"✅ 完成: pred_root={pred_root}, roi_num={roi_num}")
    print(f"✅ 保存: {out_path}")
    print(f"nonzero voxels: {np.count_nonzero(lbl_pred)}")


# =========================
# 4. 批量运行
# =========================
for roi_num in roi_num_list:
    for pred_root in pred_root_list:
        train_one_stardist(pred_root, roi_num)

print("\nAll StarDist3D runs finished.")