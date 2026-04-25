import os
import re
import subprocess
import pandas as pd
from pathlib import Path

from cellmap_segmentation_challenge.utils.datasplit import make_datasplit_csv


# =========================
# paths
# =========================
manifest_path = "/mnt/c/baidunetdiskdownload/empanda/data/cellmap_custom/manifest.csv"
data_root = "/mnt/c/baidunetdiskdownload/empanda/data/cellmap_custom"
out_root = "/mnt/c/baidunetdiskdownload/empanda/data/cellmap_official_2d_runs"

os.makedirs(out_root, exist_ok=True)


# =========================
# easy-to-change settings
# =========================
patch_size_xy = 80

batch_size = 8
epochs = 100
iterations_per_epoch = 200
learning_rate = 0.0001

# only_name = "hela2_mito_100_1"
only_name = None

# 是否训练后只保留最新 checkpoint
keep_only_latest_checkpoint = True


# =========================
# helper: cleanup checkpoints
# =========================
def get_epoch_from_checkpoint(path: Path, model_name: str):
    """
    从 checkpoint 文件名中解析 epoch。
    适配:
        {model_name}_{epoch}.pth
    """
    name = path.name

    pattern = rf"^{re.escape(model_name)}_(\d+)\.pth$"
    m = re.match(pattern, name)

    if m is not None:
        return int(m.group(1))

    # fallback: 取文件名中最后一个数字
    nums = re.findall(r"\d+", name)
    if nums:
        return int(nums[-1])

    return None


def cleanup_checkpoints_keep_latest(run_dir, model_name):
    """
    只保留 run_dir/checkpoints 里当前 model_name 最新 epoch 的 checkpoint。
    """
    ckpt_dir = Path(run_dir) / "checkpoints"

    if not ckpt_dir.exists():
        print(f"[Checkpoint cleanup] skip, not found: {ckpt_dir}")
        return

    ckpt_files = sorted(ckpt_dir.glob(f"{model_name}_*.pth"))

    if len(ckpt_files) <= 1:
        print(f"[Checkpoint cleanup] skip, checkpoint num = {len(ckpt_files)}")
        return

    parsed = []

    for p in ckpt_files:
        epoch = get_epoch_from_checkpoint(p, model_name)

        if epoch is None:
            print(f"[Checkpoint cleanup] skip unrecognized file: {p.name}")
            continue

        parsed.append((epoch, p))

    if len(parsed) <= 1:
        print(f"[Checkpoint cleanup] skip, parsed checkpoint num = {len(parsed)}")
        return

    parsed.sort(key=lambda x: x[0])

    keep_epoch, keep_path = parsed[-1]
    delete_items = parsed[:-1]

    print("\n[Checkpoint cleanup]")
    print(f"Checkpoint dir: {ckpt_dir}")
    print(f"Keep latest: epoch={keep_epoch}, file={keep_path.name}")

    for epoch, p in delete_items:
        print(f"Delete: epoch={epoch}, file={p.name}")
        p.unlink()

    print(f"Deleted {len(delete_items)} old checkpoint(s).")


# =========================
# load manifest
# =========================
df = pd.read_csv(manifest_path)

if only_name is not None:
    df = df[df["name"] == only_name]


# =========================
# train config template
# =========================
train_template = r'''
from upath import UPath
from cellmap_segmentation_challenge.models import UNet_2D

learning_rate = {learning_rate}
batch_size = {batch_size}

input_array_info = {{
    "shape": ({patch_size_xy}, {patch_size_xy}),
    "scale": (8, 8),
}}

target_array_info = {{
    "shape": ({patch_size_xy}, {patch_size_xy}),
    "scale": (8, 8),
}}

epochs = {epochs}
iterations_per_epoch = {iterations_per_epoch}
random_seed = 42

classes = ["mito"]

model_name = "{model_name}"
model_to_load = "{model_name}"
model = UNet_2D(1, len(classes))

load_model = "latest"

logs_save_path = UPath("{run_dir}/tensorboard/{model_name}").path
model_save_path = UPath("{run_dir}/checkpoints/{model_name}" + "_{{epoch}}.pth").path
datasplit_path = "{datasplit_path}"

spatial_transforms = {{
    "mirror": {{"axes": {{"x": 1.0, "y": 1.0}}}},
    "transpose": {{"axes": ["x", "y"]}},
    "rotate": {{"axes": {{"x": [-180, 180], "y": [-180, 180]}}}},
}}

validation_time_limit = 0
validation_batch_limit = 0
filter_by_scale = False

crops = "crop0"
filter_classes = False

if __name__ == "__main__":
    from cellmap_segmentation_challenge import train
    train(__file__)
'''


# =========================
# predict config template
# =========================
predict_template = r'''
from cellmap_segmentation_challenge import predict

config_path = __file__.replace("predict", "train")

predict(
    config_path,
    crops="crop0",
    overwrite=True,
    filter_classes=False,
)
'''


# =========================
# batch run
# =========================
for _, row in df.iterrows():
    name = row["name"]

    run_dir = os.path.join(out_root, name)
    os.makedirs(run_dir, exist_ok=True)

    datasplit_path = os.path.join(run_dir, "datasplit.csv")
    train_config_path = os.path.join(run_dir, "train_2D.py")
    predict_config_path = os.path.join(run_dir, "predict_2D.py")

    model_name = f"cellmap_2d_unet_{name}"

    if os.path.exists(datasplit_path):
        os.remove(datasplit_path)

    search_path = os.path.join(
        data_root,
        "{dataset}",
        "{dataset}.zarr",
        "recon-1",
        "{name}",
    )

    make_datasplit_csv(
        classes=["mito"],
        datasets=[name],
        crops=["crop0"],
        search_path=search_path,
        raw_name="em/fibsem-uint8",
        crop_name="labels/groundtruth/{crop}/{label}",
        csv_path=datasplit_path,
        validation_prob=0.0,
        force_all_classes=False,
    )

    with open(train_config_path, "w") as f:
        f.write(train_template.format(
            learning_rate=learning_rate,
            batch_size=batch_size,
            patch_size_xy=patch_size_xy,
            epochs=epochs,
            iterations_per_epoch=iterations_per_epoch,
            model_name=model_name,
            run_dir=run_dir,
            datasplit_path=datasplit_path,
        ))

    with open(predict_config_path, "w") as f:
        f.write(predict_template)

    print("\n============================================================")
    print("Run:", name)
    print("Mode: 2D")
    print("Patch size:", patch_size_xy)
    print("Train config:", train_config_path)
    print("Predict config:", predict_config_path)
    print("Datasplit:", datasplit_path)
    print("Model name:", model_name)
    print("============================================================")

    subprocess.run(["python", train_config_path], check=True)

    if keep_only_latest_checkpoint:
        cleanup_checkpoints_keep_latest(run_dir, model_name)

    subprocess.run(["python", predict_config_path], check=True)

    if keep_only_latest_checkpoint:
        cleanup_checkpoints_keep_latest(run_dir, model_name)


print("\nAll official CellMap 2D train + predict runs finished.")