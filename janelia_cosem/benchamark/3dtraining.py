import os
import glob
import re
import subprocess
import pandas as pd

from cellmap_segmentation_challenge.utils.datasplit import make_datasplit_csv


def keep_only_latest_checkpoint(ckpt_dir, model_name):
    ckpts = glob.glob(os.path.join(ckpt_dir, f"{model_name}_*.pth"))

    if len(ckpts) <= 1:
        return

    def get_epoch(p):
        base = os.path.basename(p)
        m = re.search(r"_(\d+)\.pth$", base)
        return int(m.group(1)) if m else -1

    latest = max(ckpts, key=get_epoch)

    for p in ckpts:
        if p != latest:
            os.remove(p)

    print("Kept checkpoint:", latest)


# =========================
# paths
# =========================
manifest_path = "/mnt/c/baidunetdiskdownload/empanda/data/cellmap_custom/manifest.csv"
data_root = "/mnt/c/baidunetdiskdownload/empanda/data/cellmap_custom"
out_root = "cellmap_official_3d_runs"

os.makedirs(out_root, exist_ok=True)


# =========================
# easy-to-change settings
# =========================
patch_size_z = 16
patch_size_xy = 80

batch_size = 2
epochs = 100
iterations_per_epoch = 200
learning_rate = 0.0001

# only_name = "hela2_mito_100_1"
only_name = None


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
from cellmap_segmentation_challenge.models import UNet_3D

learning_rate = {learning_rate}
batch_size = {batch_size}

input_array_info = {{
    "shape": ({patch_size_z}, {patch_size_xy}, {patch_size_xy}),
    "scale": (8, 8, 8),
}}

target_array_info = {{
    "shape": ({patch_size_z}, {patch_size_xy}, {patch_size_xy}),
    "scale": (8, 8, 8),
}}

epochs = {epochs}
iterations_per_epoch = {iterations_per_epoch}
random_seed = 42

classes = ["mito"]

model_name = "{model_name}"
model_to_load = "{model_name}"
model = UNet_3D(1, len(classes))

load_model = "latest"

logs_save_path = UPath("{run_dir}/tensorboard/{model_name}").path
model_save_path = UPath("{run_dir}/checkpoints/{model_name}" + "_{{epoch}}.pth").path
datasplit_path = "{datasplit_path}"

spatial_transforms = {{
    "mirror": {{"axes": {{"x": 1.0, "y": 1.0, "z": 1.0}}}},
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


for _, row in df.iterrows():
    name = row["name"]

    run_dir = os.path.join(out_root, name)
    os.makedirs(run_dir, exist_ok=True)

    datasplit_path = os.path.join(run_dir, "datasplit.csv")
    train_config_path = os.path.join(run_dir, "train_3D.py")

    model_name = f"cellmap_3d_unet_{name}"

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
            patch_size_z=patch_size_z,
            patch_size_xy=patch_size_xy,
            epochs=epochs,
            iterations_per_epoch=iterations_per_epoch,
            model_name=model_name,
            run_dir=run_dir,
            datasplit_path=datasplit_path,
        ))

    print("\n============================================================")
    print("Run:", name)
    print("Mode: 3D")
    print("Patch size:", (patch_size_z, patch_size_xy, patch_size_xy))
    print("Train config:", train_config_path)
    print("Datasplit:", datasplit_path)
    print("============================================================")

    subprocess.run(["python", train_config_path], check=True)

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    keep_only_latest_checkpoint(ckpt_dir, model_name)

print("\nAll official CellMap 3D training runs finished.")