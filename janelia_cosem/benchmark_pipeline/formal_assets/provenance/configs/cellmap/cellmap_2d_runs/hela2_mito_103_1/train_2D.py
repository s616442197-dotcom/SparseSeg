
from upath import UPath
from cellmap_segmentation_challenge.models import UNet_2D

learning_rate = 0.0001
batch_size = 8

input_array_info = {
    "shape": (80, 80),
    "scale": (8, 8),
}

target_array_info = {
    "shape": (80, 80),
    "scale": (8, 8),
}

epochs = 100
iterations_per_epoch = 200
random_seed = 42

classes = ["mito"]

model_name = "cellmap_2d_unet_hela2_mito_103_1"
model_to_load = "cellmap_2d_unet_hela2_mito_103_1"
model = UNet_2D(1, len(classes))

load_model = "latest"

logs_save_path = UPath("/mnt/c/baidunetdiskdownload/empanda/data/cellmap_official_2d_runs/hela2_mito_103_1/tensorboard/cellmap_2d_unet_hela2_mito_103_1").path
model_save_path = UPath("/mnt/c/baidunetdiskdownload/empanda/data/cellmap_official_2d_runs/hela2_mito_103_1/checkpoints/cellmap_2d_unet_hela2_mito_103_1" + "_{epoch}.pth").path
datasplit_path = "/mnt/c/baidunetdiskdownload/empanda/data/cellmap_official_2d_runs/hela2_mito_103_1/datasplit.csv"

spatial_transforms = {
    "mirror": {"axes": {"x": 1.0, "y": 1.0}},
    "transpose": {"axes": ["x", "y"]},
    "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
}

validation_time_limit = 0
validation_batch_limit = 0
filter_by_scale = False

crops = "crop0"
filter_classes = False

if __name__ == "__main__":
    from cellmap_segmentation_challenge import train
    train(__file__)
