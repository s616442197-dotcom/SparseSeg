
from cellmap_segmentation_challenge import predict

config_path = __file__.replace("predict", "train")

predict(
    config_path,
    crops="crop0",
    overwrite=True,
    filter_classes=False,
)
