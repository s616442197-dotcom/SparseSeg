import numpy as np
from tqdm import tqdm
import argparse
from scipy.ndimage import binary_closing,binary_erosion, binary_dilation, label as nd_label,gaussian_filter, distance_transform_edt
import tifffile as tiff
from utils import compute_statistical_mask,process_volume,local_contrast_normalize,local_standardize,filter_connected_regions_shape,intersect_regions,soften_center_mask_dilated
from skimage.transform import downscale_local_mean
import torch

def save_volume_with_masks_as_rgb_tiff(volume, mask1, mask2, path="output.tiff"):
    """
    volume, mask1, mask2: [D, H, W] 的 numpy 数组，float32 或 uint8，范围归一化到 [0, 1] 或 [0, 255]
    保存为 [D, H, W, 3] 的 RGB 格式 tiff
    - volume 映射到 R 通道
    - mask1 映射到 G 通道
    - mask2 映射到 B 通道
    """
    volume = volume.astype(np.float32)
    mask1 = mask1.astype(np.float32)
    mask2 = mask2.astype(np.float32)

    # 归一化到 0~1
    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    volume_n = norm(volume)
    mask1_n = norm(mask1)
    mask2_n = norm(mask2)

    rgb = np.stack([
        volume_n,  # R: 原图
        mask1_n,  # G: mask1
        mask2_n  # B: mask2
    ], axis=-1)  # [D, H, W, 3]

    # 转为 8-bit
    rgb_uint8 = (rgb * 255).astype(np.uint8)

    # 保存为多帧 RGB tiff
    tiff.imwrite(path, rgb_uint8, photometric='rgb')

    print(f"✅ 保存成功：{path}")


def save_model(model, ckpt_path):
    """保存模型参数"""
    torch.save(model.state_dict(), ckpt_path)
    print(f"✅ 模型参数已保存: {ckpt_path}")