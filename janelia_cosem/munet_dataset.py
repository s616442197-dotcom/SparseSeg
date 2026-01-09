import numpy as np
from tqdm import tqdm
import argparse
from scipy.ndimage import binary_erosion, binary_dilation, gaussian_filter, distance_transform_edt
import tifffile as tiff
from utils import process_volume,local_contrast_normalize,filter_connected_regions_shape,intersect_regions
from skimage.transform import downscale_local_mean
import os
from Loss_func import total_loss_fn
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from save_function import save_volume_with_masks_as_rgb_tiff
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader,RandomSampler
from datetime import datetime
from get_inputfeature import extract_2d_features_from_patch
from scipy import ndimage

def get_edge_mask(mask, edge_width=1):
    """
    计算 mask 的 XY 平面边界区域：仅在平面内膨胀和腐蚀（不跨 Z）。
    Args:
        mask: 二值数组 (0/1)，形状 [H, W] 或 [Z, H, W]
        edge_width: 边界宽度（膨胀/腐蚀迭代次数）
    Returns:
        edge: 浮点型边界掩码 (0/1)
    """
    mask = (mask > 0)

    if mask.ndim == 3:
        # 3D 情况，只在 XY 平面扩张
        struct_xy = np.zeros((3, 3, 3), dtype=bool)
        struct_xy[1, :, :] = True  # 仅影响同一 Z 层的像素
        dilated = ndimage.binary_dilation(mask, structure=struct_xy, iterations=edge_width)
        eroded = ndimage.binary_erosion(mask, structure=struct_xy, iterations=edge_width)
    else:
        # 2D 情况，正常处理
        dilated = ndimage.binary_dilation(mask, iterations=edge_width)
        eroded = ndimage.binary_erosion(mask, iterations=edge_width)

    edge = np.logical_xor(dilated, eroded).astype(np.float32)
    return edge

class ValidPatchSliceDataset(Dataset):
    def __init__(self, volume, mask_volume, negative_volume_label, softnega, patch_size=(256, 256), threshold=0.7,
                 num_samples=1000, max_trials=5000, thickness=2):
        """
        volume: np.ndarray, shape (D, H, W), 原始图像
        mask_volume: np.ndarray, shape (D, H, W), 对应的 mask
        """
        self.thickness = thickness
        self.volume = volume.astype(np.float32)
        self.mask_volume = mask_volume.astype(np.float32)
        self.negative_mask_volume = negative_volume_label.astype(np.float32)
        self.softnega = softnega.astype(np.float32)
        # Step 1: 对第一个维度求和，得到一个二维图像
        mask_2d = np.sum(self.mask_volume, axis=0)  # shape: (H, W)

        self.patch_size = patch_size
        # self.threshold = threshold
        self.samples = []

        D, H, W = self.volume.shape
        nz_coords = np.argwhere(self.mask_volume > 0)  # (N,3), 每行(z,x,y)
        soft_coords = np.argwhere((self.negative_mask_volume > 0) | (self.softnega > 0))
        # soft_coords = np.argwhere((self.negative_mask_volume > 0))
        if len(soft_coords) > 0:
            n_extra = max(1, int(0.01 * threshold * len(nz_coords)))  # 取 nz_coords 数量的 10%
            n_extra = min(n_extra, len(soft_coords))  # 防止 soft_coords 不足
            idx_extra = np.random.choice(len(soft_coords), n_extra, replace=False)
            soft_sample = soft_coords[idx_extra]
            # 合并
            nz_coords = np.concatenate([nz_coords, soft_sample], axis=0)
        # nz_coords = np.argwhere((self.mask_volume > 0) | (self.softnega > 0))/

        # self.mask_volume[negative_volume_label == 1] = 0

        print("🔍 正在构建符合非零占比要求的 patch 索引...")

        def uniform_sample_from_nz(nz_coords, volume_shape, num_samples,
                                   patch_size=(128, 128), thickness=0):
            """
            从非零体素中随机采样 num_samples 个 patch 起点 (z, x0, y0)，
            确保 patch 不越界。
            """
            D, H, W = volume_shape
            ph, pw = patch_size

            # --- 合法中心范围 ---
            z_min, z_max = thickness, D - thickness - 1
            x_min, x_max = ph // 2, H - ph // 2 - 1
            y_min, y_max = pw // 2, W - pw // 2 - 1

            coords = np.array(nz_coords)

            # --- 过滤掉越界的点 ---
            valid = coords[
                (coords[:, 0] >= z_min) & (coords[:, 0] <= z_max) &
                (coords[:, 1] >= x_min) & (coords[:, 1] <= x_max) &
                (coords[:, 2] >= y_min) & (coords[:, 2] <= y_max)
                ]
            if len(valid) == 0:
                raise ValueError("❌ 没有合法的非零体素点用于采样")

            # --- 均匀随机采样 ---
            idx = np.linspace(0, len(valid) - 1, num_samples, dtype=int)
            np.random.shuffle(idx)
            sampled = valid[idx[:num_samples]]

            # --- 转为 patch 左上角坐标 ---
            z = sampled[:, 0]
            x = (sampled[:, 1] - ph // 2).clip(0, H - ph)
            y = (sampled[:, 2] - pw // 2).clip(0, W - pw)
            return np.stack([z, x, y], axis=1)

        sample_coords = uniform_sample_from_nz(
            nz_coords,
            volume_shape=(D, H, W),
            num_samples=num_samples,
            patch_size=(patch_size[0], patch_size[1]),
            thickness=thickness
        )

        for z, x, y in sample_coords:
            self.samples.append((z, x, y))
            # patch = self.volume[z, x:x + patch_size[0], y:y + patch_size[1]]
        print(f"✅ 构建完成，共采样 {len(self.samples)} 个有效 patch")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        def apply_random_flip_rotate(patch_list):
            """
            对多个 patch（例如 img_patch, negative_mask_patch）
            做一致的随机增强（翻转 + 旋转），保持同步。
            输出保证为 C-contiguous，不会产生负 stride。
            """

            # 0: 不翻转, 1: 水平翻转, 2: 垂直翻转
            flip_mode = np.random.choice([0, 1, 2])

            # 0: 不转, 1: 90°, 2: 180°, 3: 270°
            rot_k = np.random.choice([0, 1, 2, 3])

            out = []

            for patch in patch_list:
                aug = patch  # 不 copy，先轻度操作，最后统一 copy

                # === 翻转（安全模式，不出现负 stride）===
                if flip_mode == 1:  # 水平 flips W 轴
                    aug = np.flip(aug, axis=-1)
                elif flip_mode == 2:  # 垂直 flips H 轴
                    aug = np.flip(aug, axis=-2)

                # === 旋转（safe，np.rot90 本身 safe，但 stride 变化，结尾 copy）===
                if rot_k > 0:
                    aug = np.rot90(aug, k=rot_k, axes=(-2, -1))

                # === ⚠ 最关键的一步：copy() 保证没有负 stride ===
                out.append(aug.copy())

            return out

        z, x, y = self.samples[idx]

        # 取 3 张 slice 构成 3 通道图像
        img_patch = self.volume[z - self.thickness:z + self.thickness + 1, x:x + self.patch_size[0],
                    y:y + self.patch_size[1]]  # shape [3, H, W]
        # 取中心切片对应 mask
        mask_patch = self.mask_volume[z:z + 1, x:x + self.patch_size[0], y:y + self.patch_size[1]]  # shape [H, W]
        mask_patch = (mask_patch > 0.5).astype(np.float32)  # 二值化（如果原本不是）

        edge_patch = get_edge_mask(mask_patch, edge_width=1)

        negative_mask_patch = self.negative_mask_volume[z:z + 1, x:x + self.patch_size[0], y:y + self.patch_size[1]]
        soft_negative_mask_patch = self.softnega[z:z + 1, x:x + self.patch_size[0], y:y + self.patch_size[1]]

        img_patch, mask_patch, edge_patch, negative_mask_patch, soft_negative_mask_patch = apply_random_flip_rotate([
            img_patch, mask_patch, edge_patch, negative_mask_patch, soft_negative_mask_patch
        ])

        # img_patch = (img_patch - img_patch.min()) / (img_patch.max() - img_patch.min() + 1e-6)
        _, feats_stack = extract_2d_features_from_patch(
            img_patch,  # [Z,H,W] 例如 thickness=1 -> Z=3
            aggregate_mode="gaussian",
            sigma_z=0.8,
            denoise_tv=0.05,  # 可设 0 关闭
            sigmas_gauss=(1.0, 2.0, 4.0),
            sigmas_hessian=(1.0, 2.0, 4.0),
            win_local_stats=9,
            st_sigma=1.0,
        )

        return torch.tensor(feats_stack), torch.tensor(mask_patch), torch.tensor(negative_mask_patch), torch.tensor(
            soft_negative_mask_patch), torch.tensor(edge_patch)  # (3, H, W), (1, H, W)