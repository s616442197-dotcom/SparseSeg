import numpy as np
from tqdm import tqdm
import argparse
from scipy.ndimage import binary_erosion, binary_dilation, gaussian_filter, distance_transform_edt
import tifffile as tiff
from utils import process_volume,local_contrast_normalize,filter_connected_regions_shape,intersect_regions
from skimage.transform import downscale_local_mean
import os
from Loss_func import projection_by_mean_diff,build_dilated_rings
import torch
from torch.utils.data import Dataset, DataLoader,RandomSampler
from scipy import ndimage
def projection_by_mean_diff_volume(input_img, target, negative, eps=1e-8):
    """
    input_img : (D,F,H,W)
    target    : (D,H,W) {0,1}
    negative  : (D,H,W) {0,1}

    return
    ------
    project_img : (D,H,W)   normalized to 0~1
    w           : (F,)
    """

    D, F, H, W = input_img.shape

    # ---- reshape features ----
    X = input_img.permute(0,2,3,1).reshape(-1, F)  # (N,F)

    pos_mask = target.reshape(-1) > 0
    neg_mask = negative.reshape(-1) > 0

    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        raise ValueError("target 或 negative 没有正样本")

    # ---- 均值向量 ----
    mu_pos = X[pos_mask].mean(0)
    mu_neg = X[neg_mask].mean(0)

    # ---- projection direction ----
    w = mu_pos - mu_neg
    w = w / (w.norm() + eps)

    # ---- 投影 ----
    project_img = (input_img * w.view(1, F, 1, 1)).sum(dim=1)  # (D,H,W)

    # ---- normalize to 0~1 ----
    min_v = project_img.min()
    max_v = project_img.max()

    project_img = (project_img - min_v) / (max_v - min_v + eps)

    return project_img, w

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

# class ValidPatchSliceDataset(Dataset):
#     def __init__(self, volume, mask_volume,feature_volume, negative_volume_label, softnega, patch_size=(256, 256), threshold=0.7,
#                  num_samples=1000, max_trials=5000, thickness=2):
#         """
#         volume: np.ndarray, shape (D, H, W), 原始图像
#         mask_volume: np.ndarray, shape (D, H, W), 对应的 mask
#         """
#         self.thickness = thickness
#         self.volume = volume.astype(np.float32)
#         self.mask_volume = mask_volume.astype(np.float32)
#         self.negative_mask_volume = negative_volume_label.astype(np.float32)
#         self.softnega = softnega.astype(np.float32)
#         # Step 1: 对第一个维度求和，得到一个二维图像
#         mask_2d = np.sum(self.mask_volume, axis=0)  # shape: (H, W)
#         self.feature = feature_volume
#
#         self.patch_size = patch_size
#         # self.threshold = threshold
#         self.samples = []
#         _,negative_ring = build_dilated_rings(self.mask_volume, 1, kernel_size=1, edge_size=4)
#         self.area_ref, _ = projection_by_mean_diff_volume(self.feature, self.mask_volume, (self.negative_mask_volume > 0) | (negative_ring > 0))
#
#         D, H, W = self.volume.shape
#         nz_coords = np.argwhere(self.mask_volume > 0)  # (N,3), 每行(z,x,y)
#         soft_coords = np.argwhere((self.negative_mask_volume > 0) | (self.softnega > 0))
#         # soft_coords = np.argwhere((self.negative_mask_volume > 0))
#         if len(soft_coords) > 0:
#             n_extra = max(1, int(0.01 * threshold * len(nz_coords)))  # 取 nz_coords 数量的 10%
#             n_extra = min(n_extra, len(soft_coords))  # 防止 soft_coords 不足
#             idx_extra = np.random.choice(len(soft_coords), n_extra, replace=False)
#             soft_sample = soft_coords[idx_extra]
#             # 合并
#             nz_coords = np.concatenate([nz_coords, soft_sample], axis=0)
#         # nz_coords = np.argwhere((self.mask_volume > 0) | (self.softnega > 0))/
#
#         # self.mask_volume[negative_volume_label == 1] = 0
#
#         print("🔍 正在构建符合非零占比要求的 patch 索引...")
#
#         def uniform_sample_from_nz(nz_coords, volume_shape, num_samples,
#                                    patch_size=(128, 128), thickness=0):
#             """
#             从非零体素中随机采样 num_samples 个 patch 起点 (z, x0, y0)，
#             确保 patch 不越界。
#             """
#             D, H, W = volume_shape
#             ph, pw = patch_size
#
#             # --- 合法中心范围 ---
#             z_min, z_max = thickness, D - thickness - 1
#             x_min, x_max = ph // 2, H - ph // 2 - 1
#             y_min, y_max = pw // 2, W - pw // 2 - 1
#
#             coords = np.array(nz_coords)
#
#             # --- 过滤掉越界的点 ---
#             valid = coords[
#                 (coords[:, 0] >= z_min) & (coords[:, 0] <= z_max) &
#                 (coords[:, 1] >= x_min) & (coords[:, 1] <= x_max) &
#                 (coords[:, 2] >= y_min) & (coords[:, 2] <= y_max)
#                 ]
#             if len(valid) == 0:
#                 raise ValueError("❌ 没有合法的非零体素点用于采样")
#
#             # --- 均匀随机采样 ---
#             idx = np.linspace(0, len(valid) - 1, num_samples, dtype=int)
#             np.random.shuffle(idx)
#             sampled = valid[idx[:num_samples]]
#
#             # --- 转为 patch 左上角坐标 ---
#             z = sampled[:, 0]
#             x = (sampled[:, 1] - ph // 2).clip(0, H - ph)
#             y = (sampled[:, 2] - pw // 2).clip(0, W - pw)
#             return np.stack([z, x, y], axis=1)
#
#         sample_coords = uniform_sample_from_nz(
#             nz_coords,
#             volume_shape=(D, H, W),
#             num_samples=num_samples,
#             patch_size=(patch_size[0], patch_size[1]),
#             thickness=thickness
#         )
#
#         for z, x, y in sample_coords:
#             self.samples.append((z, x, y))
#             # patch = self.volume[z, x:x + patch_size[0], y:y + patch_size[1]]
#         print(f"✅ 构建完成，共采样 {len(self.samples)} 个有效 patch")
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         def apply_random_flip_rotate(patch_list):
#             """
#             对 patch_list 中：
#             - 所有 patch：做 H/W 维度的 flip + rotate
#             - 仅 patch_list[0]：额外允许 axis=0 的反转
#             """
#
#             flip_mode = np.random.choice([0, 1, 2])  # H/W flip
#             rot_k = np.random.choice([0, 1, 2, 3])  # H/W rotate
#
#             # ⭐ 是否对第一个 patch 的 axis=0 反转
#             flip_z0 = np.random.choice([False, True])
#
#             out = []
#
#             for i, patch in enumerate(patch_list):
#                 aug = patch
#
#                 # =========================
#                 # ① 仅对第一个 patch 做 axis=0 反转
#                 # =========================
#                 if i == 0 and flip_z0:
#                     aug = np.flip(aug, axis=0)
#
#                 # =========================
#                 # ② H / W 翻转（所有 patch）
#                 # =========================
#                 if flip_mode == 1:
#                     aug = np.flip(aug, axis=-1)
#                 elif flip_mode == 2:
#                     aug = np.flip(aug, axis=-2)
#
#                 # =========================
#                 # ③ H / W 旋转（所有 patch）
#                 # =========================
#                 if rot_k > 0:
#                     aug = np.rot90(aug, k=rot_k, axes=(-2, -1))
#
#                 out.append(aug.copy())
#
#             return out
#
#         z, x, y = self.samples[idx]
#
#         # ===============================
#         # image patch（如果你还要用）
#         # ===============================
#         img_patch = self.volume[
#                     z - self.thickness: z + self.thickness + 1,
#                     x: x + self.patch_size[0],
#                     y: y + self.patch_size[1]
#                     ]  # (Z, H, W)
#
#         # ===============================
#         # ⭐ 正确的 feature patch
#         # ===============================
#         feature_patch = self.feature[
#                         z, :,  # 取中心 z + 所有 F
#                         x: x + self.patch_size[0],
#                         y: y + self.patch_size[1]
#                         ]  # (F, H, W)
#
#         # ===============================
#         # mask / edge / negative
#         # ===============================
#         mask_patch = self.mask_volume[
#                      z: z + 1,
#                      x: x + self.patch_size[0],
#                      y: y + self.patch_size[1]
#                      ]
#         mask_patch = (mask_patch > 0.5).astype(np.float32)
#
#         edge_patch = get_edge_mask(mask_patch, edge_width=1)
#
#         negative_mask_patch = self.negative_mask_volume[
#                               z: z + 1,
#                               x: x + self.patch_size[0],
#                               y: y + self.patch_size[1]
#                               ]
#         soft_negative_mask_patch = self.softnega[
#                                    z: z + 1,
#                                    x: x + self.patch_size[0],
#                                    y: y + self.patch_size[1]
#                                    ]
#         area_ref_patch = self.area_ref[
#                                    z: z + 1,
#                                    x: x + self.patch_size[0],
#                                    y: y + self.patch_size[1]
#                                    ]
#         # ===============================
#         # 同步数据增强（feature 一起）
#         # ===============================
#         (
#             img_patch,
#             feature_patch,
#             mask_patch,
#             edge_patch,
#             negative_mask_patch,
#             soft_negative_mask_patch,
#             area_ref_patch
#         ) = apply_random_flip_rotate([
#             img_patch,
#             feature_patch,
#             mask_patch,
#             edge_patch,
#             negative_mask_patch,
#             soft_negative_mask_patch,
#             area_ref_patch
#         ])
#
#         return (
#             torch.from_numpy(feature_patch),  # (F, H, W)
#             torch.from_numpy(mask_patch),  # (1, H, W)
#             torch.from_numpy(negative_mask_patch),
#             torch.from_numpy(soft_negative_mask_patch),
#             torch.from_numpy(edge_patch),
#             torch.from_numpy(area_ref_patch),
#         )


class ValidPatchSliceDataset(Dataset):

    def __init__(self, volume, mask_volume, feature_volume,
                 negative_volume_label, softnega,
                 patch_size=(256,256), threshold=0.7,
                 num_samples=1000, thickness=2):

        self.thickness = thickness
        self.patch_size = patch_size

        # ---------- 全部转 torch ----------
        self.volume = torch.as_tensor(volume).float()
        self.mask_volume = torch.as_tensor(mask_volume).float()
        self.negative_mask_volume = torch.as_tensor(negative_volume_label).float()
        self.softnega = torch.as_tensor(softnega).float()
        self.feature = torch.as_tensor(feature_volume).float()

        D,H,W = self.volume.shape
        ph,pw = patch_size

        # ---------- 构造 negative ring ----------
        self.edge,eroded,_,negative_ring = build_dilated_rings(
            self.mask_volume.unsqueeze(1),
            kernel_size=3,edge_size=4
        )
        negative_ring = negative_ring.squeeze(1)
        self.edge_ref, _ = projection_by_mean_diff_volume(
            self.feature,
            self.edge.squeeze(1),
            (self.negative_mask_volume > 0) | (negative_ring > 0)|(eroded.squeeze(1)>0)
        )
        # ---------- projection ----------
        self.area_ref,_ = projection_by_mean_diff_volume(
            self.feature,
            self.mask_volume,
            (self.negative_mask_volume>0)|(negative_ring>0)
        )

        # ---------- nz_coords ----------
        nz_coords = torch.nonzero(self.mask_volume>0, as_tuple=False)

        soft_coords = torch.nonzero(
            (self.negative_mask_volume>0) |
            (self.softnega>0),
            as_tuple=False
        )

        if len(soft_coords) > 0:

            n_extra = max(1,int(0.01*threshold*len(nz_coords)))
            n_extra = min(n_extra,len(soft_coords))

            idx = torch.randperm(len(soft_coords))[:n_extra]
            soft_sample = soft_coords[idx]

            nz_coords = torch.cat([nz_coords,soft_sample],dim=0)

        print("🔍 正在构建 patch 索引...")

        # ---------- 合法坐标 ----------
        z_min,z_max = thickness, D-thickness-1
        x_min,x_max = ph//2, H-ph//2-1
        y_min,y_max = pw//2, W-pw//2-1

        valid = nz_coords[
            (nz_coords[:,0]>=z_min)&(nz_coords[:,0]<=z_max)&
            (nz_coords[:,1]>=x_min)&(nz_coords[:,1]<=x_max)&
            (nz_coords[:,2]>=y_min)&(nz_coords[:,2]<=y_max)
        ]

        if len(valid)==0:
            raise ValueError("❌ 没有合法采样点")

        # ---------- 均匀采样 ----------
        idx = torch.linspace(
            0,len(valid)-1,
            steps=num_samples
        ).long()

        idx = idx[torch.randperm(len(idx))]
        sampled = valid[idx[:num_samples]]

        # ---------- patch 左上角 ----------
        z = sampled[:,0]
        x = (sampled[:,1]-ph//2).clamp(0,H-ph)
        y = (sampled[:,2]-pw//2).clamp(0,W-pw)

        self.samples = torch.stack([z,x,y],dim=1)

        print(f"✅ 构建完成，共采样 {len(self.samples)} 个 patch")


    def __len__(self):
        return len(self.samples)


    def __getitem__(self,idx):

        z,x,y = self.samples[idx]

        ph,pw = self.patch_size

        # ---------- image ----------
        img_patch = self.volume[
            z-self.thickness:z+self.thickness+1,
            x:x+ph,
            y:y+pw
        ]

        # ---------- feature ----------
        feature_patch = self.feature[
            z,
            :,
            x:x+ph,
            y:y+pw
        ]

        # ---------- mask ----------
        mask_patch = self.mask_volume[
            z:z+1,
            x:x+ph,
            y:y+pw
        ]
        mask_patch = (mask_patch>0.5).float()

        edge_patch = self.edge[
            z:z+1,0,
            x:x+ph,
            y:y+pw
        ]
        # edge_patch = get_edge_mask(mask_patch,edge_width=1)
        # if isinstance(edge_patch, np.ndarray):
        #     edge_patch = torch.from_numpy(edge_patch)

        negative_patch = self.negative_mask_volume[
            z:z+1,
            x:x+ph,
            y:y+pw
        ]

        soft_negative_patch = self.softnega[
            z:z+1,
            x:x+ph,
            y:y+pw
        ]

        area_ref_patch = self.area_ref[
            z:z+1,
            x:x+ph,
            y:y+pw
        ]
        edge_ref_patch = self.edge_ref[
            z:z+1,
            x:x+ph,
            y:y+pw
        ]
        # ---------- augmentation ----------
        patches = [
            img_patch,
            feature_patch,
            mask_patch,
            negative_patch,
            soft_negative_patch,
            edge_patch,
            area_ref_patch,
            edge_ref_patch
        ]

        patches = self.apply_random_flip_rotate(patches)
        return tuple(patches[1:])


    def apply_random_flip_rotate(self,patch_list):

        flip_mode = torch.randint(0,3,(1,)).item()
        rot_k = torch.randint(0,4,(1,)).item()
        flip_z0 = torch.randint(0,2,(1,)).item()

        out = []

        for i,patch in enumerate(patch_list):

            aug = patch

            if i==0 and flip_z0:
                aug = torch.flip(aug,[0])

            if flip_mode==1:
                aug = torch.flip(aug,[-1])
            elif flip_mode==2:
                aug = torch.flip(aug,[-2])

            if rot_k>0:
                aug = torch.rot90(aug,k=rot_k,dims=(-2,-1))

            out.append(aug)

        return out