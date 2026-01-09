#%%
import numpy as np
from tqdm import tqdm
import argparse
from scipy.ndimage import binary_closing,binary_erosion, binary_dilation, label as nd_label,gaussian_filter, distance_transform_edt
import tifffile as tiff
from utils import compute_statistical_mask,process_volume,local_contrast_normalize,local_standardize,filter_connected_regions_shape,intersect_regions,soften_center_mask_dilated
from skimage.transform import downscale_local_mean
import os

def setup_model(model_class, model_args=None, checkpoint_folder="checkpoints", model_name="unet_model.pt",
                device="cuda"):
    """
    创建或加载模型参数。
    - 若 checkpoint_folder 下存在 model_name 文件，则加载继续训练；
    - 否则新建模型并保存初始权重。

    参数:
        model_class: nn.Module 类，如 SimpleUNet
        model_args: 初始化模型的参数 dict（可为空）
        checkpoint_folder: 模型保存的文件夹路径
        model_name: 模型文件名
        device: 'cuda' 或 'cpu'

    返回:
        model: 加载/新建的模型
        ckpt_path: 模型参数保存路径
    """
    os.makedirs(checkpoint_folder, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_folder, model_name)

    model = model_class(**(model_args or {})).to(device)

    if os.path.exists(ckpt_path):
        print(f"🔄 检测到已有模型参数，正在加载: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print(f"🆕 未检测到已有模型，新建并保存初始参数到: {ckpt_path}")
        torch.save(model.state_dict(), ckpt_path)

    return model

def save_model(model, ckpt_path):
    """保存模型参数"""
    torch.save(model.state_dict(), ckpt_path)
    print(f"✅ 模型参数已保存: {ckpt_path}")
# def downsample_volume(volume, factor=2):
#     """
#     将3D volume 每个维度缩小 factor 倍，体积变为原来的1/(factor^3)
#     """
#     # assert all(s % factor == 0 for s in volume.shape), "shape 必须能整除 factor"
#     return downscale_local_mean(volume, (1, factor, factor))
def dilate_z_binary(volume,size=(3, 1, 1)):
    """
    使用3D结构元素在z方向膨胀
    """
    struct = np.ones((size), dtype=np.uint8)
    # struct[:,0,0] = 1
    return binary_dilation(volume, structure=struct).astype(volume.dtype)
def erode_z_binary(volume, size=(1, 3, 3)):
    """
    使用3D结构元素进行收缩
    参数:
        volume: 3D ndarray (Z, H, W)，二值体数据
        size: tuple/list，结构元素大小，例如 (1,3,3) 表示只在xy收缩
    返回:
        eroded: 3D ndarray，收缩后的体
    """
    struct = np.ones(size, dtype=np.uint8)
    return binary_erosion(volume, structure=struct).astype(volume.dtype)
# def build_gaussian_contrast_mask(temp_base, sigma=3):
#     """
#     temp_base: (D,H,W) 0/1 ndarray
#     sigma: 高斯模糊尺度（以 voxel 为单位）
#     """
#     temp_base = temp_base.astype(np.float32)
#
#     # 1. 高斯模糊
#     temp_base_blur = gaussian_filter(temp_base, sigma=sigma)
#
#     # 2. 全局均值
#     mean_val = 0.5*temp_base_blur.max()
#
#     # 3. 构建 mask
#     mask = 0.01*(1 - temp_base_blur/mean_val).clip(min=0)
#
#     return mask

def build_distance_mask(temp_base, R=30, mode="sigmoid"):
    dist = distance_transform_edt(1 - temp_base)

    if mode == "linear":
        mask = (dist / R).clip(0, 1)

    elif mode == "gaussian":
        mask = 1 - np.exp(-(dist**2) / (2 * R**2))

    elif mode == "sigmoid":
        k = R / 6
        mask = (1 / (1 + np.exp(-(dist - R) / k))-0.01).clip(min=0)

    else:
        raise ValueError("Unknown mode")

    return 0.1*mask


parser = argparse.ArgumentParser(description="Unet Training Script")
parser.add_argument("--interation_idx", type=int, default=0, help="Model type")
parser.add_argument("--filer_method", type=int, default=2, help="Model type")
parser.add_argument("--z_threshold", type=int, default=10, help="Model type")
parser.add_argument("--patch_scale", type=int, default=140, help="Model type")

parser.add_argument("--raw_name", type=str, default='jurkat_em_s3', help="Model type")
parser.add_argument("--mask_name", type=str, default='label_jurkat_er_30', help="Model type")
parser.add_argument("--area_coef", type=float, default=1, help="Model type")
parser.add_argument("--edge_coef", type=float, default=0.5, help="Model type")
parser.add_argument("--iou_thresh", type=float, default=0.6, help="Model type")
parser.add_argument("--threshold", type=float, default=0.5, help="Model type")
parser.add_argument("--negative_threshold", type=float, default=1, help="Model type")
parser.add_argument("--low_weight_coeff", type=float, default=10, help="Model type")
parser.add_argument("--sparsity_weight", type=float, default=0.0, help="Model type")

args = parser.parse_args()
patch_scale=args.patch_scale
raw_name=args.raw_name
mask_name=args.mask_name
interation_idx = args.interation_idx
area_coef = args.area_coef
edge_coef = args.edge_coef
filer_method = args.filer_method
z_threshold= args.z_threshold
iou_thresh= args.iou_thresh
threshold=args.threshold
negative_threshold=args.negative_threshold
sparsity_weight=args.sparsity_weight

# area_coef=1.0
# edge_coef=0.5
# raw_name='11416_raw'
# filer_method=2
#
# interation_idx=1
# mask_name='11416_vescilemask'
# z_threshold=40
# threshold=0.95
# iou_thresh=0.8
# patch_scale=140 # 4倍数
# negative_threshold=20
# low_weight_coeff=0
# sparsity_weight=0.5

# raw_name='11416_raw'
# mask_name='11416_mitomask'
# area_coef=1.0
# edge_coef=0.5
# filer_method=2
# z_threshold=1

patchsize=(patch_scale,patch_scale)
low_weight_coeff=args.low_weight_coeff
repeated_epoch=60
# filtering

# mask2unit8
# msk_threshold=0.7-interation_idx*0.04
msk_threshold=0.9
#volume self
vol0 = tiff.imread(f'inputdata/{raw_name}.tif')
# vol0 = tiff.imread('inputdata/main_p_c_raw.tif')

# vol0 = np.transpose(tiff.imread('inputdata/janelia_raw.tif'), (1, 0, 2))

vol1=process_volume(vol0)
test_volume=(local_contrast_normalize(vol0))
# test_volume=vol1

# mask_name='11416_mitomask'

# mask_name='main_p_c_mask_mito'


# test_volume_label_base=dilate_z_binary(test_volume_label_base,size=(3, 1, 1))
base0 = tiff.imread(f'inputdata/{mask_name}.tif')
base0 = (base0 > 0).astype(np.uint8)
if interation_idx==0:
    test_volume_label = tiff.imread(f'inputdata/{mask_name}.tif')
    test_volume_label_base = tiff.imread(f'inputdata/{mask_name}.tif')
    test_volume_label_base = (test_volume_label_base > 0).astype(np.uint8)
    # test_volume_label=dilate_z_binary(test_volume_label,size=(3, 1, 1))
else:
    test_volume_label_base = tiff.imread(f'{mask_name}/{mask_name}_{interation_idx-1}_base.tif')
    test_volume_label_base = (test_volume_label_base > 0).astype(np.uint8)
    test_volume_label = tiff.imread(f'{mask_name}/{mask_name}_{interation_idx-1}.tif')
    # test_volume_label = erode_z_binary(test_volume_label,size=(1, 5, 5))
    # test_volume_label=dilate_z_binary(test_volume_label,size=(1, 5, 5))

test_volume_label_new=filter_connected_regions_shape(test_volume_label_base, base0, threshold=threshold,min_ratio=0.8,max_height=z_threshold)
test_volume_label_new[base0>0]=1
# tiff.imwrite(f'outfig/remain_mask.tif', 255*test_volume_label_new)
# tiff.imwrite(f'outfig/old_mask.tif', 255*test_volume_label)
# test_volume_label=dilate_z_binary(test_volume_label,size=(3, 1, 1))

mask_path = f'inputdata/negative_{mask_name}.tif'
if os.path.exists(mask_path):
    nega_test_volume_label = tiff.imread(mask_path)
    nega_test_volume_label = dilate_z_binary(nega_test_volume_label, size=(1, 1, 1))
else:
    mask_path = f'inputdata/negative_{raw_name}.tif'
    if os.path.exists(mask_path):
        nega_test_volume_label = tiff.imread(mask_path)
        nega_test_volume_label = dilate_z_binary(nega_test_volume_label, size=(1, 1, 1))
    else:
        nega_test_volume_label = np.zeros_like(test_volume_label, dtype=np.uint8)

nega_test_volume_label=(nega_test_volume_label>0).astype(np.uint8)

# soft nega
# softnega=(test_volume_label >0).astype(np.uint8)
# softnega[test_volume_label_new>0] = 0
softnega=build_distance_mask(test_volume_label_base, R=low_weight_coeff)

# 找出 mask_volume 在 z 维度上的有效切片
volume=test_volume

#%%
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
        eroded  = ndimage.binary_erosion(mask, structure=struct_xy, iterations=edge_width)
    else:
        # 2D 情况，正常处理
        dilated = ndimage.binary_dilation(mask, iterations=edge_width)
        eroded  = ndimage.binary_erosion(mask, iterations=edge_width)

    edge = np.logical_xor(dilated, eroded).astype(np.float32)
    return edge

class ValidPatchSliceDataset(Dataset):
    def __init__(self, volume, mask_volume, negative_volume_label,softnega,patch_size=(256,256), threshold=0.7, num_samples=1000, max_trials=5000,thickness=2):
        """
        volume: np.ndarray, shape (D, H, W), 原始图像
        mask_volume: np.ndarray, shape (D, H, W), 对应的 mask
        """
        self.thickness=thickness
        self.volume = volume.astype(np.float32)
        self.mask_volume = mask_volume.astype(np.float32)
        self.negative_mask_volume = negative_volume_label.astype(np.float32)
        self.softnega=softnega.astype(np.float32)
        # Step 1: 对第一个维度求和，得到一个二维图像
        mask_2d = np.sum(self.mask_volume, axis=0)  # shape: (H, W)

        self.patch_size = patch_size
        # self.threshold = threshold
        self.samples = []

        D, H, W = self.volume.shape
        nz_coords = np.argwhere(self.mask_volume  > 0)  # (N,3), 每行(z,x,y)
        soft_coords = np.argwhere((self.negative_mask_volume > 0)| (self.softnega > 0))
        # soft_coords = np.argwhere((self.negative_mask_volume > 0))
        if len(soft_coords) > 0:
            n_extra = max(1, int(0.01 *threshold* len(nz_coords)))  # 取 nz_coords 数量的 10%
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
        img_patch = self.volume[z-self.thickness:z+self.thickness+1, x:x+self.patch_size[0], y:y+self.patch_size[1]]  # shape [3, H, W]
        # 取中心切片对应 mask
        mask_patch = self.mask_volume[z:z+1, x:x+self.patch_size[0], y:y+self.patch_size[1]]  # shape [H, W]
        mask_patch = (mask_patch > 0.5).astype(np.float32)  # 二值化（如果原本不是）

        edge_patch = get_edge_mask(mask_patch, edge_width=1)

        negative_mask_patch=self.negative_mask_volume[z:z+1, x:x+self.patch_size[0], y:y+self.patch_size[1]]
        soft_negative_mask_patch = self.softnega[z:z+1, x:x + self.patch_size[0], y:y + self.patch_size[1]]

        img_patch, mask_patch, edge_patch,negative_mask_patch, soft_negative_mask_patch = apply_random_flip_rotate([
            img_patch, mask_patch, edge_patch,negative_mask_patch, soft_negative_mask_patch
        ])

        # img_patch = (img_patch - img_patch.min()) / (img_patch.max() - img_patch.min() + 1e-6)
        _, feats_stack = extract_2d_features_from_patch(
            img_patch,                 # [Z,H,W] 例如 thickness=1 -> Z=3
            aggregate_mode="gaussian",
            sigma_z=0.8,
            denoise_tv=0.05,          # 可设 0 关闭
            sigmas_gauss=(1.0,2.0,4.0),
            sigmas_hessian=(1.0,2.0,4.0),
            win_local_stats=9,
            st_sigma=1.0,
        )



        return torch.tensor(feats_stack), torch.tensor(mask_patch) ,torch.tensor(negative_mask_patch) ,torch.tensor(soft_negative_mask_patch),torch.tensor(edge_patch) # (3, H, W), (1, H, W)

# edge_patch = get_edge_mask(test_volume_label, edge_width=1)
# tiff.imwrite(f'{mask_name}/edge0_{interation_idx}.tif', edge_patch)
# ============================
line_coef=1.2*(get_edge_mask(test_volume_label_new).sum())/(test_volume_label_new.sum())
print(line_coef)
thickness=2
# dataset = ValidPatchSliceDataset(volume,test_volume_label_new,nega_test_volume_label,softnega, patch_size=patchsize, threshold=negative_threshold, num_samples=300,thickness=thickness)
# sampler = RandomSampler(dataset, replacement=True, num_samples=300)
# loader = DataLoader(dataset, batch_size=12, sampler=sampler)

#%%
class AttentionGate(nn.Module):
    """
    g: gating（来自 decoder 的上采样特征）
    x: skip（来自 encoder 的特征）
    F_g: gating 通道数
    F_l: skip 通道数
    F_int: 中间通道数（压缩比例）
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 将 g 映射到中间通道；必要时插值到 x 的空间分辨率
        g1 = self.W_g(g)
        if g1.shape[-2:] != x.shape[-2:]:
            g1 = F.interpolate(g1, size=x.shape[-2:], mode='bilinear', align_corners=False)

        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)                   # 注意力权重 (B,1,H,W)
        out = x * psi                         # 对 skip 做加权
        return out
class MultiKernelConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(3, 5, 15)):
        super().__init__()
        padding = [(k // 2) for k in kernel_sizes]
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=p),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for k, p in zip(kernel_sizes, padding)
        ])
        self.merge = nn.Sequential(
            nn.Conv2d(out_channels * len(kernel_sizes), out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        concat = torch.cat(branch_outputs, dim=1)
        out = self.merge(concat)
        return out
class MultiKernelUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = MultiKernelConvBlock(in_channels, 64)
        self.enc2 = MultiKernelConvBlock(64, 128)
        self.enc3 = MultiKernelConvBlock(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = MultiKernelConvBlock(256, 128)   # cat(128, 128)
        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = MultiKernelConvBlock(128, 64)    # cat(64, 64)

        # ★ Attention Gates：对 e2, e1 进行筛选
        self.ag2 = AttentionGate(F_g=128, F_l=128, F_int=64)  # 对应 d2 与 e2
        self.ag1 = AttentionGate(F_g=64,  F_l=64,  F_int=32)  # 对应 d1 与 e1

        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # x_pad, original_hw = pad_to_multiple(x, multiple=4)

        e1 = self.enc1(x)               # (B, 64,  H,   W)
        e2 = self.enc2(self.pool(e1))   # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool(e2))   # (B, 256, H/4, W/4)

        d2 = self.up2(e3)               # (B, 128, H/2, W/2)
        # e2_att = self.ag2(d2, e2)       # 注意力筛选后的 skip
        # d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)               # (B, 64, H, W)

        # e1_att = self.ag1(d1, e1)       # 注意力筛选后的 skip
        # d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        out = self.final(d1)

        # H, W = original_hw
        # out = out[:, :, :H, :W]
        return out

from model import SwinUNetLike


model = MultiKernelUNet(in_channels=2*thickness+31, out_channels=1).cuda()
if interation_idx==0:
    model = MultiKernelUNet(in_channels=2*thickness+31, out_channels=1+1).cuda()
    # model = ExpandTransformerUNet(in_channels=2 * thickness + 31, out_channels=1).cuda()
    # model = SwinUNetLike(in_channels=2*thickness+31, out_channels=2).cuda()

else:
    model = setup_model(MultiKernelUNet,model_args={"in_channels": 2 * thickness + 31, "out_channels": 1+1}, checkpoint_folder=mask_name, model_name=f'model_{interation_idx-1}.pt')
    # model = setup_model(ExpandTransformerUNet, model_args={"in_channels": 2 * thickness + 31, "out_channels": 1}, checkpoint_folder=mask_name, model_name=f'model_{interation_idx - 1}.pt')
    # model = setup_model(SwinUNetLike, model_args={"in_channels": 2 * thickness + 31, "out_channels": 1}, checkpoint_folder=mask_name, model_name=f'model_{interation_idx - 1}.pt')


from Loss_func import region_consistency_loss, region_contrast_loss,masked_soft_bce_loss,smoothness_loss,edge_local_bce_loss

def total_loss_fn(pred, target, input_img, negative_label, softnega,edge_mask, model,
                  bce_weight=10.0, corr_weight=0.1, smooth_weight=0.1, sparsity_weight=0.01,
                  l1_weight=0.05, high_weight=1.0, low_weight=0.5, area_coef=1 ,edge_coef=0 ,thickness=5):

    _,_, H, W = input_img.shape

    bce = masked_soft_bce_loss(pred[:, (0):(1), :, :], target, negative_target=negative_label, softnega=softnega,
                               high_weight=high_weight, low_weight=low_weight,kernel_size=3,edge_size=4)

    # contrast = correlation_loss(pred, input_img[:, (thickness):( thickness + 1), :, :])
    # contrast = smoothness_loss(pred, input_img, slice_idx=thickness)
    smooth = smoothness_loss(pred[:, (0):(1), :, :],  input_img[:, (thickness):( thickness + 1), :, :])

    area_correlation = region_consistency_loss(pred[:, (1):(2), :, :], input_img[:,  (2*thickness+30):( 2*thickness+31), :, :])
    areea_contrast = 0.1* region_contrast_loss(pred[:, (1):(2), :, :], input_img[:,  (2*thickness+30):( 2*thickness+31), :, :])
    edge_correlation = region_consistency_loss(pred[:, (0):(1), :, :], input_img[:,  (thickness):( thickness+1), :, :])
    edge_contrast = 0.1* region_contrast_loss(pred[:, (0):(1), :, :], input_img[:,  (thickness):( thickness+1), :, :])

    correlation = area_coef*area_correlation+edge_coef*edge_correlation
    contrast = area_coef*areea_contrast+edge_coef*edge_contrast


    # edge_loss = edge_local_bce_loss(pred[:, 1:2, :, :], edge_mask, radius=3)
    edge_loss = masked_soft_bce_loss(pred[:, (1):(2), :, :], edge_mask, negative_target=negative_label, softnega=softnega,
                         high_weight=high_weight, low_weight=low_weight,kernel_size=3,edge_size=4)
    # edge_loss = smooth
    # contrast = region_contrast_loss(pred, input_img, slice_idx=2 * thickness + 1)
    # 🔹 L1 on model weights
    l1 = 0.0
    for p in model.parameters():
        l1 += torch.sum(torch.abs(p))
    l1 = l1 / sum(p.numel() for p in model.parameters())

    l1 += sparsity_weight * torch.sigmoid(pred[:, (0):(2), :, :]).mean()

    total = (bce_weight *area_coef* bce +
             corr_weight * correlation +
             corr_weight * contrast +
             smooth_weight * smooth +
             bce_weight* edge_coef * edge_loss +
             l1_weight * l1).mean()

    loss_dict = {
        "bce": bce.mean().item(),
        "correlation": correlation.mean().item(),
        "contrast": contrast.mean().item(),
        "smooth": smooth.item(),
        "edge": edge_loss.item(),
        "l1": l1.item()
    }

    return total, loss_dict

# dataset_edge = EdgeDataset(dataset)
# loader = DataLoader(dataset_edge, batch_size=8, shuffle=True)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(repeated_epoch):
    model.train()
    total_loss = 0
    loss_log = {"bce": 0.0, "correlation": 0.0, "smooth": 0.0,"l1":0.0, "contrast":0.0,"edge":0.0}
    batch_count = 0

    dataset = ValidPatchSliceDataset(volume, test_volume_label_new, nega_test_volume_label, softnega,
                                     patch_size=patchsize, threshold=negative_threshold, num_samples=600,
                                     thickness=thickness)
    loader = DataLoader(dataset, batch_size=12, shuffle=True, drop_last=True)

    for x, y,z,softnega_p,edge in loader:
        x, y,z,softnega_p,edge = x.cuda(), y.cuda(),z.cuda(),softnega_p.cuda(),edge.cuda()
        pred = model(x)

        loss, loss_dict = total_loss_fn(pred, y, x,z,softnega_p,edge,model,low_weight=low_weight_coeff,thickness=thickness,area_coef=area_coef ,edge_coef=edge_coef,sparsity_weight=sparsity_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for k in loss_log:
            loss_log[k] += loss_dict[k]

        batch_count += 1

    # 计算每项 loss 的平均值
    avg_loss_log = {k: v / batch_count for k, v in loss_log.items()}
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{now} Epoch {epoch} - Total: {total_loss:.4f} | "
          f"BCE: {avg_loss_log['bce']:.4f}, correlation: {avg_loss_log['correlation']:.4f},contrast: {avg_loss_log['contrast']:.4f} Smooth: {avg_loss_log['smooth']:.4f}, l1: {avg_loss_log['l1']:.4f}, edge: {avg_loss_log['edge']:.4f}")

#%%
def pad_to_multiple_of(volume, multiple=4):
    """
    将 volume 的 H 和 W 补零到 multiple 的倍数
    volume: [D, H, W]
    return: padded_volume, pad_info
    """
    D, H, W = volume.shape
    new_H = ((H + multiple - 1) // multiple) * multiple
    new_W = ((W + multiple - 1) // multiple) * multiple

    pad_top = (new_H - H) // 2
    pad_bottom = new_H - H - pad_top
    pad_left = (new_W - W) // 2
    pad_right = new_W - W - pad_left

    padded = np.pad(volume,
                    ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='constant', constant_values=0)

    pad_info = {
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right
    }

    return padded, pad_info
def unpad_volume(padded_volume, pad_info):
    """
    还原为原始尺寸
    """
    return padded_volume[:,
                         pad_info["pad_top"]:-pad_info["pad_bottom"] or None,
                         pad_info["pad_left"]:-pad_info["pad_right"] or None]
def infer_volume_edges(volume_np, model, threshold=0.5, patch_size=(128, 128), stride=(64, 64),thickness=2):
    """
    滑窗方式推理整个 volume，支持二维 patch 尺寸和步长。

    Args:
        volume_np: [D, H, W], 输入 3D volume，float32，归一化到 [0,1]
        model: 训练好的 UNet 模型，输入 [1, 3, H, W] → 输出 [1, 1, H, W]
        threshold: float，边缘概率阈值
        patch_size: tuple(int, int)，patch 高度和宽度
        stride: tuple(int, int)，滑窗的垂直和水平步长

    Returns:
        edge_volume: [D, H, W]，uint8，0/1 mask
    """
    patch_h, patch_w = patch_size
    stride_h, stride_w = stride

    padded_volume, pad_info = pad_to_multiple_of(volume_np, multiple=64)

    D, H, W = padded_volume.shape
    edge_volume = np.zeros((D, H, W), dtype=np.uint8)

    model.eval()
    model.cuda()

    with torch.no_grad():
        for z in tqdm(range(thickness, D - thickness), desc="Predicting edges per slice"):
            slice_img = padded_volume[(z - thickness):(z + thickness+1)]  # [3, H, W]
            slice_pred = np.zeros((H, W), dtype=np.float32)
            slice_count = np.zeros((H, W), dtype=np.float32)

            for i in range(0, H - patch_h + 1, stride_h):
                for j in range(0, W - patch_w + 1, stride_w):
                    patch = slice_img[:, i:i + patch_h, j:j + patch_w]
                    _, feats_stack = extract_2d_features_from_patch(
                        patch,  # [Z,H,W] 例如 thickness=1 -> Z=3
                        aggregate_mode="gaussian",
                        sigma_z=0.8,
                        denoise_tv=0.05,  # 可设 0 关闭
                        sigmas_gauss=(1.0, 2.0, 4.0),
                        sigmas_hessian=(1.0, 2.0, 4.0),
                        win_local_stats=9,
                        st_sigma=1.0,
                    )
                    patch_tensor = torch.from_numpy(feats_stack).unsqueeze(0).float().cuda()  # [1, 3, patch_h, patch_w]
                    pred = model(patch_tensor)  # [1, 1, patch_h, patch_w]
                    pred_prob = torch.sigmoid(pred).squeeze().cpu().numpy()  # [patch_h, patch_w]

                    slice_pred[i:i + patch_h, j:j + patch_w] += pred_prob
                    slice_count[i:i + patch_h, j:j + patch_w] += 1

                    # slice_pred[i:i + patch_h, j:j + patch_w] = np.maximum(
                    #     slice_pred[i:i + patch_h, j:j + patch_w],
                    #     (pred_prob > threshold).astype(np.uint8)
                    # )

            # 平均融合预测结果
            # slice_count[slice_count == 0] = 1
            slice_avg = slice_pred / slice_count
            edge_mask = (slice_avg > threshold).astype(np.uint8)

            # edge_mask=slice_pred

            edge_volume[z] = edge_mask

    restored = unpad_volume(edge_volume, pad_info)
    return restored
def infer_volume_edges_whole(volume_np, model, thickness=2):
    """
    volume_np: [D, H, W], float32, normalized to [0, 1]
    model: model(input: [1, 1, H, W]) → pred [1, 1, H, W]
    Returns:
        edge_volume: [D, H, W], uint8 0/1 mask
    """
    padded_volume, pad_info = pad_to_multiple_of(volume_np, multiple=4)
    D, H, W = padded_volume.shape

    pred_volume = np.zeros((D, H, W), dtype=np.float32)
    edge_volume = np.zeros((D, H, W), dtype=np.float32)
    count_volume = np.zeros((D, H, W), dtype=np.float32)

    model.eval()
    model.cuda()

    with torch.no_grad():
        for z in tqdm(range(thickness, D - thickness), desc="Predicting edges per slice"):
            slice_img = padded_volume[(z - thickness):(z + thickness + 1)]  # [3, H, W]
            # slice_img = padded_volume[(z-1):(z+2)]  # [H, W]
            _, feats_stack = extract_2d_features_from_patch(
                slice_img,  # [Z,H,W] 例如 thickness=1 -> Z=3
                aggregate_mode="gaussian",
                sigma_z=0.8,
                denoise_tv=0.05,  # 可设 0 关闭
                sigmas_gauss=(1.0, 2.0, 4.0),
                sigmas_hessian=(1.0, 2.0, 4.0),
                win_local_stats=9,
                st_sigma=1.0,
            )
            input_tensor = torch.from_numpy(feats_stack).unsqueeze(0).float().cuda()  # [1, 1, H, W]

            pred = model(input_tensor)  # [1, 1, H, W]
            pred_prob_stack = torch.sigmoid(pred).squeeze(0).cpu().numpy()  # [2*thickness+1, H, W]

            # === 对齐并累积回 volume ===
            z_start = z - 0
            z_end = z + 0 + 1
            edge_volume[z_start:z_end] = pred_prob_stack[1:2,:,:]
            pred_volume[z_start:z_end] = pred_prob_stack[0:1,:,:]
            count_volume[z_start:z_end] += 1

            # === 平均重叠区域 ===
        count_volume[count_volume == 0] = 1
        edge_volume /= count_volume
        pred_volume /= count_volume
        # === 去除 padding 并平滑 ===
        restored = unpad_volume(pred_volume, pad_info)
        restored2 = unpad_volume(edge_volume, pad_info)
        blurred = gaussian_filter(restored, sigma=(1, 3, 3), mode="reflect")

    return blurred,restored2

# edge_vol = infer_volume_edges(volume, model, patch_size=patchsize,threshold=0.5,thickness=thickness)
edge_vol,edge_Line = infer_volume_edges_whole(volume, model, thickness=thickness)

# 保存结果
# np.save("edge_volume.npy", edge_vol)

# 可视化中间层
# import matplotlib.pyplot as plt
# 中间切片索引
z = volume.shape[0] // 2

# 提取原图和边缘mask
img = volume[z]               # [H, W]
mask = edge_vol[z]            # [H, W]

# 归一化原图（确保 ∈ [0,1]）
if img.max() > 1:
    img = img / 255.0

# 构建 RGB 叠加图
overlay = np.stack([img, img, img], axis=-1)  # → [H, W, 3]
overlay[mask == 1] = [1.0, 0.0, 0.0]           # 红色边缘

# 显示并保存叠加图
# plt.imshow(mask)
# plt.title("Edge Overlay")
# plt.axis("off")
# plt.savefig("cell_edges_overlay.png", dpi=300, bbox_inches='tight', pad_inches=0)
# plt.show()


#%%
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
        mask1_n,   # G: mask1
        mask2_n    # B: mask2
    ], axis=-1)  # [D, H, W, 3]

    # 转为 8-bit
    rgb_uint8 = (rgb * 255).astype(np.uint8)

    # 保存为多帧 RGB tiff
    tiff.imwrite(path, rgb_uint8, photometric='rgb')

    print(f"✅ 保存成功：{path}")
def save_multichannel_tiff(volume, mask1, mask2, path="multichannel.tiff"):
    """
    保存为多通道灰度 TIFF，每通道可在 ImageJ 中独立查看。
    volume, mask1, mask2: [D, H, W]，归一化到 [0, 1]
    """
    volume = volume.astype(np.float32)
    mask1 = mask1.astype(np.float32)
    mask2 = mask2.astype(np.float32)

    # Stack as [D, C, H, W]
    stack = np.stack([volume, mask1, mask2], axis=1)  # shape: [D, 3, H, W]

    # 转成 uint8（ImageJ 显示友好）
    stack = (stack * 255).clip(0, 255).astype(np.uint8)

    # 保存为 multi-channel，ImageJ 可识别
    tiff.imwrite(
        path,
        stack,
        photometric='minisblack',
        metadata={'axes': 'ZCYX'},  # 通道顺序 ZCYX = depth, channel, height, width
        planarconfig='separate'
    )

    print(f"✅ 已保存为多通道 TIFF: {path}")

# test_volume_label0 = tiff.imread(f'inputdata/{mask_name}.tif')
# test_volume_label0=(test_volume_label0 >0).astype(np.uint8)
# test_volume_label0 = expand_mask_3d(test_volume_label0, radius=2)

if not os.path.exists(mask_name):
    os.makedirs(mask_name)
# from edge_extract import get_edge_region
from edge_extract import get_edge_region,filter_edge_area_by_perimeter_fast,filter_edge_area_by_bbox_iou_2d_vectorized

def shrink_expand_xy(volume, size=2):
    """
    对3D二值体素数据仅在XY平面进行“缩小后扩大”（腐蚀→膨胀）

    参数:
        volume: np.ndarray, 0/1 体数据 [Z,H,W]
        size: int, XY平面的核半径（越大变化越强）
    返回:
        np.ndarray, 处理后的体
    """
    # 结构元素只在XY方向有效，Z方向厚度为1
    struct = np.ones((1, size, size), dtype=np.uint8)
    eroded = binary_erosion(volume, structure=struct)
    dilated = binary_dilation(eroded, structure=struct)
    return dilated.astype(np.uint8)

edge_new = edge_vol
thresh_value = np.percentile(edge_new, 100* msk_threshold)

if filer_method==0:
    edge_area = get_edge_region(edge_Line)
    vol010 = ((edge_new >= max(thresh_value, 0.5)) & (edge_area > 0.5)).astype(np.uint8)
elif filer_method==1:
    edge_area = get_edge_region(edge_Line)
    vol010 = intersect_regions((edge_area > 0.5),(edge_new >= max(thresh_value, 0.5)),overlap_ratio=0.01)
elif filer_method==2:
    vol010 = (edge_new >= max(thresh_value, 0.5))


# vol01=keep_connected_regions_inside_edge(vol010, edge_Line)
# vol01=shrink_expand_xy(vol010,size=5)
# volume_label0=shrink_expand_xy(test_volume_label_base,size=5)
vol01=(vol010)
volume_label0=(test_volume_label_base)

volume_label0=(volume_label0>0).astype(np.uint8)

test_volume_label_shape=filter_connected_regions_shape( vol01, base0, threshold=threshold,min_ratio=0.8,max_height=z_threshold)
test_volume_label_new = filter_edge_area_by_bbox_iou_2d_vectorized((edge_Line > 0.5), test_volume_label_shape, iou_thresh=iou_thresh,line_fill_thresh=line_coef)

# test_volume_label_shape = filter_edge_area_by_bbox_iou_2d_vectorized((edge_Line > 0.5), vol01, iou_thresh=iou_thresh,line_fill_thresh=0.9,method=0)
# test_volume_label_new=filter_connected_regions_shape( test_volume_label_shape, base0, threshold=threshold,min_ratio=0.8,max_height=z_threshold)

test_volume_label_save = 1.0 * test_volume_label_new + test_volume_label_base
test_volume_label_save = np.clip(test_volume_label_save, 0, 1.0)
test_volume_label_save[nega_test_volume_label>0] = 0

# tiff.imwrite(f'{mask_name}/remain_mask_{interation_idx}.tif', 255*test_volume_label_new)
# save_volume_with_masks_as_rgb_tiff(volume, edge_vol, test_volume_label_save, f'{mask_name}/remain_mask_{interation_idx}.tif')
save_volume_with_masks_as_rgb_tiff(volume, edge_vol, base0, f"{mask_name}/volume_mask_pred_{interation_idx}.tiff")

# save_volume_with_masks_as_rgb_tiff(volume, vol01, volume_with_edge, f"{mask_name}/volume_mask_pred_{interation_idx}.tiff")


test_volume_label_save=test_volume_label_save.astype(np.uint8)

# savemat(f"{mask_name}.mat", {"vol0": vol01})
# tiff.imwrite(f'inputdata/{mask_name}_{interation_idx}.tif', vol01)
tiff.imwrite(f'{mask_name}/{mask_name}_{interation_idx}.tif', vol01)
tiff.imwrite(f'{mask_name}/{mask_name}_{interation_idx}_base.tif', test_volume_label_save)

# tiff.imwrite(f'inputdata/{mask_name}_{interation_idx}.tif', vol01)

# tiff.imwrite('mask.tif', vol01)
# tiff.imwrite('base.tif', volume_label0)
# tiff.imwrite(f'{mask_name}/edge_{interation_idx}.tif', (edge_Line))

# tiff.imwrite(f'{mask_name}/{mask_name}2_{interation_idx}.tif', 255*vol01)
save_model(model, f'{mask_name}/model_{interation_idx}.pt')

