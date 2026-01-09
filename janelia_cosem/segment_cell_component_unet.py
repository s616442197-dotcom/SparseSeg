#%%
import numpy as np
from tqdm import tqdm
import argparse
from scipy.ndimage import binary_closing,binary_erosion, binary_dilation, label as nd_label,gaussian_filter
import tifffile as tiff
from utils import compute_statistical_mask,process_volume,local_contrast_normalize,local_standardize,filter_connected_regions_shape
from skimage.transform import downscale_local_mean
from skimage.measure import block_reduce
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
def downsample_volume(volume, factor=2):
    """
    将3D volume 每个维度缩小 factor 倍，体积变为原来的1/(factor^3)
    """
    # assert all(s % factor == 0 for s in volume.shape), "shape 必须能整除 factor"
    return downscale_local_mean(volume, (1, factor, factor))
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

parser = argparse.ArgumentParser(description="Unet Training Script")
parser.add_argument("--interation_idx", type=int, default=0, help="Model type")
parser.add_argument("--raw_name", type=str, default='main_raw', help="Model type")
parser.add_argument("--mask_name", type=str, default='main_mask_mito', help="Model type")
args = parser.parse_args()
raw_name=args.raw_name
mask_name=args.mask_name
interation_idx = args.interation_idx
patchsize=(140,140)
low_weight_coeff=10
repeated_epoch=50
# filtering
threshold=0.8
z_threshold=20
# mask2unit8
msk_threshold=0.6
#volume self
vol0 = tiff.imread(f'inputdata/{raw_name}.tif')
# vol0 = tiff.imread('inputdata/main_p_c_raw.tif')

# vol0 = np.transpose(tiff.imread('inputdata/janelia_raw.tif'), (1, 0, 2))

vol1=process_volume(vol0)
test_volume=(local_contrast_normalize(vol0))
# test_volume=vol1

# mask_name='11416_mitomask'

# mask_name='main_p_c_mask_mito'

test_volume_label_base = tiff.imread(f'inputdata/{mask_name}.tif')
# test_volume_label_base=dilate_z_binary(test_volume_label_base,size=(3, 1, 1))
if interation_idx==0:
    test_volume_label = tiff.imread(f'inputdata/{mask_name}.tif')
    test_volume_label=dilate_z_binary(test_volume_label,size=(3, 1, 1))
else:
    test_volume_label = tiff.imread(f'inputdata/{mask_name}_{interation_idx-1}.tif')
    # test_volume_label = erode_z_binary(test_volume_label,size=(1, 5, 5))
    # test_volume_label=dilate_z_binary(test_volume_label,size=(1, 5, 5))

test_volume_label_new=filter_connected_regions_shape(test_volume_label, test_volume_label_base, threshold=threshold,z_threshold=z_threshold,min_ratio=0.5,sim_thresh=0.5)
test_volume_label_new[test_volume_label_base>0]=1
tiff.imwrite(f'outfig/remain_mask.tif', 255*test_volume_label_new)
tiff.imwrite(f'outfig/old_mask.tif', 255*test_volume_label)
# test_volume_label=dilate_z_binary(test_volume_label,size=(3, 1, 1))

mask_path = f'inputdata/negative_{mask_name}.tif'
if os.path.exists(mask_path):
    nega_test_volume_label = tiff.imread(mask_path)
    nega_test_volume_label = dilate_z_binary(nega_test_volume_label, size=(5, 1, 1))
else:
    # 如果不存在，就用全 0（shape 与 test_volume_label 相同）
    nega_test_volume_label = np.zeros_like(test_volume_label, dtype=np.uint8)

test_volume_label0=(test_volume_label >0).astype(np.uint8)

test_volume_label=(test_volume_label_new >0).astype(np.uint8)
# test_volume_label=(test_volume_label >0).astype(np.uint8)
test_volume_label[nega_test_volume_label == 1] = 0
test_volume_label[test_volume_label_base>0]=1

nega_test_volume_label=(nega_test_volume_label>0).astype(np.uint8)
# soft nega
softnega=test_volume_label0
softnega[test_volume_label_new>0] = 0
# 找出 mask_volume 在 z 维度上的有效切片
valid_z = np.where(test_volume_label.sum(axis=(1, 2)) > 0)[0]
volume=test_volume
volume_label=test_volume_label
volume_pred=test_volume_label
negative_volume_label=nega_test_volume_label
D, H, W = volume.shape



#%%
# import numpy as np
# import matplotlib.pyplot as plt
#
# # ========== 输入数据 ==========
# # 示例：使用 volume[z]（电镜图层）
# # 如果是 PNG 文件，也可以使用 imageio.imread("path.png")
# img = volume[0]  # 或者你选 volume[z]
# plt.imsave("cellpose_seg_overlay.png", img)

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
# ============================
# 2. Dataset from volume
# ============================
def sample_nonzero_point(mask_volume, patch_size, thickness, seed=None):
    """
    从 mask_volume 中随机选取一个非零点 (z, x, y)，
    若超出范围则clip到边界。
    """
    rng = np.random.default_rng(seed)
    D, H, W = mask_volume.shape
    ph, pw = patch_size

    # 随机一个非零点
    nz_coords = np.argwhere(mask_volume != 0)  # (N,3), 每行(z,x,y)
    if nz_coords.size == 0:
        raise ValueError("mask_volume 全为0")
    z, x, y = nz_coords[rng.integers(len(nz_coords))]

    # 合法范围
    z_min, z_max = thickness, D - thickness
    x_min, x_max = 0, H - ph
    y_min, y_max = 0, W - pw

    # clip 到合法范围
    z = int(np.clip(z, z_min, z_max))
    # 限制在 valid_z 里，找最近的合法值
    x = int(np.clip(x, x_min, x_max))
    y = int(np.clip(y, y_min, y_max))

    return z, x, y
class ValidPatchSliceDataset(Dataset):
    def __init__(self, volume, mask_volume, negative_volume_label,softnega,patch_size=(256,256), threshold=0.7, num_samples=1000, max_trials=5000,thickness=2,valid_z=np.array([1,2])):
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
        self.threshold = threshold
        self.samples = []

        D, H, W = self.volume.shape
        # nz_coords = np.argwhere(self.mask_volume  > 0)  # (N,3), 每行(z,x,y)
        nz_coords = np.argwhere((self.mask_volume > 0) | (self.softnega > 0))

        # self.mask_volume[negative_volume_label == 1] = 0

        print("🔍 正在构建符合非零占比要求的 patch 索引...")
        attempts = 0
        while len(self.samples) < num_samples:
            success = False
            for _ in range(max_trials):
                z, x, y = nz_coords[np.random.randint(0,len(nz_coords))]

                # 合法范围
                z_min, z_max = thickness, D - thickness-1
                x_min, x_max = 0, H - patch_size[0]
                y_min, y_max = 0, W - patch_size[1]
                # clip 到合法范围
                z = int(np.clip(z, z_min, z_max))
                x = int(np.clip(x-patch_size[0]/2, x_min, x_max))
                y = int(np.clip(y-patch_size[1]/2, y_min, y_max))

                # z = sample_valid_z(valid_z, D, thickness)
                # # bounds=nonzero_bounds(mask_volume, z)
                # x = np.random.randint(0, H - patch_size[0])
                # y = np.random.randint(0, W - patch_size[1])
                patch = self.volume[z, x:x+patch_size[0], y:y+patch_size[1]]
                patch2 = self.mask_volume[z,x:x + patch_size[0], y:y + patch_size[1]]
                # print(z,x,y,patch2.size)
                # patch2 = mask_2d[x:x + patch_size[0], y:y + patch_size[1]]

                # Step 3: 计算绝对值大于 0.0001 的比例
                nonzero_ratio = np.count_nonzero(np.abs(patch) > 0.0001) / patch.size
                nonzero_ratio2 = np.count_nonzero(np.abs(patch2) > 0.0001) / patch2.size


                if nonzero_ratio >= threshold:
                    if nonzero_ratio2>=0.00:
                        self.samples.append((z, x, y))
                        success = True
                        # if nonzero_ratio2==0.00:
                            # print('exist')
                    break
            if not success:
                attempts += 1
                if attempts > max_trials:
                    print(f"⚠️ 尝试超过限制，当前采样数为 {len(self.samples)} / {num_samples}")
                    break

        print(f"✅ 构建完成，共采样 {len(self.samples)} 个有效 patch")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        z, x, y = self.samples[idx]

        # 取 3 张 slice 构成 3 通道图像
        img_patch = self.volume[z-self.thickness:z+self.thickness+1, x:x+self.patch_size[0], y:y+self.patch_size[1]]  # shape [3, H, W]
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
        # 取中心切片对应 mask
        mask_patch = self.mask_volume[z, x:x+self.patch_size[0], y:y+self.patch_size[1]]  # shape [H, W]
        mask_patch = (mask_patch > 0).astype(np.float32)  # 二值化（如果原本不是）
        negative_mask_patch=self.negative_mask_volume[z, x:x+self.patch_size[0], y:y+self.patch_size[1]]
        soft_negative_mask_patch = self.softnega[z, x:x + self.patch_size[0], y:y + self.patch_size[1]]
        return torch.tensor(feats_stack), torch.tensor(mask_patch).unsqueeze(0) ,torch.tensor(negative_mask_patch).unsqueeze(0) ,torch.tensor(soft_negative_mask_patch).unsqueeze(0) # (3, H, W), (1, H, W)

# ============================
thickness=2
dataset = ValidPatchSliceDataset(volume,test_volume_label,negative_volume_label,softnega, patch_size=patchsize, threshold=0.0, num_samples=1200,thickness=thickness,valid_z=valid_z)
sampler = RandomSampler(dataset, replacement=True, num_samples=1200)
loader = DataLoader(dataset, batch_size=12, sampler=sampler)

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


        return self.final(d1)
# model = MultiKernelUNet(in_channels=2*thickness+31, out_channels=1).cuda()
if interation_idx==0:
    model = MultiKernelUNet(in_channels=2*thickness+31, out_channels=1).cuda()
else:
    model = setup_model(MultiKernelUNet,model_args={"in_channels": 2 * thickness + 31, "out_channels": 1}, checkpoint_folder=mask_name, model_name=f'model_{interation_idx-1}.pt')

# loss
def masked_soft_bce_loss(pred, target, negative_target=None,softnega=None,
                         high_weight=1.0, low_weight=0.1, kernel_size=1):
    """
    pred: raw logits, shape [B,1,H,W]
    target: binary mask, [B,1,H,W], only 1 is reliable positive
    negative_target: binary mask, [B,1,H,W], 1 表示可靠负样本
    """
    B, C, H, W = target.shape

    edge_size=4

    dilation_kernel = torch.ones((1, 1, kernel_size, kernel_size), device=target.device)
    dilation_kernel2 = torch.ones((1, 1, kernel_size+edge_size, kernel_size+edge_size), device=target.device)
    # 正样本膨胀
    target_bin = (target == 1).float()
    dilated = F.conv2d(target_bin, dilation_kernel, padding=kernel_size//2)
    dilated = (dilated > 0).float()

    dilated2 = F.conv2d(target_bin, dilation_kernel2, padding=(kernel_size+edge_size)//2)
    dilated2 = (dilated2 > 0).float()
    dilated_extra = (dilated - target_bin).clamp(min=0)
    dilated_extra2 = (dilated2 - dilated).clamp(min=0)

    weight = torch.full_like(target, 0.0)

    weight[dilated_extra2 > 0] = low_weight
    weight[dilated_extra > 0] = 0
    weight[target_bin > 0] = high_weight
    weight[negative_target > 0]=high_weight
    weight[softnega>0]=0.001
    # # ★ 把可靠负区域并入高权重区
    # if negative_target is not None:
    #     dilated = torch.clamp(dilated + (negative_target > 0).float(), 0, 1)
    #
    # # 权重图
    # weight = torch.where(
    #     dilated == 1,
    #     torch.full_like(target, high_weight),
    #     torch.full_like(target, low_weight)
    # )

    return F.binary_cross_entropy_with_logits(pred, target, weight=weight)
def masked_dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    mask = (target >= 0).float()  # 全部参与
    pred = pred * mask
    target = target * mask
    num = 2 * (pred * target).sum(dim=(2, 3))
    den = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - (num + smooth) / (den + smooth)
def smoothness_loss(pred_mask, input_image, slice_idx=7, win_size=5):
    """
    让 pred_mask 的局部方差与 input_image 的局部方差尽可能相似。

    参数:
        pred_mask: [B, 1, H, W] - 模型预测 (logits)
        input_image: [B, C, H, W] - 输入图像
        slice_idx: int - 选择的输入通道索引
        win_size: int - 计算局部方差的窗口大小 (默认 5)

    返回:
        loss: 方差匹配损失（越小表示预测的纹理与输入一致）
    """
    pred = torch.sigmoid(pred_mask)  # [B,1,H,W]
    img = input_image[:, slice_idx:slice_idx+1, :, :]  # [B,1,H,W]

    pad = win_size // 2

    # --- 局部方差计算函数 ---
    def local_variance(x):
        mean = F.avg_pool2d(x, kernel_size=win_size, stride=1, padding=pad)
        mean_sq = F.avg_pool2d(x ** 2, kernel_size=win_size, stride=1, padding=pad)
        return mean_sq - mean ** 2

    # --- 计算输入和预测的局部方差 ---
    var_pred = local_variance(pred)
    var_img  = local_variance(img)

    # --- 归一化以防数值差异过大 ---
    var_pred = var_pred / (var_pred.mean() + 1e-6)
    var_img  = var_img / (var_img.mean() + 1e-6)

    # --- 方差匹配损失（用 L1 或 L2） ---
    # loss = F.l1_loss(var_pred, var_img)
    # 或者更平滑的 L2：
    loss = F.mse_loss(var_pred, var_img)

    return loss
def smoothness_loss0(pred_mask, input_image, slice_idx=None):
    """
    图像引导平滑损失：
    - 在输入图像梯度大的地方，允许mask有变化；
    - 在平滑区域，鼓励mask也平滑。

    参数:
        pred_mask: [B, 1, H, W]，预测的 mask（未sigmoid 或已sigmoid均可）
        input_image: [B, C, H, W]，原始输入图像（用于计算梯度权重）
        slice_idx: int，可选，用于指定哪一个通道 slice 作为引导
    返回:
        scalar loss
    """
    pred = torch.sigmoid(pred_mask)

    # 选择引导图
    if slice_idx is not None and input_image.shape[1] > slice_idx:
        guide = input_image[:, slice_idx:slice_idx+1, :, :]  # [B,1,H,W]
    else:
        guide = input_image[:, :1, :, :]  # 默认第一个通道

    # input 的梯度越大 → 权重越小（允许边缘变化）
    with torch.no_grad():
        dx_img = torch.exp(-torch.abs(guide[:, :, :, :-1] - guide[:, :, :, 1:]))
        dy_img = torch.exp(-torch.abs(guide[:, :, :-1, :] - guide[:, :, 1:, :]))

    # pred 的梯度越大 → 惩罚平滑性
    dx_pred = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
    dy_pred = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])

    loss = (dx_pred * dx_img).mean() + (dy_pred * dy_img).mean()
    return loss
def region_consistency_loss(pred_mask, input_image, slice_idx=None, eps=1e-6):
    """
    让预测 mask=1 的区域对应的 input_image 内部一致（低方差或高相似性）
    """
    pred = torch.sigmoid(pred_mask)
    if slice_idx is not None:
        img = input_image[:, slice_idx:slice_idx+1]
    else:
        img = input_image[:, :1]  # 默认第一个通道

    # 将预测转为 soft mask 权重
    w = pred / (pred.sum(dim=(2,3), keepdim=True) + eps)

    # 区域加权均值与方差
    mean_region = (img * w).sum(dim=(2,3), keepdim=True)
    var_region = ((img - mean_region) ** 2 * w).sum(dim=(2,3)).mean()

    return var_region
def region_contrast_loss(pred_mask, input_image, slice_idx=None, eps=1e-6):
    pred = torch.sigmoid(pred_mask)
    if slice_idx is not None:
        img = input_image[:, slice_idx:slice_idx+1]
    else:
        img = input_image[:, :1]

    fg_mean = (img * pred).sum(dim=(2,3)) / (pred.sum(dim=(2,3)) + eps)
    bg_mean = (img * (1 - pred)).sum(dim=(2,3)) / ((1 - pred).sum(dim=(2,3)) + eps)

    contrast = torch.abs(fg_mean - bg_mean).mean()
    return -contrast  # 我们希望对比度大，所以取负号让 loss 越小越好

def correlation_loss(x, y, reduction="mean"):
    """
    计算 x 和 y 各通道间的相关性损失：
      对每个通道 c： corr_c = corr(x, y[:, c])
      loss = 1 - mean(corr_c)

    参数:
        x: [B, 1, H, W]
        y: [B, C, H, W]
        reduction: "mean" 或 "sum"
    返回:
        loss: scalar
    """
    B, C, H, W = y.shape
    # 展平空间维度
    x_flat = x.view(B, -1)  # [B, H*W]
    y_flat = y.view(B, C, -1)  # [B, C, H*W]

    # 去均值
    x_centered = x_flat - x_flat.mean(dim=1, keepdim=True)  # [B, HW]
    y_centered = y_flat - y_flat.mean(dim=2, keepdim=True)  # [B, C, HW]

    # 对每个通道计算 corr
    numerator = (y_centered * x_centered.unsqueeze(1)).sum(dim=2)  # [B, C]
    denominator = (
        torch.norm(x_centered, dim=1, keepdim=True) * torch.norm(y_centered, dim=2)
        + 1e-6
    )  # [B, C]
    corr = numerator / denominator  # [B, C]

    loss = 1 - corr  # 越大越相关 → 损失越小
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss  # 不聚合，返回 [B,C]
def total_loss_fn(pred, target, input_img, negative_label, softnega, model,
                  bce_weight=10.0, corr_weight=0.1, smooth_weight=0.05,
                  l1_weight=0.05, high_weight=1.0, low_weight=0.5, thickness=5):
    bce = masked_soft_bce_loss(pred, target, negative_target=negative_label, softnega=softnega,
                               high_weight=high_weight, low_weight=low_weight)

    # contrast = correlation_loss(pred, input_img[:, (0):(2 * thickness + 1), :, :])
    smooth = smoothness_loss0(pred, input_img, slice_idx=thickness)
    correlation = region_consistency_loss(pred, input_img, slice_idx=2 * thickness + 1)
    contrast = region_contrast_loss(pred, input_img, slice_idx=2 * thickness + 1)
    # 🔹 L1 on model weights
    l1 = 0.0
    for p in model.parameters():
        l1 += torch.sum(torch.abs(p))
    l1 = l1 / sum(p.numel() for p in model.parameters())

    total = (bce_weight * bce +
             corr_weight * correlation +
             0.0 * contrast +
             smooth_weight * smooth +
             l1_weight * l1).mean()

    loss_dict = {
        "bce": bce.mean().item(),
        "correlation": correlation.mean().item(),
        "contrast": contrast.mean().item(),
        "smooth": smooth.item(),
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
    loss_log = {"bce": 0.0, "correlation": 0.0, "smooth": 0.0,"l1":0.0, "contrast":0.0}
    batch_count = 0

    for x, y,z,softnega_p in loader:
        x, y,z,softnega_p = x.cuda(), y.cuda(),z.cuda(),softnega_p.cuda()
        pred = model(x)

        loss, loss_dict = total_loss_fn(pred, y, x,z,softnega_p,model,low_weight=0.1*low_weight_coeff,thickness=thickness)

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
          f"BCE: {avg_loss_log['bce']:.4f}, correlation: {avg_loss_log['correlation']:.4f},contrast: {avg_loss_log['contrast']:.4f} Smooth: {avg_loss_log['smooth']:.4f}, l1: {avg_loss_log['l1']:.4f}")
#%%
model.eval()
with torch.no_grad():
    x, y_true,z,_ = dataset[40]  # 某一张图
    pred = model(x.unsqueeze(0).cuda())  # [1, 1, H, W]
    pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()
    y_true = y_true.squeeze().numpy()


    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 3, 1); plt.imshow(x[thickness].squeeze(), cmap='gray'); plt.title("Input")
    # plt.subplot(1, 3, 2); plt.imshow(y_true, cmap='gray'); plt.title("Canny Mask")
    # plt.subplot(1, 3, 3); plt.imshow(pred_mask > 0.5, cmap='gray'); plt.title("Predicted")
    # plt.show()

# model.eval()
# with torch.no_grad():
#     position=(50,50,200)
#
#     x=torch.from_numpy(volume[(position[0]-1):(position[0]+2),position[1]:(position[1]+patchsize[0]),position[2]:(position[2]+patchsize[1])]).float()
#     y_true = volume_label[position[0], position[1]:(position[1] + patchsize[0]),
#         position[2]:(position[2] + patchsize[1])]
#     pred = model(x.unsqueeze(0).cuda())  # [1, 1, H, W]
#     pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()
#     # y_true = y_true.squeeze().numpy()
#
#
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 3, 1); plt.imshow(x[1].squeeze(), cmap='gray'); plt.title("Input")
#     plt.subplot(1, 3, 2); plt.imshow(y_true, cmap='gray'); plt.title("Canny Mask")
#     plt.subplot(1, 3, 3); plt.imshow(pred_mask > 0.5, cmap='gray'); plt.title("Predicted")
#     plt.show()
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
def infer_volume_edges_whole(volume_np, model, threshold=0.5,thickness=2):
    """
    volume_np: [D, H, W], float32, normalized to [0, 1]
    model: model(input: [1, 1, H, W]) → pred [1, 1, H, W]
    Returns:
        edge_volume: [D, H, W], uint8 0/1 mask
    """
    padded_volume, pad_info = pad_to_multiple_of(volume_np, multiple=4)
    D, H, W = padded_volume.shape

    edge_volume = np.zeros((D, H, W), dtype=np.float32)

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
            pred_prob = torch.sigmoid(pred).squeeze().cpu().numpy()  # [H, W]

            # pred_prob_smooth = gaussian_filter(pred_prob, sigma=1)
            # edge_mask = (pred_prob_smooth > threshold).astype(np.uint8)

            edge_mask = pred_prob

            edge_volume[z] = edge_mask
    restored = unpad_volume(edge_volume, pad_info)

    blurred = gaussian_filter(restored, sigma=(1, 3, 3), mode="reflect")

    return blurred

# edge_vol = infer_volume_edges(volume, model, patch_size=patchsize,threshold=0.5,thickness=thickness)
edge_vol = infer_volume_edges_whole(volume, model, threshold=0.5,thickness=thickness)

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

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1);
plt.imshow(img);
plt.title("Input")
plt.subplot(1, 3, 2);
plt.imshow(mask);
plt.title("Predicted")
plt.subplot(1, 3, 3);
plt.imshow(volume_label[z]);
plt.title("labeled")
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

test_volume_label0 = tiff.imread(f'inputdata/{mask_name}.tif')
test_volume_label0=(test_volume_label0 >0).astype(np.uint8)
# test_volume_label0 = expand_mask_3d(test_volume_label0, radius=2)
volume_label0 = test_volume_label0

if not os.path.exists(mask_name):
    os.makedirs(mask_name)
save_volume_with_masks_as_rgb_tiff(volume, edge_vol, volume_label0, f"{mask_name}/volume_mask_pred_{interation_idx}.tiff")


# save_volume_with_masks_as_rgb_tiff(volume, edge_vol, volume_label, f"{mask_name}/volume_mask_pred2_{interation_idx}.tiff")
#
# vol0 = tiff.imread('../data/archive/archive/10515/data/test/cropped drifting_correction_noback.tif')
#
# vol1=process_volume(vol0)
# test_volume=(local_contrast_normalize(vol0))
# test_volume_label = tiff.imread('../data/archive/archive/10515/data/test/label_mask.tif')
# test_volume_label=(test_volume_label != 0).astype(np.uint8)
# test_volume_label = expand_mask_3d(test_volume_label, radius=2)
# factor=1
# test_volume_small=downsample_volume(test_volume,factor=factor)
# test_volume_label_small = block_reduce(test_volume_label, block_size=(1, factor, factor), func=np.max)
#

#
# z = test_volume.shape[0] // 2
#
# img = test_volume[z]               # [H, W]
# mask = test_edge_vol[z]            # [H, W]
#
# # 归一化原图（确保 ∈ [0,1]）
# if img.max() > 1:
#     img = img / 255.0
#
# # 构建 RGB 叠加图
# overlay = np.stack([img, img, img], axis=-1)  # → [H, W, 3]
# overlay[mask == 1] = [1.0, 0.0, 0.0]           # 红色边缘
#
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 3, 1);
# plt.imshow(img);
# plt.title("Input")
# plt.subplot(1, 3, 2);
# plt.imshow(mask);
# plt.title("Predicted")
# plt.subplot(1, 3, 3);
# plt.imshow(test_volume_label[z]);
# plt.title("labeled")
# plt.show()
# test_edge_vol = infer_volume_edges(test_volume_small, model, patch_size=patchsize,threshold=0.5,thickness=thickness)
# save_volume_with_masks_as_rgb_tiff(test_volume_small, test_edge_vol, test_volume_label_small, "outfig/volume_transfer_pred.tiff")

from scipy.io import savemat
vol01=(edge_vol > msk_threshold).astype(np.uint8)
test_volume_label_new=filter_connected_regions_shape( vol01, test_volume_label_base, threshold=threshold,z_threshold=z_threshold,min_ratio=0.5,sim_thresh=0.5)
tiff.imwrite(f'{mask_name}/remain_mask.tif', 255*test_volume_label_new)
# savemat(f"{mask_name}.mat", {"vol0": vol01})
tiff.imwrite(f'inputdata/{mask_name}_{interation_idx}.tif', vol01)
save_model(model, f'{mask_name}/model_{interation_idx}.pt')