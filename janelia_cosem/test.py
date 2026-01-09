#%%
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader,RandomSampler
from datetime import datetime
from tqdm import tqdm
from get_inputfeature import MultiScaleExtractor
from scipy.ndimage import binary_closing, binary_dilation, label as nd_label
import tifffile as tiff
from utils import compute_statistical_mask,process_volume,local_contrast_normalize,local_standardize
from skimage.transform import downscale_local_mean
from skimage.measure import block_reduce
def load_model(model_path="region_classifier.pth"):
    model = MultiScaleExtractor()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
def downsample_volume(volume, factor=2):
    """
    将3D volume 每个维度缩小 factor 倍，体积变为原来的1/(factor^3)
    """
    # assert all(s % factor == 0 for s in volume.shape), "shape 必须能整除 factor"
    return downscale_local_mean(volume, (1, factor, factor))
def expand_mask_3d(volume_pred, radius=2):
    """
    将3D mask中为1的区域扩张一定半径（像素）。

    参数:
        volume_pred: np.ndarray，三维二值数组 [D, H, W]
        radius: int，扩张半径（单位：像素）

    返回:
        dilated_mask: np.ndarray，扩张后的二值数组
    """
    # 创建一个球形结构元素
    from scipy.ndimage import generate_binary_structure, iterate_structure

    struct = generate_binary_structure(rank=3, connectivity=1)  # 3D 十字结构
    struct = iterate_structure(struct, radius)  # 扩张 radius 次

    dilated = binary_dilation(volume_pred, structure=struct)
    return dilated.astype(np.uint8)

#volume self
vol0 = tiff.imread('cropped drifting_correction_noback.tif')
vol1=process_volume(vol0)
test_volume=(local_contrast_normalize(vol0))
test_volume_label = tiff.imread('label_mask.tif')
test_volume_label=(test_volume_label != 0).astype(np.uint8)
test_volume_label = expand_mask_3d(test_volume_label, radius=2)
factor=1
test_volume_small=downsample_volume(test_volume,factor=factor)
test_volume_label_small = block_reduce(test_volume_label, block_size=(1, factor, factor), func=np.max)

# 找出 mask_volume 在 z 维度上的有效切片
valid_z = np.where(test_volume_label.sum(axis=(1, 2)) > 0)[0]

volume=test_volume_small[valid_z]
volume_label=test_volume_label_small[valid_z]
volume_pred=test_volume_label_small[valid_z]
D, H, W = volume.shape
patchsize=(100,100)


#%%
import matplotlib.pyplot as plt

# ========== 输入数据 ==========
# 示例：使用 volume[z]（电镜图层）
# 如果是 PNG 文件，也可以使用 imageio.imread("path.png")
img = volume[0]  # 或者你选 volume[z]
plt.imsave("cellpose_seg_overlay.png", img)

#%%
class ValidPatchSliceDataset(Dataset):
    def __init__(self, volume, mask_volume, patch_size=(256,256), threshold=0.7, num_samples=5000, max_trials=200000,thickness=2):
        """
        volume: np.ndarray, shape (D, H, W), 原始图像
        mask_volume: np.ndarray, shape (D, H, W), 对应的 mask
        """
        self.thickness=thickness
        self.volume = volume.astype(np.float32)
        self.mask_volume = mask_volume.astype(np.float32)
        # Step 1: 对第一个维度求和，得到一个二维图像
        mask_2d = np.sum(self.mask_volume, axis=0)  # shape: (H, W)

        self.patch_size = patch_size
        self.threshold = threshold
        self.samples = []

        D, H, W = self.volume.shape

        print("🔍 正在构建符合非零占比要求的 patch 索引...")
        attempts = 0
        while len(self.samples) < num_samples:
            success = False
            for _ in range(max_trials):


                z = np.random.randint(self.thickness, D - self.thickness)
                x = np.random.randint(0, H - patch_size[0])
                y = np.random.randint(0, W - patch_size[1])
                patch = self.volume[z, x:x+patch_size[0], y:y+patch_size[1]]
                patch2 = self.mask_volume[z,x:x + patch_size[0], y:y + patch_size[1]]
                # patch2 = mask_2d[x:x + patch_size[0], y:y + patch_size[1]]

                # Step 3: 计算绝对值大于 0.0001 的比例
                nonzero_ratio = np.count_nonzero(np.abs(patch) > 0.0001) / patch.size
                nonzero_ratio2 = np.count_nonzero(np.abs(patch2) > 0.0001) / patch2.size


                if nonzero_ratio >= threshold:
                    if nonzero_ratio2>=0.001:
                        self.samples.append((z, x, y))
                        success = True
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

        # 取中心切片对应 mask
        mask_patch = self.mask_volume[z, x:x+self.patch_size[0], y:y+self.patch_size[1]]  # shape [H, W]
        mask_patch = (mask_patch > 0).astype(np.float32)  # 二值化（如果原本不是）

        return torch.tensor(img_patch), torch.tensor(mask_patch).unsqueeze(0)  # (3, H, W), (1, H, W)

# ============================
thickness=2
dataset = ValidPatchSliceDataset(volume,volume_label, patch_size=patchsize, threshold=0.0, num_samples=5000,thickness=thickness)
sampler = RandomSampler(dataset, replacement=True, num_samples=1200)
loader = DataLoader(dataset, batch_size=12, sampler=sampler)

#%%
class MultiKernelConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(3, 7, 15)):
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

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = MultiKernelConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = MultiKernelConvBlock(128, 64)

        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)
model = MultiKernelUNet(in_channels=2*thickness+1, out_channels=1).cuda()

# loss
def masked_soft_bce_loss(pred, target, high_weight=1.0, low_weight=0.1, kernel_size=5):
    """
    pred: raw logits, shape [B, 1, H, W]
    target: binary mask [B, 1, H, W], only target==1 is reliable
    kernel_size: must be odd, e.g., 11 means上下左右扩5像素范围
    """
    B, C, H, W = target.shape

    # Create dilation kernel
    dilation_kernel = torch.ones((1, 1, kernel_size, kernel_size), device=target.device)

    # Binary: 1 where target==1
    target_bin = (target == 1).float()

    # Dilate the 1-valued region
    dilated = F.conv2d(target_bin, dilation_kernel, padding=kernel_size//2)
    dilated = (dilated > 0).float()  # Get binary mask for surrounding

    # Generate weight map
    weight = torch.where(
        dilated == 1,
        torch.full_like(target, high_weight),
        torch.full_like(target, low_weight)
    )

    return F.binary_cross_entropy_with_logits(pred, target, weight=weight)
def masked_dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    mask = (target >= 0).float()  # 全部参与
    pred = pred * mask
    target = target * mask
    num = 2 * (pred * target).sum(dim=(2, 3))
    den = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - (num + smooth) / (den + smooth)
def smoothness_loss(pred_mask, input_image):
    pred = torch.sigmoid(pred_mask)

    # 用 input 生成边权时不需要反向传播
    with torch.no_grad():
        dx_img = torch.exp(-torch.abs(input_image[:, :, :, :-1] - input_image[:, :, :, 1:]))
        dy_img = torch.exp(-torch.abs(input_image[:, :, :-1, :] - input_image[:, :, 1:, :]))

    # pred 的梯度需要保留
    dx_mask = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
    dy_mask = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])

    loss = (dx_mask * dx_img).mean() + (dy_mask * dy_img).mean()
    return loss
def total_loss_fn(pred, target, input_img,
                  bce_weight=10.0, dice_weight=0.00, smooth_weight=1.0,
                  high_weight=1.0, low_weight=0.1):
    bce = masked_soft_bce_loss(pred, target, high_weight, low_weight)
    dice = masked_dice_loss(pred, target)
    smooth = 0
    smooth = smoothness_loss(pred, input_img)

    total = (bce_weight * bce + dice_weight * dice + smooth_weight * smooth).mean()
    loss_dict = {
        "bce": bce.mean().item(),
        "dice": dice.mean().item(),
        "smooth": smooth.item()
    }
    return total, loss_dict

# dataset_edge = EdgeDataset(dataset)
# loader = DataLoader(dataset_edge, batch_size=8, shuffle=True)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    model.train()
    total_loss = 0
    loss_log = {"bce": 0.0, "dice": 0.0, "smooth": 0.0}
    batch_count = 0

    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        pred = model(x)

        loss, loss_dict = total_loss_fn(pred, y, x)

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
          f"BCE: {avg_loss_log['bce']:.4f}, Dice: {avg_loss_log['dice']:.4f}, Smooth: {avg_loss_log['smooth']:.4f}")

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
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).float().cuda()  # [1, 3, patch_h, patch_w]
                    pred = model(patch_tensor)  # [1, 1, patch_h, patch_w]
                    pred_prob = torch.sigmoid(pred).squeeze().cpu().numpy()  # [patch_h, patch_w]

                    # slice_pred[i:i + patch_h, j:j + patch_w] += pred_prob
                    # slice_count[i:i + patch_h, j:j + patch_w] += 1

                    slice_pred[i:i + patch_h, j:j + patch_w] = np.maximum(
                        slice_pred[i:i + patch_h, j:j + patch_w],
                        (pred_prob > threshold).astype(np.uint8)
                    )

            # 平均融合预测结果
            # slice_count[slice_count == 0] = 1
            # slice_avg = slice_pred / slice_count
            # edge_mask = (slice_avg > threshold).astype(np.uint8)

            edge_mask=slice_pred

            edge_volume[z] = edge_mask

    restored = unpad_volume(edge_volume, pad_info)
    return restored
def infer_volume_edges_whole(volume_np, model, threshold=0.5):
    """
    volume_np: [D, H, W], float32, normalized to [0, 1]
    model: model(input: [1, 1, H, W]) → pred [1, 1, H, W]
    Returns:
        edge_volume: [D, H, W], uint8 0/1 mask
    """
    padded_volume, pad_info = pad_to_multiple_of(volume_np, multiple=4)
    D, H, W = padded_volume.shape

    edge_volume = np.zeros((D, H, W), dtype=np.uint8)

    model.eval()
    model.cuda()

    with torch.no_grad():
        for z in tqdm(range(1,D-1), desc="Predicting edges per slice"):
            slice_img = padded_volume[(z-1):(z+2)]  # [H, W]
            input_tensor = torch.from_numpy(slice_img).unsqueeze(0).float().cuda()  # [1, 1, H, W]

            pred = model(input_tensor)  # [1, 1, H, W]
            pred_prob = torch.sigmoid(pred).squeeze().cpu().numpy()  # [H, W]

            edge_mask = (pred_prob > threshold).astype(np.uint8)
            edge_volume[z] = edge_mask
    restored = unpad_volume(edge_volume, pad_info)
    return restored

edge_vol = infer_volume_edges(volume, model, patch_size=patchsize,threshold=0.5,thickness=thickness)


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

save_volume_with_masks_as_rgb_tiff(volume, edge_vol, volume_pred, "volume_mask_pred.tiff")


test_edge_vol = infer_volume_edges(test_volume_small, model, patch_size=patchsize,threshold=0.5,thickness=thickness)
save_volume_with_masks_as_rgb_tiff(test_volume_small, test_edge_vol, test_volume_label_small, "volume_transfer_pred.tiff")