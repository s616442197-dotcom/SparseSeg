#%%
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from tqdm import tqdm
from get_inputfeature import MultiScaleExtractor
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import binary_closing, binary_dilation, label as nd_label
import tifffile as tiff
from utils import compute_statistical_mask,process_volume,local_contrast_normalize,local_standardize

def load_model(model_path="region_classifier.pth"):
    model = MultiScaleExtractor()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 假设 volume 已是 numpy 数组，形状为 (D, H, W)，数值范围任意
def local_normalize_gpu(volume, window_size=20, eps=1e-6, device='cuda'):
    """
    使用滑动窗口方式在 GPU 上对 3D volume 每一层做局部归一化。
    每个像素的值会除以其窗口内的 max-min 值。

    参数:
        volume: np.ndarray (D, H, W)
        window_size: int，局部窗口半径
        eps: float，小常数避免除零
        device: 'cuda' or 'cpu'

    返回:
        normalized: np.ndarray (D, H, W)
    """
    window_size=window_size*2-1
    D, H, W = volume.shape
    volume_tensor = torch.from_numpy(volume).float().to(device).unsqueeze(1)  # (D,1,H,W)

    pad = window_size // 2
    kernel = torch.ones((1, 1, window_size, window_size), device=device)

    # 局部最大和最小（用 max_pool2d 和 min_pool2d）
    local_max = F.max_pool2d(volume_tensor, kernel_size=window_size, stride=1, padding=pad)
    local_min = -F.max_pool2d(-volume_tensor, kernel_size=window_size, stride=1, padding=pad)

    normalized = (volume_tensor - local_min) / (local_max - local_min + eps)
    normalized = normalized.squeeze(1).clamp(0, 1)  # (D, H, W)

    return normalized.cpu().numpy().astype(np.float32)


input_dir = "/home/sbw/VEMModel/.venv/data/archive/archive/10515/data/raw_control_release"
output_dir = "output_data"
volume_path = os.path.join(output_dir, "raw_volume.npy")
label_path = os.path.join(output_dir, "weak_label.npy")
os.makedirs(output_dir, exist_ok=True)
volume = tiff.imread('inputdata/groundtruth_volume.tif')
volume = np.transpose(volume, (1, 0, 2))

volume0 = volume.astype(np.float32)
# volume = process_volume(volume0)
volume = local_standardize(volume0)
D, H, W = volume.shape


model = MultiScaleExtractor(kernel_sizes=[3, 7, 11])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss(ignore_index=255)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = load_model().to(device)
all_features = []
with torch.no_grad():
    for i in tqdm(range(D), desc="提取特征"):
        img = torch.tensor(volume[i:i+1], dtype=torch.float32).unsqueeze(0).to(device)  # (1,1,H,W)
        _, feat = model(img)  # (1, 32, H/4, W/4)

        # 上采样回原图大小 (H,W)
        feat_up = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
        feat = feat_up.squeeze(0).permute(1, 2, 0).cpu().numpy().reshape(-1, 32)
        all_features.append(feat)

all_features_flat = np.concatenate(all_features, axis=0)  # (D*H*W, 32)

# ===============================
# 5. 对所有像素做 KMeans 聚类
# ===============================
print("开始聚类...")
def relabel_kmeans_by_region_priority(region_labels, y_range, x_range):
    """
    将 region_labels 的 label 重新排序，使得 [y_range, x_range] 区域中最常见的 label 映射为 0

    参数:
        region_labels: (D, H, W) array，聚类标签图
        y_range: tuple，例如 (500, 1000)
        x_range: tuple，例如 (1000, 2500)

    返回:
        relabeled: 同 shape，标签已重新编号
    """
    D, H, W = region_labels.shape
    # 聚合目标区域所有切片的 label
    region_focus_labels = region_labels[:, y_range[0]:y_range[1], x_range[0]:x_range[1]].flatten()

    # 统计 label 出现频率
    unique_labels, counts = np.unique(region_focus_labels, return_counts=True)
    sorted_labels = unique_labels[np.argsort(-counts)]  # 按出现频率从高到低排序

    # 构建 relabel 映射字典（让最多的成为0）
    remap_dict = {old_label: new_label for new_label, old_label in enumerate(sorted_labels)}
    remaining_labels = set(np.unique(region_labels)) - set(sorted_labels)
    for old_label in sorted(remaining_labels):
        remap_dict[old_label] = len(remap_dict)

    # 应用映射
    relabeled = np.vectorize(remap_dict.get)(region_labels)
    return relabeled

kmeans = KMeans(n_clusters=3, random_state=0).fit(all_features_flat)
all_labels_flat = kmeans.labels_  # (D*H*W,)
region_labels = all_labels_flat.reshape(D, H, W)
region_labels = relabel_kmeans_by_region_priority(
    region_labels,
    y_range=(300, 500),
    x_range=(100, 200)
)
print("聚类完成，输出 region_labels.shape =", region_labels.shape)

#%%
# ===============================
# 6. 可视化部分 slice
# ===============================

# ------------------------------------
# 可视化
# ------------------------------------
def show_slice_with_seg(idx, region_labels_cleaned):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(volume[idx], cmap='gray')
    axes[0].set_title(f'Slice {idx}')
    axes[1].imshow(region_labels_cleaned[idx], cmap='tab10')
    axes[1].set_title('Clustered Regions')
    plt.tight_layout()
    plt.show()

# ------------------------------------
# Mode filter on GPU
# ------------------------------------
def mode_filter_gpu(region_labels, num_classes=3, window_size=50, device='cuda'):
    window_size=2*window_size-1
    if isinstance(region_labels, np.ndarray):
        region_labels = torch.from_numpy(region_labels)
    region_labels = region_labels.to(device)
    D, H, W = region_labels.shape

    one_hot = F.one_hot(region_labels.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    pooled = F.avg_pool2d(one_hot, kernel_size=window_size, stride=1, padding=window_size//2)
    smoothed = torch.argmax(pooled, dim=1)
    smoothed_np = smoothed.cpu().numpy()
    return smoothed_np.astype(np.int32)

def refine_label_region(region_labels, target_label=1, min_area=10000,
                        fill_kernel_size=7, shrink_ratio=0.25,
                        use_polygon=True, epsilon_ratio=0.01):
    """
    精修指定标签区域：
    1. 闭运算填洞 + 缩放平滑；
    2. 筛除小区域；
    3. 膨胀扩张；
    4. 边界轮廓简化（可选）。

    参数:
        region_labels: (D, H, W) 标签图
        target_label: int, 目标标签
        min_area: int, 最小保留面积
        background_label: int, 用于替换小区域或空白区域
        fill_kernel_size: 闭运算核大小
        shrink_ratio: 缩放系数（<1 以去噪）
        dilation_radius: 扩张像素数
        use_polygon: 是否使用approxPolyDP简化轮廓
        epsilon_ratio: approxPolyDP 的简化程度（建议 0.003 ~ 0.01）

    返回:
        refined: np.ndarray (D, H, W)
    """


    refined = region_labels.copy()
    D, H, W = region_labels.shape
    structure = np.ones((fill_kernel_size, fill_kernel_size), dtype=np.uint8)

    for z in range(D):
        mask = (region_labels[z] == target_label).astype(np.uint8)

        # Step 1: 缩小再放大（去噪）
        if shrink_ratio < 1.0:
            small = cv2.resize(mask, (0, 0), fx=shrink_ratio, fy=shrink_ratio, interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)

        # Step 2: 闭运算填孔洞
        mask_closed = binary_closing(mask, structure=structure)

        # Step 3: 连通区域标记 + 面积筛选
        labeled_array, num = nd_label(mask_closed)
        keep_mask = np.zeros_like(mask, dtype=np.uint8)
        for i in range(1, num + 1):
            region = (labeled_array == i)
            if region.sum() >= min_area:
                keep_mask[region] = 1



        # Step 4: 可选轮廓简化（多边形逼近）
        if use_polygon:
            convex_mask = np.zeros_like(keep_mask, dtype=np.uint8)
            contours, _ = cv2.findContours(keep_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if len(cnt) >= 3:
                    epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    cv2.drawContours(convex_mask, [approx], 0, 1, thickness=-1)
            final_mask = convex_mask
        else:
            final_mask = keep_mask.astype(np.uint8)

        # Step 6: 写回 refined map
        refined[z] = final_mask

    return refined

def binary_dilate_gpu(region_labels, kernel_size=11, device='cuda'):
    """
    将二值标签图中 label==1 的区域进行 GPU 膨胀。

    参数:
        region_labels: np.ndarray or torch.Tensor, shape (D, H, W), 值为0或1
        kernel_size: 膨胀核尺寸（奇数），例如11表示扩张5像素
        device: 'cuda' 或 'cpu'

    返回:
        dilated: np.ndarray, 膨胀后的二值图（0/1）
    """
    kernel_size=2*kernel_size-1
    if isinstance(region_labels, np.ndarray):
        region_labels = torch.from_numpy(region_labels)
    region_labels = region_labels.to(device).float().unsqueeze(1)  # (D,1,H,W)

    dilated = F.max_pool2d(region_labels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)  # (D,1,H,W)
    dilated = dilated.squeeze(1).round().clamp(0, 1)  # 保证仍为0或1
    return dilated.cpu().numpy().astype(np.uint8)
# ------------------------------------
# 应用所有步骤
# ------------------------------------
from datetime import datetime
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"⏳ {now}Step 1: GPU滑动窗口平滑中...")
region_labels_smoothed = mode_filter_gpu(region_labels, num_classes=3, window_size=30, device='cuda')

region_labels_smoothed = (region_labels_smoothed != 0).astype(np.uint8)
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"⏳ {now}Step 2: 清除小面积的编号为0区域...")
region_labels_refined = refine_label_region(
    region_labels_smoothed,
    target_label=1,
    min_area=250,
    fill_kernel_size=7,
    shrink_ratio=0.25,  # 缩放去小点
    use_polygon=False,
)
region_labels_refined0=region_labels_refined
region_labels_dilated = binary_dilate_gpu(region_labels_refined0, kernel_size=15, device='cuda')

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"⏳ {now}over")
np.save( os.path.join(output_dir, "region_labels_dilated.npy"), region_labels_dilated)
np.save(os.path.join(output_dir, "volume_masked.npy"), volume * (region_labels_dilated == 1))
#%%
# ------------------------------------
# 可视化结果
# ------------------------------------
show_slice_with_seg(24, region_labels_smoothed)

