import numpy as np
from scipy.ndimage import label

from scipy.ndimage import label, binary_erosion

def uniform_connected_labels(volume, num_bins=10, erosion_size=2):
    """
    将3D体数据按数值区间均匀划分为若干段，并对每个区间的连通区域独立编号。
    可选地对每个mask执行腐蚀操作以去除噪声边缘。

    参数:
        volume: np.ndarray, 任意浮点型3D数组
        num_bins: int, 区间数量（例如10表示10等分）
        erosion_size: int or tuple or None
            - None: 不执行腐蚀（默认）
            - int: 使用 3D 立方结构元素（大小为 erosion_size）
            - tuple: 分别指定 (z, y, x) 三个方向的腐蚀大小

    返回:
        label_volume: np.ndarray (int32)，每个连通区域有唯一label id
        bin_edges: np.ndarray，区间边界
    """
    assert volume.ndim == 3, "输入必须是3D体数据"
    volume = np.nan_to_num(volume)

    vmin, vmax = float(volume.min()), float(volume.max())
    bin_edges = np.linspace(vmin, vmax, num_bins + 1)

    label_volume = np.zeros_like(volume, dtype=np.int32)
    current_label = 1

    # === 设置腐蚀核 ===
    if erosion_size is not None:
        if isinstance(erosion_size, int):
            structure = np.ones((erosion_size,) * 3, dtype=np.uint8)
        else:
            structure = np.ones(erosion_size, dtype=np.uint8)
    else:
        structure = None

    # === 遍历每个区间 ===
    for i in range(num_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (volume > low) & (volume <= high)

        # === 腐蚀操作 ===
        # if structure is not None:
        #     mask = binary_erosion(mask, structure=structure, iterations=2)

        # === 连通区域标记 ===
        labeled, n = label(mask, structure=np.ones((3, 3, 3)))
        if n == 0:
            continue

        labeled[labeled > 0] += current_label - 1
        label_volume += labeled
        current_label += n

        print(f"区间 {i+1}/{num_bins} ({low:.3f}, {high:.3f}]: {n} 个连通区域")

    print(f"✅ 总共生成 {current_label - 1} 个连通区域。")
    return label_volume, bin_edges ,current_label

def randomly_remove_regions(label_vol, remove_ratio=0.3, seed=None):
    """
    从 label_vol 中随机移除一定比例的连通区域（将其值置为0）。

    参数:
        label_vol : np.ndarray
            连通区域标注（每个区域有唯一 label 编号）
        remove_ratio : float
            删除比例 (0~1)，例如 0.3 表示删除 30% 区域
        seed : int | None
            随机种子（可选，保证可重复）

    返回:
        new_label_vol : np.ndarray
            删除后的label体
        removed_labels : list
            被移除的label编号
    """
    assert label_vol.ndim == 3, "label_vol 必须是3D"
    if seed is not None:
        np.random.seed(seed)

    labels = np.unique(label_vol)
    labels = labels[labels > 0]  # 排除背景
    n_labels = len(labels)
    if n_labels == 0:
        print("⚠️ 没有可删除的区域。")
        return label_vol.copy(), []

    n_remove = int(n_labels * remove_ratio)
    removed_labels = np.random.choice(labels, n_remove, replace=False)

    mask_remove = np.isin(label_vol, removed_labels)
    new_label_vol = label_vol.copy()
    new_label_vol[mask_remove] = 0

    print(f"共 {n_labels} 个区域，随机删除 {n_remove} 个 ({remove_ratio*100:.1f}%)。")
    return new_label_vol, removed_labels

def randomly_remove_regions_by_z(
    label_vol,
    z_remove_ratio=0.3,
    seed=None
):
    """
    对每个被选中的连通区域，仅删除其内部一部分 z-slices。

    参数:
        label_vol : np.ndarray (3D)
        remove_ratio : float
            被选中的 region 比例
        z_remove_ratio : float
            每个 region 内，被删除的 z-slice 比例
        seed : int | None

    返回:
        new_label_vol : np.ndarray
        removed_info : dict {label: removed_z_indices}
    """
    assert label_vol.ndim == 3
    if seed is not None:
        np.random.seed(seed)

    labels = np.unique(label_vol)
    labels = labels[labels > 0]
    n_labels = len(labels)

    if n_labels == 0:
        print("⚠️ 没有可删除的区域。")
        return label_vol.copy(), {}

    # 1. 选择要“部分删除”的 region
    n_remove = n_labels
    selected_labels = np.random.choice(labels, n_remove, replace=False)

    new_label_vol = label_vol.copy()
    removed_info = {}

    for lb in selected_labels:
        z_indices = np.unique(np.where(label_vol == lb)[0])
        if len(z_indices) <= 1:
            continue

        n_z_remove = max(1, int(len(z_indices) * z_remove_ratio))
        removed_z = np.random.choice(z_indices, n_z_remove, replace=False)

        for z in removed_z:
            new_label_vol[z][new_label_vol[z] == lb] = 0

        removed_info[lb] = sorted(removed_z.tolist())

    print(
        f"共 {n_labels} 个区域，"
        f"选择 {n_remove} 个进行 z-slice 局部删除，"
        f"每个删除约 {z_remove_ratio*100:.1f}% 的 z。"
    )

    return new_label_vol, removed_info


import tifffile
remove_ratio_list=[30,50,70,80,95]

cell_type='hela2'
# cell_type='jurkat'
# cell_type='macrophage'

type='golgi'
type='lyso'
# type='nucleus'
# type='endo'
# type='er'
# type='mito'
# type='cent'

for remove_ratio in remove_ratio_list:
    vol = tifffile.imread(f"download/{cell_type}_{type}_s3.tif")
    label_vol, qvals, current_label = uniform_connected_labels(vol, num_bins=8)
    # tifffile.imwrite(f"download/label_{cell_type}_{type}_full.tif", label_vol)
    if current_label <500:
        label_sparse, removed = randomly_remove_regions_by_z(label_vol, z_remove_ratio=remove_ratio/100, seed=42)
    else:
        label_sparse, removed = randomly_remove_regions(
            label_vol,
            remove_ratio=remove_ratio / 100,
            seed=42
        )

    tifffile.imwrite(f"download/label_{cell_type}_{type}_volume_{remove_ratio}.tif", label_sparse.astype(np.uint16))
    print("保存完成，适用于 StarDist 训练。")

    structure = np.ones((3, 3, 3), dtype=np.uint8)  # 可调整腐蚀核大小
    eroded = binary_erosion(label_sparse > 0, structure=structure, iterations=0)

    label_binary = (label_sparse > 0).astype(np.uint8)  # 二值化：大于0的为1

    tifffile.imwrite(f"download/label_{cell_type}_{type}_{remove_ratio}.tif", label_binary)
    print("✅ label_volume2.tif 保存完成。")