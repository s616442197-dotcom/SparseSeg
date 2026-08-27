from scipy.ndimage import maximum_filter, minimum_filter,binary_dilation ,gaussian_filter,uniform_filter,label
import cv2
import numpy as np
from skimage.measure import moments_hu, moments_central, moments,regionprops
from scipy.ndimage import binary_opening, generate_binary_structure

def compute_statistical_mask(volume, window_size=20):
    """
    计算3D图像volume每个像素点的局部方差（只在Y×X方向做 2D 滑窗）。

    参数：
        volume (np.ndarray): 输入3D图像 [Z, Y, X]
        window_size (int): 局部窗口大小（默认20）

    返回：
        var_local (np.ndarray): 局部方差图，shape 与 volume 相同
    """
    volume = volume.astype(np.float32)

    # 计算局部均值和局部平方均值
    local_mean = uniform_filter(volume, size=(1, window_size, window_size), mode='reflect')
    local_sq_mean = uniform_filter(volume ** 2, size=(1, window_size, window_size), mode='reflect')

    # 局部方差 = E[x^2] - (E[x])^2
    var_local = local_sq_mean - local_mean ** 2
    var_local
    return var_local

def process_volume(volume):
    vol = volume.astype(float)  # 转为 float 以支持 NaN

    # # 1. 计算25th percentile
    # q25 = np.percentile(vol, 10)
    # q75 = np.percentile(vol, 90)
    # # 2. 将小于 q25 的值设为 NaN
    # vol[vol < q25] = np.nan
    # vol[vol > q75] = np.nan

    # 3. 忽略 NaN 执行归一化
    vmin = np.nanmin(vol)
    vmax = np.nanmax(vol)
    vol = (vol - vmin) / (vmax - vmin)

    # 4. 将 NaN 替换为 0
    vol = np.nan_to_num(vol, nan=0.0)

    return vol

def local_contrast_normalize(volume, kernel_size=20, eps=1e-5):
    footprint = (3, kernel_size, kernel_size)

    local_max = maximum_filter(volume, size=footprint, mode='reflect')
    local_min = minimum_filter(volume, size=footprint, mode='reflect')

    volume_norm = (volume - local_min) / (local_max - local_min + eps)
    return volume_norm

def local_standardize(volume, kernel_size=15, eps=1e-2):
    footprint = (kernel_size, kernel_size, 3)
    footprint2 = (kernel_size*2, kernel_size*2, 3)
    # 局部均值
    local_mean = uniform_filter(volume, size=footprint, mode='reflect')

    # 局部平方的均值
    local_sq_mean = uniform_filter(volume ** 2, size=footprint2, mode='reflect')

    # std = sqrt(E[x^2] - (E[x])^2)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))

    # 标准化
    volume_standardized = (volume - local_mean)
    volume_standardized = (volume - local_mean) / (local_std + eps)
    return volume_standardized

def intersect_regions(M1, M2, min_v=None, max_v=None, overlap_ratio=0.1):
    """
    从 M2 中提取与 M1 有显著交集的连通区域。
    若区域与 M1 的交叠比例 >= overlap_ratio（例如 0.1），则保留。
    """
    assert M1.shape == M2.shape, "M1 和 M2 必须同 shape"
    M1_bin = (M1 > 0).astype(np.uint8)
    M2_bin = (M2 > 0).astype(np.uint8)
    structure = np.zeros((3, 3, 3), dtype=int)
    structure[1, 1, 1] = 1  # 当前中心点
    structure[0, 1, 1] = 1  # 上
    structure[2, 1, 1] = 1  # 下
    structure[1, 0, 1] = 1  # 前
    structure[1, 2, 1] = 1  # 后
    structure[1, 1, 0] = 1  # 左
    structure[1, 1, 2] = 1  # 右
    labeled, num = label(M2_bin, structure=structure)
    if num == 0:
        return np.zeros_like(M2_bin)

    # (1) 每个区域体积
    sizes = np.bincount(labeled.ravel(), minlength=num + 1)

    # (2) 每个区域与 M1 的交集体积
    # overlap_mask = (M1_bin & M2_bin)
    overlap_labels = labeled[M1_bin > 0]
    overlap_counts = np.bincount(overlap_labels.ravel(), minlength=num + 1)

    # (3) 计算交叠比例（避免除0）
    overlap_ratio_per_label = np.zeros(num + 1, dtype=np.float32)
    valid = sizes > 0
    overlap_ratio_per_label[valid] = overlap_counts[valid] / sizes[valid]

    # (4) 根据阈值过滤
    keep = overlap_ratio_per_label >= overlap_ratio
    if min_v is not None:
        keep &= (sizes >= min_v)
    if max_v is not None:
        keep &= (sizes <= max_v)

    valid_labels = np.where(keep)[0]
    valid_labels = valid_labels[valid_labels > 0]  # 排除背景标签0

    # (5) 生成输出
    M2_new = np.isin(labeled, valid_labels).astype(np.uint8)
    return M2_new

def intersect_regions_zexpand(M1, M2, z_expand=2):
    """
    对 M1 在 z 方向上进行形态学膨胀（而非简单复制层），
    然后取与 M2 的交集。

    Args:
        M1, M2 : np.ndarray, shape (D,H,W)
        z_expand : int, z方向膨胀半径（默认2 → 上下各扩2层）

    Returns:
        M2_new : np.ndarray, shape (D,H,W)
    """
    assert M1.shape == M2.shape, "M1 和 M2 必须同 shape"

    M1_bin = (M1 > 0).astype(np.uint8)
    M2_bin = (M2 > 0).astype(np.uint8)

    # === 构造3D结构元素 ===
    # 仅在z方向膨胀（可以改成全3D膨胀）
    struct = np.ones((2 * z_expand + 1, 2 * z_expand + 1, 2 * z_expand + 1), dtype=np.uint8)


    # === 进行z方向膨胀 ===
    M1_dilated = binary_dilation(M1_bin, structure=struct).astype(np.uint8)

    # === 取交集 ===
    M2_new = (M2_bin & M1_dilated).astype(np.uint8)
    return M2_new

def pca_normalized_iou(region1, region2):
    """
    旋转&缩放不变的形状相似度（PCA 主轴对齐 + IoU）
    """
    def align(mask):
        mask = (mask > 0).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        if len(xs) < 5:
            return mask
        coords = np.stack([xs - xs.mean(), ys - ys.mean()], axis=1)
        cov = np.cov(coords, rowvar=False)
        eigvec = np.linalg.eigh(cov)[1][:, 1]  # 主轴方向
        angle = np.degrees(np.arctan2(eigvec[1], eigvec[0]))
        M = cv2.getRotationMatrix2D((mask.shape[1]/2, mask.shape[0]/2), -angle, 1.0)
        return cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)

    r1, r2 = align(region1), align(region2)
    inter = np.logical_and(r1, r2).sum()
    union = np.logical_or(r1, r2).sum()
    return inter / (union + 1e-8)

def hu_moments_similarity(region1, region2):
    """region1, region2: 二值 mask"""
    cnts1, _ = cv2.findContours(region1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2, _ = cv2.findContours(region2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts1) == 0 or len(cnts2) == 0:
        return 0.0

    m1 = cv2.HuMoments(cv2.moments(cnts1[0])).flatten()
    m2 = cv2.HuMoments(cv2.moments(cnts2[0])).flatten()

    # log transform，常用于稳定比较
    m1 = -np.sign(m1) * np.log10(np.abs(m1) + 1e-10)
    m2 = -np.sign(m2) * np.log10(np.abs(m2) + 1e-10)

    # 余弦相似度
    sim = np.dot(m1, m2) / (np.linalg.norm(m1) * np.linalg.norm(m2) + 1e-10)
    return sim

def fast_hu_from_mask(mask):
    """快速计算单个二值 mask 的 Hu 矩（log尺度）"""
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() < 5:
        return np.zeros(7)
    m = cv2.moments(mask)
    hu = cv2.HuMoments(m).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu

def hu_similarity_from_mask(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    计算两个二值区域的 Hu 矩相似度（旋转、缩放、平移不变）。
    返回 [0,1] 之间的相似度值。
    """
    hu1 = fast_hu_from_mask(mask1)
    hu2 = fast_hu_from_mask(mask2)

    if np.all(hu1 == 0) or np.all(hu2 == 0):
        return 0.0

    # 使用余弦相似度衡量形状一致性
    sim = np.dot(hu1, hu2) / (np.linalg.norm(hu1) * np.linalg.norm(hu2) + 1e-8)
    return float(np.clip(sim, 0, 1))

def filter_connected_regions_shape_hu(test_volume_label, mask_label, threshold=0.5, min_ratio=0.2,max_height=10):
    """
    保留 test_volume_label 中与 mask_label 中任意层连通块形状相似的区域。
    过滤过小区域，并且在 intersect_regions 中应用 z_extent 过滤。

    参数:
        test_volume_label: np.ndarray, [D,H,W]
        mask_label: np.ndarray, [D,H,W]
        threshold: float，相似度阈值
        min_ratio: float，最小面积比例阈值
        z_threshold: int，Z方向最大厚度
    返回:
        test_out: np.ndarray, [D,H,W]
    """

    D, H, W = test_volume_label.shape
    test_new = np.zeros_like(test_volume_label, dtype=np.uint8)

    # === 1️⃣ 预提取 mask_label 所有区域的 Hu 特征 ===
    mask_regions, hu_templates, size_templates = [], [], []
    for z in range(D):
        labeled_v2, num2 = label(mask_label[z])
        for j in range(1, num2 + 1):
            region_v2 = (labeled_v2 == j)
            if region_v2.sum() > 0:
                mask_regions.append(region_v2)
                hu_templates.append(fast_hu_from_mask(region_v2))
                size_templates.append(region_v2.sum())
    if len(hu_templates) == 0:
        return np.zeros_like(test_volume_label)

    hu_templates = np.stack(hu_templates)
    hu_templates /= np.linalg.norm(hu_templates, axis=1, keepdims=True) + 1e-8

    min_size_v2, max_size_v2 = np.min(size_templates), np.max(size_templates)

    # === 2️⃣ 遍历 test_volume_label 每层 ===
    for z in range(D):
        # print(D)
        labeled_v1, num1 = label(test_volume_label[z])
        for i in range(1, num1 + 1):
            region_v1 = (labeled_v1 == i)
            size_v1 = region_v1.sum()
            if size_v1 < 10:
                continue

            # 面积过滤
            if(
                size_v1 < min_size_v2 * min_ratio or size_v1 > max_size_v2 / min_ratio
            ):
                continue

            hu1 = fast_hu_from_mask(region_v1)
            if np.all(hu1 == 0):
                continue

            hu1 = hu1 / (np.linalg.norm(hu1) + 1e-8)
            sims = hu_templates @ hu1  # 向量化余弦相似度

            if np.max(sims) >= threshold:
                test_new[z][region_v1] = 1

    # test_out = intersect_regions(test_new, test_volume_label,min_v=min_size_v2,max_v=max_size_v2*max_height,overlap_ratio=0.2)
    # test_out = intersect_regions_zexpand(test_new, test_volume_label)

    return test_new

def filter_connected_regions_shape_shape(test_volume_label, mask_label, threshold=0.5, min_ratio=0.2,max_height=10):
    D, H, W = test_volume_label.shape
    test_new = np.zeros_like(test_volume_label, dtype=np.uint8)

    # === 1️⃣ 提取 mask_label 的几何特征模板 ===
    def get_shape_vec(p):
        """从 regionprops 属性提取几何特征"""
        area = p.area
        perim = p.perimeter
        circularity = 4 * np.pi * area / (perim**2 + 1e-8)
        aspect = (
            p.major_axis_length / (p.minor_axis_length + 1e-8)
            if p.minor_axis_length > 0
            else 0
        )
        compact = area / (p.bbox_area + 1e-8)
        ecc = p.eccentricity
        return np.array([area, circularity, aspect, compact, ecc], dtype=np.float32)

    mask_feats = []
    mask_sizes = []
    for z in range(D):
        labeled_v2, num2 = label(mask_label[z])
        props = regionprops(labeled_v2)
        for p in props:
            vec = get_shape_vec(p)
            mask_feats.append(vec / (np.linalg.norm(vec) + 1e-8))
            mask_sizes.append(p.area)

    if len(mask_feats) == 0:
        return np.zeros_like(test_volume_label)
    mask_feats = np.stack(mask_feats)
    min_size_v2, max_size_v2 = np.min(mask_sizes), np.max(mask_sizes)

    # === 2️⃣ 遍历 test_volume_label 各层 ===
    for z in range(D):
        labeled_v1, _ = label(test_volume_label[z])
        props_v1 = regionprops(labeled_v1)
        for p in props_v1:
            if p.area < 10:
                continue

            # 面积过滤
            if p.area < min_size_v2 * min_ratio or p.area > max_size_v2 / min_ratio:
                continue

            vec = get_shape_vec(p)
            if np.all(vec == 0):
                continue

            vec = vec / (np.linalg.norm(vec) + 1e-8)
            sims = mask_feats @ vec  # 与模板批量余弦相似度

            if np.max(sims) >= threshold:
                test_new[z][labeled_v1 == p.label] = 1

    test_out = intersect_regions(test_new, test_volume_label,min_v=min_size_v2,max_v=max_size_v2*max_height)
    # test_out = intersect_regions_zexpand(test_new, test_volume_label)

    return test_out

def filter_connected_regions_shape_old(
    test_volume_label, mask_label,
    threshold=0.5, min_ratio=0.2, max_height=10
):
    """
    综合版：结合 Hu 矩 + 几何 shape 特征的形状相似筛选。

    参数:
        test_volume_label: np.ndarray, [D,H,W]
        mask_label: np.ndarray, [D,H,W]
        threshold: float，相似度阈值（余弦相似度）
        min_ratio: float，最小面积比例阈值
        max_height: int，Z方向最大厚度（传给 intersect 过滤）
    返回:
        test_out: np.ndarray, [D,H,W]
    """

    D, H, W = test_volume_label.shape
    test_new = np.zeros_like(test_volume_label, dtype=np.uint8)

    # === 1️⃣ 定义综合 shape+Hu 向量提取函数 ===
    def get_shape_vec(p):
        """
        从 skimage.measure.regionprops 的 RegionProperties 提取形状特征 + Hu 矩

        返回：
            feat: np.ndarray[float32], 形状 (D,)
        """

        eps = 1e-8

        # --- 基本几何特征 ---
        area = float(p.area)
        perim = float(p.perimeter) if p.perimeter is not None else 0.0

        # 圆度 (0~1，越接近1越像圆)
        if perim > 0:
            circularity = 4.0 * np.pi * area / (perim ** 2 + eps)
        else:
            circularity = 0.0

        # 长宽比 (主轴/次轴) —— 拉伸程度
        if p.minor_axis_length is not None and p.minor_axis_length > 0:
            aspect = float(p.major_axis_length) / float(p.minor_axis_length + eps)
        else:
            aspect = 1.0  # 看成接近圆

        # bbox 紧致程度：区域面积 / 外接矩形面积
        # extent 本身就是 area / bbox_area
        compact = float(p.extent) if p.extent is not None else 0.0

        # 凸性：区域面积 / 凸包面积
        solidity = float(p.solidity) if p.solidity is not None else 0.0

        # 偏心率：0 接近圆，接近 1 强拉伸
        ecc = float(p.eccentricity) if p.eccentricity is not None else 0.0

        # Euler 数：连通成分数 - 孔洞数（可以体现孔洞多不多）
        euler = float(p.euler_number)
        # 如果你希望“形状不含尺寸信息”，可以不放 area；
        # 如果也希望考虑大小，可以加一个 log 尺寸特征：
        log_area = np.log1p(area)  # 这个是尺寸相关的特征，可选

        # --- Hu 矩（自带平移/缩放/旋转不变性） ---
        # skimage 已经帮你算好 p.moments_hu
        hu = np.array(p.moments_hu, dtype=np.float64).ravel()  # 长度 7
        hu_norm=hu[0:3]
        # # log 变换（标准做法，稳定数量级）
        # hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
        #
        # # 依据经验对不同阶 Hu 做一个粗略缩放，避免某一维量级过大
        # # 大致假定：
        # #   hu_log[0] ~ O(1),
        # #   hu_log[1] ~ O(2),
        # #   hu_log[2..3] ~ O(3),
        # #   hu_log[4..5] ~ O(4),
        # #   hu_log[6] ~ O(5)
        # scale_factors = np.array([1, 2, 3, 3, 4, 4, 5], dtype=np.float32)
        # hu_norm = (hu_log / scale_factors).astype(np.float32)

        # --- 拼接特征 ---
        # 这里把“易解释的几何特征”放前面，Hu 放后面
        # 你之前给 circularity 乘了 5，我这里也放大一点权重
        geom_feat = np.array([
            circularity,  # 放大圆度权重
            (aspect),  # aspect 比较大，用 log 压一下
            compact,  # [0,1]
            solidity,  # [0,1]
            ecc,  # [0,1)
            # log_area,  # 尺寸信息（如不想考虑大小，可去掉这一项）
            euler,  # 孔洞/拓扑信息
        ], dtype=np.float32)

        feat = np.concatenate([geom_feat, hu_norm]).astype(np.float32)

        return feat

    def check_vec_condition(mask_feats, vec, threshold=0.3, eps=1e-8):
        """
        mask_feats : (N, D)
        vec        : (D,)
        threshold  : float

        return : bool
        """

        mask_feats = np.asarray(mask_feats)
        vec = np.asarray(vec)

        # -------------------------
        # 条件 1：存在某个模板完全匹配
        # -------------------------
        error = np.abs(vec[None, :] - mask_feats) / (np.abs(mask_feats) + eps)

        cond1 = np.any(np.all(error <= threshold, axis=1))

        # -------------------------
        # 条件 2：靠近整体统计范围
        # -------------------------
        mean_feat = mask_feats.mean(axis=0)
        max_feat = mask_feats.max(axis=0)
        min_feat = mask_feats.min(axis=0)

        alpha = (1.0 + threshold) / 2.0

        range_feat = max_feat - min_feat

        cond2 = np.all(
            np.abs(vec - mean_feat) <
            alpha * (np.abs(range_feat) + eps)
        )

        # -------------------------
        return cond1 and cond2
    # === 2️⃣ 提取 mask_label 模板特征 ===
    mask_feats = []
    mask_sizes = []
    for z in range(D):
        labeled_v2, num2 = label(mask_label[z])
        props = regionprops(labeled_v2)
        for p in props:
            vec = get_shape_vec(p)
            mask_feats.append(vec)
            mask_sizes.append(p.area)
    # print(len(mask_feats))
    if len(mask_feats) == 0:
        return np.zeros_like(test_volume_label)
    mask_feats = np.stack(mask_feats)
    # min_feat = mask_feats.min(axis=0)  # [D]
    # max_feat = mask_feats.max(axis=0)  # [D]
    min_size_v2, max_size_v2 = np.min(mask_sizes), np.max(mask_sizes)
    # print(np.shape(mask_feats))
    # === 3️⃣ 遍历 test_volume_label 每层 ===
    for z in range(D):
        labeled_v1, _ = label(test_volume_label[z])
        props_v1 = regionprops(labeled_v1)
        for p in props_v1:
            # if p.area < 10:
            #     continue
            if p.euler_number != 1:
                # print('euler mismatch')
                continue
            # 面积过滤
            if p.area < min_size_v2 * min_ratio or p.area > max_size_v2 / min_ratio:
                # print('area mismatch')
                continue

            vec = get_shape_vec(p)
            if np.all(vec == 0):
                # print('none vec')
                continue

            # vec = vec
            #
            # eps = 1e-8
            # diff = np.abs(vec[None, :] - mask_feats)  # [N, D]
            # rel_diff2 = (diff ** 2) / (mask_feats ** 2 + eps)  # 相对平方差
            # dist = np.max(rel_diff2, axis=1)  # 每个模板的平均相对误差
            # # print(dist)
            # sims = 1.0 - np.clip(dist, 0.0, 1.0)  # 转为“相似度”
            #
            # if np.max(sims) >= threshold:
            #     # print(dist)
            #     test_new[z][labeled_v1 == p.label] = 1
            # if np.all((vec >= min_feat) & (vec <= max_feat)):
            #     test_new[z][labeled_v1 == p.label] = 1
            if check_vec_condition(mask_feats, vec, 1-threshold):
                test_new[z][labeled_v1 == p.label] = 1
    # === 4️⃣ 应用交集和体积过滤 ===
    if max_height>5:
        test_out = intersect_regions(
            test_new, test_volume_label,
            min_v=min_size_v2,
            max_v=max_size_v2 * max_height
        )
    else:
        test_out = test_new

    return test_out

def filter_connected_regions_shape(
    test_volume_label,
    mask_label,
    threshold=0.5,
    min_ratio=0.2,
    max_height=10,
    w_error=1.0,
    w_range=1.0,
    num_threshold=500,
    candidate_cap="legacy",
):
    """
    根据 mask_label 的已标注区域形状特征，对 test_volume_label 中的候选区域进行打分排序，
    选择最像 mask_label 的前若干个区域。

    选择数量:
        target_num = max(num_mask + add_num, int(expand_ratio * num_mask))

    参数:
        test_volume_label: np.ndarray, [D,H,W]
        mask_label: np.ndarray, [D,H,W]
        threshold: 保留接口兼容，当前排序版本中不直接作为硬阈值使用
        min_ratio: 面积过滤比例
        max_height: Z方向最大厚度，传给 intersect_regions
        w_error: score1 权重，越大越重视与单个模板的最小误差
        w_range: score2 权重，越大越重视与整体统计范围的偏离
        add_num: 至少比原始 mask 区域多选多少个
        expand_ratio: 至少扩张到原始 mask 区域数量的多少倍
        candidate_cap: ``"legacy"`` 保留原先 500/100 上限；整数指定
            固定上限；``None`` 不额外截断。优化 profile 使用 ``None``，
            让 score cutoff 而不是旧的任意数量上限决定候选。

    返回:
        test_out: np.ndarray, [D,H,W], uint8
    """

    D, H, W = test_volume_label.shape
    eps = 1e-8
    if threshold <= 0:
        print("threshold <= 0, skip shape filtering")
        return (mask_label > 0).astype(np.uint8)

    def _label_image(binary):
        """
        兼容 scipy.ndimage.label 和 skimage.measure.label
        """
        labeled = label(binary)

        if isinstance(labeled, tuple):
            labeled = labeled[0]

        return labeled

    def get_shape_vec(p):
        area = float(p.area)
        perim = float(p.perimeter) if p.perimeter is not None else 0.0

        if perim > 0:
            circularity = 4.0 * np.pi * area / (perim ** 2 + eps)
        else:
            circularity = 0.0

        if p.minor_axis_length is not None and p.minor_axis_length > 0:
            aspect = float(p.major_axis_length) / float(p.minor_axis_length + eps)
        else:
            aspect = 1.0

        compact = float(p.extent) if p.extent is not None else 0.0
        solidity = float(p.solidity) if p.solidity is not None else 0.0
        ecc = float(p.eccentricity) if p.eccentricity is not None else 0.0
        euler = float(p.euler_number)

        hu = np.array(p.moments_hu, dtype=np.float64).ravel()
        hu_norm = hu[:3]

        geom_feat = np.array([
            circularity,
            aspect,
            compact,
            solidity,
            ecc,
            euler,
        ], dtype=np.float32)

        feat = np.concatenate([geom_feat, hu_norm]).astype(np.float32)

        return feat


    def shape_score(mask_feats, vec):
        """
        score1 和 score2 是负的相对误差，数值越接近 0 表示越相似。
        最终 score 取两者中的较大值，因此只要候选区域满足其中一种相似性标准，就会获得较高分数。

        score1:
            当前 vec 与所有 mask_feats 的相对误差，
            对每个模板取 max(error)，再取 min。

        score2:
            当前 vec 相对 mask_feats 整体均值和范围的偏离。
        """

        mask_feats_np = np.asarray(mask_feats, dtype=np.float32)
        vec_np = np.asarray(vec, dtype=np.float32)

        error = np.abs(vec_np[None, :] - mask_feats_np) / (
            np.abs(mask_feats_np) + eps
        )

        score1 = -1*np.min(np.max(error, axis=1))

        mean_feat = mask_feats_np.mean(axis=0)
        max_feat = mask_feats_np.max(axis=0)
        min_feat = mask_feats_np.min(axis=0)
        range_feat = max_feat - min_feat

        score2 = -1*np.max(
            np.abs(vec_np - mean_feat) / (np.abs(mean_feat) + eps)
        )

        return float(score1), float(score2)


    mask_feats = []
    mask_sizes = []

    for z in range(D):
        labeled_mask = _label_image(mask_label[z] > 0)
        props = regionprops(labeled_mask)

        for p in props:
            if p.area <= 0:
                continue

            vec = get_shape_vec(p)

            if np.all(vec == 0):
                continue

            mask_feats.append(vec)
            mask_sizes.append(p.area)

    if len(mask_feats) == 0:
        return np.zeros_like(test_volume_label, dtype=np.uint8)

    mask_feats = np.stack(mask_feats, axis=0)
    mask_sizes = np.asarray(mask_sizes)

    num_mask = len(mask_feats)
    min_size_v2 = np.min(mask_sizes)
    max_size_v2 = np.max(mask_sizes)


    candidates = []

    for z in range(D):
        labeled_test = _label_image(test_volume_label[z] > 0)
        props_test = regionprops(labeled_test)

        for p in props_test:

            if p.euler_number != 1:
                continue
            if p.area < min_size_v2 * min_ratio:
                continue
            if p.area > max_size_v2 / min_ratio:
                continue
            vec = get_shape_vec(p)
            if np.all(vec == 0):
                continue
            coords = np.where(labeled_test == p.label)

            overlap_ratio = np.mean(mask_label[z][coords] > 0)

            if overlap_ratio >= 0.8:
                continue
            score1, score2 = shape_score(mask_feats, vec)
            candidates.append({
                "score": np.max([score1*w_error,score2*w_range]),
                "score1": score1,
                "score2": score2,
                "z": z,
                "coords": coords,
                "area": p.area,
            })

    if len(candidates) == 0:
        return np.zeros_like(test_volume_label, dtype=np.uint8)

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    scores = np.array([c["score"] for c in candidates], dtype=np.float32)

    score_min = scores.min()
    score_max = scores.max()
    score_range = score_max - score_min

    score_cutoff = score_max - score_range * threshold
    target_num = int(np.sum(scores >= score_cutoff))

    if candidate_cap == "legacy":
        if num_mask < num_threshold:
            add_num_max = 500
        else:
            add_num_max = 100
        target_num = min(target_num, add_num_max)
    elif candidate_cap is not None:
        candidate_cap = int(candidate_cap)
        if candidate_cap < 1:
            raise ValueError("candidate_cap must be positive, None, or 'legacy'")
        target_num = min(target_num, candidate_cap)
    target_num = min(target_num, len(candidates))

    test_new = np.zeros_like(test_volume_label, dtype=np.uint8)

    for c in candidates[:target_num]:
        z = c["z"]
        coords = c["coords"]
        test_new[z][coords] = 1

    print(f"mask num = {num_mask},add num= {target_num}")
    return test_new.astype(np.uint8)


def smooth_and_threshold(volume, sigma=(1, 2, 2), threshold=0.7):
    """
    对3D二值体数据进行高斯模糊，并选取高于阈值的区域。

    参数:
        volume: np.ndarray, shape (D,H,W), 值为0/1
        sigma: tuple(float), 高斯滤波标准差 (z方向, y方向, x方向)
        threshold: float, 阈值（选取模糊后 > threshold 的体素）

    返回:
        smoothed_mask: np.ndarray, 经过滤波+阈值后的二值体
    """
    assert volume.ndim == 3, "输入必须是3D数组"
    volume = volume.astype(np.float32)

    print(f"⛓️ 对体数据进行高斯平滑：sigma={sigma}")
    blurred = gaussian_filter(volume, sigma=sigma, mode="reflect")

    print(f"📊 平滑后范围: min={blurred.min():.3f}, max={blurred.max():.3f}")
    smoothed_mask = (blurred > threshold).astype(np.uint8)
    print(f"✅ 阈值 {threshold} 后保留比例: {smoothed_mask.mean():.4f}")

    return smoothed_mask

def break_thin_connections(volume, radius=1):
    """
    使用3D形态学开运算 (Erosion + Dilation) 去掉细连接。
    参数:
        volume: np.ndarray, 3D二值图
        radius: int, 控制断开的尺度（越大断得越狠）
    返回:
        断开细连接后的mask
    """
    struct = generate_binary_structure(3, 1)  # 6邻域
    for _ in range(radius):
        volume = binary_opening(volume, structure=struct)
    return volume.astype(np.uint8)

def normalize_shape(volume):
    """
    将三维mask标准化为旋转、平移、缩放不变的形状表示。
    步骤:
      1. 提取体素坐标
      2. 去中心化
      3. PCA 主轴对齐
      4. 缩放到固定边界
      5. 重采样为固定大小的标准体
    返回:
      norm_vol: np.ndarray [N,N,N]
    """
    coords = np.argwhere(volume > 0)
    if len(coords) == 0:
        return np.zeros((32, 32, 32), dtype=np.float32)

    # 去中心化
    coords = coords - coords.mean(axis=0, keepdims=True)

    # PCA 对齐主轴
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, np.argsort(-eigvals)]  # 降序排列主轴
    aligned = coords @ eigvecs

    # 归一化尺度（等比缩放到 [-1,1] 立方体）
    max_extent = np.max(np.abs(aligned))
    if max_extent < 1e-5:
        return np.zeros((32, 32, 32), dtype=np.float32)
    aligned /= max_extent

    # 投到标准体素网格
    grid_size = 32
    norm_vol = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    indices = np.clip(((aligned + 1) * (grid_size - 1) / 2).round().astype(int), 0, grid_size - 1)
    norm_vol[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
    return norm_vol

def dice_similarity(a, b):
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return 2 * inter / denom if denom > 0 else 0.0

def filter_by_3d_shape_invariant(M2_new, M2_new_temp, sim_thresh=0.1,
                                 size_min_ratio=0.5, size_max_ratio=2.0):
    """
    根据旋转&缩放不变的3D形状相似度筛选区域
    """
    from scipy.ndimage import label

    labeled_temp, num_temp = label(M2_new_temp, structure=np.ones((3, 3, 3)))
    temp_regions = [(labeled_temp == j).astype(np.uint8) for j in range(1, num_temp + 1)]
    if len(temp_regions) == 0:
        print("⚠️ 模板为空，返回空 mask")
        return np.zeros_like(M2_new)

    temp_norms = [normalize_shape(r) for r in temp_regions]
    temp_sizes = [r.sum() for r in temp_regions]
    min_size, max_size = min(temp_sizes), max(temp_sizes)

    labeled_new, num_new = label(M2_new, structure=np.ones((3, 3, 3)))
    M2_filtered = np.zeros_like(M2_new)

    for i in range(1, num_new + 1):
        # print(i)
        region = (labeled_new == i)
        size = region.sum()
        if size < min_size * size_min_ratio or size > max_size * size_max_ratio:
            continue

        norm_r = normalize_shape(region)
        sims = [dice_similarity(norm_r, tnorm) for tnorm in temp_norms]
        if len(sims) > 0 and max(sims) >= sim_thresh:
            M2_filtered[region] = 1

    return M2_filtered

def soften_center_mask_dilated(
    softnega, density=0.05, center_bias=3.0, seed=None, dilate_iter=2
):
    """
    对每层随机置0（中心更密集），然后对置零点进行膨胀。

    参数:
        softnega : np.ndarray [D,H,W]，原始 mask (0/1)
        density : float，随机去掉比例
        center_bias : float，中心加权大小
        seed : int，固定随机种子
        dilate_iter : int，膨胀迭代次数
    """
    if seed is not None:
        np.random.seed(seed)

    D, H, W = softnega.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    cy, cx = H / 2, W / 2
    dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    dist_norm = dist / dist.max()

    # 中心高、边缘低
    weight = np.exp(-center_bias * dist_norm**2)
    weight = weight / weight.sum()

    modified = softnega.copy()
    total_pixels = H * W

    for z in range(D):
        # === 1) 随机掉点 ===
        n_drop = int(total_pixels * density)
        flat_idx = np.random.choice(
            total_pixels, size=n_drop, replace=False, p=weight.ravel()
        )
        y, x = np.unravel_index(flat_idx, (H, W))

        # 记录这些掉点
        drop_mask = np.zeros((H, W), dtype=bool)
        drop_mask[y, x] = True

        # === 2) 对掉点膨胀 ===
        if dilate_iter > 0:
            drop_mask = binary_dilation(drop_mask, iterations=dilate_iter)

        # === 3) 应用到 mask ===
        modified[z][drop_mask] = 0

    return modified
import tifffile as tiff

# vol01 = tiff.imread('mask.tif')
# volume_label0 = tiff.imread('base.tif')
# test_volume_label_new=filter_connected_regions_shape( vol01, volume_label0, threshold=0.9,min_ratio=0.5)
#
# tiff.imwrite(f'out.tif', 255*test_volume_label_new)


def compute_metrics(gt, pred):
    """给定两个 0/1 mask，计算 IoU 和假阳性率 FPR"""

    gt = gt.astype(bool)
    pred = pred.astype(bool)

    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    iou = intersection / (union + 1e-8)

    fp = np.logical_and(~gt, pred).sum()
    tn = np.logical_and(~gt, ~pred).sum()
    fpr = fp / (fp + tn + 1e-8)

    return iou, fpr

def compute_metrics0(gt, pred, overlap_thr=0.1):
    """
    iou : float
    fpr : float
    """

    gt = gt.astype(bool)
    pred = pred.astype(bool)

    # =========================
    # 1. 连通域标记 gt
    # =========================
    labeled_gt, num_gt = label(gt)

    intersection = 0

    for label_id in range(1, num_gt + 1):
        comp = labeled_gt == label_id
        comp_area = comp.sum()

        if comp_area == 0:
            continue

        overlap = np.logical_and(comp, pred).sum()
        overlap_ratio = overlap / comp_area

        if overlap_ratio > overlap_thr:
            intersection += comp_area

    # =========================
    # 2. union 仍然按原来的 pixel-level union
    # =========================
    union = np.logical_or(gt, pred).sum()
    iou = intersection / (union + 1e-8)

    # =========================
    # 3. FPR 仍然按 pixel-level false positive rate
    # =========================
    fp = np.logical_and(~gt, pred).sum()
    tn = np.logical_and(~gt, ~pred).sum()
    fpr = fp / (fp + tn + 1e-8)

    return iou, fpr

import os
def evaluate_model_specs(
    gt,
    roi_num,
    pred_root_list,
    model_specs,
    *,
    celltype,
    organelletype,
    pred_filename,
    anchor_model="SparseSeg",
    verbose=True,
    label_specific=None,
):
    """
    返回:
        metrics_dict:
            {
                "SparseSeg": {"iou": [...], "fpr": [...]},
                "MitoNet":   {"iou": [...], "fpr": [...]},
                ...
            }

        labels:
            boxplot 横轴标签，例如 [1, 5, 10]

        missing_files:
            缺失文件记录
    """

    metrics_dict = {
        spec["name"]: {
            "iou": [],
            "fpr": [],
        }
        for spec in model_specs
    }

    labels = []
    missing_files = []
    ii=0
    for folder in roi_num:

        if verbose:
            print("\n" + "=" * 80)
            print(f"ROI folder = {folder}")
            print("=" * 80)

        per_folder_metrics = {
            spec["name"]: {
                "iou": [],
                "fpr": [],
            }
            for spec in model_specs
        }

        for i in pred_root_list:

            if verbose:
                print(f"\nPred root = {i}")

            # ------------------------------------------------------------
            # 保持和你原始逻辑一致：
            # 如果 SparseSeg 这个 i/folder 不存在，就跳过这个重复实验。
            # ------------------------------------------------------------
            if anchor_model is not None:
                anchor_spec = next(
                    spec for spec in model_specs
                    if spec["name"] == anchor_model
                )

                anchor_path = anchor_spec["path_template"].format(
                    celltype=celltype,
                    organelletype=organelletype,
                    i=i,
                    folder=folder,
                    pred_filename=pred_filename,
                )

                if not os.path.exists(anchor_path):
                    if verbose:
                        print(f"[skip repeat] missing anchor {anchor_model}: {anchor_path}")

                    missing_files.append({
                        "model": anchor_model,
                        "i": i,
                        "folder": folder,
                        "path": anchor_path,
                    })
                    continue

            # ------------------------------------------------------------
            # 对所有模型计算 metrics
            # ------------------------------------------------------------
            for spec in model_specs:

                model_name = spec["name"]
                path_template = spec["path_template"]
                loader = spec["loader"]
                loader_kwargs = spec.get("loader_kwargs", {})

                pred_path = path_template.format(
                    celltype=celltype,
                    organelletype=organelletype,
                    i=i,
                    folder=folder,
                    pred_filename=pred_filename,
                )

                if not os.path.exists(pred_path):
                    if verbose:
                        print(f"[skip] {model_name}: {pred_path}")

                    missing_files.append({
                        "model": model_name,
                        "i": i,
                        "folder": folder,
                        "path": pred_path,
                    })
                    continue

                pred_bin = loader(pred_path, **loader_kwargs)

                if pred_bin.shape != gt.shape:
                    raise ValueError(
                        f"Shape mismatch for {model_name}\n"
                        f"path       : {pred_path}\n"
                        f"gt shape   : {gt.shape}\n"
                        f"pred shape : {pred_bin.shape}"
                    )

                iou, fpr = compute_metrics(gt, pred_bin)

                per_folder_metrics[model_name]["iou"].append(iou)
                per_folder_metrics[model_name]["fpr"].append(fpr)

                if verbose:
                    print(
                        f"[ok] {model_name:<10s} "
                        f"IoU={iou:.4f}, FPR={fpr:.6f}"
                    )

        # ------------------------------------------------------------
        # 如果 anchor model 在这个 folder 下至少有一个结果，则保留这个 folder
        # ------------------------------------------------------------
        if anchor_model is not None:
            if len(per_folder_metrics[anchor_model]["iou"]) == 0:
                if verbose:
                    print(f"[skip folder] {folder}: no valid {anchor_model} result")
                continue
        if label_specific is None:
            labels.append(folder)
        else:
            labels.append(label_specific[ii])
            ii = ii+1
            # labels.append(label_specific[folder])

        for spec in model_specs:
            model_name = spec["name"]
            metrics_dict[model_name]["iou"].append(
                per_folder_metrics[model_name]["iou"]
            )
            metrics_dict[model_name]["fpr"].append(
                per_folder_metrics[model_name]["fpr"]
            )

    return metrics_dict, labels, missing_files
