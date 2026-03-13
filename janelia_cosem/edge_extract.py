import numpy as np
import tifffile as tiff
from scipy.ndimage import label, binary_dilation,convolve,binary_erosion,binary_closing
from tqdm import tqdm
import cv2

def filter_by_erosion_ratio(slice_bin, min_size=50, max_size=10000, ratio_thresh=0.4):
    """
    过滤掉块状区域，仅保留细长或边缘状区域。
    基于腐蚀交集比： ratio = |A ∩ erosion(A)| / |A|
    若 ratio < ratio_thresh，则认为是细结构（被腐蚀掉较多）。

    参数:
        slice_bin: 2D 二值图 (H, W)
        min_size, max_size: 面积过滤阈值
        ratio_thresh: 交集比例阈值 (默认 0.7，越小保留越细的区域)

    返回:
        mask_out: 过滤后的二值图
    """
    labeled, num = label(slice_bin)
    if num == 0:
        return np.zeros_like(slice_bin, dtype=np.uint8)

    sizes = np.bincount(labeled.ravel())
    if len(sizes) <= 1:
        return np.zeros_like(slice_bin, dtype=np.uint8)

    # 整体腐蚀
    eroded = binary_erosion(slice_bin, structure=np.ones((3, 3)))

    mask_out = np.zeros_like(slice_bin, dtype=bool)

    for i in range(1, len(sizes)):
        area = sizes[i]
        if not (min_size <= area <= max_size):
            continue

        region_mask = (labeled == i)
        eroded_overlap = (region_mask & eroded).sum()
        ratio = eroded_overlap / (area + 1e-6)

        # 腐蚀后剩得少（细） → 保留
        if ratio < ratio_thresh:
            mask_out |= region_mask

    return mask_out.astype(np.uint8)
def closed_region_fill_2d(img2d, dilate_iter=1, close_iter=2):
    """
    1. dilation + closing 修补断裂
    2. 从边缘所有 0 区域 flood fill
    3. 取反得到封闭区域
    """
    img_bin = (img2d > 0).astype(np.uint8)

    # --- 1. 膨胀 + 闭合 ---
    # dilated = binary_dilation(img_bin, structure=np.ones((3,3)), iterations=dilate_iter)
    closed = binary_closing(img_bin, structure=np.ones((3,3)), iterations=close_iter).astype(np.uint8)

    # --- 2. flood fill 从所有边界像素开始 ---
    padded = np.pad(closed, pad_width=1, mode='constant', constant_values=0)
    flood = padded.copy()
    mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8)

    H, W = flood.shape
    # 从四条边扫描并 flood fill 所有外部0
    for y in range(H):
        for x in [0, W-1]:
            if flood[y, x] == 0:
                cv2.floodFill(flood, mask, (x, y), 1)
    for x in range(W):
        for y in [0, H-1]:
            if flood[y, x] == 0:
                cv2.floodFill(flood, mask, (x, y), 1)

    # --- 3. 取反得到封闭区域 ---
    # flood_inv = cv2.bitwise_not(flood)
    filled = flood[1:-1, 1:-1] > 0  # 去掉padding
    out=(filled<1).astype(np.uint8)
    eroded = binary_erosion(out, structure=np.ones((3, 3)), iterations=2)
    dilated = binary_dilation(eroded, structure=np.ones((3, 3)), iterations=2)
    return dilated
def closed_region_fill_volume(volume, close_iter=2):
    """
    对3D体逐层执行 closed_region_fill_2d
    """
    D, H, W = volume.shape
    result = np.zeros_like(volume, dtype=np.uint8)
    for z in range(D):
        result[z] = closed_region_fill_2d(volume[z], close_iter)
        # if z % 10 == 0:
        #     print(f"处理切片 {z+1}/{D}")
    return result

def get_edge_region(input_image, threshold = 0.7,min_size = 100,max_size = 20000,ratio_thresh = 0.25):

    edge_vol = input_image.astype(np.float32)

    if edge_vol.max() > 1:
        edge_vol /= edge_vol.max()
    D, H, W = edge_vol.shape
    print(f"✅ 数据维度: {edge_vol.shape}")
    filtered = np.zeros_like(edge_vol, dtype=np.uint8)
    print("🧠 正在逐层处理 ...")
    for z in tqdm(range(D)):
        img = edge_vol[z]
        binary = (img > threshold).astype(np.uint8)
        filtered[z] = filter_by_erosion_ratio(
            binary,
            min_size=min_size,
            max_size=max_size,
        )
    closed_only = closed_region_fill_volume(filtered, close_iter=1)

    return closed_only

from skimage.measure import label, regionprops,perimeter
def filter_edge_area_by_perimeter_fast(edge_Line, edge_Area, ratio_low=1.8, ratio_high=2.2):
    """
    高效版：按周长比例筛选 edge_Area 区域。
    利用 regionprops + 标签矩阵一次性统计交集。

    参数:
        edge_Line, edge_Area: np.ndarray(bool) 3D 二值图像
        ratio_low, ratio_high: float, 周长比例范围 (例如 1.8 ~ 2.2)

    返回:
        filtered_Area: np.ndarray(bool)
    """
    assert edge_Line.shape == edge_Area.shape, "shape 不匹配"
    D, H, W = edge_Area.shape

    # === 1️⃣ 连通区域标记 ===
    labeled_line, n_line = label(edge_Line, return_num=True, connectivity=1)
    labeled_area, n_area = label(edge_Area, return_num=True, connectivity=1)
    print(f"检测到 {n_line} 个 line 区域, {n_area} 个 area 区域")

    # === 2️⃣ 提取每个区域的周长 ===
    perim_line = np.zeros(n_line + 1)
    perim_area = np.zeros(n_area + 1)

    for z in range(D):
        if np.any(edge_Line[z]):
            for prop in regionprops(labeled_line[z]):
                perim_line[prop.label] += perimeter(prop.image, neighbourhood=8)
        if np.any(edge_Area[z]):
            for prop in regionprops(labeled_area[z]):
                perim_area[prop.label] += perimeter(prop.image, neighbourhood=8)

    # === 3️⃣ 统计每个 area 与哪些 line 相交 ===
    overlap_pairs = np.stack([labeled_area.ravel(), labeled_line.ravel()], axis=1)
    overlap_pairs = overlap_pairs[(overlap_pairs[:, 0] > 0) & (overlap_pairs[:, 1] > 0)]

    # 利用唯一组合提取交集关系
    unique_pairs = np.unique(overlap_pairs, axis=0)

    # === 4️⃣ 检查周长比例条件 ===
    keep_areas = set()
    for a_id, l_id in unique_pairs:
        c2 = perim_area[a_id]
        c1 = perim_line[l_id]
        if c2 == 0 or c1 == 0:
            continue
        ratio = c1 / (c2 + 1e-6)
        if ratio_low <= ratio <= ratio_high:
            keep_areas.add(a_id)

    # === 5️⃣ 输出保留区域 ===
    filtered = np.isin(labeled_area, list(keep_areas)).astype(np.uint8)
    print(f"保留 {len(keep_areas)} 个符合条件的区域。")
    return filtered

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image
from scipy import ndimage
def fill_edge_volume_by_region(edge_volume, min_size=5, max_ratio=3.0):
    """
    对 3D edge_volume [D,H,W] 逐 slice 调用 fill_edge_slice_by_region
    """

    D = edge_volume.shape[0]
    filled_volume = np.zeros_like(edge_volume, dtype=bool)

    for z in range(D):
        if np.any(edge_volume[z]):
            filled_volume[z] = fill_edge_slice_by_region(
                edge_volume[z],
                min_size=min_size,
                max_ratio=max_ratio
            )

    return filled_volume
def fill_edge_slice_by_region(edge_slice, min_size=5, max_ratio=3.0):

    labeled, num = ndimage.label(edge_slice)
    filled = np.zeros_like(edge_slice, dtype=bool)

    props = regionprops(labeled)

    for prop in props:

        area = prop.area
        if area < min_size:
            continue

        minr, minc, maxr, maxc = prop.bbox

        # 只取 bbox 区域
        region = (labeled[minr:maxr, minc:maxc] == prop.label)

        hull = convex_hull_image(region)
        hull_area = hull.sum()

        if hull_area >= max_ratio * area:
            filled[minr:maxr, minc:maxc] = True

    return filled

def filter_edge_area_by_bbox_iou_2d_vectorized(edge_Line, edge_Area, iou_thresh=0.8, line_fill_thresh=0.5,method=0):
    """
    每层 2D：使用 NumPy 向量化计算 IoU，
    筛选与 edge_Line 有高重合度 (IoU > 阈值) 的 edge_Area 区域。
    同时过滤掉在自身 bbox 内填充比例过大的 line 区域（非细线）。

    参数:
        edge_Line, edge_Area: np.ndarray(bool) 3D 二值图像
        iou_thresh: float, IoU 阈值（默认 0.8）
        line_fill_thresh: float, line 区域在自身 bbox 中的填充比例上限（默认 0.8）

    返回:
        filtered_Area: np.ndarray(uint8)
    """
    assert edge_Line.shape == edge_Area.shape, "shape 不匹配"
    D, H, W = edge_Area.shape
    filtered = np.zeros_like(edge_Area, dtype=np.uint8)
    total_kept_regions = 0

    for z in range(D):
        line_z = edge_Line[z]
        area_z = edge_Area[z]
        if not (np.any(line_z) and np.any(area_z)):
            continue
        # line_z = fill_edge_slice_by_region(line_z)
        labeled_line, n_line = label(line_z, return_num=True, connectivity=1)
        labeled_area, n_area = label(area_z, return_num=True, connectivity=1)

        min_size=20
        keep_line = np.zeros_like(line_z, dtype=bool)
        keep_area = np.zeros_like(area_z, dtype=bool)

        # area 整体 mask（用于 line 计算重合）
        area_mask = area_z.astype(bool)
        line_mask = line_z.astype(bool)

        # # ---------- 2️⃣ 处理 line regions ----------
        # for i in range(1, n_line + 1):
        #     region = (labeled_line == i)
        #     size = region.sum()
        #
        #     if size < min_size:
        #         continue
        #
        #     overlap = np.logical_and(region, area_mask).sum()
        #     overlap_ratio = overlap / size
        #
        #     if overlap_ratio >= iou_thresh:
        #         keep_line |= region

        # ---------- 3️⃣ 处理 area regions ----------
        for j in range(1, n_area + 1):
            region = (labeled_area == j)
            size = region.sum()

            if size < min_size:
                continue

            overlap = np.logical_and(region, line_mask).sum()
            overlap_ratio = overlap / size

            if overlap_ratio >= iou_thresh:
                keep_area |= region


        filtered[z] = keep_line | keep_area
        _, z_num = label(filtered, return_num=True, connectivity=1)
        total_kept_regions+=z_num
        # if n_line == 0 or n_area == 0:
        #     continue
        #
        # props_line = regionprops(labeled_line)
        # props_area = regionprops(labeled_area)
        #
        # # === 提取 bbox ===
        # bboxes_area = np.array([p.bbox for p in props_area])  # [Nₐ, 4]
        # area_ids = [p.label for p in props_area]
        #
        # # === 对 line 过滤：去掉在自身 bbox 中填充比例过大的 ===
        # bboxes_line, valid_line_labels = [], []
        # for p in props_line:
        #
        #     if method ==0:
        #         fill_ratio = p.area / (p.area_filled + 1e-6)
        #     elif method ==1:
        #         if p.euler_number != 1:
        #             continue
        #         fill_ratio = p.solidity
        #     elif method ==2:
        #         if p.euler_number != 1:
        #             continue
        #         y1, x1, y2, x2 = p.bbox
        #         bbox_area = (y2 - y1) * (x2 - x1)
        #         fill_ratio=p.area / (bbox_area + 1e-6)
        #     if fill_ratio <= line_fill_thresh:
        #         bboxes_line.append(p.bbox)
        #         valid_line_labels.append(p.label)
        # # print('a')
        # if len(bboxes_line) == 0:
        #     continue  # 该层没有合法线
        #
        # bboxes_line = np.array(bboxes_line)
        #
        # # === 计算 IoU（矢量化）===
        # ya1, xa1, ya2, xa2 = np.split(bboxes_area[:, None, :], 4, axis=2)
        # yl1, xl1, yl2, xl2 = np.split(bboxes_line[None, :, :], 4, axis=2)
        #
        # inter_h = np.maximum(0, np.minimum(ya2, yl2) - np.maximum(ya1, yl1))
        # inter_w = np.maximum(0, np.minimum(xa2, xl2) - np.maximum(xa1, xl1))
        # inter_area = inter_h * inter_w
        # area_a = (ya2 - ya1) * (xa2 - xa1)
        # area_l = (yl2 - yl1) * (xl2 - xl1)
        # iou = inter_area / (area_a + area_l - inter_area + 1e-6)
        # # iou = inter_area / (area_a + 1e-6)
        #
        # # === 取每个 area 的最大 IoU ===
        # max_iou_per_area = iou.max(axis=1).ravel()
        # keep_mask = max_iou_per_area >= iou_thresh
        # keep_labels = [area_ids[i] for i in np.where(keep_mask)[0]]
        #
        # total_kept_regions += len(keep_labels)
        #
        # # === 输出该层 ===
        # if len(keep_labels) > 0:
        #     filtered[z] = np.isin(labeled_area, keep_labels).astype(np.uint8)

    print(f"✅ 处理完成，共保留 {total_kept_regions} 个符合 IoU>{iou_thresh} 的区域")
    return filtered
