import torch
import numpy as np
from tqdm import tqdm
from scipy.ndimage import binary_closing,binary_erosion, binary_dilation, label as nd_label,gaussian_filter, distance_transform_edt
from skimage.transform import downscale_local_mean
from get_inputfeature import extract_2d_features_from_patch
from get_inputfeature_new import extract_stack_features

def pad_to_multiple_of(volume, multiple=4):
    """
    将 volume 的最后两个维度 (H, W) pad 到 multiple 的倍数
    其余维度保持不变

    volume: np.ndarray [..., H, W]
    return:
        padded_volume: np.ndarray [..., H_new, W_new]
        pad_info: dict
    """
    *prefix_dims, H, W = volume.shape

    new_H = ((H + multiple - 1) // multiple) * multiple
    new_W = ((W + multiple - 1) // multiple) * multiple

    pad_top = (new_H - H) // 2
    pad_bottom = new_H - H - pad_top
    pad_left = (new_W - W) // 2
    pad_right = new_W - W - pad_left

    # 构造 pad_width，只 pad 最后两个维度
    pad_width = [(0, 0)] * len(prefix_dims) + [
        (pad_top, pad_bottom),
        (pad_left, pad_right),
    ]

    padded = np.pad(
        volume,
        pad_width=pad_width,
        mode="constant",
        constant_values=0,
    )

    pad_info = {
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
    }

    return padded, pad_info

def unpad_volume(padded_volume, pad_info):
    """
    对 padded_volume 的最后两个维度 (H, W) 进行 unpad
    其余维度保持不变

    padded_volume: (..., H, W)
    pad_info: dict with keys pad_top, pad_bottom, pad_left, pad_right
    """
    pad_top    = pad_info["pad_top"]
    pad_bottom = pad_info["pad_bottom"]
    pad_left   = pad_info["pad_left"]
    pad_right  = pad_info["pad_right"]

    h_slice = slice(
        pad_top,
        -pad_bottom if pad_bottom > 0 else None
    )
    w_slice = slice(
        pad_left,
        -pad_right if pad_right > 0 else None
    )

    return padded_volume[..., h_slice, w_slice]

def feature_volume_generation(volume_np, thickness=2):
    """
    volume_np: np.ndarray [D, H, W], float32, normalized to [0, 1]

    Returns:
        restored_feature_volume: np.ndarray [D, F, H, W]
    """
    padded_volume = volume_np
    # padded_volume, pad_info = pad_to_multiple_of(volume_np, multiple=4)
    D, H, W = padded_volume.shape
    feature_volume = None  # 先不分配，等知道 F 再说

    for z in tqdm(
        range(thickness, D - thickness),
        desc="Extracting features per slice"
    ):
        # === 取 Z-stack ===
        slice_img = padded_volume[
            z - thickness : z + thickness + 1
        ]  # shape: (2*thickness+1, H, W)

        # === 提特征（你原来的逻辑）===
        # _, feats_stack = extract_2d_features_from_patch(
        #     slice_img,
        #     aggregate_mode="gaussian",
        #     sigma_z=0.8,
        #     denoise_tv=0.05,
        #     sigmas_gauss=(1.0, 2.0, 4.0),
        #     sigmas_hessian=(1.0, 2.0, 4.0),
        #     win_local_stats=9,
        #     st_sigma=1.0,
        # )
        feats_stack = extract_stack_features(slice_img)

        # feats_stack: (F, H, W)

        # === 第一次才知道 F，初始化 volume ===
        if feature_volume is None:
            F = feats_stack.shape[0]
            feature_volume = np.zeros(
                (D, F, H, W),
                dtype=np.float32
            )

        # === 放回对应 z ===
        feature_volume[z] = feats_stack

    # === 去 padding（只去 Z/H/W，不动 feature 维）===
    # restored_feature_volume = unpad_volume(feature_volume, pad_info)

    return feature_volume

def infer_volume_edges(volume_np, model, threshold=0.5, patch_size=(128, 128), stride=(64, 64), thickness=2):
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
            slice_img = padded_volume[(z - thickness):(z + thickness + 1)]  # [3, H, W]
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

# def infer_volume_edges_whole(feature_volume, model, thickness=2):
#     """
#     volume_np: [D, H, W], float32, normalized to [0, 1]
#     model: model(input: [1, 1, H, W]) → pred [1, 1, H, W]
#     Returns:
#         edge_volume: [D, H, W], uint8 0/1 mask
#     """
#     # volume_np = feature_volume[:, 0, :, :]
#     padded_volume, pad_info = pad_to_multiple_of(feature_volume, multiple=4)
#     D,F, H, W = padded_volume.shape
#
#     pred_volume = np.zeros((D, H, W), dtype=np.float32)
#     edge_volume = np.zeros((D, H, W), dtype=np.float32)
#     count_volume = np.zeros((D, H, W), dtype=np.float32)
#
#     model.eval()
#     model.cuda()
#
#     # feature_volume = feature_volume_generation(volume_np, thickness=2)
#     with torch.no_grad():
#         for z in tqdm(range(thickness, D - thickness), desc="Predicting edges per slice"):
#             # slice_img = padded_volume[(z - thickness):(z + thickness + 1)]  # [3, H, W]
#             # # slice_img = padded_volume[(z-1):(z+2)]  # [H, W]
#             # _, feats_stack = extract_2d_features_from_patch(
#             #     slice_img,  # [Z,H,W] 例如 thickness=1 -> Z=3
#             #     aggregate_mode="gaussian",
#             #     sigma_z=0.8,
#             #     denoise_tv=0.05,  # 可设 0 关闭
#             #     sigmas_gauss=(1.0, 2.0, 4.0),
#             #     sigmas_hessian=(1.0, 2.0, 4.0),
#             #     win_local_stats=9,
#             #     st_sigma=1.0,
#             # )
#             feats_stack=padded_volume[z]
#
#             input_tensor = torch.from_numpy(feats_stack).unsqueeze(0).float().cuda()  # [1, F, H, W]
#
#             pred = model(input_tensor)  # [1, 1, H, W]
#             pred_prob_stack = torch.sigmoid(pred).squeeze(0).cpu().numpy()  # [2*thickness+1, H, W]
#
#             # === 对齐并累积回 volume ===
#             z_start = z - 0
#             z_end = z + 0 + 1
#             edge_volume[z_start:z_end] = pred_prob_stack[1:2, :, :]
#             pred_volume[z_start:z_end] = pred_prob_stack[0:1, :, :]
#             count_volume[z_start:z_end] += 1
#
#             # === 平均重叠区域 ===
#         count_volume[count_volume == 0] = 1
#         edge_volume /= count_volume
#         pred_volume /= count_volume
#         # === 去除 padding 并平滑 ===
#         restored = unpad_volume(pred_volume, pad_info)
#         restored2 = unpad_volume(edge_volume, pad_info)
#         blurred = gaussian_filter(restored, sigma=(1, 3, 3), mode="reflect")
#
#     return blurred, restored2

def infer_volume_edges_whole(feature_volume, model, thickness=2, batch_size=8):

    D, F, H, W = feature_volume.shape

    pred_volume = np.zeros((D, H, W), dtype=np.float32)
    edge_volume = np.zeros((D, H, W), dtype=np.float32)

    model.eval().cuda()

    with torch.no_grad():

        zs = list(range(thickness, D - thickness))

        for i in tqdm(range(0, len(zs), batch_size), desc="Batch inference"):

            batch_z = zs[i:i + batch_size]

            batch_feats = []
            pad_infos = []

            # =========================
            # 1️⃣ 读取 batch
            # =========================
            for z in batch_z:
                feats = feature_volume[z]  # lazy load
                feats, pad_info = pad_to_multiple_of(feats, multiple=4)

                batch_feats.append(feats)
                pad_infos.append(pad_info)

            batch_feats = np.stack(batch_feats, axis=0)  # (B,F,H,W)

            # =========================
            # 2️⃣ 一次性丢进 GPU
            # =========================
            input_tensor = torch.from_numpy(batch_feats).float().cuda()

            pred = model(input_tensor)  # (B,2,H,W)
            pred_prob = torch.sigmoid(pred).cpu().numpy()

            # =========================
            # 3️⃣ 写回
            # =========================
            for j, z in enumerate(batch_z):

                out = pred_prob[j]  # (2,H,W)

                out = unpad_volume(out, pad_infos[j])

                edge_volume[z] = out[1]
                pred_volume[z] = out[0]

    # =========================
    # 4️⃣ smooth
    # =========================
    blurred = gaussian_filter(pred_volume, sigma=(1, 3, 3), mode="reflect")

    return blurred, edge_volume



def infer_volume_edges_patchwise_old(
    feature_volume,
    model,
    thickness=2,
    patch_size=160,
    stride=120
):

    D, F, H, W = feature_volume.shape

    pred_volume = np.zeros((D, H, W), dtype=np.float32)
    edge_volume = np.zeros((D, H, W), dtype=np.float32)

    model.eval().cuda()

    with torch.no_grad():

        for z in tqdm(range(thickness, D - thickness), desc="Patch inference"):

            feat = feature_volume[z]
            if not isinstance(feat, torch.Tensor):
                feat = torch.from_numpy(feat)
            feat = feat.float().numpy()   # (F,H,W)

            # ========= slice buffer =========
            pred_slice = np.zeros((H, W), dtype=np.float32)
            edge_slice = np.zeros((H, W), dtype=np.float32)
            count_slice = np.zeros((H, W), dtype=np.float32)

            # ========= patch =========
            for x in range(0, H, stride):
                x = min(x, H - patch_size)

                for y in range(0, W, stride):
                    y = min(y, W - patch_size)

                    patch = feat[:, x:x+patch_size, y:y+patch_size]

                    input_tensor = torch.from_numpy(patch).unsqueeze(0).float().cuda()

                    with torch.amp.autocast("cuda"):
                        pred = model(input_tensor)

                    pred_prob = torch.sigmoid(pred).squeeze(0).cpu().numpy()

                    pred_slice[x:x+patch_size, y:y+patch_size] += pred_prob[0]
                    edge_slice[x:x+patch_size, y:y+patch_size] += pred_prob[1]
                    count_slice[x:x+patch_size, y:y+patch_size] += 1

            # ========= average =========
            count_slice[count_slice == 0] = 1
            pred_slice /= count_slice
            edge_slice /= count_slice

            pred_volume[z] = pred_slice
            edge_volume[z] = edge_slice

    # ========= smooth =========
    blurred = gaussian_filter(pred_volume, sigma=(1, 3, 3), mode="reflect")

    return blurred, edge_volume

def _make_starts(length, patch_size, stride):
    """
    生成 patch 起点，保证覆盖到最后，同时避免重复起点。
    """
    if length <= patch_size:
        return [0]

    starts = list(range(0, length - patch_size + 1, stride))

    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)

    return sorted(set(starts))

def infer_volume_edges_patchwise(
    feature_volume,
    model,
    thickness=2,
    patch_size=160,
    stride=120,
    batch_size=16,
    device="cuda",
    use_amp=True,
    amp_dtype=torch.float16,
    smooth_sigma=(1, 3, 3),
):
    """
    优化点：
    1. 每个 slice 内 batch 推理多个 patch，而不是一个 patch forward 一次
    2. 每个 slice 的 feature 只搬到 GPU 一次
    3. pred_slice / edge_slice / count_slice 在 GPU 上累加，最后一次性拷回 CPU
    4. 使用 torch.inference_mode()
    5. 可选 AMP 混合精度
    """

    D, F, H, W = feature_volume.shape

    pred_volume = np.zeros((D, H, W), dtype=np.float32)
    edge_volume = np.zeros((D, H, W), dtype=np.float32)

    model = model.to(device)
    model.eval()

    # 固定输入尺寸时，cuDNN benchmark 通常有助于卷积选择更快算法
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    x_starts = _make_starts(H, patch_size, stride)
    y_starts = _make_starts(W, patch_size, stride)
    coords = [(x, y) for x in x_starts for y in y_starts]

    print(f"Patch num per slice: {len(coords)}")
    print(f"Patch batch size: {batch_size}")

    with torch.inference_mode():
        for z in tqdm(range(thickness, D - thickness), desc="Patch inference"):

            feat = feature_volume[z]

            if isinstance(feat, torch.Tensor):
                feat_t = feat.to(device=device, dtype=torch.float32, non_blocking=True)
            else:
                feat_t = torch.as_tensor(feat, dtype=torch.float32, device=device)

            # shape: (F, H, W)
            if feat_t.ndim != 3:
                raise ValueError(f"Expected feature slice shape (F,H,W), got {feat_t.shape}")

            pred_slice = torch.zeros((H, W), dtype=torch.float32, device=device)
            edge_slice = torch.zeros((H, W), dtype=torch.float32, device=device)
            count_slice = torch.zeros((H, W), dtype=torch.float32, device=device)

            # ========= batched patch inference =========
            for start in range(0, len(coords), batch_size):
                batch_coords = coords[start:start + batch_size]

                patches = []
                for x, y in batch_coords:
                    patch = feat_t[:, x:x + patch_size, y:y + patch_size]
                    patches.append(patch)

                input_tensor = torch.stack(patches, dim=0)  # (B, F, patch, patch)

                if use_amp and device.startswith("cuda"):
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        logits = model(input_tensor)
                else:
                    logits = model(input_tensor)

                probs = torch.sigmoid(logits).float()  # (B, C, patch, patch)

                for bi, (x, y) in enumerate(batch_coords):
                    pred_slice[x:x + patch_size, y:y + patch_size] += probs[bi, 0]
                    edge_slice[x:x + patch_size, y:y + patch_size] += probs[bi, 1]
                    count_slice[x:x + patch_size, y:y + patch_size] += 1.0

            pred_slice = pred_slice / count_slice.clamp_min(1.0)
            edge_slice = edge_slice / count_slice.clamp_min(1.0)

            pred_volume[z] = pred_slice.detach().cpu().numpy()
            edge_volume[z] = edge_slice.detach().cpu().numpy()

    blurred = gaussian_filter(pred_volume, sigma=smooth_sigma, mode="reflect")

    return blurred, edge_volume
# def infer_volume_edges_whole(feature_volume, model, thickness=2):
#
#     # feature_volume: zarr array [D,F,H,W]
#
#     D, F, H, W = feature_volume.shape
#
#     pred_volume = np.zeros((D, H, W), dtype=np.float32)
#     edge_volume = np.zeros((D, H, W), dtype=np.float32)
#     count_volume = np.zeros((D, H, W), dtype=np.float32)
#
#     model.eval()
#     model.cuda()
#
#     with torch.no_grad():
#         for z in tqdm(range(thickness, D - thickness), desc="Predicting edges per slice"):
#
#             # =========================
#             # 1️⃣ 从 Zarr 读取一个 slice
#             # =========================
#             feats_stack = feature_volume[z]   # 🔥 lazy读取 (F,H,W)
#
#             # =========================
#             # 2️⃣ pad 单 slice（不是整个volume）
#             # =========================
#             feats_stack, pad_info = pad_to_multiple_of(feats_stack, multiple=4)
#
#             # =========================
#             # 3️⃣ forward
#             # =========================
#             input_tensor = torch.from_numpy(feats_stack).unsqueeze(0).float().cuda()
#
#             pred = model(input_tensor)  # [1,1,H,W]
#             pred_prob = torch.sigmoid(pred).squeeze(0).cpu().numpy()
#
#             # =========================
#             # 4️⃣ 去 padding
#             # =========================
#             pred_prob = unpad_volume(pred_prob, pad_info)
#
#             # =========================
#             # 5️⃣ 写回 volume
#             # =========================
#             edge_volume[z] = pred_prob[1]
#             pred_volume[z] = pred_prob[0]
#             count_volume[z] += 1
#
#     # =========================
#     # 6️⃣ normalize
#     # =========================
#     count_volume[count_volume == 0] = 1
#     edge_volume /= count_volume
#     pred_volume /= count_volume
#
#     # =========================
#     # 7️⃣ smooth
#     # =========================
#     blurred = gaussian_filter(pred_volume, sigma=(1, 3, 3), mode="reflect")
#
#     return blurred, edge_volume

