import torch
import numpy as np
from tqdm import tqdm
from scipy.ndimage import binary_closing,binary_erosion, binary_dilation, label as nd_label,gaussian_filter, distance_transform_edt
from skimage.transform import downscale_local_mean
from get_inputfeature import extract_2d_features_from_patch

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
            edge_volume[z_start:z_end] = pred_prob_stack[1:2, :, :]
            pred_volume[z_start:z_end] = pred_prob_stack[0:1, :, :]
            count_volume[z_start:z_end] += 1

            # === 平均重叠区域 ===
        count_volume[count_volume == 0] = 1
        edge_volume /= count_volume
        pred_volume /= count_volume
        # === 去除 padding 并平滑 ===
        restored = unpad_volume(pred_volume, pad_info)
        restored2 = unpad_volume(edge_volume, pad_info)
        blurred = gaussian_filter(restored, sigma=(1, 3, 3), mode="reflect")

    return blurred, restored2
