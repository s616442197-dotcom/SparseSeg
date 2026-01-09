import torch
import torch.nn.functional as F

def _avg_pool2d(x, win_size):
    pad = win_size // 2
    return F.avg_pool2d(x, win_size, stride=1, padding=pad)

def local_mean_var(x, win_size=3, eps=1e-6):
    """
    计算局部均值和局部方差图（逐通道）
    输入:  x ∈ [B, C, H, W]
    输出:  mean_map, var_map (同形状)
    """
    mean = _avg_pool2d(x, win_size)
    mean2 = _avg_pool2d(x**2, win_size)
    var = (mean2 - mean**2).clamp_min(eps)
    return mean, var


def region_consistency_loss(pred_mask, input_image, win_size=3, eps=1e-6, alpha=0.9):
    """
    一致性损失（基于局部均值 + 方差）
    pred_mask, input_image: [B, C, H, W]
    在 mask 内每个通道的局部 mean/var 都应当一致（低波动）。
    """
    pred = torch.sigmoid(pred_mask)  # [B,C,H,W]
    mean_map, var_map = local_mean_var(input_image, win_size=win_size, eps=eps)

    # soft 权重归一化（逐通道）
    w = pred / (pred.sum(dim=(2, 3), keepdim=True) + eps)

    # 方差一致性（方差的方差）
    mu_var = (var_map * w).sum(dim=(2, 3), keepdim=True)
    var_of_var = ((var_map - mu_var)**2 * w).sum(dim=(2, 3))

    # 均值一致性（均值的方差）
    mu_mean = (mean_map * w).sum(dim=(2, 3), keepdim=True)
    var_of_mean = ((mean_map - mu_mean)**2 * w).sum(dim=(2, 3))

    # 融合均值与方差一致性
    loss = (1 - alpha) * var_of_var.mean() + alpha * var_of_mean.mean()
    return loss


def region_contrast_loss(pred_mask, input_image, win_size=3, eps=1e-6, alpha=0.9):
    """
    对比损失（基于局部均值 + 方差）
    mask 区域与背景区域的 mean/var 应当差异大。
    pred_mask, input_image: [B, C, H, W]
    """
    pred = torch.sigmoid(pred_mask)
    mean_map, var_map = local_mean_var(input_image, win_size=win_size, eps=eps)

    # 计算每通道前景/背景的加权均值
    fg_sum = pred.sum(dim=(2, 3)) + eps
    bg_sum = (1 - pred).sum(dim=(2, 3)) + eps

    fg_mean = (mean_map * pred).sum(dim=(2, 3)) / fg_sum
    bg_mean = (mean_map * (1 - pred)).sum(dim=(2, 3)) / bg_sum
    fg_var = (var_map * pred).sum(dim=(2, 3)) / fg_sum
    bg_var = (var_map * (1 - pred)).sum(dim=(2, 3)) / bg_sum

    # 差异：希望越大越好（取负号）
    mean_diff = torch.abs(fg_mean - bg_mean).mean()
    var_diff = torch.abs(fg_var - bg_var).mean()

    loss = -((1 - alpha) * var_diff + alpha * mean_diff)
    return loss

def masked_soft_bce_loss(pred, target, negative_target=None,softnega=None,
                         high_weight=1.0, low_weight=0.1, kernel_size=1, edge_size=4):
    """
    pred: raw logits, shape [B,1,H,W]
    target: binary mask, [B,1,H,W], only 1 is reliable positive
    negative_target: binary mask, [B,1,H,W], 1 表示可靠负样本
    """
    B, C, H, W = target.shape

    dilation_kernel = torch.ones((C, 1, kernel_size, kernel_size), device=target.device)
    dilation_kernel2 = torch.ones((C, 1, kernel_size+edge_size, kernel_size+edge_size), device=target.device)
    # 正样本膨胀
    target_bin = (target > 0).float()
    dilated = F.conv2d(target_bin, dilation_kernel, padding=kernel_size//2)
    dilated = (dilated > 0).float()

    dilated2 = F.conv2d(target_bin, dilation_kernel2, padding=(kernel_size+edge_size)//2)
    dilated2 = (dilated2 > 0).float()
    dilated_extra = (dilated - target_bin).clamp(min=0)
    dilated_extra2 = (dilated2 - dilated).clamp(min=0)

    weight = target.clone()
    weight += softnega
    # soft_mask = (softnega > 0)
    # rand_mask = torch.rand_like(softnega)
    # weight[soft_mask] = torch.where(
    #     rand_mask[soft_mask] < 0.01*low_weight,  # 20% 概率
    #     torch.tensor(0.01, device=softnega.device),
    #     torch.tensor(0.00, device=softnega.device)
    # )

    weight[dilated_extra2 > 0] = high_weight
    weight[dilated_extra > 0] = 0
    weight[target_bin > 0] = high_weight
    weight[negative_target > 0]=high_weight
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

    bce_loss = F.binary_cross_entropy_with_logits(pred, target_bin, weight=weight)
    push_term = -(torch.sigmoid(pred) * target).mean()  # pred大 → loss小

    mse_loss = F.mse_loss(pred, target_bin, reduction="none")
    if weight is not None:
        weighted_loss = (mse_loss * weight).sum() / (weight.sum() + 1e-8)
    else:
        weighted_loss = mse_loss.mean()

    loss = bce_loss + 0.0*weighted_loss +1.0*push_term

    return loss

def smoothness_loss(pred_mask, input_image, edge_strength=1.0, eps=1e-6):
    """
    图像引导平滑损失 (Edge-aware smoothness loss)
    pred_mask: [B, C, H, W], raw logits or sigmoid
    input_image: [B, C, H, W], same spatial size
    edge_strength: float, controls suppression on strong image edges
    """
    # ---- Sigmoid to get probabilities ----
    pred = torch.sigmoid(pred_mask)

    # ---- Compute image gradient (edge guidance) ----
    with torch.no_grad():
        dx_img = torch.exp(-edge_strength * torch.abs(input_image[:, :, :, 1:] - input_image[:, :, :, :-1]))
        dy_img = torch.exp(-edge_strength * torch.abs(input_image[:, :, 1:, :] - input_image[:, :, :-1, :]))

        # Pad to original size
        dx_img = F.pad(dx_img, (0, 1, 0, 0))
        dy_img = F.pad(dy_img, (0, 0, 0, 1))

    # ---- Compute prediction gradient ----
    dx_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    dy_pred = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])

    dx_pred = F.pad(dx_pred, (0, 1, 0, 0))
    dy_pred = F.pad(dy_pred, (0, 0, 0, 1))

    # ---- Combine with edge-aware weights ----
    # 平滑区域(∇I 小 → dx_img≈1) → 惩罚更强；边缘区域(∇I 大 → dx_img≈0) → 惩罚弱
    loss = ((1 - dx_img) * dx_pred + (1 - dy_img) * dy_pred).mean()

    return loss

def edge_local_bce_loss(pred, edge_mask, radius=1):
    # pred_edge = pred[:, 1:2, :, :]  # [B,1,H,W]

    # 构造 region_mask：只在边界附近 ±radius 内有效
    kernel_size = 2 * radius + 1
    dilation_kernel = torch.ones((1, 1, kernel_size, kernel_size), device=edge_mask.device)
    region_mask = F.conv2d((edge_mask > 0.5).float(), dilation_kernel, padding=radius)
    region_mask = (region_mask > 0).float()

    # ✅ 用 region_mask 作为 BCE 的逐像素权重
    bce_loss = F.binary_cross_entropy_with_logits(pred, edge_mask, weight=region_mask)

    return bce_loss

def total_loss_fn(pred, target, input_img, negative_label, softnega, edge_mask, model,
                  bce_weight=10.0, corr_weight=0.1, smooth_weight=0.1, sparsity_weight=0.01,
                  l1_weight=0.05, high_weight=1.0, low_weight=0.5, area_coef=1, edge_coef=0, thickness=5):
    _, _, H, W = input_img.shape

    bce = masked_soft_bce_loss(pred[:, (0):(1), :, :], target, negative_target=negative_label, softnega=softnega,
                               high_weight=high_weight, low_weight=low_weight, kernel_size=3, edge_size=4)

    # contrast = correlation_loss(pred, input_img[:, (thickness):( thickness + 1), :, :])
    # contrast = smoothness_loss(pred, input_img, slice_idx=thickness)
    smooth = smoothness_loss(pred[:, (0):(1), :, :], input_img[:, (thickness):(thickness + 1), :, :])

    area_correlation = region_consistency_loss(pred[:, (1):(2), :, :],
                                               input_img[:, (2 * thickness + 30):(2 * thickness + 31), :, :])
    areea_contrast = 0.1 * region_contrast_loss(pred[:, (1):(2), :, :],
                                                input_img[:, (2 * thickness + 30):(2 * thickness + 31), :, :])
    edge_correlation = region_consistency_loss(pred[:, (0):(1), :, :],
                                               input_img[:, (thickness):(thickness + 1), :, :])
    edge_contrast = 0.1 * region_contrast_loss(pred[:, (0):(1), :, :],
                                               input_img[:, (thickness):(thickness + 1), :, :])

    correlation = area_coef * area_correlation + edge_coef * edge_correlation
    contrast = area_coef * areea_contrast + edge_coef * edge_contrast

    # edge_loss = edge_local_bce_loss(pred[:, 1:2, :, :], edge_mask, radius=3)
    edge_loss = masked_soft_bce_loss(pred[:, (1):(2), :, :], edge_mask, negative_target=negative_label,
                                     softnega=softnega,
                                     high_weight=high_weight, low_weight=low_weight, kernel_size=3, edge_size=4)
    # edge_loss = smooth
    # contrast = region_contrast_loss(pred, input_img, slice_idx=2 * thickness + 1)
    # 🔹 L1 on model weights
    l1 = 0.0
    for p in model.parameters():
        l1 += torch.sum(torch.abs(p))
    l1 = l1 / sum(p.numel() for p in model.parameters())

    l1 += sparsity_weight * torch.sigmoid(pred[:, (0):(2), :, :]).mean()

    total = (bce_weight * area_coef * bce +
             corr_weight * correlation +
             corr_weight * contrast +
             smooth_weight * smooth +
             bce_weight * edge_coef * edge_loss +
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