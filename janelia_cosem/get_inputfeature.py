import numpy as np
from typing import Dict, Tuple
from scipy.ndimage import gaussian_filter, gaussian_laplace, sobel, uniform_filter
from skimage.feature import structure_tensor
from skimage.restoration import denoise_tv_chambolle
from scipy.signal import wiener

# ---------- 小工具 ----------

def _normalize(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    img = img.astype(np.float32)
    m, s = img.mean(), img.std()
    if s < eps:
        return np.zeros_like(img, dtype=np.float32)
    return (img - m) / (s + eps)

def _aggregate_z(img_patch: np.ndarray, mode: str = "gaussian", sigma_z: float = 0.8) -> np.ndarray:
    """
    将 [Z,H,W] 聚合成 [H,W] 基图。
      - 'center' 取中间切片
      - 'mean'   z 均值
      - 'gaussian' z 高斯加权
    """
    assert img_patch.ndim == 3, "img_patch must be [Z,H,W]"
    Z, H, W = img_patch.shape
    if mode == "center":
        base = img_patch[Z // 2]
    elif mode == "mean":
        base = img_patch.mean(axis=0)
    elif mode == "gaussian":
        z = np.arange(Z)
        w = np.exp(-0.5 * ((z - Z//2) / float(sigma_z))**2)
        w = w / w.sum()
        base = (img_patch * w[:, None, None]).sum(axis=0)
    else:
        raise ValueError("aggregate_mode must be one of {'center','mean','gaussian'}")
    return base.astype(np.float32)

def _local_stats(img: np.ndarray, win: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    """局部均值 / 方差（同尺寸）"""
    mu = uniform_filter(img, size=win, mode="reflect")
    mu2 = uniform_filter(img**2, size=win, mode="reflect")
    var = np.clip(mu2 - mu**2, 0, None)
    return mu.astype(np.float32), var.astype(np.float32)

def _gradient_mag(img: np.ndarray) -> np.ndarray:
    gx = sobel(img, axis=1, mode="reflect")
    gy = sobel(img, axis=0, mode="reflect")
    return np.hypot(gx, gy).astype(np.float32)

def _structure_tensor_eigvals(img: np.ndarray, sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    结构张量特征值（2D 问题解析解）：
      M = [[Axx, Axy], [Axy, Ayy]]
      l1,2 = (Axx + Ayy)/2 ± sqrt(((Axx - Ayy)/2)^2 + Axy^2)
    返回已排序 |l1|<=|l2|
    """
    Axx, Axy, Ayy = structure_tensor(img, sigma=sigma, mode="reflect")
    tmp = np.sqrt(((Axx - Ayy) * 0.5) ** 2 + Axy**2)
    l1 = 0.5 * (Axx + Ayy) - tmp
    l2 = 0.5 * (Axx + Ayy) + tmp
    return l1.astype(np.float32), l2.astype(np.float32)

def _hessian_matrix(img: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算 2D Hessian（同尺寸），用高斯二阶导近似：
      I_xx = ∂^2(G_sigma * I) / ∂x^2 等价于对 I 先做高斯平滑，再二次差分。
    这里用高斯平滑后再用 sobel 近似二阶（简洁稳健）；如需更精准可自定义二阶高斯核。
    """
    # 高斯预平滑
    sm = gaussian_filter(img, sigma=sigma, mode="reflect")
    # 一阶
    Ix = sobel(sm, axis=1, mode="reflect")
    Iy = sobel(sm, axis=0, mode="reflect")
    # 二阶：对一阶再求导
    Ixx = sobel(Ix, axis=1, mode="reflect")
    Ixy = sobel(Ix, axis=0, mode="reflect")  # 等价于 d/dy(d/dx)
    Iyy = sobel(Iy, axis=0, mode="reflect")
    return Ixx.astype(np.float32), Ixy.astype(np.float32), Iyy.astype(np.float32)

def _hessian_eigvals(Hxx: np.ndarray, Hxy: np.ndarray, Hyy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hessian 2×2 的特征值（解析解），返回已排序 |l1|<=|l2|"""
    tmp = np.sqrt(((Hxx - Hyy) * 0.5) ** 2 + Hxy**2)
    l1 = 0.5 * (Hxx + Hyy) - tmp
    l2 = 0.5 * (Hxx + Hyy) + tmp
    return l1.astype(np.float32), l2.astype(np.float32)

def _frangi_like_from_hessian(l1: np.ndarray, l2: np.ndarray, beta=0.5, c=15.0) -> np.ndarray:
    """
    简化 Frangi 响应：对细长/管状结构敏感（σ 依赖）
      Ra = |l1| / (|l2| + eps)   （各向异性）
      S  = sqrt(l1^2 + l2^2)     （结构强度）
      resp = exp(-(Ra^2)/(2*beta^2)) * (1 - exp(-(S^2)/(2*c^2)))
    """
    eps = 1e-8
    Ra = np.abs(l1) / (np.abs(l2) + eps)
    S  = np.sqrt(l1**2 + l2**2)
    resp = np.exp(-(Ra**2)/(2*beta**2)) * (1 - np.exp(-(S**2)/(2*c**2)))
    return resp.astype(np.float32)

def normalize(img, eps=1e-8, clip_percentile=99.5):
    img = img.astype(np.float32)

    # 去掉 NaN / Inf
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    # 抑制极端值（CTF / deconv 非常需要）
    vmax = np.percentile(img, clip_percentile)
    vmin = np.percentile(img, 100 - clip_percentile)
    img = np.clip(img, vmin, vmax)

    denom = img.max() - img.min()
    if denom < eps:
        return np.zeros_like(img)

    return (img - img.min()) / (denom + eps)
# ---------- 主函数 ----------

def extract_2d_features_from_patch(
    img_patch: np.ndarray,                # [Z,H,W] 例如 Z=3
    aggregate_mode: str = "gaussian",     # 'center' | 'mean' | 'gaussian'
    sigma_z: float = 0.8,
    denoise_tv: float = 0.0,              # >0 启用 TV 去噪（如 0.05）
    sigmas_gauss = (1.0, 2.0, 4.0),
    sigmas_hessian = (1.0, 4.0, 2.0),
    win_local_stats: int = 9,
    st_sigma: float = 1.0,
    return_stack: bool = True,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    返回:
      feats_dict: {name: [H,W]}
      feats_stack: [C,H,W]（按添加顺序堆叠；若 return_stack=False 则为 None）
    """
    assert img_patch.ndim == 3, "img_patch must be [Z,H,W]"
    base = _aggregate_z(img_patch, mode=aggregate_mode, sigma_z=sigma_z)

    # 可选去噪（论文 ROI 例子里 TV 去噪后再做特征）
    if denoise_tv and denoise_tv > 0:
        base_d = denoise_tv_chambolle(base, weight=denoise_tv, channel_axis=None)
    else:
        base_d = base

    # base_n = _normalize(base_d)
    base_n=base_d
    H, W = base_n.shape

    feats: Dict[str, np.ndarray] = {}
    stack = []

    for zi in range(img_patch.shape[0]):
        raw_slice = img_patch[zi].astype(np.float32)
        feats[f"raw_z{zi}"] = raw_slice
        stack.append(raw_slice)

    # 3) 梯度幅值 + 局部统计
    gradmag = _gradient_mag(base_n)
    feats["grad_mag"] = gradmag
    stack.append(gradmag)

    # 0) 原图（标准化）
    feats["base_norm"] = base_n
    stack.append(base_n)


    z_var = img_patch.var(axis=0).astype(np.float32)  # [H,W]
    feats["z_var"] = z_var
    stack.append(z_var)

    # 1) 多尺度高斯平滑
    for s in sigmas_gauss:
        g = gaussian_filter(base_n, sigma=float(s), mode="reflect").astype(np.float32)
        feats[f"gauss_sigma{float(s):.1f}"] = g
        stack.append(g)


    # 2) DoG & LoG
    for i in range(len(sigmas_gauss)-1):
        s1, s2 = float(sigmas_gauss[i]), float(sigmas_gauss[i+1])
        g1 = gaussian_filter(base_n, sigma=s1, mode="reflect")
        g2 = gaussian_filter(base_n, sigma=s2, mode="reflect")
        dog = (g1 - g2).astype(np.float32)
        feats[f"dog_{s1:.1f}_{s2:.1f}"] = dog
        stack.append(dog)

    for s in sigmas_gauss:
        log = gaussian_laplace(base_n, sigma=float(s), mode="reflect").astype(np.float32)
        feats[f"log_sigma{float(s):.1f}"] = log
        stack.append(log)



    mu, var = _local_stats(base_n, win=int(win_local_stats))
    feats[f"local_mean_w{int(win_local_stats)}"] = mu
    feats[f"local_var_w{int(win_local_stats)}"]  = var
    stack += [mu, var]

    # 4) 结构张量特征值
    l1_st, l2_st = _structure_tensor_eigvals(base_n, sigma=float(st_sigma))
    feats[f"st_l1_sigma{float(st_sigma):.1f}"] = l1_st
    feats[f"st_l2_sigma{float(st_sigma):.1f}"] = l2_st
    stack += [l1_st, l2_st]

    # 5) Hessian 多尺度 + Frangi-like
    for s in sigmas_hessian:
        Hxx, Hxy, Hyy = _hessian_matrix(base_n, sigma=float(s))
        l1, l2 = _hessian_eigvals(Hxx, Hxy, Hyy)
        trace = (Hxx + Hyy).astype(np.float32)
        det   = (Hxx * Hyy - Hxy * Hxy).astype(np.float32)

        feats[f"hess_l1_sigma{float(s):.1f}"]   = l1
        feats[f"hess_l2_sigma{float(s):.1f}"]   = l2
        feats[f"hess_trace_sigma{float(s):.1f}"] = trace
        feats[f"hess_det_sigma{float(s):.1f}"]   = det
        stack += [l1, l2, trace, det]

        fr = _frangi_like_from_hessian(l1, l2, beta=0.5, c=15.0)
        feats[f"frangi_like_sigma{float(s):.1f}"] = fr
        stack.append(fr)

    for wr in [1,5,10]:
        wiener_p=normalize(wiener(base_n, (wr, wr)))
        stack.append(wiener_p)
    feats_stack = np.stack(stack, axis=0) if return_stack else None
    return feats, feats_stack

# ---------- 使用示例 ----------
# feats_dict, feats_stack = extract_2d_features_from_patch(
#     img_patch,                 # [Z,H,W] 例如 thickness=1 -> Z=3
#     aggregate_mode="gaussian",
#     sigma_z=0.8,
#     denoise_tv=0.05,          # 可设 0 关闭
#     sigmas_gauss=(1.0,2.0,4.0),
#     sigmas_hessian=(1.0,2.0,4.0),
#     win_local_stats=9,
#     st_sigma=1.0,
# )
# feats_stack 形状 [C,H,W]，可直接作为 RF/SVM 或浅层/深度模型的输入通道。
