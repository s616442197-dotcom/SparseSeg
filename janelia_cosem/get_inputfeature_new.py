import numpy as np
from scipy.ndimage import gaussian_filter, median_filter,convolve,sobel, uniform_filter,gaussian_laplace
from scipy.signal import wiener
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage.filters import hessian
from skimage.filters.rank import gradient
from skimage.morphology import disk

def fft_filter(img, mask):
    F = fftshift(fft2(img))
    return np.real(ifft2(ifftshift(F * mask)))


def make_radial_mask(H, W, r_low=None, r_high=None):
    y, x = np.ogrid[:H, :W]
    cy, cx = H // 2, W // 2
    r = np.sqrt((y - cy)**2 + (x - cx)**2)

    mask = np.ones((H, W), dtype=np.float32)
    if r_low is not None:
        mask[r < r_low] = 0
    if r_high is not None:
        mask[r > r_high] = 0
    return mask


def pseudo_bfactor(img, B=50):
    F = fftshift(fft2(img))
    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    cy, cx = img.shape[0]//2, img.shape[1]//2
    r2 = (y-cy)**2 + (x-cx)**2
    return np.real(ifft2(ifftshift(F * np.exp(B * r2))))

def frangi_like_from_hessian(l1, l2, beta=0.5, c=15.0, eps=1e-12):
    """
    l1, l2: Hessian eigenvalues, |l1| <= |l2|
    return: Frangi-like response
    """
    # line-likeness
    RA = np.abs(l1) / (np.abs(l2) + eps)

    # structureness
    S = np.sqrt(l1**2 + l2**2)

    frangi = np.exp(-(RA**2) / (2 * beta**2)) * \
             (1 - np.exp(-(S**2) / (2 * c**2)))

    return frangi.astype(np.float32)

def hessian_eigvals_from_smoothed(sm):
    """
    sm: Gaussian-smoothed image (G_sigma * I)
    """
    Ixx = gaussian_filter(sm, sigma=0, order=(2, 0))
    Iyy = gaussian_filter(sm, sigma=0, order=(0, 2))
    Ixy = gaussian_filter(sm, sigma=0, order=(1, 1))

    tmp = np.sqrt((Ixx - Iyy)**2 + 4 * Ixy**2)
    l1 = 0.5 * (Ixx + Iyy - tmp)
    l2 = 0.5 * (Ixx + Iyy + tmp)

    swap = np.abs(l1) > np.abs(l2)
    l1[swap], l2[swap] = l2[swap], l1[swap]

    return l1.astype(np.float32), l2.astype(np.float32)


def extract_2d_features(img):
    """
    img: [H, W] float32
    return: [C, H, W]
    """
    H, W = img.shape
    feats = []

    # ========= 0. Base =========
    img = img.astype(np.float32)
    # feats.append(base)

    # ========= 2. Local statistics（⭐ 新增） =========
    win = 5
    mu  = uniform_filter(img, size=win, mode="reflect")
    mu2 = uniform_filter(img * img, size=win, mode="reflect")
    var = mu2 - mu * mu
    gx, gy = np.gradient(img)
    grad_mag=np.sqrt(gx**2 + gy**2)
    feats += [mu, grad_mag, var]

    # ========= 1. Multi-scale Gaussian + Hessian / Frangi =========
    for s in (3, 5, 7):
        sm = gaussian_filter(img, sigma=s, mode="reflect")
        feats.append(sm)

        l1, l2 = hessian_eigvals_from_smoothed(sm)
        fr = frangi_like_from_hessian(l1, l2, beta=0.5, c=15.0)
        feats.append(fr)

        # ---- LoG（⭐ 新增）----
        log = gaussian_laplace(img, sigma=s, mode="reflect")
        feats.append(log)



    # ========= 3. Frequency domain =========
    low  = fft_filter(img, make_radial_mask(H, W, r_high=H//8))
    high = fft_filter(img, make_radial_mask(H, W, r_low=H//8))
    band = fft_filter(img, make_radial_mask(H, W, r_low=H//16, r_high=H//6))
    feats += [band, high, low]

    # ========= 4. Wiener =========
    for k in (1, 3, 5):
        feats.append(wiener(img, (k, k)))



    feats = np.nan_to_num(
        np.stack(feats, axis=0),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return feats


def extract_stack_features(image_stack):
    """
    image_stack: (Z, H, W)
    return: (51, H, W)
    """
    Z, H, W = image_stack.shape
    center = Z // 2

    stack_feats = []
    for z in range(Z):
        stack_feats.append(extract_2d_features(image_stack[z]))

    stack_feats = np.stack(stack_feats, axis=1)
    # shape: (17, Z, H, W)

    # --- Z aggregation ---

    # feat_center = stack_feats[:, center]
    feat_mean = stack_feats.mean(axis=1)
    feat_var = stack_feats.var(axis=1)

    out = np.concatenate(
        [feat_mean, feat_var],
        axis=0
    )  # (51, H, W)

    return out.astype(np.float32)
