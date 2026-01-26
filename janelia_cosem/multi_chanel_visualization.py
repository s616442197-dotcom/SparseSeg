import os
import numpy as np
import tifffile as tiff

celltype = 'hela2'
celltype = 'macrophage'

mask_name='10392_hemozin_mask'
mask_name='10442_mito_mask'
mask_name='11415_mitomask'
mask_name='11416_mitomask'
mask_name='11416_vescilemask'
mask_name='11417_mito_mask'
# mask_name='11419_mito_mask'
mask_name='11420_heart_mask'

pred_filename = "volume_mask_pred_2.tiff"
organelletypes = ['golgi', 'lyso', 'nucleus', 'endo', 'er', 'mito']
organelletypes=['er']
# ===== organelle -> RGB 颜色映射（0–255）=====
color_map = {
    'mito':     np.array([220,  20,  60]),   # red (crimson)
    'nucleus':  np.array([186,  85, 211]),   # purple (medium orchid)
    'golgi':    np.array([ 64, 224, 208]),   # turquoise / cyan
    'lyso':     np.array([255, 215,   0]),   # yellow (gold)
    'endo':     np.array([200, 200, 200]),   # light gray
    'er':       np.array([124, 252,   0]),   # green (lawn green)
}


rgb_volume = None

# for organelletype in organelletypes:
#     pred_path = os.path.join(
#         f'/mnt/d/vem_data/label_{celltype}_{organelletype}_80',
#         pred_filename
#     )
#
#     pred_ori = tiff.imread(pred_path)
#     pred = pred_ori[:, :, :, 1].astype(np.float32) / 255.0  # [D,H,W] in [0,1]
#
#     if rgb_volume is None:
#         D, H, W = pred.shape
#         rgb_volume = np.zeros((D, H, W, 3), dtype=np.float32)
#
#     color = color_map[organelletype] / 255.0  # -> [0,1]
#
#     # ===== 核心：x 映射为 x * RGB 并叠加 =====
#     rgb_volume += pred[..., None] * color[None, None, None, :]

for organelletype in organelletypes:
    pred_path = os.path.join(
        # f'/mnt/d/vem_data/label_{celltype}_{organelletype}_80',
        # f'/mnt/d/vem_data/{mask_name}',
        f'{mask_name}',
        pred_filename
    )

    pred_ori = tiff.imread(pred_path).astype(np.float32) / 255.0
    # pred_ori shape: [D, H, W, 2]  (0=background, 1=organelle)

    background = pred_ori[:, :, :, 0]   # [D,H,W]
    pred = pred_ori[:, :, :, 1]          # [D,H,W]

    if rgb_volume is None:
        D, H, W = pred.shape
        rgb_volume = np.zeros((D, H, W, 3), dtype=np.float32)

        # ===== 用 background 初始化（灰度）=====
        rgb_volume[..., 0] = background
        rgb_volume[..., 1] = background
        rgb_volume[..., 2] = background

    # organelle 颜色（0–1）
    color = color_map[organelletype] / 255.0  # shape (3,)

    # ===== 核心：mask 位置直接替换颜色 =====
    mask = pred > 0.5   # 阈值你可以改

    rgb_volume[mask, 0] = color[0]
    rgb_volume[mask, 1] = color[1]
    rgb_volume[mask, 2] = color[2]


# 防止溢出
rgb_volume = np.clip(rgb_volume, 0, 1)

# 转回 uint8
rgb_volume_uint8 = (rgb_volume * 255).astype(np.uint8)

# 保存
out_path = f'{mask_name}_mc.tiff'
tiff.imwrite(out_path, rgb_volume_uint8)

print(f"Saved combined RGB volume to: {out_path}")
