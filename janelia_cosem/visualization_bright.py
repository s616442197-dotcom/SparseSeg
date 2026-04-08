import numpy as np
import tifffile as tiff
from skimage import measure
from scipy.ndimage import gaussian_filter, zoom, label,distance_transform_edt,gaussian_laplace
import os


mask_name='11416_vescilemask'
mask_name='11416_nucleusmask'

mask_name='11415_mitomask'
mask_name='10392_hemozin_mask'
mask_name='10442_mito_mask'
mask_name='11415_mitomask'
mask_name='11416_mitomask'
mask_name='11417_mito_mask'
mask_name='11419_mito_mask_new'
# mask_name='10311_mito_mask'
# mask_name='main_control_mitomask'
# mask_name='main_patient_mitomask'
# mask_name='11420_heart_mask'

# mask_name='label_jurkat_mito_80'
# mask_name='label_macrophage_mito_80'


threshold = 90
min_voxels=5000
level=1.0

# mask_name='10442_mito_mask'
# threshold = 0.5
# min_voxels=1000
# level=2.0

basedir=f'{mask_name}'
basedir=f'/mnt/d/vem_data/{mask_name}'

vol00 = tiff.imread(f'{basedir}/volume_mask_pred_5.tiff')

output_z=500
import numpy as np
from scipy.ndimage import uniform_filter

def local_dog_2d_per_slice(
    volume,
    sigma1=4.0,
    sigma2=8.0,
    eps=1e-8,
):
    """
    对 3D volume 的每一层 (2D) 计算 DoG 轮廓能量

    参数
    ----
    volume : np.ndarray (Z, H, W)
    sigma1, sigma2 : float
        DoG 两个尺度（sigma2 > sigma1）
    eps : float

    返回
    ----
    dog_energy : np.ndarray (Z, H, W)
    """

    volume = volume.astype(np.float32)
    Z, H, W = volume.shape

    dog_energy = np.zeros_like(volume, dtype=np.float32)

    for z in range(Z):
        img = volume[z]
        g1 = gaussian_filter(img, sigma=sigma1, mode='reflect')
        g2 = gaussian_filter(img, sigma=sigma2, mode='reflect')
        dog_energy[z] = np.abs(g1 - g2)

    emean = dog_energy.mean()
    dog_energy = dog_energy / (emean + eps)

    return dog_energy# 假设数据是 [Z, H, W, C]

def local_log_2d_per_slice(
    volume,
    sigma=10.0,
    eps=1e-8,
):
    """
    对 3D volume 的每一层 (2D) 计算 LoG 轮廓能量

    参数
    ----
    volume : np.ndarray (Z, H, W)
        输入体数据
    sigma : float
        LoG 的尺度（决定轮廓宽度）
    eps : float
        防止除零

    返回
    ----
    log_energy : np.ndarray (Z, H, W)
        LoG 轮廓能量（已做全局归一化）
    """

    volume = volume.astype(np.float32)
    Z, H, W = volume.shape

    log_energy = np.zeros_like(volume, dtype=np.float32)

    for z in range(Z):
        img = volume[z]
        log = gaussian_laplace(img, sigma=sigma, mode='reflect')
        log_energy[z] = np.abs(log)

    # === 全局归一化（与你原来的 var / mean 一致）===
    emean = log_energy.mean()
    log_energy = log_energy / (emean + eps)

    return log_energy

volume0 = vol00[:, :, :, 0]   # (Z, H, W)

var2d = local_dog_2d_per_slice(
    volume0,
)
Z, H, W = volume0.shape

# zoom_factors = (output_z / Z, 1.0, 1.0)
zoom_factors = (min(4.0,output_z / Z), 1.0, 1.0)

volume = zoom(
    volume0,
    zoom=zoom_factors,
    order=3   # cubic interpolation
)


# === 2. 平滑体数据 ===
volume_f = gaussian_filter(
    volume.astype(np.float32),
    sigma=5
)

# 2. threshold
p70 = np.percentile(volume_f, threshold)
binary = volume_f > p70

# 3. 连通区域筛选（高效版）
labeled, _ = label(binary)
sizes = np.bincount(labeled.ravel())
keep = sizes >= min_voxels
keep[0] = 0  # background

binary_clean = keep[labeled]

# 4. 距离场（核心）
dist = distance_transform_edt(binary_clean)

# 5. （可选）轻度平滑距离场
dist = gaussian_filter(dist, sigma=2.0)

# 6. marching cubes
verts, faces, normals, values = measure.marching_cubes(
    dist,
    level=level
)

# === 4. 移动到 volume 空间中心 ===
# marching_cubes 输出的坐标是以 voxel 索引为单位 (z,y,x)
# 我们让 (Z/2, Y/2, X/2) 作为原点
z_dim, y_dim, x_dim = binary_clean.shape
space_center = np.array([z_dim / 2, y_dim / 2, x_dim / 2], dtype=np.float32)

# 平移到空间中心，并适当缩放
verts_centered = (verts - space_center) / 100.0   # 除以100可缩放比例

def save_as_obj_with_color(
    obj_path,
    verts,
    faces,
    color_rgb,        # np.array([R,G,B]) 0-255
    material_name="mat"
):
    obj_path = os.path.abspath(obj_path)
    base, _ = os.path.splitext(obj_path)
    mtl_path = base + ".mtl"

    # === 归一化颜色到 0–1 ===
    color = color_rgb.astype(np.float32) / 255.0

    # === 写 .mtl 文件 ===
    with open(mtl_path, "w") as f:
        f.write(f"newmtl {material_name}\n")
        f.write(f"Kd {color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n")  # diffuse
        f.write("Ka 0.1 0.1 0.1\n")   # ambient
        f.write("Ks 0.2 0.2 0.2\n")   # specular
        f.write("Ns 10.0\n")          # shininess
        f.write("d 1.0\n")            # opacity

    # === 写 .obj 文件 ===
    with open(obj_path, "w") as f:
        f.write(f"mtllib {os.path.basename(mtl_path)}\n")
        f.write(f"usemtl {material_name}\n")

        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

color_map = {
    'mito':     np.array([220,  20,  60]),   # red
    'nucleus':  np.array([186,  85, 211]),   # purple
    'golgi':    np.array([ 64, 224, 208]),   # cyan
    'lyso':     np.array([255, 215,   0]),   # yellow
    'endo':     np.array([200, 200, 200]),   # gray
    'er':       np.array([124, 252,   0]),   # green
    'bright':       np.array([255, 255,   255]),   # green
}

organelle = 'bright'   # or nucleus / golgi / er / ...

save_as_obj_with_color(
    obj_path=f"{basedir}/3D_volume_bright.obj",
    verts=verts_centered,
    faces=faces,
    color_rgb=color_map[organelle],
    material_name=organelle
)

tiff.imwrite(f"{basedir}/volume_bright.tif", volume_f)
print("✅ OBJ 导出完成：", f"{basedir}/3D_volume_bright.obj")
