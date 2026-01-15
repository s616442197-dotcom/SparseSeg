import numpy as np
import tifffile as tiff
from skimage import measure
from scipy.ndimage import gaussian_filter, zoom, label,distance_transform_edt

# === 1. 读取 volume ===
mask_name = '11416_mitomask'
mask_name = '11416_vescilemask'

celltype = 'jurkat'
organelletype = 'endo'
threshold = 0.3
min_voxels=1000
level=2.0

organelletype = 'er'
threshold = 0.4
min_voxels=1000
level=2.0

organelletype = 'mito'
threshold = 0.8
min_voxels=1000
level=1.0

organelletype = 'nucleus'
threshold = 0.8
min_voxels=50000
level=0.0

organelletype = 'lyso'
threshold = 0.3
min_voxels=100
level=2.0

# organelletype = 'golgi'
# threshold = 0.3
# min_voxels=100
# level=2.0

# celltype = 'hela2'
# organelletype = 'golgi'
# threshold = 0.3
# min_voxels=100
# level=2.0
# #
# organelletype = 'mito'
# threshold = 0.7
# min_voxels=1000
# level=2.0
# #
# organelletype = 'nucleus'
# threshold = 0.8
# min_voxels=50000
# level=0.0
#
# organelletype = 'endo'
# threshold = 0.3
# min_voxels=100
# level=2.0
#
# organelletype = 'er'
# threshold = 0.1
# min_voxels=100
# level=2.0
#
# organelletype = 'lyso'
# threshold = 0.3
# min_voxels=100
# level=2.0

celltype = 'macrophage'
organelletype = 'mito'
threshold = 0.4
min_voxels=1000
level=2.0


organelletype = 'golgi'
threshold = 0.5
min_voxels=1000
level=2.0

organelletype = 'lyso'
threshold = 0.4
min_voxels=10
level=2.0

organelletype = 'nucleus'
threshold = 0.8
min_voxels=50000
level=2.0

organelletype = 'endo'
threshold = 0.5
min_voxels=100
level=2.0
#
organelletype = 'er'
threshold = 0.25
min_voxels=100
level=2.0

# mask_name = f'label_{celltype}_{organelletype}_80'

mask_name='11416_vescilemask'
mask_name='11416_nucleusmask'
threshold = 0.7
min_voxels=10000
level=2.0

# mask_name='10442_mito_mask'
# threshold = 0.5
# min_voxels=1000
# level=2.0

basedir=f'{mask_name}'
basedir=f'/mnt/d/vem_data/{mask_name}'

vol00 = tiff.imread(f'{basedir}/volume_mask_pred_2.tiff')

output_z=500

# 假设数据是 [Z, H, W, C]
volume0 = vol00[:, :, :, 0]
Z, H, W = volume0.shape

zoom_factors = (output_z / Z, 1.0, 1.0)

volume = zoom(
    volume0,
    zoom=zoom_factors,
    order=3   # cubic interpolation
)


# === 2. 平滑体数据 ===
volume_f = gaussian_filter(
    volume.astype(np.float32) / 255.0,
    sigma=2
)

# 2. threshold
binary = volume_f > threshold

# 3. 连通区域筛选（高效版）
labeled, _ = label(binary)
sizes = np.bincount(labeled.ravel())
keep = sizes >= min_voxels
keep[0] = 0  # background

binary_clean = keep[labeled]

# 4. 距离场（核心）
dist = distance_transform_edt(binary_clean)

# 5. （可选）轻度平滑距离场
dist = gaussian_filter(dist, sigma=1.0)

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

# === 5. 导出 OBJ ===
def save_as_obj(filename, verts, faces):
    with open(filename, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

save_as_obj(f"{basedir}/volume_bright.obj", verts_centered, faces)

print("✅ OBJ 导出完成：", f"{mask_name}/volume_mesh_smooth.obj")
