import zarr
import dask.array as da
import tifffile as tiff

download_Janelia_url='s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5'
# download_Janelia_url='s3://janelia-cosem-datasets/jrc_jurkat-1/jrc_jurkat-1.n5'
# download_Janelia_url='s3://janelia-cosem-datasets/jrc_macrophage-2/jrc_macrophage-2.n5/'

cell_type='hela2'
# cell_type='jurkat'
# cell_type='macrophage'
type='golgi'
type='lyso'
# type='nucleus'
type='endo'
type='er'

# type='ribo'
# type='mito'
# type='cent'
# ========================================================
# 1. 打开 S3 上的 jrc_hela-2 n5 文件
# ========================================================
print("Opening Janelia COSEM dataset (jrc_hela-2)...")
resolution='s3'
root = zarr.open(
    zarr.N5FSStore(
        download_Janelia_url,
        anon=True
    ),
    mode='r'
)

# ========================================================
# 2. 读取 EM（raw）数据，分辨率 s1
# ========================================================
print("Reading EM s1...")

em_s1_path = f"em/fibsem-uint16/{resolution}"
em_s1_zarr = root[em_s1_path]
em_s1 = da.from_array(em_s1_zarr, chunks=em_s1_zarr.chunks)

print("EM s1 shape:", em_s1.shape)
print("EM dtype:", em_s1.dtype)


# ========================================================
# 3. 读取 mitochondria segmentation mask（s1）
#    你的实际路径：labels/mito_seg/s1
# ========================================================
print("Reading mitochondria mask s1...")



mito_s1_path = f"labels/{type}_seg/{resolution}"
mito_s1_zarr = root[mito_s1_path]
mito_s1 = da.from_array(mito_s1_zarr, chunks=mito_s1_zarr.chunks)

print("Mito mask s1 shape:", mito_s1.shape)
print("Mito dtype:", mito_s1.dtype)


# ========================================================
# 4. 保存成 TIFF（3D stack）
# ========================================================
print("Saving TIFF files (this may take some minutes)...")

em_np = em_s1.compute()
mito_np = mito_s1.compute()
em_np = em_np.transpose(1, 2, 0)
mito_np = mito_np.transpose(1, 2, 0)
tiff.imwrite(f"download/{cell_type}_em_{resolution}.tif", em_np, compression="zlib")
tiff.imwrite(f"download/{cell_type}_{type}_{resolution}.tif", mito_np.astype("uint8"), compression="zlib")

print("Saved:")
print("  hela2_em_s1.tif")
print(f"  hela2_{type}_s1.tif")
print("Done.")
