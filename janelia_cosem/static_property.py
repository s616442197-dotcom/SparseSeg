import numpy as np
import tifffile as tiff
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.measure import label, regionprops
from skimage import measure
from scipy.stats import mannwhitneyu
from scipy.ndimage import gaussian_filter, zoom, binary_erosion


# =========================
# 3D 工具函数
# =========================

def compute_surface_area(binary_crop, structure=None):
    """
    用一次 erosion 估计表面体素数
    """
    if binary_crop.sum() == 0:
        return np.nan

    if structure is None:
        structure = np.ones((3, 3, 3), dtype=bool)  # 6/18/26 邻域近似

    eroded = binary_erosion(binary_crop, structure=structure)
    surface_voxels = binary_crop.sum() - eroded.sum()

    return surface_voxels

def compute_sphericity(volume, surface_area):
    if surface_area <= 0 or np.isnan(surface_area):
        return np.nan
    return (volume) ** (2 / 3) / surface_area


def analyze_3d_connected_components(volume_bin, min_volume=100, connectivity=1):
    labeled = label(volume_bin, connectivity=connectivity)
    records = []
    i=0
    for region in regionprops(labeled):
        i=i+1
        print(i)
        if region.area < min_volume:
            continue

        volume = region.area
        extent = region.extent
        solidity = region.solidity

        z0, y0, x0, z1, y1, x1 = region.bbox
        dz, dy, dx = z1 - z0, y1 - y0, x1 - x0
        bbox_aspect_ratio = max(dx, dy, dz) / max(1, min(dx, dy, dz))

        eigvals = np.sort(region.inertia_tensor_eigvals)[::-1]
        if eigvals[2] > 0:
            elongation = eigvals[0] / eigvals[2]
            flatness = eigvals[1] / eigvals[2]
        else:
            elongation = np.nan
            flatness = np.nan

        crop = (labeled[z0:z1, y0:y1, x0:x1] == region.label)
        surface_area = compute_surface_area(crop)

        records.append({
            "volume": volume,
            "extent": extent,
            "solidity": solidity,
            "elongation": elongation,
            "flatness": flatness,
            "bbox_aspect_ratio": bbox_aspect_ratio,
            "surface_area": surface_area,
            "surface_volume_ratio": surface_area / volume if volume > 0 else np.nan,
            "sphericity": compute_sphericity(volume, surface_area),
        })

    return pd.DataFrame(records)


# =========================
# 2D 工具函数
# =========================
def analyze_2d_connected_components(volume_bin, connectivity=1,min_volume=400):
    records = []

    for z in range(volume_bin.shape[0]):
        labeled = label(volume_bin[z], connectivity=connectivity)

        for region in regionprops(labeled):

            area = region.area
            if area>min_volume:
                extent = region.extent
                solidity = region.solidity

                y0, x0, y1, x1 = region.bbox
                dy, dx = y1 - y0, x1 - x0
                bbox_aspect_ratio = max(dx, dy) / max(1, min(dx, dy))

                if region.axis_minor_length > 0:
                    elongation = region.axis_major_length / region.axis_minor_length
                else:
                    elongation = np.nan

                records.append({
                    "area": area,
                    "extent": extent,
                    "solidity": solidity,
                    "elongation": elongation,
                    "bbox_aspect_ratio": bbox_aspect_ratio,
                    "eccentricity": region.eccentricity,
                })

    return pd.DataFrame(records)


# =========================
# 可视化
# =========================
def plot_distributions(df, features, title_prefix):
    plt.figure(figsize=(15, 10))

    for i, feat in enumerate(features):
        plt.subplot(2, 3, i + 1)

        for group in df["type"].unique():
            data = df[df["type"] == group][feat].dropna()

            if feat in ["volume", "area", "elongation", "bbox_aspect_ratio",'sphericity']:
                data = np.log10(data + 1e-6)

            plt.hist(
                data,
                # bins=50,
                alpha=0.6,
                density=True,
                label=group,
            )

        plt.title(f"{title_prefix} {feat}")
        plt.legend()

    plt.tight_layout()
    plt.show()


def run_stats(df, features, label):
    print(f"\nStatistical comparison ({label})")
    for feat in features:
        a = df[df["type"] == "control"][feat].dropna()
        b = df[df["type"] == "exp"][feat].dropna()

        if len(a) > 0 and len(b) > 0:
            _, p = mannwhitneyu(a, b, alternative="two-sided")
            print(f"{feat}: p = {p:.3e}")


# =========================
# Main analysis flow
# =========================


mask_dirs = {
    "control": "/mnt/d/vem_data/main_control_mitomask",
    "exp": "/mnt/d/vem_data/main_patient_mitomask",
}

min_area = {
    "control": 400,
    "exp": 100,
}

min_volume = {
    "control": 5000,
    "exp": 1000,
}
# mask_dirs = {
#     "hela2": "label_hela2_mito_80",
#     "jurkat": "label_jurkat_mito_80",
#     "macrophage": "label_macrophage_mito_80",
# }

dfs_3d, dfs_2d = [], []

for group, mask_dir in mask_dirs.items():
    print(f"Processing {group}")

    vol = tiff.imread(f"{mask_dir}/volume_mask_pred_2.tiff")[:, :, :, 1]/255
    volume_bin = vol > 0.3
    volume_z = zoom(
        vol,
        zoom=[5,1,1],
        order=3  # cubic interpolation
    )
    volume_z=volume_z>0.3
    print(f"Processing 3D")
    df3d = analyze_3d_connected_components(
        volume_z,
        min_volume=min_volume[group],
        connectivity=1
    )
    df3d["type"] = group
    dfs_3d.append(df3d)
    print(f"Processing 2D")
    df2d = analyze_2d_connected_components(
        volume_bin,
        connectivity=1,
        min_volume=min_area[group],
    )
    df2d["type"] = group
    dfs_2d.append(df2d)

df_3d = pd.concat(dfs_3d, ignore_index=True)
df_2d = pd.concat(dfs_2d, ignore_index=True)

df_3d.to_csv("regionprops_3d_control_vs_exp.csv", index=False)
df_2d.to_csv("regionprops_2d_control_vs_exp.csv", index=False)

features_3d = [
    "volume", "elongation", "bbox_aspect_ratio",
    "sphericity", "extent", "surface_volume_ratio"
]

features_2d = [
    "area", "elongation", "bbox_aspect_ratio",
    "eccentricity", "extent", "solidity"
]
#%%

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import pearsonr


from itertools import combinations
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt


def plot_distributions(df, features, title_prefix, show_legend=False):
    plt.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    plt.figure(figsize=(21, 12))

    groups = df["type"].unique()

    for i, feat in enumerate(features):
        ax = plt.subplot(2, 3, i + 1)

        # -------- collect data per group --------
        data_dict = {}
        for group in groups:
            data = df[df["type"] == group][feat].dropna().values

            if feat in ["volume", "area", "elongation", "bbox_aspect_ratio"]:
                data = np.log10(data + 1e-6)

            if len(data) > 0:
                data_dict[group] = data

        # -------- shared bins --------
        all_data = np.concatenate(list(data_dict.values()))
        bins = np.histogram_bin_edges(all_data, bins=50)

        hist_dict = {}

        # -------- plot histograms --------
        for group, data in data_dict.items():
            hist, _ = np.histogram(data, bins=bins, density=True)
            hist_dict[group] = hist

            ax.hist(
                data,
                bins=bins,
                alpha=0.4,
                density=True,
                label=group,
            )

        # -------- compute histogram correlations --------
        corrs = []
        for g1, g2 in combinations(hist_dict.keys(), 2):
            if np.std(hist_dict[g1]) > 0 and np.std(hist_dict[g2]) > 0:
                r, _ = pearsonr(hist_dict[g1], hist_dict[g2])
                corrs.append(r)

        # -------- annotate --------
        # if len(corrs) > 0:
        #     text = (
        #         f"mean r = {np.mean(corrs):.2f}\n"
        #         f"min = {np.min(corrs):.2f}, max = {np.max(corrs):.2f}"
        #     )
        #     ax.text(
        #         0.92, 0.92,
        #         text,
        #         transform=ax.transAxes,
        #         ha="right",
        #         va="top",
        #         fontsize=20,
        #         bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        #     )

        ax.set_title(f"{title_prefix} {feat}")

        # -------- legend switch --------
        if show_legend:
            ax.legend()

    plt.tight_layout()
    plt.show()




plot_distributions(df_3d, features_3d, "3D",show_legend=True)
plot_distributions(df_2d, features_2d, "2D",show_legend=True)

# run_stats(df_3d, features_3d, "3D")
# run_stats(df_2d, features_2d, "2D")