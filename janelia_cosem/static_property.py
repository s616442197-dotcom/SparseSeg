import numpy as np
import tifffile as tiff
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.measure import label, regionprops
from skimage import measure
from scipy.stats import mannwhitneyu
from scipy.ndimage import gaussian_filter, zoom, binary_erosion
import os
import pandas as pd
import tifffile as tiff
from scipy.ndimage import zoom

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

csv_3d_path = "regionprops_3d_control_vs_exp.csv"
csv_2d_path = "regionprops_2d_control_vs_exp.csv"

features_3d = [
    "volume", "elongation", "bbox_aspect_ratio",
    "sphericity", "extent", "surface_volume_ratio"
]

features_2d = [
    "area", "elongation", "bbox_aspect_ratio",
    "eccentricity", "extent", "solidity"
]

# ============================================================
# 如果已经计算过，就直接读取
# ============================================================
if os.path.exists(csv_3d_path) and os.path.exists(csv_2d_path):
    print("Existing regionprops CSV files found. Loading...")
    df_3d = pd.read_csv(csv_3d_path)
    df_2d = pd.read_csv(csv_2d_path)

else:
    print("Regionprops CSV files not found. Computing...")

    dfs_3d, dfs_2d = [], []

    for group, mask_dir in mask_dirs.items():
        print(f"Processing {group}")

        mask_path = f"{mask_dir}/volume_mask_pred_2.tiff"

        vol = tiff.imread(mask_path)[:, :, :, 1] / 255.0

        # -----------------------------
        # 2D binary mask
        # -----------------------------
        volume_bin = vol > 0.3

        # -----------------------------
        # Z-rescaled 3D binary mask
        # -----------------------------
        volume_z = zoom(
            vol,
            zoom=[5, 1, 1],
            order=3
        )
        volume_z = volume_z > 0.3

        print("Processing 3D")
        df3d = analyze_3d_connected_components(
            volume_z,
            min_volume=min_volume[group],
            connectivity=1
        )
        df3d["type"] = group
        dfs_3d.append(df3d)

        print("Processing 2D")
        df2d = analyze_2d_connected_components(
            volume_bin,
            connectivity=1,
            min_volume=min_area[group],
        )
        df2d["type"] = group
        dfs_2d.append(df2d)

    df_3d = pd.concat(dfs_3d, ignore_index=True)
    df_2d = pd.concat(dfs_2d, ignore_index=True)

    df_3d.to_csv(csv_3d_path, index=False)
    df_2d.to_csv(csv_2d_path, index=False)

    print(f"Saved: {csv_3d_path}")
    print(f"Saved: {csv_2d_path}")
#%%

# import numpy as np
# import matplotlib.pyplot as plt
# from itertools import combinations
# from scipy.stats import pearsonr
#
#
# from itertools import combinations
# from scipy.stats import pearsonr
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def plot_distributions(df, features, title_prefix, show_legend=False):
#     plt.rcParams.update({
#         "font.size": 20,
#         "axes.titlesize": 20,
#         "axes.labelsize": 20,
#         "xtick.labelsize": 14,
#         "ytick.labelsize": 14,
#     })
#
#     plt.figure(figsize=(21, 12))
#
#     groups = df["type"].unique()
#
#     for i, feat in enumerate(features):
#         ax = plt.subplot(2, 3, i + 1)
#
#         # -------- collect data per group --------
#         data_dict = {}
#         for group in groups:
#             data = df[df["type"] == group][feat].dropna().values
#
#             if feat in ["volume", "area", "elongation", "bbox_aspect_ratio"]:
#                 data = np.log10(data + 1e-6)
#
#             if len(data) > 0:
#                 data_dict[group] = data
#
#         # -------- shared bins --------
#         all_data = np.concatenate(list(data_dict.values()))
#         bins = np.histogram_bin_edges(all_data, bins=50)
#
#         hist_dict = {}
#
#         # -------- plot histograms --------
#         for group, data in data_dict.items():
#             hist, _ = np.histogram(data, bins=bins, density=True)
#             hist_dict[group] = hist
#
#             ax.hist(
#                 data,
#                 bins=bins,
#                 alpha=0.4,
#                 density=True,
#                 label=group,
#             )
#
#         # -------- compute histogram correlations --------
#         corrs = []
#         for g1, g2 in combinations(hist_dict.keys(), 2):
#             if np.std(hist_dict[g1]) > 0 and np.std(hist_dict[g2]) > 0:
#                 r, _ = pearsonr(hist_dict[g1], hist_dict[g2])
#                 corrs.append(r)
#
#         # -------- annotate --------
#         # if len(corrs) > 0:
#         #     text = (
#         #         f"mean r = {np.mean(corrs):.2f}\n"
#         #         f"min = {np.min(corrs):.2f}, max = {np.max(corrs):.2f}"
#         #     )
#         #     ax.text(
#         #         0.92, 0.92,
#         #         text,
#         #         transform=ax.transAxes,
#         #         ha="right",
#         #         va="top",
#         #         fontsize=20,
#         #         bbox=dict(boxstyle="round", fc="white", alpha=0.8),
#         #     )
#
#         ax.set_title(f"{title_prefix} {feat}")
#
#         # -------- legend switch --------
#         if show_legend:
#             ax.legend()
#
#     plt.tight_layout()
#     plt.show()
#
#
# plot_distributions(df_3d, features_3d, "3D",show_legend=True)
# plot_distributions(df_2d, features_2d, "2D",show_legend=True)
#
# # run_stats(df_3d, features_3d, "3D")
# # run_stats(df_2d, features_2d, "2D")
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu, kruskal
from itertools import combinations


def format_p(p):
    if pd.isna(p):
        return "NA"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def p_to_star(p):
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def bh_fdr_correction(pvals):
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full_like(pvals, np.nan, dtype=float)

    valid = ~np.isnan(pvals)
    p = pvals[valid]

    if len(p) == 0:
        return qvals

    order = np.argsort(p)
    ranked_p = p[order]
    n = len(ranked_p)

    ranked_q = ranked_p * n / (np.arange(n) + 1)
    ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
    ranked_q = np.clip(ranked_q, 0, 1)

    q = np.empty_like(ranked_q)
    q[order] = ranked_q

    qvals[valid] = q
    return qvals


def transform_feature(data, feat, log_features=None, eps=1e-6):
    if log_features is None:
        log_features = ["volume", "area", "elongation", "bbox_aspect_ratio"]

    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]

    if feat in log_features:
        data = data[data + eps > 0]
        data = np.log10(data + eps)

    return data


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu, kruskal


def format_p(p):
    if pd.isna(p):
        return "NA"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def format_stat(x):
    if pd.isna(x):
        return "NA"
    if abs(x) >= 1e4:
        return f"{x:.2e}"
    return f"{x:.1f}"


def format_effect(x):
    if pd.isna(x):
        return "NA"
    return f"{x:.2f}"


def p_to_star(p):
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def bh_fdr_correction(pvals):
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full_like(pvals, np.nan, dtype=float)

    valid = ~np.isnan(pvals)
    p = pvals[valid]

    if len(p) == 0:
        return qvals

    order = np.argsort(p)
    ranked_p = p[order]
    n = len(ranked_p)

    ranked_q = ranked_p * n / (np.arange(n) + 1)
    ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
    ranked_q = np.clip(ranked_q, 0, 1)

    q = np.empty_like(ranked_q)
    q[order] = ranked_q

    qvals[valid] = q
    return qvals


def transform_feature(data, feat, log_features=None, eps=1e-6):
    if log_features is None:
        log_features = ["volume", "area", "elongation", "bbox_aspect_ratio"]

    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]

    if feat in log_features:
        data = data[data + eps > 0]
        data = np.log10(data + eps)

    return data


def summarize_values(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if len(x) == 0:
        return {
            "mean": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "iqr": np.nan,
        }

    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)

    return {
        "mean": np.mean(x),
        "sd": np.std(x, ddof=1) if len(x) > 1 else np.nan,
        "median": np.median(x),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }


def rank_biserial_from_u(U, n1, n2):
    """
    Rank-biserial correlation for Mann-Whitney U.
    Positive value means group_1 tends to have larger values than group_2.
    """
    if n1 <= 0 or n2 <= 0:
        return np.nan
    return (2.0 * U) / (n1 * n2) - 1.0


def compute_object_level_stats(
    df,
    features,
    group_col="type",
    log_features=None,
):
    """
    Object-level / cross-section-level statistics.
    Each connected mitochondrial object or 2D cross-section is treated as one observation.

    Statistics are computed on the transformed scale for features in log_features.
    Raw-scale descriptive statistics are also reported for source data.
    """

    if log_features is None:
        log_features = ["volume", "area", "elongation", "bbox_aspect_ratio"]

    groups = list(df[group_col].dropna().unique())
    stats_rows = []

    for feat in features:
        raw_dict = {}
        test_dict = {}

        for group in groups:
            raw = df[df[group_col] == group][feat].dropna().values
            test_values = transform_feature(raw, feat, log_features=log_features)

            if len(test_values) > 0:
                raw_dict[group] = np.asarray(raw, dtype=float)
                test_dict[group] = test_values

        if len(test_dict) < 2:
            stats_rows.append({
                "feature": feat,
                "test": "NA",
                "group_1": None,
                "group_2": None,
                "n_1": np.nan,
                "n_2": np.nan,
                "U_statistic": np.nan,
                "H_statistic": np.nan,
                "p_value": np.nan,
                "rank_biserial_r": np.nan,
                "value_scale": "log10" if feat in log_features else "raw",
            })
            continue

        # -----------------------------
        # Two groups: Mann-Whitney U
        # -----------------------------
        if len(test_dict) == 2:
            g1, g2 = list(test_dict.keys())
            x = test_dict[g1]
            y = test_dict[g2]

            U, p = mannwhitneyu(x, y, alternative="two-sided")
            r_rb = rank_biserial_from_u(U, len(x), len(y))

            raw_s1 = summarize_values(raw_dict[g1])
            raw_s2 = summarize_values(raw_dict[g2])
            test_s1 = summarize_values(x)
            test_s2 = summarize_values(y)

            row = {
                "feature": feat,
                "test": "Mann-Whitney U",
                "alternative": "two-sided",
                "group_1": g1,
                "group_2": g2,
                "n_1": len(x),
                "n_2": len(y),
                "U_statistic": U,
                "H_statistic": np.nan,
                "p_value": p,
                "rank_biserial_r": r_rb,
                "value_scale": "log10(x+1e-6)" if feat in log_features else "raw",

                # raw-scale descriptive statistics
                "mean_1_raw": raw_s1["mean"],
                "sd_1_raw": raw_s1["sd"],
                "median_1_raw": raw_s1["median"],
                "q1_1_raw": raw_s1["q1"],
                "q3_1_raw": raw_s1["q3"],
                "iqr_1_raw": raw_s1["iqr"],

                "mean_2_raw": raw_s2["mean"],
                "sd_2_raw": raw_s2["sd"],
                "median_2_raw": raw_s2["median"],
                "q1_2_raw": raw_s2["q1"],
                "q3_2_raw": raw_s2["q3"],
                "iqr_2_raw": raw_s2["iqr"],

                # test-scale descriptive statistics
                "mean_1_test_scale": test_s1["mean"],
                "sd_1_test_scale": test_s1["sd"],
                "median_1_test_scale": test_s1["median"],
                "q1_1_test_scale": test_s1["q1"],
                "q3_1_test_scale": test_s1["q3"],
                "iqr_1_test_scale": test_s1["iqr"],

                "mean_2_test_scale": test_s2["mean"],
                "sd_2_test_scale": test_s2["sd"],
                "median_2_test_scale": test_s2["median"],
                "q1_2_test_scale": test_s2["q1"],
                "q3_2_test_scale": test_s2["q3"],
                "iqr_2_test_scale": test_s2["iqr"],

                "median_diff_test_scale": test_s1["median"] - test_s2["median"],
            }

            stats_rows.append(row)

        # -----------------------------
        # More than two groups: Kruskal-Wallis
        # -----------------------------
        else:
            values = [v for v in test_dict.values() if len(v) > 0]
            H, p = kruskal(*values)

            row = {
                "feature": feat,
                "test": "Kruskal-Wallis",
                "alternative": "two-sided",
                "group_1": "all",
                "group_2": "all",
                "n_1": sum(len(v) for v in values),
                "n_2": np.nan,
                "U_statistic": np.nan,
                "H_statistic": H,
                "p_value": p,
                "rank_biserial_r": np.nan,
                "value_scale": "log10(x+1e-6)" if feat in log_features else "raw",
            }

            stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows)
    stats_df["q_value"] = bh_fdr_correction(stats_df["p_value"].values)
    stats_df["significance"] = stats_df["q_value"].apply(p_to_star)

    return stats_df


def plot_distributions(
    df,
    features,
    title_prefix,
    show_legend=False,
    group_col="type",
    log_features=None,
    save_path=None,
    stats_csv_path=None,
):
    plt.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    if log_features is None:
        log_features = ["volume", "area", "elongation", "bbox_aspect_ratio"]

    groups = list(df[group_col].dropna().unique())

    stats_df = compute_object_level_stats(
        df=df,
        features=features,
        group_col=group_col,
        log_features=log_features,
    )

    if stats_csv_path is not None:
        stats_df.to_csv(stats_csv_path, index=False)
        print(f"Saved statistics table: {stats_csv_path}")

    stats_map = {row["feature"]: row for _, row in stats_df.iterrows()}

    n_features = len(features)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5.5 * n_rows))
    axes = np.asarray(axes).ravel()

    for i, feat in enumerate(features):
        ax = axes[i]

        data_dict = {}

        for group in groups:
            raw = df[df[group_col] == group][feat].dropna().values
            data = transform_feature(raw, feat, log_features=log_features)

            if len(data) > 0:
                data_dict[group] = data

        if len(data_dict) == 0:
            ax.set_title(f"{title_prefix} {feat}")
            ax.text(
                0.5, 0.5,
                "No valid data",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            continue

        all_data = np.concatenate(list(data_dict.values()))
        bins = np.histogram_bin_edges(all_data, bins=50)

        for group, data in data_dict.items():
            ax.hist(
                data,
                bins=bins,
                alpha=0.4,
                density=True,
                label=group,
            )

        ax.set_title(f"{title_prefix} {feat}")

        if feat in log_features:
            ax.set_xlabel(f"log10({feat})")
        else:
            ax.set_xlabel(feat)

        ax.set_ylabel("Density")

        # ---------- statistical annotation ----------
        row = stats_map.get(feat, None)

        if row is not None:
            test = row["test"]
            p = row["p_value"]
            q = row["q_value"]
            sig = row["significance"]

            if test == "Mann-Whitney U":
                text = (
                    f"{sig}\n"
                    f"Mann-Whitney U\n"
                    f"n={int(row['n_1'])} vs {int(row['n_2'])}\n"
                    f"U={format_stat(row['U_statistic'])}\n"
                    f"r={format_effect(row['rank_biserial_r'])}\n"
                    f"p={format_p(p)}\n"
                    f"q={format_p(q)}"
                )
            elif test == "Kruskal-Wallis":
                text = (
                    f"{sig}\n"
                    f"Kruskal-Wallis\n"
                    f"n={int(row['n_1'])}\n"
                    f"H={format_stat(row['H_statistic'])}\n"
                    f"p={format_p(p)}\n"
                    f"q={format_p(q)}"
                )
            else:
                text = "No test"

            ax.text(
                0.97, 0.97,
                text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=12,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85),
            )

        if show_legend:
            ax.legend(fontsize=12)

    for j in range(n_features, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {save_path}")

    plt.show()

    return stats_df


# ============================================================
# Run plotting and statistics
# ============================================================

stats_3d = plot_distributions(
    df_3d,
    features_3d,
    "3D",
    show_legend=True,
    save_path="mitochondria_morphometry_3d.png",
    stats_csv_path="mitochondria_morphometry_stats_3d.csv",
)

stats_2d = plot_distributions(
    df_2d,
    features_2d,
    "2D",
    show_legend=True,
    save_path="mitochondria_morphometry_2d.png",
    stats_csv_path="mitochondria_morphometry_stats_2d.csv",
)

print("\n3D statistics")
print(stats_3d)

print("\n2D statistics")
print(stats_2d)