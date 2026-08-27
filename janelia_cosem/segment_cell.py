#%%
import numpy as np
from tqdm import tqdm
import argparse
from scipy.ndimage import binary_erosion, binary_dilation, gaussian_filter, distance_transform_edt
import tifffile as tiff
from utils import process_volume,local_contrast_normalize,filter_connected_regions_shape,intersect_regions
from skimage.transform import downscale_local_mean
import os
import json
import time
from Loss_func import total_loss_fn
import torch
from torch import optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from edge_extract import (
    get_edge_region,
    filter_edge_area_by_bbox_iou_2d_vectorized,
    fill_edge_volume_by_region,
    filter_edge_area_by_bbox_iou_2d_corrected,
    fill_edge_volume_by_region_corrected,
)
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_fill_holes
import zarr
from save_function import save_volume_with_masks_as_rgb_tiff,save_model
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader,RandomSampler
from datetime import datetime
from scipy import ndimage
from munet_dataset import get_edge_mask, ValidPatchSliceDataset
from MUNET_model import MultiKernelUNet,SimpleViTSeg
from prediction_func import infer_volume_edges_whole,feature_volume_generation,infer_volume_edges_patchwise
from get_inputfeature_new import extract_stack_features
from functools import lru_cache
from pathlib import Path
from adaptive_iterated_mask import (
    generate_adaptive_iterated_mask,
    infer_case as infer_adaptive_case,
    source_balanced_dataset_class,
)


# Historical candidate 7 is retained only for explicit diagnostics.  It was
# disqualified because cross-trial pseudo-label validation did not preserve a
# nonempty zero/very-low-FP gate. It must never be the default workflow.
DISQUALIFIED_CANDIDATE7_REFINEMENT = {
    "shape_threshold": 0.2,
    "shape_min_ratio": 0.5,
    "shape_candidate_cap": None,
    "edge_fill_mode": "raw",
    "edge_min_size": 3,
    "edge_max_ratio": None,
    "edge_z_expand": 5,
    "bbox_iou_threshold": 0.001,
    "line_fill_threshold": 1.0,
}

@lru_cache(maxsize=4)
def load_feature_volume_cached(feature_path, preload=True):
    """
    读取已有 zarr feature，并缓存到内存中。
    同一个 feature_path 第二次调用时不会重复读取。

    返回:
        preload=True  -> numpy array, shape (D,F,H,W)
        preload=False -> zarr array handle
    """
    feature_path = os.path.abspath(feature_path)

    print(f"✅ Loading feature from: {feature_path}")
    z = zarr.open(feature_path, mode="r")

    print("feature shape:", z.shape)
    print("feature chunks:", z.chunks)
    print("feature dtype:", z.dtype)

    if preload:
        nbytes = np.prod(z.shape) * np.dtype(z.dtype).itemsize
        print(f"feature size: {nbytes / 1024**3:.2f} GB")
        print("⏳ Preloading feature into RAM...")
        z = np.asarray(z, dtype=np.float32)
        print("✅ Preloaded:", z.shape, z.dtype)

    return z
def get_or_build_feature_volume(volume, feature_path, thickness=2):
    """
    feature_path: xxx.zarr
    返回: zarr array (D,F,H,W)
    """

    start_time = datetime.now()
    print(f"⏱️ 开始时间: {start_time}")

    D, H, W = volume.shape

    # =========================
    # 1️⃣ 已存在 → 直接打开
    # =========================
    if os.path.exists(feature_path):
        print("✅ 使用已有 Zarr feature / cache")
        return load_feature_volume_cached(feature_path, preload=True)

    # =========================
    # 2️⃣ 创建 Zarr
    # =========================
    print("⚠️ 构建 Zarr feature...")

    os.makedirs(os.path.dirname(feature_path), exist_ok=True)

    # 先算一个 slice 确定 F
    test = extract_stack_features(volume[thickness:thickness*2+1])
    F = test.shape[0]

    z = zarr.open(
        feature_path,
        mode='w',
        shape=(D, F, H, W),
        chunks=(1, F, 256, 256),   # 🔥 关键：patch级chunk
        dtype='float32'
    )

    # =========================
    # 3️⃣ 写入
    # =========================
    for z_idx in range(thickness, D - thickness):
        slice_img = volume[z_idx-thickness:z_idx+thickness+1]
        feats = extract_stack_features(slice_img)
        z[z_idx] = feats

    end_time = datetime.now()
    print(f"⏱️ 完成，用时: {(end_time-start_time).total_seconds():.2f}s")

    return z
def setup_model(model_class, model_args=None, checkpoint_folder="checkpoints", model_name="unet_model.pt",
                device="cuda", rank=0):
    os.makedirs(checkpoint_folder, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_folder, model_name)

    model = model_class(**(model_args or {})).to(device)

    if os.path.exists(ckpt_path):
        if rank == 0:
            print(f"🔄 检测到已有模型参数，正在加载: {ckpt_path}", flush=True)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        if rank == 0:
            print(f"🆕 未检测到已有模型，新建并保存初始参数到: {ckpt_path}", flush=True)
            torch.save(model.state_dict(), ckpt_path)
        # 等 rank0 写完再读
        if dist.is_initialized():
            dist.barrier()
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint)

    return model

def dilate_z_binary(volume,size=(3, 1, 1)):
    """
    使用3D结构元素在z方向膨胀
    """
    struct = np.ones((size), dtype=np.uint8)
    # struct[:,0,0] = 1
    return binary_dilation(volume, structure=struct).astype(volume.dtype)
def erode_z_binary(volume, size=(1, 3, 3)):
    """
    使用3D结构元素进行收缩
    参数:
        volume: 3D ndarray (Z, H, W)，二值体数据
        size: tuple/list，结构元素大小，例如 (1,3,3) 表示只在xy收缩
    返回:
        eroded: 3D ndarray，收缩后的体
    """
    struct = np.ones(size, dtype=np.uint8)
    return binary_erosion(volume, structure=struct).astype(volume.dtype)
def build_distance_mask(temp_base, R=30, mode="sigmoid"):
    dist = distance_transform_edt(1 - temp_base)

    if mode == "linear":
        mask = (dist / R).clip(0, 1)

    elif mode == "gaussian":
        mask = 1 - np.exp(-(dist**2) / (2 * R**2))

    elif mode == "sigmoid":
        k = R / 6
        mask = (1 / (1 + np.exp(-(dist - R) / k))-0.01).clip(min=0)

    else:
        raise ValueError("Unknown mode")

    return 0.1*mask

def ddp_setup():
    """
    通吃版：
    - torchrun: 读取 RANK/WORLD_SIZE/LOCAL_RANK
    - slurm+srun: 读取 SLURM_PROCID/SLURM_NTASKS/SLURM_LOCALID
    - 直接 python: 自动退化为单进程(不初始化 dist)
    """
    # ---------- Case 1: torchrun ----------
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        distributed = world_size > 1

    # ---------- Case 2: SLURM srun ----------
    elif "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        distributed = world_size > 1

        # SLURM 下建议用首个节点作为 master（如果有 SLURM_NODELIST）
        if "MASTER_ADDR" not in os.environ:
            # 简单做法：用 localhost（单节点最常见），多节点可自行替换为解析 nodelist 的主节点
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ.setdefault("MASTER_PORT", "29500")

    # ---------- Case 3: 兼容 PMI (有些 MPI/PMI 环境) ----------
    elif "PMI_RANK" in os.environ and "PMI_SIZE" in os.environ:
        rank = int(os.environ["PMI_RANK"])
        world_size = int(os.environ["PMI_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        distributed = world_size > 1
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

    # ---------- Case 4: 普通 python 运行（非分布式） ----------
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        distributed = False

    # 设定 device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # 初始化进程组（仅当确实是多进程）
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    return rank, world_size, local_rank, device

def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank: int) -> bool:
    return rank == 0
def predict_packed(
    model_path,
    feature_volume,
    device="cuda",
    thickness=2
):
    """
    输入:
        model_path: 训练好的模型路径 (.pt)
        raw_name: 原始数据名（不带.tif）

    输出:
        edge_vol: 预测的 edge volume
    """

    # ======================
    # 1️⃣ 读取 raw volume
    # ======================
    D, F, H, W = feature_volume.shape

    # ======================
    # 3️⃣ 加载模型
    # ======================
    model = MultiKernelUNet(in_channels=F, out_channels=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # ======================
    # 4️⃣ 推理
    # ======================
    with torch.no_grad():
        edge_vol, edge_line = infer_volume_edges_patchwise(
            feature_volume,
            model,
            thickness=thickness
        )

    return edge_vol

def main(
    interation_idx=0,
    *,
    filer_method=2,
    z_threshold=10,
    patch_scale=140,
    raw_name="jurkat_em_s3",
    mask_name="label_jurkat_er_30",
    folder_name="label_jurkat_er_30",
    area_coef=1.0,
    edge_coef=0.5,
    iou_thresh=0.6,
    threshold=0.5,
    negative_threshold=1.0,
    low_weight_coeff=10.0,
    sparsity_weight=0.0,
    repeated_epoch=50,
    batch_size=12,
    num_samples=1000,
    thickness=2,
    inference_stride=None,
    base_folder="inputdata",
    kernel_sizes=(3,5,7),
    Loss_list=[10,0.1,0.1,0.01],
    if_Vit=False,
    area_probability_quantile=99.0,
    edge_filter_uses_probability=False,
    refinement_profile="adaptive_iterated",
    evaluation_probability_quantile=98.95,
    pseudo_label_core_quantile=99.9,
    adaptive_trial=None,
    adaptive_run_name=None,
    adaptive_backend_dir=None,
    adaptive_continuous_selector=None,
    adaptive_frozen_actions=None,
    adaptive_sampling_policy="source_base85_850_120_30",
    adaptive_seed_offset=1400000,
):
    # ========= DDP init =========
    rank, world_size, local_rank, device = ddp_setup()
    main_proc = is_main_process(rank)

    patchsize = (patch_scale, patch_scale)
    if inference_stride is None:
        inference_stride = max(1, patch_scale // 2)
    if inference_stride > patch_scale:
        raise ValueError(
            f"inference_stride ({inference_stride}) must not exceed "
            f"patch_scale ({patch_scale})"
        )
    valid_refinement_profiles = {
        "adaptive_iterated",
        "safe_abstain",
        "experimental_candidate7",
        "legacy",
        "optimized_low_fp",
    }
    if refinement_profile not in valid_refinement_profiles:
        raise ValueError(
            f"refinement_profile must be one of {sorted(valid_refinement_profiles)}"
        )
    if refinement_profile == "optimized_low_fp":
        raise RuntimeError(
            "The former optimized_low_fp candidate was disqualified by "
            "cross-trial pseudo-label FP validation. Use safe_abstain, or "
            "explicitly request experimental_candidate7 for diagnostics only."
        )

    adaptive_case = None
    if refinement_profile == "adaptive_iterated":
        adaptive_case = infer_adaptive_case(
            mask_name, adaptive_trial, adaptive_run_name
        )

    if main_proc:
        print("=" * 70, flush=True)
        print(f"[DDP] world_size         = {world_size}", flush=True)
        print(f"[DDP] rank/local_rank    = {rank}/{local_rank}", flush=True)
        print(f"[INFO] interation_idx    = {interation_idx}", flush=True)
        print(f"[INFO] raw_name          = {raw_name}", flush=True)
        print(f"[INFO] mask_name         = {mask_name}", flush=True)
        print(f"[INFO] folder_name       = {folder_name}", flush=True)
        print(f"[INFO] patch_scale       = {patch_scale}", flush=True)
        print(f"[INFO] inference_stride  = {inference_stride}", flush=True)
        print(f"[INFO] z_threshold       = {z_threshold}", flush=True)
        print(f"[INFO] iou_thresh        = {iou_thresh}", flush=True)
        print(f"[INFO] threshold         = {threshold}", flush=True)
        print(f"[INFO] negative_threshold= {negative_threshold}", flush=True)
        print(f"[INFO] low_weight_coeff  = {low_weight_coeff}", flush=True)
        print(f"[INFO] sparsity_weight   = {sparsity_weight}", flush=True)
        print(f"[INFO] area_prob_quantile= {area_probability_quantile}", flush=True)
        print(f"[INFO] edge_prob_to_hook = {edge_filter_uses_probability}", flush=True)
        print(f"[INFO] refinement_profile= {refinement_profile}", flush=True)
        print(f"[INFO] eval_prob_quantile= {evaluation_probability_quantile}", flush=True)
        print(f"[INFO] new2_core_quantile= {pseudo_label_core_quantile}", flush=True)
        if adaptive_case is not None:
            print(f"[INFO] adaptive_trial/run = {adaptive_case}", flush=True)
            print(f"[INFO] adaptive_sampler   = {adaptive_sampling_policy}", flush=True)
        print("=" * 70, flush=True)


    # ===== 数据读取：最小改动，所有 rank 都读（稳）=====
    vol0 = tiff.imread(os.path.join(base_folder, f"{raw_name}.tif"))
    volume = local_contrast_normalize(vol0)

    mask_thd=0.5

    base0 = tiff.imread(os.path.join(base_folder, f"{mask_name}.tif"))
    base0 = (base0 > mask_thd).astype(np.uint8)

    if interation_idx == 0:
        test_volume_label = tiff.imread(os.path.join(base_folder, f"{mask_name}.tif"))
        test_volume_label_base = (test_volume_label > mask_thd).astype(np.uint8)
    else:
        # test_volume_label_base = tiff.imread(f"{folder_name}/{mask_name}_{interation_idx-1}_base.tif")
        test_volume_label_base = tiff.imread(f"{folder_name}/{mask_name}_new_base.tif")
        test_volume_label_base = (test_volume_label_base > mask_thd).astype(np.uint8)
        # test_volume_label = tiff.imread(f"{folder_name}/{mask_name}_{interation_idx-1}.tif")

    # test_volume_label_new = filter_connected_regions_shape(
    #     test_volume_label_base, base0,
    #     threshold=threshold, min_ratio=0.8, max_height=z_threshold
    # )
    # test_volume_label_new[base0 > mask_thd] = 1
    test_volume_label_new = test_volume_label_base

    # negative
    mask_path = os.path.join(base_folder, f"negative_{mask_name}.tif")
    if os.path.exists(mask_path):
        nega_test_volume_label = tiff.imread(mask_path)
        nega_test_volume_label = dilate_z_binary(nega_test_volume_label, size=(1, 1, 1))
    else:
        mask_path2 = os.path.join(base_folder, f"negative_{raw_name}.tif")
        if os.path.exists(mask_path2):
            nega_test_volume_label = tiff.imread(mask_path2)
            nega_test_volume_label = dilate_z_binary(nega_test_volume_label, size=(1, 1, 1))
        else:
            nega_test_volume_label = np.zeros_like(test_volume_label_base, dtype=np.uint8)

    nega_test_volume_label = (nega_test_volume_label > mask_thd).astype(np.uint8)

    softnega = build_distance_mask(test_volume_label_base, R=low_weight_coeff)

    line_coef = 1.2 * (get_edge_mask(test_volume_label_new).sum()) / (test_volume_label_new.sum() + 1e-8)
    if main_proc:
        print("line_coef:", float(line_coef), flush=True)
    feature_path = os.path.join(base_folder, raw_name)
    feature_volume = get_or_build_feature_volume(volume, feature_path, thickness=2)
    D,F,H,W=feature_volume.shape

    # ========= Model =========
    if if_Vit:
        if interation_idx == 0:
            base_model = SimpleViTSeg(in_channels=F, out_channels=2,kernel_sizes=kernel_sizes).to(device)
        else:
            base_model = setup_model(
                SimpleViTSeg,
                model_args={"in_channels": F, "out_channels": 2, "kernel_sizes": kernel_sizes},
                checkpoint_folder=folder_name,
                model_name=f"model_{interation_idx - 1}.pt",
                device=device,
                rank=rank,
            )
    else:
        if interation_idx == 0:
            base_model = MultiKernelUNet(in_channels=F, out_channels=2,kernel_sizes=kernel_sizes).to(device)
        else:
            base_model = setup_model(
                MultiKernelUNet,
                model_args={"in_channels": F, "out_channels": 2,"kernel_sizes":kernel_sizes},
                checkpoint_folder=folder_name,
                model_name=f"model_{interation_idx-1}.pt",
                device=device,
                rank=rank,
            )

    # DDP wrap（仅 world_size>1）
    if world_size > 1:
        model = DDP(
            base_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    else:
        model = base_model

    # 统一拿“真实网络”
    net = model.module if hasattr(model, "module") else model

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    dataset_class = ValidPatchSliceDataset
    if refinement_profile == "adaptive_iterated" and interation_idx > 0:
        previous_adaptive_root = (
            Path(folder_name).resolve()
            / "adaptive_iterated_mask"
            / f"iteration_{interation_idx - 1}"
        )
        previous_new2 = (
            previous_adaptive_root / "final" / "test_volume_label_new2.tif"
        )
        previous_base = previous_adaptive_root / "input" / "base_input.tif"
        if not previous_new2.is_file() or not previous_base.is_file():
            raise FileNotFoundError(
                "adaptive iteration inputs are incomplete: "
                f"new2={previous_new2}, base={previous_base}"
            )
        adaptive_trial_value, adaptive_run_value = adaptive_case
        adaptive_roi_value = int(adaptive_run_value.rsplit("_", 1)[1])
        dataset_class = source_balanced_dataset_class(
            ValidPatchSliceDataset,
            backend_dir=adaptive_backend_dir,
            new2_path=previous_new2,
            previous_base_path=previous_base,
            audit_output_dir=(
                Path(folder_name).resolve()
                / "adaptive_sampling"
                / f"iteration_{interation_idx}"
            ),
            policy=adaptive_sampling_policy,
            seed=(
                int(adaptive_seed_offset)
                + adaptive_trial_value * 100
                + adaptive_roi_value
                + interation_idx
            ),
        )

    dataset = dataset_class(
        volume=volume, mask_volume=test_volume_label_new, feature_volume=feature_volume,
        negative_volume_label=nega_test_volume_label, softnega=softnega,
        patch_size=patchsize,
        threshold=negative_threshold,
        num_samples=num_samples,
        thickness=thickness
    )

    num_workers = max(1, 4 // max(world_size, 1))
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True  # 🔥 关键！
    )

    # ========= Train =========
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    training_started = time.perf_counter()
    for epoch in range(repeated_epoch):
        model.train()

        if sampler is not None:
            sampler.set_epoch(epoch)

        total_loss = 0.0
        batch_count = 0

        for x, y, z, softnega_p, edge,area_ref,edge_ref in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z = z.to(device, non_blocking=True)
            softnega_p = softnega_p.to(device, non_blocking=True)
            edge = edge.to(device, non_blocking=True)
            area_ref = area_ref.to(device, non_blocking=True)
            edge_ref = edge_ref.to(device, non_blocking=True)
            pred = model(x)

            loss, loss_dict = total_loss_fn(
                pred, y, x, z, softnega_p, edge, area_ref,edge_ref,
                net,  # ⭐ 单卡/多卡都正确
                low_weight=low_weight_coeff,
                thickness=thickness,
                area_coef=area_coef,
                edge_coef=edge_coef,
                bce_weight=Loss_list[0],
                corr_weight=Loss_list[1],
                smooth_weight=Loss_list[2],
                sparsity_weight=Loss_list[3],

            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            batch_count += 1

        # 只 rank0 打印，避免刷屏
        if main_proc:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            avg_loss = total_loss / max(batch_count, 1)
            print(f"{now} Epoch {epoch} avg_loss={avg_loss:.6f}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    training_wall_clock_seconds = time.perf_counter() - training_started

    # ========= 推理 + 保存：只在 rank0 =========
    if dist.is_initialized():
        dist.barrier()

    if main_proc:
        post_training_started = time.perf_counter()
        for name, value in (
            ("area_probability_quantile", area_probability_quantile),
            ("evaluation_probability_quantile", evaluation_probability_quantile),
            ("pseudo_label_core_quantile", pseudo_label_core_quantile),
        ):
            if not 0.0 < value < 100.0:
                raise ValueError(f"{name} must lie strictly between 0 and 100")

        edge_vol, edge_Line = infer_volume_edges_patchwise(
            feature_volume,
            net,
            thickness=thickness,
            patch_size=patch_scale,
            stride=inference_stride,
        )

        if not os.path.exists(folder_name):
            os.makedirs(folder_name, exist_ok=True)

        # Preserve probability calibration for direct evaluation. The RGB
        # preview remains for backward compatibility, but evaluation should
        # prefer this probability TIFF or the fixed-threshold binary TIFF.
        probability_u8 = np.rint(np.clip(edge_vol, 0.0, 1.0) * 255.0).astype(np.uint8)
        tiff.imwrite(
            f"{folder_name}/edge_vol_probability_float32.tif",
            np.asarray(edge_vol, dtype=np.float32),
            compression="zlib",
        )
        tiff.imwrite(
            f"{folder_name}/probability_uint8.tif",
            probability_u8,
            compression="zlib",
        )

        def threshold_area_probability(quantile):
            thresh_value = np.percentile(edge_vol, quantile)
            if filer_method == 0:
                binary = edge_vol >= min(thresh_value, 0.5)
            elif filer_method == 1:
                edge_area = get_edge_region(edge_Line)
                binary = intersect_regions(
                    edge_area > 0.5,
                    edge_vol >= max(thresh_value, 0.5),
                    overlap_ratio=0.01,
                ) > 0
            else:
                binary = edge_vol >= min(thresh_value, 0.5)
                for z in range(binary.shape[0]):
                    binary[z] = binary_fill_holes(binary[z])
            binary = binary.astype(np.uint8)
            binary[nega_test_volume_label > mask_thd] = 0
            return binary

        # Keep pseudo-label construction and network evaluation separate.
        # The conservative q99 area candidate feeds new2; q98.95 is the
        # frozen edge_vol operating point used only for reporting/evaluation.
        vol01 = threshold_area_probability(area_probability_quantile)
        evaluation_prediction = threshold_area_probability(
            evaluation_probability_quantile
        )
        tiff.imwrite(
            f"{folder_name}/prediction_fixed_threshold.tif",
            evaluation_prediction,
            compression="zlib",
        )

        adaptive_result = None
        if refinement_profile == "adaptive_iterated":
            adaptive_trial_value, adaptive_run_value = adaptive_case
            adaptive_result = generate_adaptive_iterated_mask(
                edge_vol=edge_vol,
                raw_path=os.path.join(base_folder, f"{raw_name}.tif"),
                feature_volume_path=feature_path,
                base_label=test_volume_label_base,
                negative_label=nega_test_volume_label,
                output_folder=folder_name,
                iteration_index=interation_idx,
                trial=adaptive_trial_value,
                run_name=adaptive_run_value,
                backend_dir=adaptive_backend_dir,
                continuous_selector=adaptive_continuous_selector,
                frozen_actions=adaptive_frozen_actions,
            )
            test_volume_label_new2 = adaptive_result.new2.astype(np.uint8)
            test_volume_label_shape = test_volume_label_new2.copy()
            edge_volume = np.zeros_like(test_volume_label_new2, dtype=np.uint8)
        elif refinement_profile == "safe_abstain":
            # No nonempty GT-blind morphology rule passed the frozen
            # cross-trial zero/very-low-FP gate. Abstaining is the only
            # automatic behavior that guarantees no false pseudo-label is
            # injected. The complete next label therefore remains the base.
            test_volume_label_shape = np.zeros_like(vol01, dtype=np.uint8)
            edge_volume = np.zeros_like(edge_Line, dtype=np.uint8)
            test_volume_label_new2 = np.zeros_like(vol01, dtype=np.uint8)
            print("[SAFETY] new2 abstained: no validated nonempty low-FP rule", flush=True)
        elif refinement_profile == "experimental_candidate7":
            config = DISQUALIFIED_CANDIDATE7_REFINEMENT
            test_volume_label_shape = filter_connected_regions_shape(
                vol01,
                test_volume_label_base,
                threshold=config["shape_threshold"],
                min_ratio=config["shape_min_ratio"],
                max_height=z_threshold,
                candidate_cap=config["shape_candidate_cap"],
            )
            edge_volume = fill_edge_volume_by_region_corrected(
                edge_Line > 0.5,
                min_size=config["edge_min_size"],
                max_ratio=config["edge_max_ratio"],
                z_expand=config["edge_z_expand"],
                fill_mode=config["edge_fill_mode"],
            )
            test_volume_label_new2 = filter_edge_area_by_bbox_iou_2d_corrected(
                edge_volume,
                test_volume_label_shape,
                iou_thresh=config["bbox_iou_threshold"],
                line_fill_thresh=config["line_fill_threshold"],
            )
            core_threshold = np.percentile(edge_vol, pseudo_label_core_quantile)
            test_volume_label_new2 = np.logical_and(
                test_volume_label_new2 > 0,
                edge_vol > core_threshold,
            ).astype(np.uint8)
        else:
            test_volume_label_shape = filter_connected_regions_shape(
                vol01,
                test_volume_label_base,
                threshold=threshold,
                min_ratio=1.0,
                max_height=z_threshold,
            )
            edge_filter_input = (
                edge_Line if edge_filter_uses_probability else (edge_Line > 0.5)
            )
            edge_volume = fill_edge_volume_by_region(
                edge_filter_input,
                min_size=5,
                max_ratio=3.0,
            )
            test_volume_label_new2 = filter_edge_area_by_bbox_iou_2d_vectorized(
                edge_volume,
                test_volume_label_shape,
                iou_thresh=iou_thresh,
                line_fill_thresh=line_coef,
            )

        test_volume_label_new2[nega_test_volume_label > mask_thd] = 0
        tiff.imwrite(
            f"{folder_name}/test_volume_label_new2.tif",
            test_volume_label_new2.astype(np.uint8),
            compression="zlib",
        )

        # ``new2`` contains only conservative additions.  The next iteration
        # must train on the complete accumulated pseudo-label, not on new2
        # alone: (new2 OR current base) AND NOT explicit negative label.
        test_volume_label_save = 1.0 * test_volume_label_new2 + test_volume_label_base
        test_volume_label_save = np.clip(test_volume_label_save, 0, 1.0)
        test_volume_label_save[nega_test_volume_label > mask_thd] = 0
        test_volume_label_save_u8 = test_volume_label_save.astype(np.uint8)
        expected_next_iteration_label = np.logical_or(
            test_volume_label_new2 > mask_thd,
            test_volume_label_base > mask_thd,
        )
        expected_next_iteration_label[nega_test_volume_label > mask_thd] = False
        if not np.array_equal(test_volume_label_save_u8 > 0, expected_next_iteration_label):
            raise RuntimeError(
                "next-iteration label must be (test_volume_label_new2 OR "
                "test_volume_label_base) with explicit negatives cleared"
            )
        if adaptive_result is not None and not np.array_equal(
            test_volume_label_save_u8 > 0,
            adaptive_result.complete_label > 0,
        ):
            raise RuntimeError(
                "segment_cell complete label differs from adaptive backend output"
            )

        # outputs
        if interation_idx >= 3:
            save_volume_with_masks_as_rgb_tiff(
                volume, edge_vol, base0,
                f"{folder_name}/volume_mask_pred.tiff"
            )
        # tiff.imwrite(f'{folder_name}/edge_mask_{interation_idx}.tif', edge_volume)
        # tiff.imwrite(f"{folder_name}/{mask_name}_{interation_idx}.tif", test_volume_label_shape)

        # tiff.imwrite(f"{folder_name}/{mask_name}_{interation_idx}_base.tif", test_volume_label_save_u8)
        tiff.imwrite(f"{folder_name}/{mask_name}_new_base.tif", test_volume_label_save_u8)

        # 保存模型：只保存真实 net（不是 DDP wrapper）
        save_model(net, f"{folder_name}/model_{interation_idx}.pt")
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        stage_timing = {
            "schema_version": 726,
            "iteration_zero_based": int(interation_idx),
            "training_wall_clock_seconds": float(training_wall_clock_seconds),
            "configured_epochs": int(repeated_epoch),
            "average_training_time_per_epoch_seconds": (
                float(training_wall_clock_seconds) / max(int(repeated_epoch), 1)
            ),
            "post_training_inference_new2_and_save_wall_clock_seconds": (
                time.perf_counter() - post_training_started
            ),
            "measurement_kind": "synchronized_training_stage_wall_clock",
            "paper_declared_parameters_changed": False,
            "created_local_time": datetime.now().isoformat(),
        }
        with open(
            f"{folder_name}/segment_cell_timing_iteration_{interation_idx}.json",
            "w", encoding="utf-8"
        ) as handle:
            json.dump(stage_timing, handle, indent=2)

        print("[DONE] rank0 saved outputs.", flush=True)

    if dist.is_initialized():
        dist.barrier()

    ddp_cleanup()
    del loader, dataset

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--interation_idx", type=int, required=True)
    parser.add_argument("--filer_method", type=int, default=2)
    parser.add_argument("--z_threshold", type=int, default=10)
    parser.add_argument("--patch_scale", type=int, default=140)
    parser.add_argument("--inference_stride", type=int, default=None)
    parser.add_argument("--raw_name", type=str, default="jurkat_em_s3")
    parser.add_argument("--mask_name", type=str, default="label_jurkat_er_30")
    parser.add_argument("--folder_name", type=str, default="label_jurkat_er_30")
    parser.add_argument("--area_coef", type=float, default=1.0)
    parser.add_argument("--edge_coef", type=float, default=0.5)
    parser.add_argument("--iou_thresh", type=float, default=0.6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--negative_threshold", type=float, default=1.0)
    parser.add_argument("--low_weight_coeff", type=float, default=10.0)
    parser.add_argument("--sparsity_weight", type=float, default=0.0)
    parser.add_argument(
        "--refinement_profile",
        choices=(
            "adaptive_iterated",
            "safe_abstain",
            "experimental_candidate7",
            "legacy",
            "optimized_low_fp",
        ),
        default="adaptive_iterated",
    )
    parser.add_argument(
        "--evaluation_probability_quantile", type=float, default=98.95
    )
    parser.add_argument(
        "--pseudo_label_core_quantile", type=float, default=99.9
    )
    parser.add_argument("--adaptive_trial", type=int)
    parser.add_argument("--adaptive_run_name")
    parser.add_argument("--adaptive_backend_dir")
    parser.add_argument("--adaptive_continuous_selector")
    parser.add_argument("--adaptive_frozen_actions")
    parser.add_argument(
        "--adaptive_sampling_policy",
        default="source_base85_850_120_30",
        choices=("source_base85_850_120_30", "source_equal_485_485_30"),
    )
    parser.add_argument("--adaptive_seed_offset", type=int, default=1400000)

    args = parser.parse_args()

    main(
        interation_idx=args.interation_idx,
        filer_method=args.filer_method,
        z_threshold=args.z_threshold,
        patch_scale=args.patch_scale,
        inference_stride=args.inference_stride,
        raw_name=args.raw_name,
        mask_name=args.mask_name,
        folder_name=args.folder_name,
        area_coef=args.area_coef,
        edge_coef=args.edge_coef,
        iou_thresh=args.iou_thresh,
        threshold=args.threshold,
        negative_threshold=args.negative_threshold,
        low_weight_coeff=args.low_weight_coeff,
        sparsity_weight=args.sparsity_weight,
        refinement_profile=args.refinement_profile,
        evaluation_probability_quantile=args.evaluation_probability_quantile,
        pseudo_label_core_quantile=args.pseudo_label_core_quantile,
        adaptive_trial=args.adaptive_trial,
        adaptive_run_name=args.adaptive_run_name,
        adaptive_backend_dir=args.adaptive_backend_dir,
        adaptive_continuous_selector=args.adaptive_continuous_selector,
        adaptive_frozen_actions=args.adaptive_frozen_actions,
        adaptive_sampling_policy=args.adaptive_sampling_policy,
        adaptive_seed_offset=args.adaptive_seed_offset,
    )
