#%%
import numpy as np
from tqdm import tqdm
import argparse
from scipy.ndimage import binary_erosion, binary_dilation, gaussian_filter, distance_transform_edt
import tifffile as tiff
from utils import process_volume,local_contrast_normalize,filter_connected_regions_shape,intersect_regions
from skimage.transform import downscale_local_mean
import os
from Loss_func import total_loss_fn
import torch
from torch import optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import torch.nn as nn
import torch.nn.functional as F

from save_function import save_volume_with_masks_as_rgb_tiff,save_model
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader,RandomSampler
from datetime import datetime
from scipy import ndimage
from munet_dataset import get_edge_mask, ValidPatchSliceDataset
from MUNET_model import MultiKernelUNet
from prediction_func import infer_volume_edges_whole



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

def main(
    interation_idx=0,
    *,
    filer_method=2,
    z_threshold=10,
    patch_scale=140,
    raw_name="jurkat_em_s3",
    mask_name="label_jurkat_er_30",
    area_coef=1.0,
    edge_coef=0.5,
    iou_thresh=0.6,
    threshold=0.5,
    negative_threshold=1.0,
    low_weight_coeff=10.0,
    sparsity_weight=0.0,
    repeated_epoch=60,
    batch_size=12,
    num_samples=600,
    thickness=2,
    base_folder="inputdata",
):
    # ========= DDP init =========
    rank, world_size, local_rank, device = ddp_setup()
    main_proc = is_main_process(rank)

    patchsize = (patch_scale, patch_scale)

    if main_proc:
        print("=" * 70, flush=True)
        print(f"[DDP] world_size         = {world_size}", flush=True)
        print(f"[DDP] rank/local_rank   = {rank}/{local_rank}", flush=True)
        print(f"[INFO] interation_idx    = {interation_idx}", flush=True)
        print(f"[INFO] raw_name          = {raw_name}", flush=True)
        print(f"[INFO] mask_name         = {mask_name}", flush=True)
        print(f"[INFO] patch_scale       = {patch_scale}", flush=True)
        print(f"[INFO] z_threshold       = {z_threshold}", flush=True)
        print(f"[INFO] iou_thresh        = {iou_thresh}", flush=True)
        print(f"[INFO] threshold         = {threshold}", flush=True)
        print(f"[INFO] negative_threshold= {negative_threshold}", flush=True)
        print(f"[INFO] low_weight_coeff  = {low_weight_coeff}", flush=True)
        print(f"[INFO] sparsity_weight   = {sparsity_weight}", flush=True)
        print("=" * 70, flush=True)

    # ===== 数据读取：最小改动，所有 rank 都读（稳）=====
    vol0 = tiff.imread(os.path.join(base_folder, f"{raw_name}.tif"))
    volume = local_contrast_normalize(vol0)

    base0 = tiff.imread(os.path.join(base_folder, f"{mask_name}.tif"))
    base0 = (base0 > 0).astype(np.uint8)

    if interation_idx == 0:
        test_volume_label = tiff.imread(os.path.join(base_folder, f"{mask_name}.tif"))
        test_volume_label_base = (test_volume_label > 0).astype(np.uint8)
    else:
        test_volume_label_base = tiff.imread(f"{mask_name}/{mask_name}_{interation_idx-1}_base.tif")
        test_volume_label_base = (test_volume_label_base > 0).astype(np.uint8)
        test_volume_label = tiff.imread(f"{mask_name}/{mask_name}_{interation_idx-1}.tif")

    test_volume_label_new = filter_connected_regions_shape(
        test_volume_label_base, base0,
        threshold=threshold, min_ratio=0.8, max_height=z_threshold
    )
    test_volume_label_new[base0 > 0] = 1

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

    nega_test_volume_label = (nega_test_volume_label > 0).astype(np.uint8)

    softnega = build_distance_mask(test_volume_label_base, R=low_weight_coeff)

    line_coef = 1.2 * (get_edge_mask(test_volume_label_new).sum()) / (test_volume_label_new.sum() + 1e-8)
    if main_proc:
        print("line_coef:", float(line_coef), flush=True)

    # ========= Model =========
    if interation_idx == 0:
        base_model = MultiKernelUNet(in_channels=2 * thickness + 31, out_channels=2).to(device)
    else:
        base_model = setup_model(
            MultiKernelUNet,
            model_args={"in_channels": 2 * thickness + 31, "out_channels": 2},
            checkpoint_folder=mask_name,
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

    # ========= Train =========
    for epoch in range(repeated_epoch):
        model.train()

        # 你现在的 dataset 每 epoch 重建一次（保持最小改动）
        dataset = ValidPatchSliceDataset(
            volume, test_volume_label_new, nega_test_volume_label, softnega,
            patch_size=patchsize,
            threshold=negative_threshold,
            num_samples=num_samples,
            thickness=thickness
        )

        if world_size > 1:
            sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=rank,
                shuffle=True, drop_last=True
            )
            sampler.set_epoch(epoch)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        # worker 数量：避免 world_size 倍增过多
        num_workers = max(1, 4 // max(world_size, 1))

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        total_loss = 0.0
        batch_count = 0

        for x, y, z, softnega_p, edge in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z = z.to(device, non_blocking=True)
            softnega_p = softnega_p.to(device, non_blocking=True)
            edge = edge.to(device, non_blocking=True)

            pred = model(x)

            loss, loss_dict = total_loss_fn(
                pred, y, x, z, softnega_p, edge,
                net,  # ⭐ 单卡/多卡都正确
                low_weight=low_weight_coeff,
                thickness=thickness,
                area_coef=area_coef,
                edge_coef=edge_coef,
                sparsity_weight=sparsity_weight,
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

    # ========= 推理 + 保存：只在 rank0 =========
    if dist.is_initialized():
        dist.barrier()

    if main_proc:
        msk_threshold = 0.9

        edge_vol, edge_Line = infer_volume_edges_whole(volume, net, thickness=thickness)

        if not os.path.exists(mask_name):
            os.makedirs(mask_name, exist_ok=True)

        from edge_extract import get_edge_region, filter_edge_area_by_bbox_iou_2d_vectorized

        thresh_value = np.percentile(edge_vol, 100 * msk_threshold)

        if filer_method == 0:
            edge_area = get_edge_region(edge_Line)
            vol010 = ((edge_vol >= max(thresh_value, 0.5)) & (edge_area > 0.5)).astype(np.uint8)
        elif filer_method == 1:
            edge_area = get_edge_region(edge_Line)
            vol010 = intersect_regions((edge_area > 0.5), (edge_vol >= max(thresh_value, 0.5)), overlap_ratio=0.01)
            vol010 = vol010.astype(np.uint8)
        else:
            vol010 = (edge_vol >= max(thresh_value, 0.5)).astype(np.uint8)

        vol01 = vol010.astype(np.uint8)

        test_volume_label_shape = filter_connected_regions_shape(
            vol01, base0, threshold=threshold, min_ratio=0.8, max_height=z_threshold
        )

        test_volume_label_new2 = filter_edge_area_by_bbox_iou_2d_vectorized(
            (edge_Line > 0.5), test_volume_label_shape,
            iou_thresh=iou_thresh, line_fill_thresh=line_coef
        )

        test_volume_label_save = 1.0 * test_volume_label_new2 + test_volume_label_base
        test_volume_label_save = np.clip(test_volume_label_save, 0, 1.0)
        test_volume_label_save[nega_test_volume_label > 0] = 0
        test_volume_label_save_u8 = test_volume_label_save.astype(np.uint8)

        # outputs
        save_volume_with_masks_as_rgb_tiff(
            volume, edge_vol, base0,
            f"{mask_name}/volume_mask_pred_{interation_idx}.tiff"
        )
        # tiff.imwrite(f'{mask_name}/edge_mask_{interation_idx}.tif', edge_Line)
        tiff.imwrite(f"{mask_name}/{mask_name}_{interation_idx}.tif", vol01)
        tiff.imwrite(f"{mask_name}/{mask_name}_{interation_idx}_base.tif", test_volume_label_save_u8)

        # 保存模型：只保存真实 net（不是 DDP wrapper）
        save_model(net, f"{mask_name}/model_{interation_idx}.pt")

        print("[DONE] rank0 saved outputs.", flush=True)

    if dist.is_initialized():
        dist.barrier()

    ddp_cleanup()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--interation_idx", type=int, required=True)
    parser.add_argument("--filer_method", type=int, default=2)
    parser.add_argument("--z_threshold", type=int, default=10)
    parser.add_argument("--patch_scale", type=int, default=140)
    parser.add_argument("--raw_name", type=str, default="jurkat_em_s3")
    parser.add_argument("--mask_name", type=str, default="label_jurkat_er_30")
    parser.add_argument("--area_coef", type=float, default=1.0)
    parser.add_argument("--edge_coef", type=float, default=0.5)
    parser.add_argument("--iou_thresh", type=float, default=0.6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--negative_threshold", type=float, default=1.0)
    parser.add_argument("--low_weight_coeff", type=float, default=10.0)
    parser.add_argument("--sparsity_weight", type=float, default=0.0)

    args = parser.parse_args()

    main(
        interation_idx=args.interation_idx,
        filer_method=args.filer_method,
        z_threshold=args.z_threshold,
        patch_scale=args.patch_scale,
        raw_name=args.raw_name,
        mask_name=args.mask_name,
        area_coef=args.area_coef,
        edge_coef=args.edge_coef,
        iou_thresh=args.iou_thresh,
        threshold=args.threshold,
        negative_threshold=args.negative_threshold,
        low_weight_coeff=args.low_weight_coeff,
        sparsity_weight=args.sparsity_weight,
    )
