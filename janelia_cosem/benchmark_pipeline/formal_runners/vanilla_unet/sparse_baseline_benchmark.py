"""Train raw-input architecture and sparse-label controls.

The two ``*_sparse_matched`` variants deliberately reuse only the two
SparseSeg controls requested for the architecture-matched benchmark:

* positive-centred patch sampling (including the small reliable/soft-negative
  supplement used by ``ValidPatchSliceDataset``), and
* the area-channel ``masked_soft_bce_loss`` weighting rule.

They do not use SparseSeg handcrafted features, edge/auxiliary heads,
correlation/smoothness/L1 terms, shape filtering, or iterative refinement.
The ``sparseseg_backbone_conventional`` control keeps only SparseSeg's
``MultiKernelUNet`` architecture and replaces its sampler/loss with uniform
sampling plus BCE and soft Dice. It likewise uses only the locally-normalized
raw z-stack and performs one train/predict pass without refinement.
Every run writes a binary TIFF that can be consumed by
``evaluation_cross_trials_extreme.py`` plus ``timing.json`` containing the
measured end-to-end wall-clock time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import socket
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, maximum_filter, minimum_filter
from torch.utils.data import DataLoader, Dataset


VARIANT_OUTPUT_DIRS = {
    "vanilla_unet": "vanilla_unet",
    "nnunet_2d": "nnUnet",
    "vanilla_unet_sparse_matched": "vanilla_unet_sparse_matched",
    "nnunet_2d_sparse_matched": "nnUnet_sparse_matched",
    "sparseseg_backbone_conventional": "sparseseg_backbone_conventional",
}
MATCHED_VARIANTS = {
    "vanilla_unet_sparse_matched",
    "nnunet_2d_sparse_matched",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class StageTimer:
    def __init__(self, device: torch.device):
        self.device = device
        self.values: Dict[str, float] = {}

    def measure(self, name: str):
        timer = self

        class _Measure:
            def __enter__(self):
                synchronize(timer.device)
                self.start = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc, tb):
                synchronize(timer.device)
                timer.values[name] = time.perf_counter() - self.start

        return _Measure()


def local_contrast_normalize(volume: np.ndarray, kernel_size: int = 20) -> np.ndarray:
    """Match the raw-volume normalization used by segment_cell.py."""
    volume_f = volume.astype(np.float32, copy=False)
    footprint = (3, kernel_size, kernel_size)
    local_max = maximum_filter(volume_f, size=footprint, mode="reflect")
    local_min = minimum_filter(volume_f, size=footprint, mode="reflect")
    normalized = (volume_f - local_min) / (local_max - local_min + 1e-5)
    return normalized.astype(np.float32, copy=False)


def build_distance_mask(mask: np.ndarray, radius: float = 50.0) -> np.ndarray:
    """Equivalent to segment_cell.build_distance_mask(..., mode='sigmoid')."""
    distance = distance_transform_edt(1 - mask.astype(np.uint8))
    k = radius / 6.0
    soft = 1.0 / (1.0 + np.exp(-(distance - radius) / k))
    soft = np.clip(soft - 0.01, 0.0, None)
    return (0.1 * soft).astype(np.float32)


def _valid_positive_centres(
    mask: np.ndarray,
    patch_size: Tuple[int, int],
    thickness: int,
) -> np.ndarray:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise ValueError("The sparse training mask has no positive voxels.")
    depth, height, width = mask.shape
    ph, pw = patch_size
    valid = coords[
        (coords[:, 0] >= thickness)
        & (coords[:, 0] <= depth - thickness - 1)
        & (coords[:, 1] >= ph // 2)
        & (coords[:, 1] <= height - ph // 2 - 1)
        & (coords[:, 2] >= pw // 2)
        & (coords[:, 2] <= width - pw // 2 - 1)
    ]
    if len(valid) == 0:
        raise ValueError("No legal positive-centred patches fit inside the volume.")
    return valid.astype(np.int32, copy=False)


def _sample_soft_negative_centres(
    eligible: np.ndarray,
    count: int,
    patch_size: Tuple[int, int],
    thickness: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniform rejection sampling without materialising all background coords."""
    if count <= 0:
        return np.empty((0, 3), dtype=np.int32)
    depth, height, width = eligible.shape
    ph, pw = patch_size
    result: List[Tuple[int, int, int]] = []
    attempts = 0
    max_attempts = max(10_000, count * 100)
    while len(result) < count and attempts < max_attempts:
        batch = min(max(256, 4 * (count - len(result))), 100_000)
        z = rng.integers(thickness, depth - thickness, size=batch)
        y = rng.integers(ph // 2, height - ph // 2, size=batch)
        x = rng.integers(pw // 2, width - pw // 2, size=batch)
        keep = eligible[z, y, x]
        result.extend(zip(z[keep].tolist(), y[keep].tolist(), x[keep].tolist()))
        attempts += batch
    if len(result) < count:
        raise ValueError(f"Could sample only {len(result)}/{count} soft-negative centres.")
    return np.asarray(result[:count], dtype=np.int32)


def make_centres(
    mask: np.ndarray,
    negative: np.ndarray,
    soft_negative: np.ndarray,
    *,
    mode: str,
    patch_size: Tuple[int, int],
    thickness: int,
    num_samples: int,
    negative_threshold: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    depth, height, width = mask.shape
    ph, pw = patch_size
    if mode == "uniform":
        centres = np.column_stack(
            [
                rng.integers(thickness, depth - thickness, size=num_samples),
                rng.integers(ph // 2, height - ph // 2, size=num_samples),
                rng.integers(pw // 2, width - pw // 2, size=num_samples),
            ]
        ).astype(np.int32)
        rng.shuffle(centres)
        return centres

    if mode == "nnunet_foreground_oversample":
        # nnU-Net's default batch sampler deliberately oversamples foreground
        # patches (approximately one third), rather than sampling all patches
        # uniformly.  Reproduce that policy at dataset-index construction time.
        positives = _valid_positive_centres(mask, patch_size, thickness)
        foreground_count = int(round(num_samples * 0.33))
        background_count = num_samples - foreground_count
        foreground = positives[
            rng.integers(0, len(positives), size=foreground_count)
        ]
        background = np.column_stack(
            [
                rng.integers(thickness, depth - thickness, size=background_count),
                rng.integers(ph // 2, height - ph // 2, size=background_count),
                rng.integers(pw // 2, width - pw // 2, size=background_count),
            ]
        ).astype(np.int32)
        centres = np.concatenate([background, foreground], axis=0)
        rng.shuffle(centres)
        return centres.astype(np.int32, copy=False)

    positives = _valid_positive_centres(mask, patch_size, thickness)
    n_extra = max(1, int(0.01 * negative_threshold * len(positives)))
    eligible = (negative > 0) | (soft_negative > 0)
    soft_centres = _sample_soft_negative_centres(
        eligible, n_extra, patch_size, thickness, rng
    )
    valid = np.concatenate([positives, soft_centres], axis=0)

    # Same evenly-spaced selection formula as ValidPatchSliceDataset.
    indices = np.arange(num_samples, dtype=np.int64) * len(valid) // num_samples
    rng.shuffle(indices)
    return valid[indices].astype(np.int32, copy=False)


class SparsePatchDataset(Dataset):
    def __init__(
        self,
        volume: np.ndarray,
        mask: np.ndarray,
        negative: np.ndarray,
        soft_negative: np.ndarray,
        centres: np.ndarray,
        patch_size: Tuple[int, int],
        thickness: int,
        matched: bool,
    ):
        self.volume = volume
        self.mask = mask
        self.negative = negative
        self.soft_negative = soft_negative
        self.centres = centres
        self.patch_size = patch_size
        self.thickness = thickness
        self.matched = matched

    def __len__(self) -> int:
        return len(self.centres)

    def __getitem__(self, index: int):
        z, cy, cx = (int(v) for v in self.centres[index])
        ph, pw = self.patch_size
        y0, x0 = cy - ph // 2, cx - pw // 2
        image = self.volume[
            z - self.thickness : z + self.thickness + 1,
            y0 : y0 + ph,
            x0 : x0 + pw,
        ]
        target = self.mask[z : z + 1, y0 : y0 + ph, x0 : x0 + pw]
        reliable_negative = self.negative[z : z + 1, y0 : y0 + ph, x0 : x0 + pw]
        soft_negative = self.soft_negative[z : z + 1, y0 : y0 + ph, x0 : x0 + pw]

        arrays = [image, target, reliable_negative, soft_negative]
        if random.randrange(2):
            arrays[0] = np.flip(arrays[0], axis=0)
        flip_mode = random.randrange(3)
        if flip_mode == 1:
            arrays = [np.flip(a, axis=-1) for a in arrays]
        elif flip_mode == 2:
            arrays = [np.flip(a, axis=-2) for a in arrays]
        rot_k = random.randrange(4)
        if rot_k:
            arrays = [np.rot90(a, k=rot_k, axes=(-2, -1)) for a in arrays]

        tensors = [torch.from_numpy(np.ascontiguousarray(a)).float() for a in arrays]
        if self.matched:
            return tuple(tensors)
        return tensors[0], tensors[1]


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VanillaUNet2D(nn.Module):
    def __init__(self, in_channels: int, features: Sequence[int] = (32, 64, 128, 256)):
        super().__init__()
        self.encoders = nn.ModuleList()
        current = in_channels
        for feature in features:
            self.encoders.append(DoubleConv(current, feature))
            current = feature
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        current = features[-1] * 2
        for feature in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(current, feature, 2, stride=2))
            self.decoders.append(DoubleConv(feature * 2, feature))
            current = feature
        self.output = nn.Conv2d(features[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([skip, x], dim=1))
        return self.output(x)


class NnUNetConvStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=True),
            nn.InstanceNorm2d(out_channels, eps=1e-5, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
            nn.InstanceNorm2d(out_channels, eps=1e-5, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class NnUNetStyle2D(nn.Module):
    """PlainConvUNet-style 2D architecture suitable for 80x80 patches.

    This intentionally uses the nnU-Net architectural conventions but not the
    official planner/trainer, because the benchmark replaces the trainer's
    sampling and loss in the matched variants.
    """

    def __init__(
        self,
        in_channels: int,
        features: Sequence[int] = (32, 64, 128, 256, 320),
    ):
        super().__init__()
        stages = []
        current = in_channels
        for index, feature in enumerate(features):
            stages.append(NnUNetConvStage(current, feature, stride=1 if index == 0 else 2))
            current = feature
        self.encoders = nn.ModuleList(stages)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for index in range(len(features) - 1, 0, -1):
            low, skip = features[index], features[index - 1]
            self.upconvs.append(nn.ConvTranspose2d(low, skip, 2, stride=2, bias=True))
            self.decoders.append(NnUNetConvStage(skip * 2, skip))
        self.output = nn.Conv2d(features[0], 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
        x = skips[-1]
        for up, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips[:-1])):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([skip, x], dim=1))
        return self.output(x)


def build_model(variant: str, in_channels: int) -> nn.Module:
    if variant == "sparseseg_backbone_conventional":
        try:
            from MUNET_model import MultiKernelUNet
        except ImportError:
            module_root = Path(__file__).resolve().parent.parent
            if str(module_root) not in sys.path:
                sys.path.insert(0, str(module_root))
            from MUNET_model import MultiKernelUNet
        return MultiKernelUNet(in_channels=in_channels, out_channels=1)
    if variant.startswith("vanilla_unet"):
        return VanillaUNet2D(in_channels)
    if variant.startswith("nnunet_2d"):
        return NnUNetStyle2D(in_channels)
    raise ValueError(f"Unknown variant: {variant}")


def build_dilated_rings(target: torch.Tensor, kernel_size: int = 3, edge_size: int = 4):
    target_bin = (target > 0).float()
    dilated = F.max_pool2d(target_bin, kernel_size, stride=1, padding=kernel_size // 2)
    outer_size = kernel_size + edge_size
    dilated2 = F.max_pool2d(target_bin, outer_size, stride=1, padding=outer_size // 2)
    dilated_extra = (dilated - target_bin).clamp(min=0)
    dilated_extra2 = (dilated2 - dilated).clamp(min=0)
    return dilated_extra, dilated_extra2


def sparse_aware_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    reliable_negative: torch.Tensor,
    soft_negative: torch.Tensor,
) -> torch.Tensor:
    """Area-channel equivalent of Loss_func.masked_soft_bce_loss."""
    target_bin = (target > 0).float()
    inner_ring, outer_ring = build_dilated_rings(target, kernel_size=3, edge_size=4)
    weight = target.clone() + soft_negative
    weight[outer_ring > 0] = 1.0
    weight[inner_ring > 0] = 0.0
    weight[target_bin > 0] = 1.0
    weight[reliable_negative > 0] = 1.0
    bce = F.binary_cross_entropy_with_logits(logits, target_bin, weight=weight)
    push = -(torch.sigmoid(logits) * target).mean()
    return bce + push


def standard_bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    target_bin = (target > 0).float()
    bce = F.binary_cross_entropy_with_logits(logits, target_bin, pos_weight=pos_weight)
    probability = torch.sigmoid(logits)
    intersection = (probability * target_bin).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target_bin.sum(dim=(1, 2, 3))
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce + dice_loss


def _autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    matched: bool,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    amp: bool,
    positive_weight: float,
    trace_path: Path,
) -> List[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp)
    pos_weight = torch.tensor([positive_weight], dtype=torch.float32, device=device)
    epoch_losses: List[float] = []
    with trace_path.open("w", encoding="utf-8") as trace:
        for epoch in range(epochs):
            model.train()
            total, batches = 0.0, 0
            epoch_start = time.perf_counter()
            for batch in loader:
                batch = [item.to(device, non_blocking=True) for item in batch]
                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(device, amp):
                    logits = model(batch[0])
                    if matched:
                        loss = sparse_aware_loss(logits, batch[1], batch[2], batch[3])
                    else:
                        loss = standard_bce_dice_loss(logits, batch[1], pos_weight)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total += float(loss.detach().cpu())
                batches += 1
            average = total / max(1, batches)
            epoch_losses.append(average)
            record = {
                "epoch": epoch,
                "average_loss": average,
                "batches": batches,
                "wall_clock_seconds": time.perf_counter() - epoch_start,
            }
            trace.write(json.dumps(record) + "\n")
            trace.flush()
            print(
                f"[{utc_now()}] epoch={epoch + 1}/{epochs} "
                f"loss={average:.6f} batches={batches}",
                flush=True,
            )
    return epoch_losses


def tile_starts(length: int, tile: int, overlap: int) -> List[int]:
    if tile >= length:
        return [0]
    stride = tile - overlap
    starts = list(range(0, max(1, length - tile + 1), stride))
    final = length - tile
    if starts[-1] != final:
        starts.append(final)
    return starts


@torch.no_grad()
def infer_volume(
    model: nn.Module,
    volume: np.ndarray,
    *,
    thickness: int,
    device: torch.device,
    amp: bool,
    threshold: float,
    tile_size: int,
    overlap: int,
    inference_batch_size: int,
) -> np.ndarray:
    model.eval()
    depth, height, width = volume.shape
    tile_h, tile_w = min(tile_size, height), min(tile_size, width)
    y_starts = tile_starts(height, tile_h, overlap)
    x_starts = tile_starts(width, tile_w, overlap)
    prediction = np.zeros((depth, height, width), dtype=np.uint8)

    for z in range(depth):
        z_indices = np.clip(
            np.arange(z - thickness, z + thickness + 1), 0, depth - 1
        )
        stack = volume[z_indices]
        probability_sum = np.zeros((height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32)
        tiles: List[np.ndarray] = []
        locations: List[Tuple[int, int]] = []

        def flush_tiles() -> None:
            if not tiles:
                return
            tensor = torch.from_numpy(np.stack(tiles)).float().to(device, non_blocking=True)
            with _autocast_context(device, amp):
                probability = torch.sigmoid(model(tensor))[:, 0]
            probability_np = probability.float().cpu().numpy()
            for prob, (y0, x0) in zip(probability_np, locations):
                probability_sum[y0 : y0 + tile_h, x0 : x0 + tile_w] += prob
                counts[y0 : y0 + tile_h, x0 : x0 + tile_w] += 1.0
            tiles.clear()
            locations.clear()

        for y0 in y_starts:
            for x0 in x_starts:
                tiles.append(np.ascontiguousarray(stack[:, y0 : y0 + tile_h, x0 : x0 + tile_w]))
                locations.append((y0, x0))
                if len(tiles) >= inference_batch_size:
                    flush_tiles()
        flush_tiles()
        prediction[z] = (probability_sum / np.maximum(counts, 1.0) >= threshold).astype(np.uint8)
        if z % 10 == 0 or z == depth - 1:
            print(f"inference z={z + 1}/{depth}", flush=True)
    return prediction


@dataclass
class RunConfig:
    variant: str
    trial: int
    roi_num: int
    raw_path: str
    label_path: str
    negative_path: str
    output_dir: str
    seed: int
    patch_size: int
    thickness: int
    num_samples: int
    epochs: int
    batch_size: int
    learning_rate: float
    negative_threshold: float
    low_weight_radius: float
    prediction_threshold: float
    inference_tile_size: int
    inference_overlap: int
    inference_batch_size: int
    num_workers: int
    amp: bool


def run(config: RunConfig, *, overwrite: bool = False) -> Path:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / (
        f"prediction_hela2_mito_{config.trial}_{config.roi_num}.tif"
    )
    timing_path = output_dir / "timing.json"
    if prediction_path.exists() and timing_path.exists() and not overwrite:
        print(f"[resume] completed output exists: {prediction_path}")
        return prediction_path

    process_start = time.perf_counter()
    start_utc = utc_now()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_determinism(config.seed)
    timer = StageTimer(device)
    matched = config.variant in MATCHED_VARIANTS

    with timer.measure("data_loading_seconds"):
        raw = np.squeeze(tifffile.imread(config.raw_path))
        mask = (np.squeeze(tifffile.imread(config.label_path)) > 0).astype(np.uint8)
        if config.negative_path and Path(config.negative_path).exists():
            negative = (np.squeeze(tifffile.imread(config.negative_path)) > 0).astype(np.uint8)
        else:
            negative = np.zeros_like(mask, dtype=np.uint8)
        if raw.shape != mask.shape or mask.shape != negative.shape:
            raise ValueError(
                f"Shape mismatch: raw={raw.shape}, mask={mask.shape}, negative={negative.shape}"
            )

    with timer.measure("preprocessing_seconds"):
        volume = local_contrast_normalize(raw)
        del raw
        if matched:
            soft_negative = build_distance_mask(mask, radius=config.low_weight_radius)
        else:
            soft_negative = np.zeros_like(mask, dtype=np.float32)

    with timer.measure("sampler_build_seconds"):
        sampler_mode = (
            "positive_centered" if matched else
            "nnunet_foreground_oversample" if config.variant == "nnunet_2d" else
            "uniform"
        )
        centres = make_centres(
            mask,
            negative,
            soft_negative,
            mode=sampler_mode,
            patch_size=(config.patch_size, config.patch_size),
            thickness=config.thickness,
            num_samples=config.num_samples,
            negative_threshold=config.negative_threshold,
            seed=config.seed,
        )
        dataset = SparsePatchDataset(
            volume,
            mask,
            negative,
            soft_negative,
            centres,
            (config.patch_size, config.patch_size),
            config.thickness,
            matched,
        )
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=config.num_workers > 0,
        )

    model = build_model(config.variant, 2 * config.thickness + 1).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    positive_fraction = float(mask.mean())
    positive_weight = min(100.0, max(1.0, (1.0 - positive_fraction) / max(positive_fraction, 1e-8)))

    with timer.measure("training_seconds"):
        losses = train_model(
            model,
            loader,
            matched=matched,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            device=device,
            amp=config.amp,
            positive_weight=positive_weight,
            trace_path=output_dir / "training_trace.jsonl",
        )

    with timer.measure("checkpoint_save_seconds"):
        checkpoint = {
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "parameter_count": parameter_count,
            "final_training_loss": losses[-1] if losses else None,
        }
        torch.save(checkpoint, output_dir / "model.pt")
        with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(asdict(config), handle, indent=2)

    with timer.measure("inference_seconds"):
        prediction = infer_volume(
            model,
            volume,
            thickness=config.thickness,
            device=device,
            amp=config.amp,
            threshold=config.prediction_threshold,
            tile_size=config.inference_tile_size,
            overlap=config.inference_overlap,
            inference_batch_size=config.inference_batch_size,
        )

    with timer.measure("prediction_save_seconds"):
        temporary_path = prediction_path.with_suffix(".tif.partial")
        tifffile.imwrite(
            temporary_path,
            prediction,
            photometric="minisblack",
            compression="zlib",
        )
        os.replace(temporary_path, prediction_path)

    synchronize(device)
    total_seconds = time.perf_counter() - process_start
    timing = {
        "schema_version": 1,
        "measurement_status": "measured",
        "definition": "process start through completed binary TIFF save",
        "variant": config.variant,
        "implementation_label": (
            "SparseSeg MultiKernelUNet backbone only; conventional training"
            if config.variant == "sparseseg_backbone_conventional"
            else "nnU-Net-style PlainConv architecture; not official nnU-Net planner/trainer"
            if config.variant.startswith("nnunet_2d")
            else "Vanilla U-Net"
        ),
        "uses_sparse_seg_handcrafted_features": False,
        "uses_sparse_seg_iterative_refinement": False,
        "trial": config.trial,
        "roi_num": config.roi_num,
        "sampler_mode": sampler_mode,
        "loss_mode": "sparse_aware" if matched else "bce_plus_soft_dice",
        "start_utc": start_utc,
        "end_utc": utc_now(),
        "end_to_end_wall_clock_seconds": total_seconds,
        "stage_wall_clock_seconds": timer.values,
        "host": socket.gethostname(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "parameter_count": parameter_count,
        "prediction_path": str(prediction_path),
        "prediction_shape": list(prediction.shape),
        "predicted_foreground_fraction": float(prediction.mean()),
    }
    with timing_path.open("w", encoding="utf-8") as handle:
        json.dump(timing, handle, indent=2)
    print(json.dumps(timing, indent=2), flush=True)
    return prediction_path


def make_synthetic_smoke_inputs(root: Path) -> Tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    shape = (12, 96, 96)
    raw = rng.normal(100.0, 12.0, size=shape).astype(np.float32)
    mask = np.zeros(shape, dtype=np.uint8)
    yy, xx = np.ogrid[:96, :96]
    for z in range(2, 10):
        region = (yy - (44 + z // 3)) ** 2 + (xx - 52) ** 2 <= 10**2
        raw[z, region] += 55.0
        if z in (4, 7):
            mask[z, region] = 1
    negative = np.zeros(shape, dtype=np.uint8)
    negative[:, :8, :8] = 1
    raw_path, mask_path, negative_path = root / "raw.tif", root / "mask.tif", root / "negative.tif"
    tifffile.imwrite(raw_path, raw)
    tifffile.imwrite(mask_path, mask)
    tifffile.imwrite(negative_path, negative)
    return raw_path, mask_path, negative_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANT_OUTPUT_DIRS), required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--roi-num", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("inputdata"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--raw-name", default="hela2_em_s3.tif")
    parser.add_argument("--negative-name", default="negative_hela2_em_s3.tif")
    parser.add_argument("--patch-size", type=int, default=80)
    parser.add_argument("--thickness", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--negative-threshold", type=float, default=3.0)
    parser.add_argument("--low-weight-radius", type=float, default=50.0)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--inference-tile-size", type=int, default=512)
    parser.add_argument("--inference-overlap", type=int, default=64)
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    if args.smoke_test:
        data_root = output_root / "synthetic_inputs"
        raw_path, label_path, negative_path = make_synthetic_smoke_inputs(data_root)
        args.epochs = min(args.epochs, 1)
        args.num_samples = min(args.num_samples, 16)
        args.batch_size = min(args.batch_size, 4)
        args.patch_size = min(args.patch_size, 64)
        args.inference_tile_size = 64
        args.inference_overlap = 16
        args.num_workers = 0
    else:
        raw_path = data_root / args.raw_name
        label_path = data_root / f"label_hela2_mito_{args.trial}_{args.roi_num}.tif"
        negative_path = data_root / args.negative_name

    output_dir = output_root / VARIANT_OUTPUT_DIRS[args.variant] / f"{args.trial}_{args.roi_num}"
    config = RunConfig(
        variant=args.variant,
        trial=args.trial,
        roi_num=args.roi_num,
        raw_path=str(raw_path),
        label_path=str(label_path),
        negative_path=str(negative_path),
        output_dir=str(output_dir),
        seed=args.seed if args.seed is not None else args.trial * 100 + args.roi_num,
        patch_size=args.patch_size,
        thickness=args.thickness,
        num_samples=args.num_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        negative_threshold=args.negative_threshold,
        low_weight_radius=args.low_weight_radius,
        prediction_threshold=args.prediction_threshold,
        inference_tile_size=args.inference_tile_size,
        inference_overlap=args.inference_overlap,
        inference_batch_size=args.inference_batch_size,
        num_workers=args.num_workers,
        amp=not args.no_amp,
    )
    run(config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
