"""Minimal official nnU-Net v2 Trainer extension for the sparse-matched control.

The planner, preprocessor, PlainConvUNet construction, optimizer, scheduler,
augmentation, checkpointing and prediction remain those of nnunetv2 2.8.1.
Only the two experimentally required controls are changed: foreground-centred
sampling and the SparseSeg area-channel sparse-aware objective.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import autocast, nn

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context


SOFT_CODE_OFFSET = 3
SOFT_CODE_SCALE = 2500.0
RELIABLE_NEGATIVE_CODE = 2


class VEMSparseAwareLoss(nn.Module):
    """Decode packaged sparse weights and match SparseSeg masked-soft BCE+push."""

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if net_output.ndim != 4 or net_output.shape[1] != 2:
            raise ValueError(f"Expected two-class 2D nnU-Net logits, got {tuple(net_output.shape)}")
        if target.ndim != 4 or target.shape[1] != 1:
            raise ValueError(f"Expected one encoded target channel, got {tuple(target.shape)}")

        codes = target[:, 0].round().long()
        positive = (codes == 1).float().unsqueeze(1)
        reliable_negative = (codes == RELIABLE_NEGATIVE_CODE).unsqueeze(1)
        soft_code = (codes >= SOFT_CODE_OFFSET) & (codes <= 253)
        soft_negative = torch.zeros_like(positive)
        decoded = (codes.float() - SOFT_CODE_OFFSET) / SOFT_CODE_SCALE
        soft_negative[:, 0][soft_code] = decoded[soft_code]

        inner_dilation = F.max_pool2d(positive, kernel_size=3, stride=1, padding=1)
        outer_dilation = F.max_pool2d(positive, kernel_size=7, stride=1, padding=3)
        inner_ring = (inner_dilation - positive).clamp(min=0)
        outer_ring = (outer_dilation - inner_dilation).clamp(min=0)

        weight = soft_negative
        weight = torch.where(outer_ring > 0, torch.ones_like(weight), weight)
        weight = torch.where(inner_ring > 0, torch.zeros_like(weight), weight)
        weight = torch.where(positive > 0, torch.ones_like(weight), weight)
        weight = torch.where(reliable_negative, torch.ones_like(weight), weight)

        # Official nnU-Net produces two softmax logits. Their difference is the
        # binary foreground logit and keeps both official output channels active.
        foreground_logit = (net_output[:, 1] - net_output[:, 0]).unsqueeze(1)
        bce = F.binary_cross_entropy_with_logits(foreground_logit, positive, weight=weight)
        push = -(torch.sigmoid(foreground_logit) * positive).mean()
        return bce + push


class nnUNetTrainerVEMSparseMatched50epochs(nnUNetTrainer):
    """Official nnUNetTrainer with only sampler percentage and loss replaced."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int | str,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50
        self.oversample_foreground_percent = 1.0

    def _build_loss(self) -> nn.Module:
        loss: nn.Module = VEMSparseAwareLoss()
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = np.asarray([1 / (2**i) for i in range(len(scales))], dtype=float)
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0.0
            weights /= weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def validation_step(self, batch: dict) -> dict:
        """Official validation structure with binary metric-only target mapping.

        The sparse weight codes must remain intact for ``self.loss``. They are
        mapped to class 0/1 only for nnU-Net's online pseudo-Dice bookkeeping,
        preventing the weight codes from being interpreted as class indices.
        """
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [item.to(self.device, non_blocking=True) for item in target]
        else:
            target = target.to(self.device, non_blocking=True)

        context = (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        )
        with context:
            output = self.network(data)
            del data
            loss = self.loss(output, target)

        if self.enable_deep_supervision:
            output = output[0]
            metric_target = target[0]
        else:
            metric_target = target
        metric_target = (metric_target.round() == 1).long()

        axes = [0] + list(range(2, output.ndim))
        output_seg = output.argmax(1)[:, None]
        predicted_onehot = torch.zeros(
            output.shape, device=output.device, dtype=torch.float16
        )
        predicted_onehot.scatter_(1, output_seg, 1)
        del output_seg
        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_onehot, metric_target, axes=axes, mask=None
        )
        tp_hard = tp.detach().cpu().numpy()[1:]
        fp_hard = fp.detach().cpu().numpy()[1:]
        fn_hard = fn.detach().cpu().numpy()[1:]
        return {
            "loss": loss.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }


class nnUNetTrainerVEMSparseMatched1epoch(nnUNetTrainerVEMSparseMatched50epochs):
    """One-epoch preflight only; formal outputs never use this class."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int | str,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        # nnUNetTrainer records constructor arguments by reflection, so every
        # Trainer subclass must retain the explicit official signature.
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1
