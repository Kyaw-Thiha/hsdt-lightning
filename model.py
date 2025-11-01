import math
import os
from typing import Any, Dict, Optional, Sequence, cast

import lightning as L
import torch
from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary
from lightning.pytorch.utilities.types import LRSchedulerConfig, OptimizerLRScheduler
from scipy.io import savemat
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

from models import build_model
from metrics.charbonnier import charbonnier_loss
from metrics.psnr import compute_batch_mpsnr
from metrics.ssim import compute_batch_mssim


class HSDTLightning(L.LightningModule):
    def __init__(
        self,
        model_name: str = "tdsat",
        model_kwargs: Optional[Dict[str, Any]] = None,
        lr: float = 3e-4,  # 3e-4 is Good for Adam + > 32 batch size
        save_test: bool = False,
    ):
        super().__init__()

        resolved_kwargs = dict(model_kwargs or {})
        self.model, resolved_kwargs = build_model(model_name, **resolved_kwargs)
        self.model_name = model_name.lower()
        self.model_kwargs = resolved_kwargs

        # For saving the hyperparameters to saved logs & checkpoints
        self.save_hyperparameters(
            {
                "model_name": self.model_name,
                "model_kwargs": self.model_kwargs,
                "lr": lr,
                "save_test": save_test,
            }
        )
        self.lr: float = lr

        # Representative tensor for Lightning summaries
        self.example_input_array = self._build_example_input(self.model_name, self.model_kwargs)

        # For saving test images
        self.save_test = save_test
        self.save_folder = "data/output"

    @staticmethod
    def _extract_spatial_dims(spatial: Any) -> tuple[int, int]:
        if isinstance(spatial, Sequence) and not isinstance(spatial, (str, bytes)):
            sequence = list(spatial)
            if len(sequence) >= 2:
                return int(sequence[0]), int(sequence[1])
            if len(sequence) == 1:
                value = int(sequence[0])
                return value, value
        if isinstance(spatial, int):
            return spatial, spatial
        return 64, 64

    @classmethod
    def _build_example_input(cls, model_name: str, params: Dict[str, Any]) -> Tensor:
        channels = params.get("in_channels")
        if channels is None:
            channels = params.get("inp_channels", 1)
        channels = int(channels)

        height, width = cls._extract_spatial_dims(params.get("img_size", 64))

        if model_name == "hdst":
            return torch.zeros(1, channels, height, width)

        depth = params.get("num_bands")
        if depth is None:
            depth = params.get("spectral_bands")
        if depth is None:
            depth = params.get("depth_dim")
        if depth is None:
            depth = 81
        depth = int(depth)

        return torch.zeros(1, channels, depth, height, width)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        input, target = batch
        output: Tensor = self.model(input)
        # loss = F.mse_loss(output, target)

        loss = charbonnier_loss(output, target, eps=1e-3)
        ssim = compute_batch_mssim(output.clamp(0, 1), target)
        ssim_l = 1.0 - ssim
        alpha = 0.95
        loss = alpha * loss + (1 - alpha) * ssim_l

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        input, target = batch
        output: Tensor = self.model(input)

        loss = charbonnier_loss(output, target, eps=1e-3)
        ssim_l = 1.0 - compute_batch_mssim(output.clamp(0, 1), target)
        alpha = 0.95
        loss = alpha * loss + (1 - alpha) * ssim_l

        output = output.clamp(0, 1)  # Model outputs overshoot 1 a bit
        ssim = compute_batch_mssim(output, target)
        psnr = compute_batch_mpsnr(output, target)

        self.log("val_loss", loss)
        self.log("val_ssim", ssim, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_psnr", psnr, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        input, target = batch
        output: Tensor = self.model(input)

        # loss = F.mse_loss(output, target)
        loss = charbonnier_loss(output, target, eps=1e-3)

        output = output.clamp(0, 1)  # Model outputs overshoot 1 a bit
        ssim = compute_batch_mssim(output.clamp(0, 1), target)
        psnr = compute_batch_mpsnr(output.clamp(0, 1), target)

        self.log("test_loss", loss)
        self.log("test_ssim", ssim, prog_bar=True)
        self.log("test_psnr", psnr, prog_bar=True)

        if self.save_test:
            os.makedirs(self.save_folder, exist_ok=True)
            output_np = output.detach().cpu().numpy()
            output_np = output_np[0][0]  # First batch, first patch
            savemat(f"{self.save_folder}/batch-{batch_idx}.mat", {"input": output_np})

        return loss

    # Logging the loss after every epoch in training
    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            print(f"Epoch-{self.current_epoch} Loss: {train_loss:.4f}")

    def on_fit_start(self) -> None:
        summary = ModelSummary(self, max_depth=2)
        print(summary)

        if self.lr:
            for optimizer in self.trainer.optimizers:
                for pg in optimizer.param_groups:
                    pg["lr"] = self.lr

    # Only for smoke test run.
    # Not used in actual training
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer: OptimizerLRScheduler = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=3e-4, betas=(0.9, 0.98))

        total_steps = 0
        if self.trainer.max_steps == -1:
            assert self.trainer.max_epochs is not None, "self.train.max_epochs is None!!!"
            steps_per_epoch = int(self.trainer.estimated_stepping_batches / self.trainer.max_epochs)
            total_steps = self.trainer.max_epochs * steps_per_epoch
        else:
            total_steps = self.trainer.max_steps
        warmup_steps = int(0.1 * total_steps)

        def lr_lambda(current_step: int):
            """
            Piecewise LR multiplier: linear warmup then cosine decay.
            Args:
                current_step (int): Number of optimizer steps taken so far (0-indexed).
                                    Lightning advances this each `optimizer.step()`.
            Returns:
                float: Multiplicative factor applied to the base LR (optimizer param group 'lr').
                       - During warmup (0..warmup_steps): linearly increases from 0 -> 1.
                       - After warmup: cosine decay from 1 -> 0 over the remaining steps.
            """
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            floor = self.lr / 1000
            progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (floor / self.lr) + (1 - floor / self.lr) * cosine

        scheduler = cast(
            LRSchedulerConfig,
            {
                "scheduler": LambdaLR(optimizer, lr_lambda),
                "interval": "step",  # call every optimizer step
                "frequency": 1,
            },
            # {
            #     "scheduler": torch.optim.lr_scheduler.OneCycleLR(
            #         optimizer,
            #         max_lr=self.lr,
            #         epochs=self.trainer.max_epochs,
            #         steps_per_epoch=steps_per_epoch,
            #         pct_start=0.35,  # 35% Warm Start
            #         anneal_strategy="cos",
            #         cycle_momentum=False,  # Make it true only for momentum based optimizers like SGD
            #         div_factor=25.0,  # start at lr/25
            #         final_div_factor=1_000.0,  # end at lr/1000
            #     ),
            #     "interval": "step",
            #     "frequency": 1,
            # },
        )
        return [optimizer], [scheduler]
