import os
from typing import List, Optional, cast
from scipy.io import savemat
import torch
from torch import Tensor
from torch.optim import Optimizer
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary
from lightning.pytorch.utilities.types import LRSchedulerConfig, OptimizerLRScheduler

from hsdt import HSDT
from metrics.psnr import compute_batch_mpsnr
from metrics.ssim import compute_batch_mssim


class HSDTLightning(L.LightningModule):
    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 16,
        encoder_count: int = 5,
        downsample_layers: List[int] = [1, 3],
        num_bands: int = 81,
        lr: float = 3e-4,  # 3e-4 is Good for Adam + > 32 batch size
        save_test: bool = False,
    ):
        super().__init__()
        self.model = HSDT(in_channels, channels, encoder_count, downsample_layers, num_bands)

        # For saving the hyperparameters to saved logs & checkpoints
        self.save_hyperparameters()
        self.lr: float = lr

        # For displaying the intermediate input and output sizes of all the layers
        batch_size = 32
        batch = 1
        channel = 1
        spectral_band = 81
        spatial_patch = 64
        self.example_input_array = torch.Tensor(batch, channel, spectral_band, spatial_patch, spatial_patch)

        # For saving test images
        self.save_test = save_test
        self.save_folder = "data/output"

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        input, target = batch
        output = self.model(input)
        loss = F.mse_loss(output, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        input, target = batch
        output = self.model(input)

        loss = F.mse_loss(output, target)
        ssim = compute_batch_mssim(output, target)
        psnr = compute_batch_mpsnr(output, target)

        self.log("val_loss", loss)
        self.log("val_ssim", ssim, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_psnr", psnr, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        input, target = batch
        output = self.model(input)

        loss = F.mse_loss(output, target)
        ssim = compute_batch_mssim(output, target)
        psnr = compute_batch_mpsnr(output, target)

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
        optimizer: OptimizerLRScheduler = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        scheduler = cast(
            LRSchedulerConfig,
            {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.lr,
                    total_steps=int(self.trainer.estimated_stepping_batches),
                    pct_start=0.1,  # 10% Warm Start
                    anneal_strategy="cos",
                    cycle_momentum=False,  # Make it true only for momentum based optimizers like SGD
                ),
                "interval": "step",
                "frequency": 1,
            },
        )
        return [optimizer], [scheduler]
