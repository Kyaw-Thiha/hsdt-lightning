import os
from typing import List, Optional
import torch
from torch import Tensor
from torch.optim import Optimizer
import torch.nn.functional as F
import lightning as L

from hsdt import HSDT
from metrics.psnr import compute_batch_mpsnr
from metrics.ssim import compute_batch_mssim


class HSDTLightning(L.LightningModule):
    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 6,
        encoder_count: int = 5,
        downsample_layers: List[int] = [1, 3],
        num_bands: int = 81,
    ):
        super().__init__()
        self.model = HSDT(in_channels, channels, encoder_count, downsample_layers, num_bands)

        # For saving the hyperparameters to saved logs & checkpoints
        self.save_hyperparameters()

        # For displaying the intermediate input and output sizes of all the layers
        batch_size = 32
        batch = 1
        channel = 1
        spectral_band = 81
        spatial_patch = 64
        self.example_input_array = torch.Tensor(batch, channel, spectral_band, spatial_patch, spatial_patch)

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

        return loss

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            print(f"Epoch-{self.current_epoch} Loss: {train_loss:.4f}")

    # Only for smoke test run.
    # Not used in actual training
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.model(x)


def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = {
        "scheduler": torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=3e-4,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy="cos",
            cycle_momentum=False,
        ),
        "interval": "step",
        "frequency": 1,
    }
    return {"optimizer": optimizer, "lr_scheduler": scheduler}
