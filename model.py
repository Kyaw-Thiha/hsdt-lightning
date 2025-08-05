import os
from typing import List
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
        in_channels: int,
        channels: int,
        encoder_count: int,
        downsample_layers: List[int],
    ):
        super().__init__()
        self.model = HSDT(in_channels, channels, encoder_count, downsample_layers)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        input, target = batch
        output = self.model(input, target)
        loss = F.mse_loss(output, target)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        input, target = batch
        output = self.model(input, target)

        loss = F.mse_loss(output, target)
        ssim = compute_batch_mssim(output, target)
        psnr = compute_batch_mpsnr(output, target)

        self.log("val_loss", loss)
        self.log("val_ssim", ssim, on_step=False, on_epoch=True)
        self.log("val_psnr", psnr, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        input, target = batch
        output = self.model(input, target)

        loss = F.mse_loss(output, target)
        ssim = compute_batch_mssim(output, target)
        psnr = compute_batch_mpsnr(output, target)

        self.log("test_loss", loss)
        self.log("test_ssim", ssim)
        self.log("test_psnr", psnr)

        return loss

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
