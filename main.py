import os
from typing import List
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L

from hsdt import HSDT


class LitAutoEncoder(L.LightningModule):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        encoder_count: int,
        downsample_layers: List[int],
    ):
        super().__init__()
        self.model = HSDT(in_channels, channels, encoder_count, downsample_layers)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # x, _ = batch
        # x = x.view(x.size(0), -1)
        # z = self.encoder(x)
        # x_hat = self.decoder(z)
        # loss = F.mse_loss(x_hat, x)
        # return loss
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
