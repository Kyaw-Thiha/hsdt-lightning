# -*- coding: utf-8 -*-
# File   : combinations.py
# Author : Guanyiman Fu, Fengchao Xiong, Member, IEEE, Jianfeng Lu, Member, IEEE, Jun Zhou, Senior Member, IEEE, Jiantao Zhou, Senior Member, IEEE, and Yuntao Qian, Senior Member, IEEE
# Date   : 09/01/2024
#
# This file is part of SSRT-Unet (https://arxiv.org/abs/2401.03885).
# https://github.com/lronkitty/SSRT/tree/main
# Distributed under MIT License.

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .utils.sync_batchnorm import SynchronizedBatchNorm2d, SynchronizedBatchNorm3d

BatchNorm3d = SynchronizedBatchNorm3d
BatchNorm2d = SynchronizedBatchNorm2d


class BNReLUConv3d(nn.Sequential):
    """BatchNorm → ReLU → 3D convolution convenience block."""

    def __init__(self, in_channels: int, channels: int, k: int = 3, s: int = 1, p: int = 1, inplace: bool = False) -> None:
        """Initialize the BN → ReLU → Conv3d block.

        Args:
            in_channels: Number of input channels.
            channels: Number of output channels.
            k: Convolution kernel size.
            s: Convolution stride.
            p: Convolution padding.
            inplace: Whether to perform the ReLU activation in-place.
        """
        super(BNReLUConv3d, self).__init__()
        self.add_module("bn", BatchNorm3d(in_channels))
        self.add_module("relu", nn.ReLU(inplace=inplace))
        self.add_module("conv", nn.Conv3d(in_channels, channels, k, s, p, bias=False))


class BNReLUConv2d(nn.Sequential):
    """BatchNorm → ReLU → 2D convolution convenience block."""

    def __init__(self, in_channels: int, channels: int, k: int = 3, s: int = 1, p: int = 1, inplace: bool = False) -> None:
        """Initialize the BN → ReLU → Conv2d block.

        Args:
            in_channels: Number of input channels.
            channels: Number of output channels.
            k: Convolution kernel size.
            s: Convolution stride.
            p: Convolution padding.
            inplace: Whether to perform the ReLU activation in-place.
        """
        super(BNReLUConv2d, self).__init__()
        self.add_module("bn", BatchNorm2d(in_channels))
        self.add_module("relu", nn.ReLU(inplace=inplace))
        self.add_module("conv", nn.Conv2d(in_channels, channels, k, s, p, bias=False))


class Conv3dBNReLU(nn.Sequential):
    """3D convolution → BatchNorm → ReLU convenience block."""

    def __init__(self, in_channels: int, channels: int, k: int = 3, s: int = 1, p: int = 1, inplace: bool = False) -> None:
        """Initialize the Conv3d → BN → ReLU block.

        Args:
            in_channels: Number of input channels.
            channels: Number of output channels.
            k: Convolution kernel size.
            s: Convolution stride.
            p: Convolution padding.
            inplace: Whether to perform the ReLU activation in-place.
        """
        super(Conv3dBNReLU, self).__init__()
        self.add_module("conv", nn.Conv3d(in_channels, channels, k, s, p, bias=False))
        self.add_module("bn", BatchNorm3d(channels))
        self.add_module("relu", nn.ReLU(inplace=inplace))


class Conv2dBNReLU(nn.Sequential):
    """2D convolution → BatchNorm → ReLU convenience block."""

    def __init__(self, in_channels: int, channels: int, k: int = 3, s: int = 1, p: int = 1, inplace: bool = False) -> None:
        """Initialize the Conv2d → BN → ReLU block.

        Args:
            in_channels: Number of input channels.
            channels: Number of output channels.
            k: Convolution kernel size.
            s: Convolution stride.
            p: Convolution padding.
            inplace: Whether to perform the ReLU activation in-place.
        """
        super(Conv2dBNReLU, self).__init__()
        self.add_module("conv", nn.Conv2d(in_channels, channels, k, s, p, bias=False))
        self.add_module("bn", BatchNorm2d(channels))
        self.add_module("relu", nn.ReLU(inplace=inplace))


class BNReLUDeConv3d(nn.Sequential):
    """BatchNorm → ReLU → ConvTranspose3d convenience block."""

    def __init__(self, in_channels: int, channels: int, k: int = 3, s: int = 1, p: int = 1, inplace: bool = False) -> None:
        """Initialize the BN → ReLU → ConvTranspose3d block.

        Args:
            in_channels: Number of input channels.
            channels: Number of output channels.
            k: Transposed convolution kernel size.
            s: Transposed convolution stride.
            p: Transposed convolution padding.
            inplace: Whether to perform the ReLU activation in-place.
        """
        super(BNReLUDeConv3d, self).__init__()
        self.add_module("bn", BatchNorm3d(in_channels))
        self.add_module("relu", nn.ReLU(inplace=inplace))
        self.add_module("deconv", nn.ConvTranspose3d(in_channels, channels, k, s, p, bias=False))


class BNReLUDeConv2d(nn.Sequential):
    """BatchNorm → ReLU → ConvTranspose2d convenience block."""

    def __init__(self, in_channels: int, channels: int, k: int = 3, s: int = 1, p: int = 1, inplace: bool = False) -> None:
        """Initialize the BN → ReLU → ConvTranspose2d block.

        Args:
            in_channels: Number of input channels.
            channels: Number of output channels.
            k: Transposed convolution kernel size.
            s: Transposed convolution stride.
            p: Transposed convolution padding.
            inplace: Whether to perform the ReLU activation in-place.
        """
        super(BNReLUDeConv2d, self).__init__()
        self.add_module("bn", BatchNorm2d(in_channels))
        self.add_module("relu", nn.ReLU(inplace=inplace))
        self.add_module("deconv", nn.ConvTranspose2d(in_channels, channels, k, s, p, bias=False))


class DeConv3dBNReLU(nn.Sequential):
    """ConvTranspose3d → BatchNorm → ReLU convenience block."""

    def __init__(self, in_channels: int, channels: int, k: int = 3, s: int = 1, p: int = 1, inplace: bool = False) -> None:
        """Initialize the ConvTranspose3d → BN → ReLU block.

        Args:
            in_channels: Number of input channels.
            channels: Number of output channels.
            k: Transposed convolution kernel size.
            s: Transposed convolution stride.
            p: Transposed convolution padding.
            inplace: Whether to perform the ReLU activation in-place.
        """
        super(DeConv3dBNReLU, self).__init__()
        self.add_module("deconv", nn.ConvTranspose3d(in_channels, channels, k, s, p, bias=False))
        self.add_module("bn", BatchNorm3d(channels))
        self.add_module("relu", nn.ReLU(inplace=inplace))


class DeConv2dBNReLU(nn.Sequential):
    """ConvTranspose2d → BatchNorm → ReLU convenience block."""

    def __init__(self, in_channels: int, channels: int, k: int = 3, s: int = 1, p: int = 1, inplace: bool = False) -> None:
        """Initialize the ConvTranspose2d → BN → ReLU block.

        Args:
            in_channels: Number of input channels.
            channels: Number of output channels.
            k: Transposed convolution kernel size.
            s: Transposed convolution stride.
            p: Transposed convolution padding.
            inplace: Whether to perform the ReLU activation in-place.
        """
        super(DeConv2dBNReLU, self).__init__()
        self.add_module("deconv", nn.ConvTranspose2d(in_channels, channels, k, s, p, bias=False))
        self.add_module("bn", BatchNorm2d(channels))
        self.add_module("relu", nn.ReLU(inplace=inplace))


class ReLUDeConv3d(nn.Sequential):
    """ReLU → ConvTranspose3d convenience block."""

    def __init__(self, in_channels: int, channels: int, k: int = 3, s: int = 1, p: int = 1, inplace: bool = False) -> None:
        """Initialize the ReLU → ConvTranspose3d block.

        Args:
            in_channels: Number of input channels.
            channels: Number of output channels.
            k: Transposed convolution kernel size.
            s: Transposed convolution stride.
            p: Transposed convolution padding.
            inplace: Whether to perform the ReLU activation in-place.
        """
        super(ReLUDeConv3d, self).__init__()
        self.add_module("relu", nn.ReLU(inplace=inplace))
        self.add_module("deconv", nn.ConvTranspose3d(in_channels, channels, k, s, p, bias=False))


class ReLUDeConv2d(nn.Sequential):
    """ReLU → ConvTranspose2d convenience block."""

    def __init__(self, in_channels: int, channels: int, k: int = 3, s: int = 1, p: int = 1, inplace: bool = False) -> None:
        """Initialize the ReLU → ConvTranspose2d block.

        Args:
            in_channels: Number of input channels.
            channels: Number of output channels.
            k: Transposed convolution kernel size.
            s: Transposed convolution stride.
            p: Transposed convolution padding.
            inplace: Whether to perform the ReLU activation in-place.
        """
        super(ReLUDeConv2d, self).__init__()
        self.add_module("relu", nn.ReLU(inplace=inplace))
        self.add_module("deconv", nn.ConvTranspose2d(in_channels, channels, k, s, p, bias=False))


class BNReLUUpsampleConv3d(nn.Sequential):
    """BatchNorm → ReLU → Upsample-then-conv block for 3D tensors."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        upsample: Tuple[int, int, int] = (1, 2, 2),
        inplace: bool = False,
    ) -> None:
        """Initialize the BN → ReLU → UpsampleConv3d block.

        Args:
            in_channels: Number of input channels.
            channels: Number of output channels.
            k: Convolution kernel size.
            s: Convolution stride.
            p: Convolution padding.
            upsample: Scale factor applied prior to the convolution.
            inplace: Whether to perform the ReLU activation in-place.
        """
        super(BNReLUUpsampleConv3d, self).__init__()
        self.add_module("bn", BatchNorm3d(in_channels))
        self.add_module("relu", nn.ReLU(inplace=inplace))
        self.add_module("upsample_conv", UpsampleConv3d(in_channels, channels, k, s, p, bias=False, upsample=upsample))


class BNReLUUpsampleConv2d(nn.Sequential):
    """BatchNorm → ReLU → Upsample-then-conv block for 2D tensors."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        upsample: Tuple[int, int] = (2, 2),
        inplace: bool = False,
    ) -> None:
        """Initialize the BN → ReLU → UpsampleConv2d block.

        Args:
            in_channels: Number of input channels.
            channels: Number of output channels.
            k: Convolution kernel size.
            s: Convolution stride.
            p: Convolution padding.
            upsample: Scale factor applied prior to the convolution.
            inplace: Whether to perform the ReLU activation in-place.
        """
        super(BNReLUUpsampleConv2d, self).__init__()
        self.add_module("bn", BatchNorm2d(in_channels))
        self.add_module("relu", nn.ReLU(inplace=inplace))
        self.add_module("upsample_conv", UpsampleConv2d(in_channels, channels, k, s, p, bias=False, upsample=upsample))


class UpsampleConv3dBNReLU(nn.Sequential):
    """Upsample-then-conv → BatchNorm → ReLU block for 3D tensors."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        upsample: Tuple[int, int, int] = (1, 2, 2),
        inplace: bool = False,
    ) -> None:
        """Initialize the UpsampleConv3d → BN → ReLU block.

        Args:
            in_channels: Number of input channels.
            channels: Number of output channels.
            k: Convolution kernel size.
            s: Convolution stride.
            p: Convolution padding.
            upsample: Scale factor applied prior to the convolution.
            inplace: Whether to perform the ReLU activation in-place.
        """
        super(UpsampleConv3dBNReLU, self).__init__()
        self.add_module("upsample_conv", UpsampleConv3d(in_channels, channels, k, s, p, bias=False, upsample=upsample))
        self.add_module("bn", BatchNorm3d(channels))
        self.add_module("relu", nn.ReLU(inplace=inplace))


class UpsampleConv2dBNReLU(nn.Sequential):
    """Upsample-then-conv → BatchNorm → ReLU block for 2D tensors."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        upsample: Tuple[int, int] = (1, 2),
        inplace: bool = False,
    ) -> None:
        """Initialize the UpsampleConv2d → BN → ReLU block.

        Args:
            in_channels: Number of input channels.
            channels: Number of output channels.
            k: Convolution kernel size.
            s: Convolution stride.
            p: Convolution padding.
            upsample: Scale factor applied prior to the convolution.
            inplace: Whether to perform the ReLU activation in-place.
        """
        super(UpsampleConv2dBNReLU, self).__init__()
        self.add_module("upsample_conv", UpsampleConv2d(in_channels, channels, k, s, p, bias=False, upsample=upsample))
        self.add_module("bn", BatchNorm2d(channels))
        self.add_module("relu", nn.ReLU(inplace=inplace))


class Conv3dReLU(nn.Sequential):
    """3D convolution optionally followed by BatchNorm, ending with ReLU."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        inplace: bool = False,
        bn: bool = False,
    ) -> None:
        """Initialize the Conv3d (+ BN) → ReLU block.

        Args:
            in_channels: Number of input channels.
            channels: Number of output channels.
            k: Convolution kernel size.
            s: Convolution stride.
            p: Convolution padding.
            inplace: Whether to perform the ReLU activation in-place.
            bn: Whether to prepend a BatchNorm layer.
        """
        super(Conv3dReLU, self).__init__()
        self.add_module("conv", nn.Conv3d(in_channels, channels, k, s, p, bias=False))
        if bn:
            self.add_module("bn", BatchNorm3d(channels))
        self.add_module("relu", nn.ReLU(inplace=inplace))


class Conv2dReLU(nn.Sequential):
    """2D convolution optionally followed by BatchNorm, ending with ReLU."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        inplace: bool = False,
        bn: bool = False,
    ) -> None:
        """Initialize the Conv2d (+ BN) → ReLU block.

        Args:
            in_channels: Number of input channels.
            channels: Number of output channels.
            k: Convolution kernel size.
            s: Convolution stride.
            p: Convolution padding.
            inplace: Whether to perform the ReLU activation in-place.
            bn: Whether to prepend a BatchNorm layer.
        """
        super(Conv2dReLU, self).__init__()
        self.add_module("conv", nn.Conv2d(in_channels, channels, k, s, p, bias=False))
        if bn:
            self.add_module("bn", BatchNorm2d(channels))
        self.add_module("relu", nn.ReLU(inplace=inplace))


class DeConv3dReLU(nn.Sequential):
    """ConvTranspose3d optionally followed by BatchNorm, ending with ReLU."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        inplace: bool = False,
        bn: bool = False,
    ) -> None:
        """Initialize the ConvTranspose3d (+ BN) → ReLU block.

        Args:
            in_channels: Number of input channels.
            channels: Number of output channels.
            k: Transposed convolution kernel size.
            s: Transposed convolution stride.
            p: Transposed convolution padding.
            inplace: Whether to perform the ReLU activation in-place.
            bn: Whether to prepend a BatchNorm layer.
        """
        super(DeConv3dReLU, self).__init__()
        self.add_module("deconv", nn.ConvTranspose3d(in_channels, channels, k, s, p, bias=False))
        if bn:
            self.add_module("bn", BatchNorm3d(channels))
        self.add_module("relu", nn.ReLU(inplace=inplace))


class DeConv2dReLU(nn.Sequential):
    """ConvTranspose2d optionally followed by BatchNorm, ending with ReLU."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        inplace: bool = False,
        bn: bool = False,
    ) -> None:
        """Initialize the ConvTranspose2d (+ BN) → ReLU block.

        Args:
            in_channels: Number of input channels.
            channels: Number of output channels.
            k: Transposed convolution kernel size.
            s: Transposed convolution stride.
            p: Transposed convolution padding.
            inplace: Whether to perform the ReLU activation in-place.
            bn: Whether to prepend a BatchNorm layer.
        """
        super(DeConv2dReLU, self).__init__()
        self.add_module("deconv", nn.ConvTranspose2d(in_channels, channels, k, s, p, bias=False))
        if bn:
            self.add_module("bn", BatchNorm2d(channels))
        self.add_module("relu", nn.ReLU(inplace=inplace))


class UpsampleConv3dReLU(nn.Sequential):
    """Upsample-then-conv with optional BatchNorm, ending with ReLU for 3D inputs."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        upsample: Tuple[int, int, int] = (1, 2, 2),
        inplace: bool = False,
        bn: bool = False,
    ) -> None:
        """Initialize the UpsampleConv3d (+ BN) → ReLU block.

        Args:
            in_channels: Number of input channels.
            channels: Number of output channels.
            k: Convolution kernel size.
            s: Convolution stride.
            p: Convolution padding.
            upsample: Scale factor applied prior to the convolution.
            inplace: Whether to perform the ReLU activation in-place.
            bn: Whether to prepend a BatchNorm layer.
        """
        super(UpsampleConv3dReLU, self).__init__()
        self.add_module("upsample_conv", UpsampleConv3d(in_channels, channels, k, s, p, bias=False, upsample=upsample))
        if bn:
            self.add_module("bn", BatchNorm3d(channels))
        self.add_module("relu", nn.ReLU(inplace=inplace))


class UpsampleConv2dReLU(nn.Sequential):
    """Upsample-then-conv ending with ReLU for 2D inputs."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        upsample: Tuple[int, int] = (1, 2),
        inplace: bool = False,
    ) -> None:
        """Initialize the UpsampleConv2d → ReLU block.

        Args:
            in_channels: Number of input channels.
            channels: Number of output channels.
            k: Convolution kernel size.
            s: Convolution stride.
            p: Convolution padding.
            upsample: Scale factor applied prior to the convolution.
            inplace: Whether to perform the ReLU activation in-place.
        """
        super(UpsampleConv2dReLU, self).__init__()
        self.add_module("upsample_conv", UpsampleConv2d(in_channels, channels, k, s, p, bias=False, upsample=upsample))
        self.add_module("relu", nn.ReLU(inplace=inplace))


class UpsampleConv3d(torch.nn.Module):
    """Upsample the signal and then apply a 3D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias: bool = True,
        upsample: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Initialize the upsample + 3D convolution block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size.
            stride: Convolution stride.
            padding: Convolution padding.
            bias: Whether to include a bias term in the convolution.
            upsample: Optional scale factor applied before the convolution.
        """
        super(UpsampleConv3d, self).__init__()
        self.upsample: Optional[Tuple[int, int, int]] = upsample
        if upsample is not None:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample, mode="trilinear", align_corners=True)
        else:
            self.upsample_layer = None

        self.conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the optional upsampling stage followed by convolution.

        Args:
            x: Input tensor of shape `(B, C, D, H, W)`.

        Returns:
            Tensor: Transformed tensor after optional upsampling and convolution.
        """
        x_in = x
        if self.upsample_layer is not None:
            x_in = self.upsample_layer(x_in)
        return self.conv3d(x_in)


class UpsampleConv2d(torch.nn.Module):
    """Upsample the signal and then apply a 2D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias: bool = True,
        upsample: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Initialize the upsample + 2D convolution block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size.
            stride: Convolution stride.
            padding: Convolution padding.
            bias: Whether to include a bias term in the convolution.
            upsample: Optional scale factor applied before the convolution.
        """
        super(UpsampleConv2d, self).__init__()
        self.upsample: Optional[Tuple[int, int]] = upsample
        if upsample is not None:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample, mode="bilinear", align_corners=True)
        else:
            self.upsample_layer = None

        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the optional upsampling stage followed by convolution.

        Args:
            x: Input tensor of shape `(B, C, H, W)`.

        Returns:
            Tensor: Transformed tensor after optional upsampling and convolution.
        """
        x_in = x
        if self.upsample_layer is not None:
            x_in = self.upsample_layer(x_in)
        return self.conv2d(x_in)


class BasicConv3d(nn.Sequential):
    """3D convolution optionally preceded by BatchNorm."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        bias: bool = False,
        bn: bool = True,
    ) -> None:
        """Initialize the basic 3D convolution block.

        Args:
            in_channels: Number of input channels.
            channels: Number of output channels.
            k: Convolution kernel size.
            s: Convolution stride.
            p: Convolution padding.
            bias: Whether to include a bias term in the convolution.
            bn: Whether to prepend a BatchNorm layer.
        """
        super(BasicConv3d, self).__init__()
        if bn:
            self.add_module("bn", BatchNorm3d(in_channels))
        self.add_module("conv", nn.Conv3d(in_channels, channels, k, s, p, bias=bias))


class BasicConv2d(nn.Sequential):
    """2D convolution optionally preceded by BatchNorm."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        bias: bool = False,
        bn: bool = True,
    ) -> None:
        """Initialize the basic 2D convolution block.

        Args:
            in_channels: Number of input channels.
            channels: Number of output channels.
            k: Convolution kernel size.
            s: Convolution stride.
            p: Convolution padding.
            bias: Whether to include a bias term in the convolution.
            bn: Whether to prepend a BatchNorm layer.
        """
        super(BasicConv2d, self).__init__()
        if bn:
            self.add_module("bn", BatchNorm2d(in_channels))
        self.add_module("conv", nn.Conv2d(in_channels, channels, k, s, p, bias=bias))


class BasicDeConv3d(nn.Sequential):
    """Transposed 3D convolution optionally preceded by BatchNorm."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        bias: bool = False,
        bn: bool = True,
    ) -> None:
        """Initialize the basic transposed 3D convolution block.

        Args:
            in_channels: Number of input channels.
            channels: Number of output channels.
            k: Transposed convolution kernel size.
            s: Transposed convolution stride.
            p: Transposed convolution padding.
            bias: Whether to include a bias term in the convolution.
            bn: Whether to prepend a BatchNorm layer.
        """
        super(BasicDeConv3d, self).__init__()
        if bn:
            self.add_module("bn", BatchNorm3d(in_channels))
        self.add_module("deconv", nn.ConvTranspose3d(in_channels, channels, k, s, p, bias=bias))


class BasicDeConv2d(nn.Sequential):
    """Transposed 2D convolution optionally preceded by BatchNorm."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        bias: bool = False,
        bn: bool = True,
    ) -> None:
        """Initialize the basic transposed 2D convolution block.

        Args:
            in_channels: Number of input channels.
            channels: Number of output channels.
            k: Transposed convolution kernel size.
            s: Transposed convolution stride.
            p: Transposed convolution padding.
            bias: Whether to include a bias term in the convolution.
            bn: Whether to prepend a BatchNorm layer.
        """
        super(BasicDeConv2d, self).__init__()
        if bn:
            self.add_module("bn", BatchNorm2d(in_channels))
        self.add_module("deconv", nn.ConvTranspose2d(in_channels, channels, k, s, p, bias=bias))


class BasicUpsampleConv3d(nn.Sequential):
    """Upsample-then-conv block optionally preceded by BatchNorm for 3D data."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        upsample: Tuple[int, int, int] = (1, 2, 2),
        bn: bool = True,
    ) -> None:
        """Initialize the basic 3D upsample-conv block.

        Args:
            in_channels: Number of input channels.
            channels: Number of output channels.
            k: Convolution kernel size.
            s: Convolution stride.
            p: Convolution padding.
            upsample: Scale factor applied prior to the convolution.
            bn: Whether to prepend a BatchNorm layer.
        """
        super(BasicUpsampleConv3d, self).__init__()
        if bn:
            self.add_module("bn", BatchNorm3d(in_channels))
        self.add_module("upsample_conv", UpsampleConv3d(in_channels, channels, k, s, p, bias=False, upsample=upsample))


class BasicUpsampleConv2d(nn.Sequential):
    """Upsample-then-conv block optionally preceded by BatchNorm for 2D data."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        upsample: Tuple[int, int] = (1, 2),
        bn: bool = True,
    ) -> None:
        """Initialize the basic 2D upsample-conv block.

        Args:
            in_channels: Number of input channels.
            channels: Number of output channels.
            k: Convolution kernel size.
            s: Convolution stride.
            p: Convolution padding.
            upsample: Scale factor applied prior to the convolution.
            bn: Whether to prepend a BatchNorm layer.
        """
        super(BasicUpsampleConv2d, self).__init__()
        if bn:
            self.add_module("bn", BatchNorm2d(in_channels))
        self.add_module("upsample_conv", UpsampleConv2d(in_channels, channels, k, s, p, bias=False, upsample=upsample))
