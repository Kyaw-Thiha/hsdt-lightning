from functools import partial
from typing import Union, Tuple, Callable
import torch
import torch.nn as nn
from torch import Tensor


def repeat(x: Union[int, Tuple[int, int, int], list]) -> list[int]:
    """
    Ensures the input is a list of three values for 3D convolution parameters.

    Args:
        x: Either a single int or a tuple/list of length 3.

    Returns:
        List[int]: A list of three integers.
    """
    if isinstance(x, (tuple, list)):
        return list(x)
    return [x] * 3


class SepConv_PD(nn.Module):
    """
    Separable convolution: Pointwise followed by Depthwise.
    """

    def __init__(
        self,
        BASECONV: Callable,
        in_ch: int,
        out_ch: int,
        k: Union[int, Tuple[int, int, int]],
        s: int = 1,
        p: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        k_0, k_1, k_2 = repeat(k)
        s_0, s_1, s_2 = repeat(s)
        p_0, p_1, p_2 = repeat(p)
        self.pw_conv = BASECONV(
            in_ch,
            out_ch,
            (k_0, 1, 1),
            (s_0, 1, 1),
            (p_0, 0, 0),
            bias=bias,
            padding_mode="zeros",
        )
        self.dw_conv = BASECONV(
            out_ch,
            out_ch,
            (1, k_1, k_2),
            (1, s_1, s_2),
            (0, p_1, p_2),
            bias=bias,
            padding_mode="zeros",
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.pw_conv(x)
        x = self.dw_conv(x)
        return x

    @staticmethod
    def of(base_conv: Callable) -> Callable:
        return partial(SepConv_PD, base_conv)


class SepConv_DP(nn.Module):
    """
    Separable convolution: Depthwise followed by Pointwise.
    """

    def __init__(
        self,
        BASECONV: Callable,
        in_ch: int,
        out_ch: int,
        k: Union[int, Tuple[int, int, int]],
        s: int = 1,
        p: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        k_0, k_1, k_2 = repeat(k)
        s_0, s_1, s_2 = repeat(s)
        p_0, p_1, p_2 = repeat(p)

        self.dw_conv = BASECONV(
            in_ch,
            out_ch,
            (1, k_1, k_2),
            (1, s_1, s_2),
            (0, p_1, p_2),
            bias=bias,
            padding_mode="zeros",
        )
        self.pw_conv = BASECONV(
            out_ch,
            out_ch,
            (k_0, 1, 1),
            (s_0, 1, 1),
            (p_0, 0, 0),
            bias=bias,
            padding_mode="zeros",
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x

    @staticmethod
    def of(base_conv: Callable) -> Callable:
        return partial(SepConv_DP, base_conv)


class SepConv_DP_CA(nn.Module):
    """
    Separable convolution: Depthwise + Pointwise with Channel Attention (Sigmoid gating).
    """

    def __init__(
        self,
        BASECONV: Callable,
        in_ch: int,
        out_ch: int,
        k: Union[int, Tuple[int, int, int]],
        s: int = 1,
        p: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        k_0, k_1, k_2 = repeat(k)
        s_0, s_1, s_2 = repeat(s)
        p_0, p_1, p_2 = repeat(p)

        self.dw_conv = BASECONV(
            in_ch,
            out_ch,
            (1, k_1, k_2),
            (1, s_1, s_2),
            (0, p_1, p_2),
            bias=bias,
            padding_mode="zeros",
        )
        self.pw_conv = BASECONV(
            out_ch,
            out_ch * 2,
            (k_0, 1, 1),
            (s_0, 1, 1),
            (p_0, 0, 0),
            bias=bias,
            padding_mode="zeros",
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        x, w = torch.chunk(x, 2, dim=1)
        return x * w.sigmoid()

    @staticmethod
    def of(base_conv: Callable) -> Callable:
        return partial(SepConv_DP_CA, base_conv)


class S3Conv(nn.Module):
    """
    Spectral-Spatial Separable Convolution block (S3Conv).

    This block consists of:
    - Two **depthwise** 3D convolutions (applied across spatial dimensions only),
      separated by LeakyReLU non-linearities.
    - One **pointwise** 3D convolution (applied across spectral/band dimension only).
    - The outputs of the depthwise and pointwise branches are added together (residual fusion).

    Args:
        BASECONV (Callable): A convolution constructor (e.g. nn.Conv3d) to use for internal layers.
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        k (Union[int, Tuple[int, int, int]]): Kernel size for 3D convolutions.
            - If int, expands to (k, k, k)
            - First element (k[0]) used for pointwise conv (spectral axis)
            - Other elements (k[1], k[2]) used for depthwise conv (spatial axes)
        s (Union[int, Tuple[int, int, int]]): Stride values for convolutions.
            - Similar usage as `k`, default is 1
        p (Union[int, Tuple[int, int, int]]): Padding values for convolutions.
            - Similar usage as `k`, default is 1
        bias (bool): Whether to use bias in convolutions. Default: False

    Example:
        conv = S3Conv(nn.Conv3d, in_ch=16, out_ch=32, k=(3, 3, 3))
        x = torch.randn(1, 16, 5, 64, 64)  # [B, C, D, H, W]
        y = conv(x)
    """

    def __init__(
        self,
        BASECONV: Callable,
        in_ch: int,
        out_ch: int,
        k: Union[int, Tuple[int, int, int]],
        s: Union[int, Tuple[int, int, int]],
        p: Union[int, Tuple[int, int, int]],
        bias: bool = False,
    ):
        super().__init__()
        k_0, k_1, k_2 = repeat(k)
        s_0, s_1, s_2 = repeat(s)
        p_0, p_1, p_2 = repeat(p)

        self.dw_conv = nn.Sequential(
            BASECONV(
                in_ch,
                out_ch,
                (1, k_1, k_2),
                (1, s_1, s_2),
                (0, p_1, p_2),
                bias=bias,
                padding_mode="zeros",
            ),
            nn.LeakyReLU(),
            BASECONV(
                out_ch,
                out_ch,
                (1, k_1, k_2),
                1,
                (0, p_1, p_2),
                bias=bias,
                padding_mode="zeros",
            ),
            nn.LeakyReLU(),
        )
        self.pw_conv = BASECONV(
            in_ch,
            out_ch,
            (k_0, 1, 1),
            (s_0, 1, 1),
            (p_0, 0, 0),
            bias=bias,
            padding_mode="zeros",
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.dw_conv(x)
        x2 = self.pw_conv(x)
        return x1 + x2

    @staticmethod
    def of(base_conv: Callable) -> Callable:
        return partial(S3Conv, base_conv)


class S3ConvSequential(nn.Module):
    """
    Sequential variant of S3Conv: Pointwise is applied after all Depthwise blocks.
    """

    def __init__(
        self,
        BASECONV: Callable,
        in_ch: int,
        out_ch: int,
        k: Union[int, Tuple[int, int, int]],
        s: int = 1,
        p: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        k_0, k_1, k_2 = repeat(k)
        s_0, s_1, s_2 = repeat(s)
        p_0, p_1, p_2 = repeat(p)

        self.dw_conv = nn.Sequential(
            BASECONV(
                in_ch,
                out_ch,
                (1, k_1, k_2),
                (1, s_1, s_2),
                (0, p_1, p_2),
                bias=bias,
                padding_mode="zeros",
            ),
            nn.LeakyReLU(),
            BASECONV(
                out_ch,
                out_ch,
                (1, k_1, k_2),
                1,
                (0, p_1, p_2),
                bias=bias,
                padding_mode="zeros",
            ),
            nn.LeakyReLU(),
        )
        self.pw_conv = BASECONV(
            out_ch,
            out_ch,
            (k_0, 1, 1),
            (s_0, 1, 1),
            (p_0, 0, 0),
            bias=bias,
            padding_mode="zeros",
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x

    @staticmethod
    def of(base_conv: Callable) -> Callable:
        return partial(S3ConvSequential, base_conv)


class S3ConvLite(nn.Module):
    """
    Simpler version of S3Conv with one Depthwise conv block and one Pointwise conv.
    """

    def __init__(
        self,
        BASECONV: Callable,
        in_ch: int,
        out_ch: int,
        k: Union[int, Tuple[int, int, int]],
        s: int = 1,
        p: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        k_0, k_1, k_2 = repeat(k)
        s_0, s_1, s_2 = repeat(s)
        p_0, p_1, p_2 = repeat(p)

        self.dw_conv = nn.Sequential(
            BASECONV(
                in_ch,
                out_ch,
                (1, k_1, k_2),
                (1, s_1, s_2),
                (0, p_1, p_2),
                bias=bias,
                padding_mode="zeros",
            ),
            nn.LeakyReLU(),
        )
        self.pw_conv = BASECONV(
            in_ch,
            out_ch,
            (k_0, 1, 1),
            (s_0, 1, 1),
            (p_0, 0, 0),
            bias=bias,
            padding_mode="zeros",
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.dw_conv(x)
        x2 = self.pw_conv(x)
        return x1 + x2

    @staticmethod
    def of(base_conv: Callable) -> Callable:
        return partial(S3ConvLite, base_conv)
