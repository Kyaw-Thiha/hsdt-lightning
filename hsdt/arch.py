from collections import OrderedDict

import torch
import torch.nn as nn
from typing import Any, Callable, List, Mapping, Optional, Tuple, Union, Dict

# External modules must be implemented elsewhere
from .attention import TransformerBlock
from .sepconv import SepConv_DP, SepConv_DP_CA, S3Conv

BatchNorm3d = nn.BatchNorm3d
Conv3d = S3Conv.of(nn.Conv3d)
IsConvImpl: bool = False
UseBN: bool = True


def PlainConv(in_ch: int, out_ch: int) -> nn.Sequential:
    """
    Creates a basic convolution block with optional batch normalization and attention.

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.

    Returns:
        nn.Sequential containing Conv3d, BatchNorm3d (optional), and TransformerBlock.
    """
    return nn.Sequential(
        OrderedDict(
            [
                ("conv", Conv3d(in_ch, out_ch, 3, 1, 1, bias=False)),
                ("bn", BatchNorm3d(out_ch) if UseBN else nn.Identity()),
                ("attn", TransformerBlock(out_ch, bias=True)),
            ]
        )
    )


def DownConv(in_ch: int, out_ch: int) -> nn.Sequential:
    """
    Creates a downsampling block using strided convolution.

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.

    Returns:
        nn.Sequential block.
    """
    return nn.Sequential(
        OrderedDict(
            [
                ("conv", nn.Conv3d(in_ch, out_ch, 3, (1, 2, 2), 1, bias=False)),
                ("bn", BatchNorm3d(out_ch) if UseBN else nn.Identity()),
                ("attn", TransformerBlock(out_ch, bias=True)),
            ]
        )
    )


def UpConv(in_ch: int, out_ch: int) -> nn.Sequential:
    """
    Creates an upsampling block using trilinear interpolation followed by convolution.

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.

    Returns:
        nn.Sequential block.
    """
    return nn.Sequential(
        OrderedDict(
            [
                (
                    "up",
                    nn.Upsample(
                        scale_factor=(1, 2, 2), mode="trilinear", align_corners=True
                    ),
                ),
                ("conv", nn.Conv3d(in_ch, out_ch, 3, 1, 1, bias=False)),
                ("bn", BatchNorm3d(out_ch) if UseBN else nn.Identity()),
                ("attn", TransformerBlock(out_ch, bias=True)),
            ]
        )
    )


class Encoder(nn.Module):
    """
    Encoder consisting of a sequence of PlainConv and DownConv layers.

    Downsampling is applied at layers specified by sample_idx.
    """

    def __init__(
        self, channels: int, num_half_layer: int, downsample_layers: List[int]
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_half_layer):
            if i in downsample_layers:
                self.layers.append(DownConv(channels, 2 * channels))
                channels *= 2
            else:
                self.layers.append(PlainConv(channels, channels))

    def forward(self, x: torch.Tensor, xs: List[torch.Tensor]) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = layer(x)
            xs.append(x)
        return self.layers[-1](x)


class Decoder(nn.Module):
    """
    Decoder that mirrors the Encoder structure.

    Applies optional fusion operations and upsampling.
    """

    def __init__(
        self,
        channels: int,
        num_half_layer: int,
        downsample_layers: List[int],
        Fusion: Optional[Callable[[int], nn.Module]] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.enable_fusion = Fusion is not None

        if self.enable_fusion and Fusion is not None:
            self.fusions = nn.ModuleList()
            ch = channels
            for i in reversed(range(num_half_layer)):
                self.fusions.append(Fusion(ch))
                if i in downsample_layers:
                    ch //= 2

        for i in reversed(range(num_half_layer)):
            if i in downsample_layers:
                self.layers.append(UpConv(channels, channels // 2))
                channels //= 2
            else:
                self.layers.append(PlainConv(channels, channels))

    def forward(self, x: torch.Tensor, xs: List[torch.Tensor]) -> torch.Tensor:
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            skip = xs.pop()
            x = self.fusions[i](x, skip) if self.enable_fusion else x + skip
            x = self.layers[i](x)
        return x


class HSDT(nn.Module):
    """
    Hierarchical Spectral-Spatial Denoising Transformer (HSDT) model.

    This model is designed for hyperspectral or volumetric data processing, particularly tasks like
    denoising or super-resolution. It follows a U-Net–like architecture with:
      - A convolutional + transformer head (`PlainConv`)
      - A hierarchical encoder with optional spatial downsampling
      - A symmetric decoder with optional fusion modules
      - A residual tail connection for reconstructing the final output

    Architecture Flow:
        input
          ↓
        head (Conv + Transformer)
          ↓
        encoder (with skip connections and optional downsampling)
          ↓
        decoder (with upsampling and fusion)
          ↓
        tail (Conv3d + residual output)

    Args:
        in_channels (int):
            Number of input channels. For hyperspectral images, this is often 1 if processed per-band,
            or the full spectral band count if joint processing is used.

        channels (int):
            The base number of feature channels in the model. This determines the width of the network
            after the initial projection by `PlainConv`. It will be doubled at each downsampling layer.

        num_half_layer (int):
            The total number of layers in both encoder and decoder halves (symmetric).
            Each layer is either a plain transformer block or down/up block, depending on `downsample_layers`.

        downsample_layers (List[int]):
            Indices of layers (0-based) in the encoder where spatial downsampling occurs.
            For example, `[1, 3]` means downsample after the 1st and 3rd encoder blocks.
            The decoder will upsample at the same layers in reverse order.

        Fusion (Optional[Callable[[int], nn.Module]]):
            Optional callable that returns a fusion module for combining encoder and decoder features
            during skip connections. If `None`, skip connections use element-wise addition.

    Inputs:
        x (Tensor): Input tensor of shape (B, C, D, H, W)

    Returns:
        Tensor: Output tensor of shape (B, 1, D, H, W), with one spectral channel restored.

    Example:
        model = HSDT(in_channels=1, channels=16, num_half_layer=5, downsample_layers=[1, 3])
        x = torch.randn(1, 1, 31, 128, 128)
        out = model(x)  # shape: (1, 1, 31, 128, 128)
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_half_layer: int,
        downsample_layers: List[int],
        Fusion: Optional[Callable[[int], nn.Module]] = None,
    ):
        super().__init__()
        self.head = PlainConv(in_channels, channels)
        self.encoder = Encoder(channels, num_half_layer, downsample_layers)
        self.decoder = Decoder(
            channels * (2 ** len(downsample_layers)),
            num_half_layer,
            downsample_layers,
            Fusion,
        )
        self.tail = nn.Conv3d(channels, 1, 3, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs: List[torch.Tensor] = [x]
        out = self.head(x)
        xs.append(out)
        out = self.encoder(out, xs)
        out = self.decoder(out, xs)
        out = out + xs.pop()  # skip connection
        out = self.tail(out)
        out = out + xs.pop()[:, 0:1, :, :, :]  # final residual connection
        return out

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        """
        Load a state dict with optional reshaping for convolutional attention weights.

        Args:
            state_dict (Mapping[str, Any]): Model state dictionary.
            strict (bool): Whether to strictly enforce that the keys match.
            assign (bool): Whether to assign the weights directly to existing tensors.
        """
        if IsConvImpl:
            new_state_dict = {}
            for k, v in state_dict.items():
                if ("attn.attn" in k) and ("weight" in k) and ("attn_proj" not in k):
                    new_state_dict[k] = v.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict

        # Call base class implementation with all arguments
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def pad_mod(x: torch.Tensor, mod: int) -> Tuple[torch.Tensor, int, int]:
    """
    Pads the spatial dimensions (height and width) of a tensor so that they are divisible by `mod`.

    This is useful for models that require input dimensions to be multiples of a certain value
    (e.g., due to strided convolution or hierarchical structure).

    Args:
        x (Tensor): Input tensor of shape (..., H, W), e.g. [B, C, H, W] or [B, C, D, H, W].
        mod (int): The modulus to which the height and width should be padded.

    Returns:
        Tuple[Tensor, int, int]: A tuple containing:
            - The padded tensor of shape (..., H_out, W_out), where H_out and W_out are the smallest multiples of `mod` ≥ H and W.
            - The original height.
            - The original width.
    """
    h, w = x.shape[-2:]
    h_out = ((h + mod - 1) // mod) * mod
    w_out = ((w + mod - 1) // mod) * mod
    out = torch.zeros(*x.shape[:-2], h_out, w_out, dtype=x.dtype, device=x.device)
    out[..., :h, :w] = x
    return out, h, w
