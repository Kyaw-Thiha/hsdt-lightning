from collections.abc import Sequence
from typing import List, Union, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from timm.layers.drop import DropPath
from timm.layers.helpers import to_2tuple
from timm.layers.weight_init import trunc_normal_


def window_partition(x, window_size):
    """Split an image tensor into non-overlapping windows.

    Args:
        x (torch.Tensor): Input tensor of shape `(B, H, W, C)`.
        window_size (int): Spatial size of each square window.

    Returns:
        torch.Tensor: Tensor of shape `(num_windows * B, window_size, window_size, C)` containing windowed patches.
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reconstruct the full image tensor from windowed patches.

    Args:
        windows (torch.Tensor): Tensor of shape `(num_windows * B, window_size, window_size, C)`.
        window_size (int): Spatial size of each square window.
        H (int): Original image height.
        W (int): Original image width.

    Returns:
        torch.Tensor: Tensor of shape `(B, H, W, C)` representing the restored image.
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    """Two-layer feed-forward network with GELU activation and dropout."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        """Initialize the MLP module.

        Args:
            in_features (int): Size of the input feature dimension.
            hidden_features (int | None): Size of the hidden layer; defaults to `in_features`.
            out_features (int | None): Output feature dimension; defaults to `in_features`.
            act_layer (Callable[[], nn.Module]): Activation constructor, defaults to `nn.GELU`.
            drop (float): Dropout probability applied after each linear layer.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Run the MLP over the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape `(B, N, C)` or `(B, C)`.

        Returns:
            torch.Tensor: Tensor matching the input leading dimensions with the output feature dimension.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def img2windows(img, H_sp, W_sp):
    """Convert an image tensor into smaller spatial windows.

    Args:
        img (torch.Tensor): Input tensor of shape `(B, C, H, W)`.
        H_sp (int): Window height in pixels.
        W_sp (int): Window width in pixels.

    Returns:
        torch.Tensor: Tensor of shape `(B * (H / H_sp) * (W / W_sp), H_sp * W_sp, C)`.
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """Merge windowed tensors back into the original image layout.

    Args:
        img_splits_hw (torch.Tensor): Tensor of shape `(B', H_sp * W_sp, C)` or reshape-compatible view.
        H_sp (int): Window height used during partition.
        W_sp (int): Window width used during partition.
        H (int): Original image height.
        W (int): Original image width.

    Returns:
        torch.Tensor: Tensor of shape `(B, H, W, C)` representing the reconstructed image.
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class LePEAttention(nn.Module):
    """Localized positional encoding attention operating on partitioned windows."""

    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0.0, qk_scale=None):
        """Set up the localized positional encoding attention branch.

        Args:
            dim (int): Input embedding dimension.
            resolution (int): Window resolution (assumes square spatial layout).
            idx (int): Axis selector (0 for horizontal, 1 for vertical splits).
            split_size (int): Size of the non-overlapping window splits.
            dim_out (int | None): Output embedding dimension; defaults to `dim`.
            num_heads (int): Number of attention heads.
            attn_drop (float): Dropout probability applied to attention weights.
            qk_scale (float | None): Optional scaling factor applied to queries.
        """
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5
        if idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        """Reshape tokens into CSWin-style windows.

        Args:
            x (torch.Tensor): Tensor of shape `(B, N, C)` representing flattened image tokens.

        Returns:
            torch.Tensor: Tensor of shape `(B * num_windows, num_heads, window_area, C / num_heads)`.
        """
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        """Compute localized positional encoding for value features.

        Args:
            x (torch.Tensor): Tensor of shape `(B, N, C)` with flattened spatial tokens.
            func (Callable[[torch.Tensor], torch.Tensor]): Depth-wise convolution applied to compute positional bias.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing the reshaped value tensor and corresponding positional encoding.
        """
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv, mask=None):
        """Apply localized positional encoding attention.

        Args:
            qkv (torch.Tensor): Tensor of shape `(3, B, L, C)` containing query, key, and value projections.
            mask (torch.Tensor | None): Optional attention mask; not used in this implementation.

        Returns:
            torch.Tensor: Tensor of shape `(B, L, C)` containing the attended features.
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        H = W = self.resolution
        B, L, C = q.shape
        # assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q)

        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        # print(q.shape,k.shape)
        attn = q @ k.transpose(-2, -1)  # B head N C @ B head C N --> B head N N

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x

    def flops(self, shape):
        """Estimate the floating-point operations for a forward pass.

        Args:
            shape (tuple[int, int]): Spatial shape `(H, W)` of the feature map.

        Returns:
            int: Estimated FLOPs for the attention branch.
        """
        flops = 0
        H, W = shape
        # q, k, v = (B* H//H_sp * W//W_sp) heads H_sp*W_sp C//heads
        flops += (
            ((H // self.H_sp) * (W // self.W_sp))
            * self.num_heads
            * (self.H_sp * self.W_sp)
            * (self.dim // self.num_heads)
            * (self.H_sp * self.W_sp)
        )
        flops += (
            ((H // self.H_sp) * (W // self.W_sp))
            * self.num_heads
            * (self.H_sp * self.W_sp)
            * (self.dim // self.num_heads)
            * (self.H_sp * self.W_sp)
        )

        return flops


class ChannelAttention(nn.Module):
    """Dictionary-guided channel attention with low-rank projections."""

    def __init__(self, num_feat, squeeze_factor=16, memory_blocks=128):
        """Set up the channel attention module.

        Args:
            num_feat (int): Number of feature channels in the input.
            squeeze_factor (int): Reduction factor for the bottleneck dimension.
            memory_blocks (int): Number of learned dictionary atoms.
        """
        super(ChannelAttention, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.subnet = nn.Sequential(
            nn.Linear(num_feat, num_feat // squeeze_factor),
            # nn.ReLU(inplace=True)
        )
        self.upnet = nn.Sequential(
            nn.Linear(num_feat // squeeze_factor, num_feat),
            # nn.Linear(num_feat, num_feat),
            nn.Sigmoid(),
        )
        self.mb = torch.nn.Parameter(torch.randn(num_feat // squeeze_factor, memory_blocks))
        self.low_dim = num_feat // squeeze_factor

    def forward(self, x):
        """Apply channel attention weights.

        Args:
            x (torch.Tensor): Tensor of shape `(B, N, C)` representing window features.

        Returns:
            torch.Tensor: Tensor of shape `(B, N, C)` with reweighted channels.
        """
        b, n, c = x.shape
        t = x.transpose(1, 2)
        y = self.pool(t).squeeze(-1)

        low_rank_f = self.subnet(y).unsqueeze(2)

        mbg = self.mb.unsqueeze(0).repeat(b, 1, 1)
        f1 = (low_rank_f.transpose(1, 2)) @ mbg
        f_dic_c = F.softmax(f1 * (int(self.low_dim) ** (-0.5)), dim=-1)  # get the similarity information
        y1 = f_dic_c @ mbg.transpose(1, 2)
        y2 = self.upnet(y1)
        out = x * y2
        return out


class CAB(nn.Module):
    """Channel attention block combining compression and dictionary guidance."""

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30, memory_blocks=128):
        """Initialize the CAB module.

        Args:
            num_feat (int): Number of feature channels.
            compress_ratio (int): Reduction factor for the first linear layer.
            squeeze_factor (int): Reduction factor inside the channel attention.
            memory_blocks (int): Number of memory blocks for the channel attention dictionary.
        """
        super(CAB, self).__init__()
        self.num_feat = num_feat
        self.cab = nn.Sequential(
            nn.Linear(num_feat, num_feat // compress_ratio),
            nn.GELU(),
            nn.Linear(num_feat // compress_ratio, num_feat),
            ChannelAttention(num_feat, squeeze_factor, memory_blocks),
        )

    def forward(self, x):
        """Apply the CAB transformation.

        Args:
            x (torch.Tensor): Tensor of shape `(B, N, C)`.

        Returns:
            torch.Tensor: Tensor of shape `(B, N, C)` after attention reweighting.
        """
        return self.cab(x)

    def flops(self, shape):
        """Estimate FLOPs for the CAB module.

        Args:
            shape (tuple[int, int]): Spatial shape `(H, W)` of the feature grid.

        Returns:
            int: Estimated number of floating-point operations.
        """
        flops = 0
        H, W = shape
        flops += self.num_feat * H * W
        return flops


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with channel attention fusion."""

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias: bool = True,
        qk_scale=None,
        memory_blocks=128,
        down_rank=16,
        weight_factor=0.1,
        attn_drop=0.0,
        proj_drop=0.0,
        split_size=1,
    ):
        """Construct the window attention module.

        Args:
            dim (int): Token embedding dimension.
            window_size (tuple[int, int]): Spatial size `(Wh, Ww)` of each window.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): Whether to include learnable bias terms in QKV projections.
            qk_scale (float | None): Optional scaling factor for queries.
            memory_blocks (int): Number of dictionary atoms in the channel attention branch.
            down_rank (int): Reduction factor used in channel attention.
            weight_factor (float): Scaling factor for channel attention fusion.
            attn_drop (float): Dropout applied to attention probabilities.
            proj_drop (float): Dropout applied after the output projection.
            split_size (int): Number of spatial splits along each axis.
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.weight_factor = weight_factor

        self.attns = nn.ModuleList(
            [
                LePEAttention(
                    dim // 2,
                    resolution=self.window_size[0],
                    idx=i,
                    split_size=split_size,
                    num_heads=num_heads // 2,
                    dim_out=dim // 2,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                )
                for i in range(2)
            ]
        )

        self.c_attns = CAB(dim, compress_ratio=4, squeeze_factor=down_rank, memory_blocks=memory_blocks)  #
        # self.c_attns_15 = CAB(dim,compress_ratio=4,squeeze_factor=15)
        # self.c_attns = Subspace(dim)

    def forward(self, x, mask=None):
        """Apply window attention and channel attention.

        Args:
            x (torch.Tensor): Input tensor of shape `(num_windows * B, N, C)`.
            mask (torch.Tensor | None): Optional attention mask of shape `(num_windows, Wh*Ww, Wh*Ww)`.

        Returns:
            torch.Tensor: Tensor of shape `(num_windows * B, N, C)` after attention.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        x1 = self.attns[0](qkv[:, :, :, : C // 2], mask)
        x2 = self.attns[1](qkv[:, :, :, C // 2 :], mask)

        attened_x = torch.cat([x1, x2], dim=2)

        attened_x = rearrange(attened_x, "b n (g d) -> b n ( d g)", g=4)

        x3 = self.c_attns(x)
        attn = attened_x + self.weight_factor * x3

        x = self.proj(attn)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        """Return a concise representation string for logging."""
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, shape):
        """Estimate FLOPs of the window attention.

        Args:
            shape (tuple[int, int]): Spatial shape `(H, W)` of the input features.

        Returns:
            int: Estimated floating-point operations.
        """
        # calculate flops for 1 window with token length of N
        flops = 0
        H, W = shape
        # qkv = self.qkv(x)
        attn_first = cast(LePEAttention, self.attns[0])
        attn_second = cast(LePEAttention, self.attns[1])
        flops += attn_first.flops((H, W))
        flops += attn_second.flops((H, W))
        flops += self.c_attns.flops([H, W])
        return flops


class ASPP(nn.Module):
    """Atrous spatial pyramid pooling with depthwise convolutions."""

    def __init__(self, in_channels, out_channels, dilation_rates=[2, 4, 6]):
        """Initialize the ASPP module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels produced by each branch.
            dilation_rates (list[int]): Dilation factors for the atrous branches.
        """
        super().__init__()
        # 维度对齐卷积
        self.pre_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.GELU())
        # 多尺度分支
        self.atrous = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        out_channels, out_channels, 3, padding=d * 2, dilation=d * 2, groups=out_channels, bias=False
                    ),  # 深度可分离卷积
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                )
                for d in dilation_rates
            ]
        )
        # 全局上下文分支
        self.global_ctx = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_channels, out_channels, 1, bias=False), nn.GELU())
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilation_rates) + 2), out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        """Forward pass through ASPP.

        Args:
            x (torch.Tensor): Input tensor of shape `(B, C_in, H, W)`.

        Returns:
            torch.Tensor: Tensor of shape `(B, out_channels, H, W)` after multi-scale fusion.
        """
        # 输入维度对齐
        x = self.pre_conv(x)
        B, C, H, W = x.shape
        # 多尺度特征提取
        features = [x]
        for conv in self.atrous:
            features.append(conv(x))
        # 全局上下文
        global_feat = self.global_ctx(x)
        global_feat = F.interpolate(global_feat, (H, W), mode="bilinear", align_corners=False)
        features.append(global_feat)
        # 特征融合
        fused = self.fusion(torch.cat(features, dim=1))
        return x + 0.2 * fused  # 残差连接


class ASPP_Enhanced(nn.Module):
    """ASPP variant with optional global context for richer multi-scale features."""

    def __init__(self, in_channels, out_channels, dilation_rates=[2, 4, 6], use_global_branch=1):
        """Enhanced ASPP with optional global context branch.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels per branch.
            dilation_rates (list[int]): Dilation rates for atrous convolutions.
            use_global_branch (int): Flag controlling inclusion of the global pooling branch.
        """
        super().__init__()
        self.use_global_branch = use_global_branch

        # 维度对齐卷积
        self.pre_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.GELU())

        # 多尺度分支
        self.atrous = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=d * 2, dilation=d * 2, groups=out_channels, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                )
                for d in dilation_rates
            ]
        )

        # 全局上下文分支（可选）
        if use_global_branch:
            self.global_ctx = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_channels, out_channels, 1, bias=False), nn.GELU()
            )

        # 动态计算融合层输入通道数
        num_branches = len(dilation_rates) + 1  # 基础分支 + 扩张卷积分支
        if use_global_branch:
            num_branches += 1  # 添加全局分支

        # 融合层（自动适应分支数量）
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * num_branches, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        """Forward pass through the enhanced ASPP block.

        Args:
            x (torch.Tensor): Input tensor of shape `(B, C_in, H, W)`.

        Returns:
            torch.Tensor: Tensor of shape `(B, out_channels, H, W)` after multi-branch fusion.
        """
        # 输入维度对齐
        x = self.pre_conv(x)
        B, C, H, W = x.shape

        # 多尺度特征提取
        features = [x]
        for conv in self.atrous:
            features.append(conv(x))

        # 可选全局上下文分支
        if self.use_global_branch:
            global_feat = self.global_ctx(x)
            global_feat = F.interpolate(global_feat, (H, W), mode="bilinear", align_corners=False)
            features.append(global_feat)

        # 特征融合
        fused = self.fusion(torch.cat(features, dim=1))
        return x + 0.2 * fused  # 残差连接


class SSMTDA(nn.Module):
    """Spatial-spectral multi-head transformer dual attention block."""

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        split_size=1,
        drop_path=0.0,
        weight_factor=0.1,
        memory_blocks=128,
        down_rank=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        ae_enable=0,
    ):
        """Initialize the SSMTDA block.

        Args:
            dim (int): Channel dimension of the input tensor.
            input_resolution (tuple[int, int]): Spatial resolution `(H, W)` of the feature map.
            num_heads (int): Number of attention heads.
            window_size (int): Window size used for window attention.
            shift_size (int): Shift amount for shifted window attention.
            split_size (int): Number of spatial splits in window attention.
            drop_path (float): Stochastic depth rate.
            weight_factor (float): Scaling factor when mixing channel attention response.
            memory_blocks (int): Number of channel-attention dictionary atoms.
            down_rank (int): Reduction factor for the channel attention.
            mlp_ratio (float): Ratio between MLP hidden dimension and `dim`.
            qkv_bias (bool): Whether to include bias in QKV projections.
            qk_scale (float | None): Optional scaling factor for Q.
            drop (float): Dropout probability applied in MLP.
            attn_drop (float): Dropout applied to attention probabilities.
            act_layer (Callable[[], nn.Module]): Activation constructor for the MLP.
            ae_enable (int): Flag controlling the after-effect branch activation.
        """
        super(SSMTDA, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.weight_factor = weight_factor

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attns = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            memory_blocks=memory_blocks,
            down_rank=down_rank,
            weight_factor=weight_factor,
            split_size=split_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.num_heads = num_heads

        self.ae_enable = ae_enable
        if self.ae_enable == 0:
            self.aspp = ASPP_Enhanced(dim, dim, [2, 4, 6])

    def forward(self, x):
        """Forward pass through the SSMTDA block.

        Args:
            x (torch.Tensor): Input tensor of shape `(B, C, H, W)`.

        Returns:
            torch.Tensor: Output tensor of shape `(B, C, H, W)` after attention and optional ASPP.
        """
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attns(x_windows)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.transpose(1, 2).view(B, C, H, W)

        if self.ae_enable == 0:
            x = self.aspp(x)  # 新增ASPP

        return x

    def extra_repr(self) -> str:
        """Return a string summarizing key hyper-parameters."""
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )

    def flops(self, shape):
        """Estimate FLOPs for this block.

        Args:
            shape (tuple[int, int]): Spatial shape `(H, W)` of the feature map.

        Returns:
            int: Estimated floating-point operations.
        """
        flops = 0
        H, W = shape
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attns.flops([self.window_size, self.window_size])
        return flops


class FrequencyDomainProcess(nn.Module):
    """Frequency-domain enhancement with learnable filtering and adaptive gating."""

    def __init__(self, dim, aspp_enable=1):
        """Initialize the frequency-domain processing block.

        Args:
            dim (int): Number of channels per frequency representation (real or imaginary).
            aspp_enable (int): Flag controlling ASPP-based gating path.
        """
        super().__init__()
        self.aspp_enable = aspp_enable
        # 频域变换层
        self.fft_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 1),  # 处理实虚部分
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, 1),
        )
        # 空间重建层
        self.recon_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.GELU())
        # 动态门控
        if self.aspp_enable == 0:
            self.gate = nn.Sequential(nn.Conv2d(dim * 2, dim // 8, 1), nn.ReLU(), nn.Conv2d(dim // 8, 1, 1), nn.Sigmoid())
        elif self.aspp_enable == 1:
            self.aspp_gate = ASPP_Enhanced(dim * 2, dim // 8, dilation_rates=[2, 4, 8])
            self.gate_fc = nn.Sequential(nn.Conv2d(dim // 8, 1, 1), nn.Sigmoid())

    def forward(self, x):
        """Apply frequency-domain filtering and gated fusion.

        Args:
            x (torch.Tensor): Input tensor of shape `(B, C, H, W)`.

        Returns:
            torch.Tensor: Tensor of shape `(B, C, H, W)` after fusing spatial and frequency features.
        """
        # 频域变换
        fft_feat = torch.fft.rfft2(x, norm="ortho")
        fft_cat = torch.cat([fft_feat.real, fft_feat.imag], dim=1)
        # 频域滤波
        fft_processed = self.fft_conv(fft_cat)
        real, imag = torch.chunk(fft_processed, 2, dim=1)
        freq_feat = torch.fft.irfft2(torch.complex(real, imag), s=x.shape[-2:])
        # 动态融合
        gate = torch.ones_like(x[:, :1])
        if self.aspp_enable == 0:
            gate = self.gate(torch.cat([x, freq_feat], dim=1))
        elif self.aspp_enable == 1:
            gate = self.gate_fc(self.aspp_gate(fft_cat))
            gate = F.interpolate(gate, size=x.shape[2:], mode="bilinear")  # 插值到原始尺寸
        return x * gate + self.recon_conv(freq_feat) * (1 - gate)


class WindowCrossAttention(nn.Module):
    """Cross-attention between spatial and frequency domains over windowed patches."""

    def __init__(self, dim, window_size=32, num_heads=4):
        """Initialize the cross-attention module.

        Args:
            dim (int): Feature dimension for attention queries/keys/values.
            window_size (int): Spatial window size used to partition the inputs.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(dim, num_heads)

    def forward(self, spa, freq):
        """Fuse spatial and frequency features through windowed cross-attention.

        Args:
            spa (torch.Tensor): Spatial-domain tensor of shape `(B, C, H, W)`.
            freq (torch.Tensor): Frequency-domain tensor of shape `(B, C, H, W)`.

        Returns:
            torch.Tensor: Tensor of shape `(B, C, H, W)` containing fused features.
        """
        B, C, H, W = spa.shape
        # 窗口划分
        spa = spa.view(B, C, H // self.window_size, self.window_size, W // self.window_size, self.window_size)
        spa = spa.permute(0, 2, 4, 1, 3, 5).reshape(-1, C, self.window_size, self.window_size)
        freq = freq.view(B, C, H // self.window_size, self.window_size, W // self.window_size, self.window_size)
        freq = freq.permute(0, 2, 4, 1, 3, 5).reshape(-1, C, self.window_size, self.window_size)

        # 窗口内注意力
        spa_flat = spa.flatten(2).permute(2, 0, 1)  # [win_size^2, B*num_win, C]
        freq_flat = freq.flatten(2).permute(2, 0, 1)
        fused, _ = self.attn(spa_flat, freq_flat, freq_flat)

        # 恢复形状
        fused = fused.permute(1, 2, 0).view(-1, C, self.window_size, self.window_size)
        fused = fused.permute(0, 2, 3, 1).reshape(B, H, W, C).permute(0, 3, 1, 2)
        return fused


class SMS_AfterEffect(nn.Module):
    """After-effect module combining spatial transformers, frequency processing, and ASPP."""

    def __init__(
        self,
        dim=90,
        window_size=8,
        depth=6,
        num_head=6,
        mlp_ratio=2,
        qkv_bias=True,
        qk_scale=None,
        weight_factor=0.1,
        memory_blocks=128,
        down_rank=16,
        drop_path: Union[Sequence[float], float] = 0.0,
        split_size=1,
        num_heads_WCA=4,
        window_size_WCA=32,
    ):
        """Configure the after-effect stack.

        Args:
            dim (int): Channel dimension throughout the module.
            window_size (int): Spatial window size for attention blocks.
            depth (int): Total number of transformer layers in the spatial block.
            num_head (int): Attention heads inside transformer layers.
            mlp_ratio (float): MLP expansion ratio for transformer blocks.
            qkv_bias (bool): Whether to use bias in QKV projections.
            qk_scale (float | None): Optional scaling factor for queries.
            weight_factor (float): Mixing weight for channel attention contributions.
            memory_blocks (int): Number of atoms in channel attention dictionaries.
            down_rank (int): Dimensionality reduction within channel attention.
            drop_path (float | Sequence[float]): Stochastic depth schedule.
            split_size (int): Window split factor for attention.
            num_heads_WCA (int): Number of heads in the window cross-attention.
            window_size_WCA (int): Window size for cross-attention fusion.
        """
        super().__init__()
        if isinstance(drop_path, float):
            drop_path_seq: List[float] = [drop_path] * depth
        else:
            drop_path = cast(Sequence[float], drop_path)
            drop_path_seq = list(drop_path)
        if len(drop_path_seq) < depth:
            raise ValueError("drop_path length must be at least the block depth.")
        self.beta = nn.Parameter(torch.tensor(0.1))  # 新增可学习参数，初始化为0
        # 空间域Transformer
        self.spatial_block = nn.Sequential(
            *[
                SSMTDA(
                    dim=dim,
                    input_resolution=window_size,
                    num_heads=num_head,
                    memory_blocks=memory_blocks,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                    weight_factor=weight_factor,
                    down_rank=down_rank,
                    split_size=split_size,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path_seq[i],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                )
                for i in range(depth - 2, depth)
            ]
        )
        # 轻量化频域处理
        self.freq_process = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1), FrequencyDomainProcess(dim // 4), nn.Conv2d(dim // 4, dim, 1)
        )
        # 跨域注意力
        self.cross_attn = WindowCrossAttention(dim, window_size_WCA, num_heads_WCA)
        # 局部-全局融合
        self.fusion = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 3, padding=1))
        self.aspp = ASPP_Enhanced(dim, dim, [2, 4, 6])

    def forward(self, x):
        """Execute the after-effect processing chain.

        Args:
            x (torch.Tensor): Input tensor of shape `(B, C, H, W)`.

        Returns:
            torch.Tensor: Tensor of shape `(B, C, H, W)` after spatial-frequency fusion.
        """
        # 空间域处理
        spa_feat = self.spatial_block(x)
        # 频域处理
        freq_feat = self.freq_process(spa_feat)
        # 显存优化版融合
        fused = self.cross_attn(spa_feat, freq_feat)
        out = spa_feat + self.beta * self.fusion(fused)
        out = self.aspp(out)
        return out


class SMSBlock(nn.Module):
    """Stack of SSMTDA blocks followed by an after-effect refinement."""

    def __init__(
        self,
        i_layer=0,
        dim=90,
        window_size=8,
        depth=6,
        num_head=6,
        mlp_ratio=2,
        qkv_bias=True,
        qk_scale=None,
        weight_factor=0.1,
        memory_blocks=128,
        down_rank=16,
        drop_path: Union[Sequence[float], float] = 0.0,
        split_size=1,
    ):
        """Initialize the SMS block.

        Args:
            i_layer (int): Index of the current stage, used for selecting cross-attention window size.
            dim (int): Channel dimension for all submodules.
            window_size (int): Spatial window size for SSMTDA blocks.
            depth (int): Number of SSMTDA layers in the block.
            num_head (int): Number of attention heads per SSMTDA.
            mlp_ratio (float): Expansion ratio for SSMTDA MLP layers.
            qkv_bias (bool): Whether to use bias in QKV projections.
            qk_scale (float | None): Optional scaling for queries.
            weight_factor (float): Mixing factor for channel attention.
            memory_blocks (int): Number of dictionary atoms for channel attention.
            down_rank (int): Reduction factor inside channel attention.
            drop_path (float | Sequence[float]): Stochastic depth schedule.
            split_size (int): Number of spatial splits for window attention.
        """
        super(SMSBlock, self).__init__()
        # if i_layer == 0:
        #     stage_level = 'shallow'
        # elif i_layer == 1:
        #     stage_level = 'medium'
        # elif i_layer == 2:
        #     stage_level = 'deep'

        # 根据网络深度动态调整窗口大小
        self.window_sizes_WCA = [32, 32, 16]  # 浅层用大窗口，深层用小窗口

        # self.pre_aspp = MultiStageASPP(dim,dim,stage_level)
        if isinstance(drop_path, float):
            drop_path_seq: List[float] = [drop_path] * depth
        else:
            drop_path = cast(Sequence[float], drop_path)
            drop_path_seq = list(drop_path)
        if len(drop_path_seq) < depth:
            raise ValueError("drop_path length must be at least the block depth.")
        self.smsblock = nn.Sequential(
            *[
                SSMTDA(
                    dim=dim,
                    input_resolution=window_size,
                    num_heads=num_head,
                    memory_blocks=memory_blocks,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                    weight_factor=weight_factor,
                    down_rank=down_rank,
                    split_size=split_size,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path_seq[i],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                )
                for i in range(depth - 2)
            ]
        )
        self.aftereffect = SMS_AfterEffect(
            dim=dim,
            window_size=window_size,
            depth=depth,
            num_head=num_head,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            weight_factor=weight_factor,
            memory_blocks=memory_blocks,
            down_rank=down_rank,
            drop_path=drop_path_seq,
            split_size=split_size,
            window_size_WCA=self.window_sizes_WCA[i_layer],
        )
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        """Forward pass through the SMS block.

        Args:
            x (torch.Tensor): Input tensor of shape `(B, C, H, W)`.

        Returns:
            torch.Tensor: Tensor of shape `(B, C, H, W)` with spatial refinement applied.
        """
        # out = self.pre_aspp(x)
        out = self.smsblock(x)
        out = self.aftereffect(out)
        out = self.conv(out) + x
        return out

    def flops(self, shape):
        """Estimate FLOPs for the SMS block.

        Args:
            shape (tuple[int, int]): Spatial shape `(H, W)` of the feature map.

        Returns:
            int: Estimated floating-point operations.
        """
        flops = 0
        for blk in self.smsblock:
            sm_block = cast(SSMTDA, blk)
            flops += sm_block.flops(shape)
        return flops


class SERT(nn.Module):
    """Spectral enhancement transformer with optional patch-wise processing."""

    def __init__(
        self,
        inp_channels=31,
        dim=96,
        window_sizes=[16, 32, 32],
        depths=[6, 6, 6],
        num_heads=[6, 6, 6],
        split_sizes=[1, 2, 4],
        mlp_ratio=2,
        down_rank=8,
        memory_blocks=64,
        qkv_bias=True,
        qk_scale=None,
        bias=False,
        drop_path_rate=0.1,
        weight_factor=0.1,
    ):
        """Initialize the SERT architecture.

        Args:
            inp_channels (int): Number of input spectral channels.
            dim (int): Base embedding dimension.
            window_sizes (list[int]): Window sizes for each transformer stage.
            depths (list[int]): Number of blocks in each stage.
            num_heads (list[int]): Attention heads per stage.
            split_sizes (list[int]): Spatial split factors per stage.
            mlp_ratio (float): Expansion ratio for MLPs.
            down_rank (int): Reduction factor used in channel attention dictionaries.
            memory_blocks (int): Number of dictionary atoms for channel attention.
            qkv_bias (bool): Whether to include bias in QKV projections.
            qk_scale (float | None): Optional scaling factor applied to queries.
            bias (bool): Whether to include bias terms in output convolutions.
            drop_path_rate (float): Maximum stochastic depth rate.
            weight_factor (float): Mixing factor for channel attention contributions.
        """
        super(SERT, self).__init__()

        for idx, heads in enumerate(num_heads):
            if heads % 2 != 0:
                raise ValueError(f"num_heads[{idx}]={heads} must be even so it can be split across attention branches.")
            if dim % heads != 0:
                raise ValueError(
                    f"Embedding dimension {dim} must be divisible by num_heads[{idx}]={heads}. "
                    "Choose matching values (e.g., dim=60 for heads=6 or heads=4/8 for dim=64)."
                )

        self.conv_first = nn.Conv2d(inp_channels, dim, 3, 1, 1)
        self.num_layers = depths
        self.layers = nn.ModuleList()
        print(len(self.num_layers))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for i_layer in range(len(self.num_layers)):
            layer = SMSBlock(
                i_layer=i_layer,
                dim=dim,
                window_size=window_sizes[i_layer],
                depth=depths[i_layer],
                num_head=num_heads[i_layer],
                weight_factor=weight_factor,
                down_rank=down_rank,
                memory_blocks=memory_blocks,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                split_size=split_sizes[i_layer],
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
            )
            self.layers.append(layer)

        self.output = nn.Conv2d(int(dim), dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv_delasta = nn.Conv2d(dim, inp_channels, 3, 1, 1)

    def forward(self, inp_img):
        """Forward pass for the SERT model supporting 4D or 5D inputs.

        Args:
            inp_img (torch.Tensor): Input tensor of shape `(B, C, H, W)` or `(B, P, C, H, W)`.

        Returns:
            torch.Tensor: Output tensor matching the input leading dimensions with spatial shape `(H, W)`.
        """
        if inp_img.dim() not in (4, 5):
            raise ValueError("SERT expects input with 4 or 5 dimensions.")

        inp_img = inp_img.contiguous()
        layout = None
        if inp_img.dim() == 5:
            expected_channels = self.conv_first.in_channels
            if inp_img.shape[1] == expected_channels and inp_img.shape[2] != expected_channels:
                inp_img = inp_img.permute(0, 2, 1, 3, 4).contiguous()
                layout = "channels_first"
            else:
                layout = "patch_first"

            if inp_img.shape[2] != expected_channels:
                print(f"Expected Channels: {expected_channels}")
                print(f"Got Channels: {inp_img.shape[2]}")
                raise ValueError("Channel dimension must match the configured number of input channels.")

            b, patches, c, h_inp, w_inp = inp_img.shape
            leading_shape = (b, patches)
            flat_inp = inp_img.view(b * patches, c, h_inp, w_inp)
        else:
            b, c, h_inp, w_inp = inp_img.shape
            leading_shape = (b,)
            flat_inp = inp_img.view(b, c, h_inp, w_inp)

        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        padded_inp = F.pad(flat_inp, (0, pad_h, 0, pad_w), "reflect")
        f1 = self.conv_first(padded_inp)
        x = f1
        for layer in self.layers:
            x = layer(x)

        x = self.output(x + f1)
        x = self.conv_delasta(x) + padded_inp
        x = x[:, :, :h_inp, :w_inp].contiguous()
        output_shape = leading_shape + (x.shape[1], h_inp, w_inp)
        x = x.view(*output_shape)
        if layout == "channels_first":
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x

    def flops(self, shape):
        """Estimate FLOPs for the full SERT stack.

        Args:
            shape (tuple[int, int]): Spatial shape `(H, W)` of the feature map.

        Returns:
            int: Estimated floating-point operations across all layers.
        """
        flops = 0
        for i, layer in enumerate(self.layers):
            sms_layer = cast(SMSBlock, layer)
            flops += sms_layer.flops(shape)
        return flops
