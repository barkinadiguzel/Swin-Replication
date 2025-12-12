import torch
import torch.nn as nn

from .window_attention import (
    WindowAttention,
    window_partition,
    window_reverse,
    compute_attn_mask
)
from .mlp_block import MLPBlock


class SwinBlock(nn.Module):
    """
    Minimal and correct Swin Transformer block for replication:
    - LayerNorm → (W-MSA or SW-MSA) → Residual
    - LayerNorm → MLP → Residual
    """

    def __init__(self, dim, input_resolution, num_heads,
                 window_size=7, shift_size=0, mlp_ratio=4.0):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution  
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(dim, mlp_ratio)

        H, W = input_resolution
        if shift_size > 0:
            mask = compute_attn_mask(H, W, window_size, shift_size)
            self.register_buffer("attn_mask", mask)
        else:
            self.attn_mask = None

    def forward(self, x):

        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input not matching declared resolution."

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x,
                                   shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
        else:
            shifted_x = x

        windows = window_partition(shifted_x, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, C)

        if self.attn_mask is not None:
            attn_windows = self.attn(windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(windows)

        attn_windows = attn_windows.view(-1,
                                         self.window_size,
                                         self.window_size,
                                         C)
        shifted_x = window_reverse(attn_windows,
                                   self.window_size,
                                   H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x,
                           shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + x
        shortcut2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut2 + x

        return x
