import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C_in, H, W)
        x = self.proj(x)                       
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)       # (B, H*W, embed_dim)
        x = self.norm(x)                       
        return x, (H, W)
