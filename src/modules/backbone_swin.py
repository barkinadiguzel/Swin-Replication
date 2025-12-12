import torch
import torch.nn as nn

from .patch_embed import PatchEmbed
from .patch_merging import PatchMerging
from .swin_block import SwinBlock


class SwinBackbone(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        patches_resolution = self.patch_embed.patches_resolution
        H, W = patches_resolution

        self.layers = nn.ModuleList()
        dim = embed_dim

        for stage in range(len(depths)):
            stage_blocks = []

            for block_idx in range(depths[stage]):
                shift = 0 if (block_idx % 2 == 0) else window_size // 2

                stage_blocks.append(
                    SwinBlock(
                        dim=dim,
                        input_resolution=(H, W),
                        num_heads=num_heads[stage],
                        window_size=window_size,
                        shift_size=shift,
                    )
                )

            self.layers.append(nn.Sequential(*stage_blocks))

            if stage < len(depths) - 1:
                self.layers.append(PatchMerging((H, W), dim))
                H, W = H // 2, W // 2
                dim = dim * 2

        self.num_features = dim

    def forward(self, x):
        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x)

        return x
