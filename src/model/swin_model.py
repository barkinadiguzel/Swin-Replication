import torch
import torch.nn as nn
from .backbone_swin import SwinBackbone


class SwinModel(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7):
        super().__init__()

        self.backbone = SwinBackbone(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size
        )

        self.norm = nn.LayerNorm(self.backbone.num_features)
        self.head = nn.Linear(self.backbone.num_features, num_classes)


    def forward(self, x):
        x = self.backbone(x)

        x = self.norm(x)
        x = x.mean(dim=1)     

        x = self.head(x)     
        return x
