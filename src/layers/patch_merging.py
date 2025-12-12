import torch
import torch.nn as nn

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.H, self.W = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  
        x1 = x[:, 0::2, 1::2, :]   
        x2 = x[:, 1::2, 0::2, :]   
        x3 = x[:, 1::2, 1::2, :]   

        x = torch.cat([x0, x1, x2, x3], dim=-1)  
        x = x.view(B, -1, 4 * C)                 
        x = self.norm(x)
        x = self.reduction(x)                    
        return x
