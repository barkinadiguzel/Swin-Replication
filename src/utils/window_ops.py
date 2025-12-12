import torch

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0,1,3,2,4,5).contiguous()
    windows = x.view(-1, window_size, window_size, C)  
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0,1,3,2,4,5).contiguous()
    x = x.view(B, H, W, -1)
    return x

def compute_mask(H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, H, W, 1), device=device)  
    cnt = 0
    for h in range(0, H, window_size):
        for w in range(0, W, window_size):
            img_mask[:, h:h+window_size, w:w+window_size, :] = cnt
            cnt += 1
    # if shifted, apply cyclic shift
    if shift_size > 0:
        img_mask = torch.roll(img_mask, shifts=(-shift_size, -shift_size), dims=(1,2))
    mask_windows = window_partition(img_mask, window_size)  # (num_windows, ws, ws, 1)
    mask_windows = mask_windows.view(-1, window_size*window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask  
