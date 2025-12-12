class SwinConfig:
    # --- Patch Embedding ---
    patch_size = 4
    in_channels = 3
    embed_dim = 96   # Stage 1 dim

    # --- Window Attention ---
    window_size = 7
    num_heads_stage1 = 3   

    # --- Model Depth ---
    # Swin-Tiny: [2, 2, 6, 2]
    depths = [2, 2, 6, 2]

    # --- Channel Multipliers ---
    dims = [
        96,
        96 * 2,
        96 * 4,
        96 * 8
    ]

    # --- MLP ---
    mlp_ratio = 4.0


def get_config():
    return SwinConfig()
