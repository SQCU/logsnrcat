# src/embedders.py - Patch and Fourier embedding/unembedding
"""
Embedding and unembedding modules for converting between pixel space and
token space. Handles both image patches and scalar fields (like logsnr).

Supports batched processing with optional attention layers for contextual
embedding. When attention is enabled, requires block_mask from context_manager.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from torch.nn.attention.flex_attention import BlockMask

from .blocks import MLPResBlock, EncoderBlock, init_linear, init_layer_norm


class FourierFeatures(nn.Module):
    """
    Projects scalar fields into high-dimensional Fourier features.

    Converts a scalar (e.g., logsnr value) into a feature vector using
    sinusoidal basis functions at exponentially spaced frequencies.

    All constants are registered as buffers to avoid Python float literals
    in traced code, which can cause inductor bounds analysis issues.
    """
    def __init__(self, fourier_dim=16, scale=1.0):
        super().__init__()
        self.fourier_dim = fourier_dim
        # Register constants as buffers - Python floats in forward() cause
        # inductor bounds analysis to use infinity during torch.compile
        self.register_buffer("scale", torch.tensor(scale))
        self.register_buffer("pi", torch.tensor(math.pi))
        # Fixed frequencies: 2^0, 2^1, ...
        self.register_buffer("freqs", 2.0 ** torch.arange(0, fourier_dim // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., 1] scalar tensor
        Returns:
            [..., fourier_dim] Fourier feature tensor
        """
        x = x * self.scale
        args = x * self.freqs * self.pi
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class FourierScaleDecoder(nn.Module):
    """
    Decodes predicted Fourier-space features back into a scalar (LogSNR/Lambda).
    Used by the Unembedder to predict the noise level.
    """
    def __init__(self, fourier_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.param_init()

    def param_init(self):
        init_linear(self.net[0])
        # Initialize output layer near zero for stability
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def forward(self, f):
        """
        Args:
            f: [..., Fourier_Dim] Fourier feature tensor
        Returns:
            [..., output_dim] decoded scalar(s)
        """
        return self.net(f)


class ContextualPatchEmbedder(nn.Module):
    """
    Embeds image patches with contextual (overlapping) windows and logsnr conditioning.

    Uses reflection padding and unfold to create overlapping context windows,
    then projects each window along with Fourier-encoded logsnr to embedding space.

    Supports both single-image and batched processing:
    - Single: x=[C, H, W], logsnr_map=[1, H, W] -> z=[L, D], grid_shape
    - Batched: x=[B, C, H, W], logsnr_map=[B, 1, H, W] -> z=[B, L, D], grid_shape

    Args:
        input_channels: Number of image channels (default: 3 for RGB)
        fourier_dim: Dimension of Fourier features for logsnr encoding
        embed_dim: Output embedding dimension
        context_size: Size of context window (patches overlap by context_size - stride)
        stride: Stride between patch centers (determines output grid size)
        mlp_depth: Number of MLP residual blocks
        n_attn_layers: Number of attention layers (0 = MLP only)
        n_heads: Number of attention heads (if n_attn_layers > 0)
        n_kv_heads: Number of KV heads for GQA (defaults to n_heads)
    """
    def __init__(
        self,
        input_channels: int = 3,
        fourier_dim: int = 16,
        embed_dim: int = 256,
        context_size: int = 4,
        stride: int = 2,
        mlp_depth: int = 1,
        n_attn_layers: int = 0,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None
    ):
        super().__init__()
        self.context_size = context_size
        self.stride = stride
        self.padding = (context_size - stride) // 2
        self.n_attn_layers = n_attn_layers

        self.fourier_enc = FourierFeatures(fourier_dim=fourier_dim)
        self.input_dim = (context_size ** 2) * input_channels + fourier_dim
        self.input_proj = nn.Linear(self.input_dim, embed_dim)
        self.res_blocks = nn.Sequential(*[MLPResBlock(embed_dim) for _ in range(max(0, mlp_depth - 1))])

        # Optional attention layers
        if n_attn_layers > 0:
            self.attn_layers = nn.ModuleList([
                EncoderBlock(dim=embed_dim, n_heads=n_heads, n_kv_heads=n_kv_heads)
                for _ in range(n_attn_layers)
            ])
        else:
            self.attn_layers = None

        self.param_init()

    def param_init(self):
        for block in self.res_blocks:
            block.param_init()
        init_linear(self.input_proj)
        if self.attn_layers is not None:
            for layer in self.attn_layers:
                layer.param_init()

    def _pad_and_patch(self, x: torch.Tensor) -> torch.Tensor:
        """Apply reflection padding and extract patches via unfold.

        Args:
            x: [C, H, W] or [B, C, H, W] input tensor
        Returns:
            patches: [C, GH, GW, P, P] or [B, C, GH, GW, P, P]
        """
        is_batched = x.dim() == 4

        # 1. Standard Reflection Pad (Context Window)
        x_pad = F.pad(x, (self.padding,)*4, mode='reflect')

        # 2. Dynamic Safety Pad (Stride Alignment)
        if is_batched:
            _, _, h, w = x_pad.shape
        else:
            _, h, w = x_pad.shape

        h_rem = (h - self.context_size) % self.stride
        w_rem = (w - self.context_size) % self.stride

        pad_bottom = (self.stride - h_rem) % self.stride
        pad_right = (self.stride - w_rem) % self.stride

        if pad_bottom > 0 or pad_right > 0:
            x_pad = F.pad(x_pad, (0, pad_right, 0, pad_bottom), mode='replicate')

        # Unfold: extract patches
        if is_batched:
            # [B, C, H, W] -> [B, C, GH, GW, P, P]
            patches = x_pad.unfold(2, self.context_size, self.stride).unfold(3, self.context_size, self.stride)
        else:
            # [C, H, W] -> [C, GH, GW, P, P]
            patches = x_pad.unfold(1, self.context_size, self.stride).unfold(2, self.context_size, self.stride)

        return patches

    def forward(
        self,
        x: torch.Tensor,
        logsnr_map: torch.Tensor,
        block_masks: Optional[List[BlockMask]] = None
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Embed an image with per-patch logsnr conditioning.

        Args:
            x: [C, H, W] or [B, C, H, W] image tensor
            logsnr_map: [1, H, W] or [B, 1, H, W] spatial logsnr field
            block_masks: Optional list of BlockMasks, one per attention layer.
                         Required if n_attn_layers > 0. For alternating patterns
                         like gemma_bigbird, each layer gets its appropriate mask.

        Returns:
            z: [L, D] or [B, L, D] patch embeddings where L = GH * GW
            grid_shape: (GH, GW) grid dimensions
        """
        is_batched = x.dim() == 4

        patches_img = self._pad_and_patch(x)

        if is_batched:
            B, C, GH, GW, P1, P2 = patches_img.shape
            # [B, C, GH, GW, P, P] -> [B, GH*GW, C*P*P]
            patches_img = patches_img.permute(0, 2, 3, 1, 4, 5).reshape(B, GH * GW, -1)

            patches_logsnr = self._pad_and_patch(logsnr_map)
            # [B, 1, GH, GW, P, P] -> [B, GH*GW, 1] (mean over patch)
            patches_logsnr = patches_logsnr.permute(0, 2, 3, 1, 4, 5).reshape(B, GH * GW, -1).mean(dim=-1, keepdim=True)
        else:
            C, GH, GW, P1, P2 = patches_img.shape
            # [C, GH, GW, P, P] -> [GH*GW, C*P*P]
            patches_img = patches_img.permute(1, 2, 0, 3, 4).reshape(GH * GW, -1)

            patches_logsnr = self._pad_and_patch(logsnr_map)
            # [1, GH, GW, P, P] -> [GH*GW, 1] (mean over patch)
            patches_logsnr = patches_logsnr.permute(1, 2, 0, 3, 4).reshape(GH * GW, -1).mean(dim=-1, keepdim=True)

        logsnr_features = self.fourier_enc(patches_logsnr)
        raw_input = torch.cat([patches_img, logsnr_features], dim=-1)
        h = self.input_proj(raw_input)
        z = self.res_blocks(h)

        # Apply attention layers if configured
        if self.attn_layers is not None:
            if not is_batched:
                # Add batch dim for attention: [L, D] -> [1, L, D]
                z = z.unsqueeze(0)

            for i, layer in enumerate(self.attn_layers):
                mask = block_masks[i] if block_masks else None
                z = layer(z, block_mask=mask)

            if not is_batched:
                # Remove batch dim: [1, L, D] -> [L, D]
                z = z.squeeze(0)

        return z, (GH, GW)


class ContextualPatchUnembedder(nn.Module):
    """
    Converts patch embeddings back to pixel space with logsnr prediction.

    Takes embeddings and grid shape, outputs reconstructed image plus
    predicted logsnr channel.

    Supports both single-image and batched processing:
    - Single: z=[L, D], shape=(GH, GW) -> [C+1, H, W]
    - Batched: z=[B, L, D], shape=(GH, GW) -> [B, C+1, H, W]

    Args:
        output_channels: Number of output image channels (default: 3 for RGB)
        fourier_dim: Dimension of Fourier features for logsnr decoding
        embed_dim: Input embedding dimension
        patch_size: Size of output patches (should match embedder stride)
        mlp_depth: Number of MLP residual blocks
        n_attn_layers: Number of attention layers (0 = MLP only)
        n_heads: Number of attention heads (if n_attn_layers > 0)
        n_kv_heads: Number of KV heads for GQA (defaults to n_heads)
    """
    def __init__(
        self,
        output_channels: int = 3,
        fourier_dim: int = 16,
        embed_dim: int = 256,
        patch_size: int = 2,
        mlp_depth: int = 1,
        n_attn_layers: int = 0,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None
    ):
        super().__init__()
        self.patch_size = patch_size
        self.output_channels = output_channels
        self.n_attn_layers = n_attn_layers
        self.raster_flat_dim = output_channels * (patch_size ** 2)

        # Optional attention layers (applied first)
        if n_attn_layers > 0:
            self.attn_layers = nn.ModuleList([
                EncoderBlock(dim=embed_dim, n_heads=n_heads, n_kv_heads=n_kv_heads)
                for _ in range(n_attn_layers)
            ])
        else:
            self.attn_layers = None

        self.res_blocks = nn.Sequential(*[MLPResBlock(embed_dim) for _ in range(max(0, mlp_depth - 1))])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim, elementwise_affine=False),
            nn.Linear(embed_dim, self.raster_flat_dim + fourier_dim)
        )
        self.logsnr_decoder = FourierScaleDecoder(fourier_dim, hidden_dim=embed_dim, output_dim=1)
        self.param_init()

    def param_init(self):
        if self.attn_layers is not None:
            for layer in self.attn_layers:
                layer.param_init()
        for block in self.res_blocks:
            block.param_init()
        init_layer_norm(self.output_proj[0])
        init_linear(self.output_proj[1])
        self.logsnr_decoder.param_init()

    def forward(
        self,
        z: torch.Tensor,
        shape: Tuple[int, int],
        block_masks: Optional[List[BlockMask]] = None
    ) -> torch.Tensor:
        """
        Convert embeddings back to pixel space.

        Args:
            z: [L, D] or [B, L, D] patch embeddings
            shape: (GH, GW) grid dimensions
            block_masks: Optional list of BlockMasks, one per attention layer.
                         Required if n_attn_layers > 0. For alternating patterns
                         like gemma_bigbird, each layer gets its appropriate mask.

        Returns:
            [C+1, H, W] or [B, C+1, H, W] reconstructed image with logsnr channel appended
        """
        is_batched = z.dim() == 3
        P = self.patch_size

        if len(shape) == 2:
            GH, GW = shape
        elif len(shape) == 1:
            GH, GW = 1, shape[0]
        else:
            GH, GW = 1, z.shape[-2] if is_batched else z.shape[0]

        # Apply attention layers first if configured
        if self.attn_layers is not None:
            if not is_batched:
                z = z.unsqueeze(0)

            for i, layer in enumerate(self.attn_layers):
                mask = block_masks[i] if block_masks else None
                z = layer(z, block_mask=mask)

            if not is_batched:
                z = z.squeeze(0)

        # MLP processing
        z = self.res_blocks(z)
        flat = self.output_proj(z)

        if is_batched:
            B = z.shape[0]
            raster_part = flat[:, :, :self.raster_flat_dim]
            fourier_part = flat[:, :, self.raster_flat_dim:]

            # [B, L, C*P*P] -> [B, GH, GW, C, P, P] -> [B, C, GH*P, GW*P]
            patches = raster_part.reshape(B, GH, GW, self.output_channels, P, P)
            patches = patches.permute(0, 3, 1, 4, 2, 5)
            rasters = patches.reshape(B, self.output_channels, GH * P, GW * P)

            logsnr_pred = self.logsnr_decoder(fourier_part)
            logsnr_grid = logsnr_pred.view(B, 1, GH, GW)
            logsnr_pixel = F.interpolate(logsnr_grid, scale_factor=P, mode='nearest')

            return torch.cat([rasters, logsnr_pixel], dim=1)
        else:
            raster_part = flat[:, :self.raster_flat_dim]
            fourier_part = flat[:, self.raster_flat_dim:]

            patches = raster_part.reshape(GH, GW, self.output_channels, P, P)
            patches = patches.permute(2, 0, 3, 1, 4)
            rasters = patches.reshape(self.output_channels, GH * P, GW * P)

            logsnr_pred = self.logsnr_decoder(fourier_part)
            logsnr_grid = logsnr_pred.view(GH, GW).unsqueeze(0)
            logsnr_pixel = F.interpolate(logsnr_grid.unsqueeze(0), scale_factor=P, mode='nearest').squeeze(0)

            return torch.cat([rasters, logsnr_pixel], dim=0)


# =============================================================================
# Latent Diffusion Batched Embedding Preparation
# =============================================================================

def prepare_latent_batch(
    group: list,
    grid_shape: Tuple[int, int],
    sparse_ae,
    patch_embedder,
    topo_config: dict,
    device: torch.device,
) -> dict:
    """
    Prepare batched embeddings for latent diffusion training.

    Creates a tidy package of data structures that can be used for:
    - Direct AE reconstruction (codes → decode → recon loss)
    - Latent diffusion (noisy codes → LDTformer → v_pred → loss)

    This decouples embedding preparation from model application, allowing
    the same embeddings to support both autoencoding and denoising paths.

    Args:
        group: List of (block, img, logsnr) tuples from group_blocks_by_grid()
        grid_shape: (GH, GW) patch grid dimensions
        sparse_ae: The sparse autoencoder model
        patch_embedder: Embedder with latent_code_proj, logsnr_proj, _get_masks
        topo_config: Dict with level_lambda, level_scale, vertical_free, window_size
        device: Target device

    Returns:
        Dict with prepared tensors:
            # Original data
            'imgs': [B, C, H, W] original images
            'blocks': Original ContextBlocks for source tracking

            # Encoded representations
            'pre_quant_flat': [B, N*L, code_dim] clean codes (pre-quantization)
            'logsnr_flat': [B, N*L, 1] logsnr per token

            # Diffusion components
            'noisy_codes': [B, N*L, code_dim] noised codes
            'target_v': [B, N*L, code_dim] target v-field
            'alpha': [B, N*L, 1] signal coefficient
            'sigma': [B, N*L, 1] noise coefficient

            # Model input (projected to model_dim)
            'h_input': [B, N*L, model_dim] input for LDTformer
            'topo_embeds': [N*L, topo_dim] topology embeddings
            'latent_mask': BlockMask for attention

            # Reconstruction support
            'decoder_masks': For sparse_ae.quantize_and_decode()

            # Geometry
            'grid_shape': (GH, GW)
            'n_patches': int
            'n_levels': int
            'batch_size': int
    """
    from .context_manager import (
        render_latent_topology_embeddings,
        get_cached_latent_mask,
    )

    H_grid, W_grid = grid_shape
    n_levels = sparse_ae.n_levels
    code_dim = sparse_ae.code_dim
    n_patches = H_grid * W_grid
    total_tokens = n_patches * n_levels
    p = sparse_ae.patch_size

    # Extract tensors from group
    blocks = [g[0] for g in group]
    imgs = torch.stack([g[1] for g in group], dim=0)  # [B, C, H, W]
    logsnrs = torch.stack([g[2] for g in group], dim=0)  # [B, 1, H, W]
    batch_size = imgs.shape[0]

    # Get encoder/decoder masks
    encoder_masks, decoder_masks = patch_embedder._get_masks(grid_shape, device)

    # Build latent diffusion topology (4D: highway, spatial_x, spatial_y, level)
    topo_embeds, level_ids, patch_ids = render_latent_topology_embeddings(
        n_patches=n_patches,
        n_levels=n_levels,
        grid_shape=grid_shape,
        device=device,
        level_scale=topo_config['level_scale'],
    )

    # Get cached level-aware attention mask
    latent_mask = get_cached_latent_mask(
        n_patches=n_patches,
        n_levels=n_levels,
        grid_shape=grid_shape,
        window_size=topo_config['window_size'],
        level_lambda=topo_config['level_lambda'],
        vertical_free=topo_config['vertical_free'],
        mode='local',
        device=device,
    )

    # === ENCODE: Batched encoding with pre_quant ===
    codes_list, level_logsnrs, prequant_list = sparse_ae.encode_with_prequant(
        imgs, logsnrs,
        grid_shape=grid_shape,
        encoder_masks=encoder_masks,
        decoder_masks=decoder_masks
    )

    # Stack and flatten pre_quants across levels: [B, n_patches, n_levels, code_dim]
    pre_quant_stacked = torch.stack(prequant_list, dim=2)  # [B, N, L, D]
    pre_quant_flat = pre_quant_stacked.view(batch_size, total_tokens, code_dim)  # [B, N*L, D]

    # Expand logsnr to per-token (each level gets same logsnr per patch)
    logsnr_pooled = F.avg_pool2d(logsnrs, kernel_size=p, stride=p)
    logsnr_patches = logsnr_pooled.flatten(2).transpose(1, 2)  # [B, N, 1]
    logsnr_flat = logsnr_patches.repeat(1, n_levels, 1)  # [B, N*L, 1]

    # === DIFFUSION: Add noise to codes ===
    noise = torch.randn_like(pre_quant_flat)

    # Compute alpha/sigma from logsnr
    alpha_sq = torch.sigmoid(logsnr_flat)  # [B, N*L, 1]
    sigma_sq = torch.sigmoid(-logsnr_flat)
    alpha = torch.sqrt(alpha_sq)
    sigma = torch.sqrt(sigma_sq)

    # Noisy codes: z_t = alpha * x_0 + sigma * noise
    noisy_codes = alpha * pre_quant_flat + sigma * noise

    # Target v-field: v = alpha * noise - sigma * x_0
    target_v = alpha * noise - sigma * pre_quant_flat

    # === PROJECT: Prepare model input ===
    h_input = patch_embedder.latent_code_proj(noisy_codes)  # [B, N*L, model_dim]
    logsnr_features = patch_embedder.logsnr_proj(logsnr_flat)  # [B, N*L, model_dim]
    h_input = h_input + logsnr_features

    return {
        # Original data
        'imgs': imgs,
        'blocks': blocks,

        # Encoded representations
        'pre_quant_flat': pre_quant_flat,
        'logsnr_flat': logsnr_flat,

        # Diffusion components
        'noisy_codes': noisy_codes,
        'target_v': target_v,
        'alpha': alpha,
        'sigma': sigma,

        # Model input
        'h_input': h_input,
        'topo_embeds': topo_embeds,
        'latent_mask': latent_mask,

        # Reconstruction support
        'encoder_masks': encoder_masks,
        'decoder_masks': decoder_masks,

        # Geometry
        'grid_shape': grid_shape,
        'n_patches': n_patches,
        'n_levels': n_levels,
        'batch_size': batch_size,
    }
