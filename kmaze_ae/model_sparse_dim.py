import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List

# Import from main codebase - use shared implementations
from src.embedders import FourierFeatures
from src.blocks import EncoderBlock, TransformerEncoder, TransformerDecoder, _uses_registers

# =============================================================================
# Sparse AE using shared components from main codebase
# =============================================================================
#
# Uses:
# - FourierFeatures from src.embedders (with proper buffer registration)
# - TransformerEncoder/TransformerDecoder from src.blocks (with proper mask building)
# - EncoderBlock which supports GQA, gated residuals, SwiGLU/MoE
#
# This avoids duplicate implementations that can have Python literal issues
# with torch.compile's inductor bounds analysis.
# =============================================================================


# FourierFeatures imported from src.embedders
# TransformerEncoder/TransformerDecoder imported from src.blocks


class BinaryFSQ(nn.Module):
    """Binary FSQ with STE."""
    def __init__(self):
        super().__init__()
        # Register threshold as buffer to avoid Python float in traced code
        self.register_buffer("threshold", torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        soft = torch.sigmoid(x)
        hard = (soft > self.threshold).float()
        return hard - soft.detach() + soft


class ThreeBitFSQ(nn.Module):
    """3-bit FSQ (8 levels: 0-7) with STE."""
    def __init__(self):
        super().__init__()
        # Register scale as buffer to avoid Python float in traced code
        self.register_buffer("scale", torch.tensor(7.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sigmoid to [0, 1], scale to [0, 7]
        soft = torch.sigmoid(x) * self.scale
        # Round to nearest integer
        hard = torch.round(soft)
        # STE: forward uses hard, backward uses soft gradient
        return hard - soft.detach() + soft


class PerDimSparsity(nn.Module):
    """
    Per-dimension sparsity with learned gating.

    For each patch, predicts which dims to keep.
    Top-k selection per patch with STE.
    """
    def __init__(self, code_dim: int = 128, k_per_patch: int = 4):
        super().__init__()
        self.code_dim = code_dim
        self.k = k_per_patch  # keep 6 of 128 = ~5% = 95% sparsity

        # Gate predictor: from latent to gate logits
        self.gate_proj = nn.Linear(code_dim, code_dim)

    def forward(self, latent: torch.Tensor, k_override: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent: (B, N, code_dim) - pre-quantization latent
            k_override: optional override for k (for k-annealing during training)
        Returns:
            sparse_latent: (B, N, code_dim) with ~95% zeros
            gate_weights: (B, N, code_dim) soft weights for logging
        """
        B, N, D = latent.shape
        k = k_override if k_override is not None else self.k

        # Predict gate from latent
        gate_logits = self.gate_proj(latent)  # (B, N, D)
        gate_weights = torch.sigmoid(gate_logits)  # (B, N, D)

        # Top-k per patch
        _, topk_idx = gate_weights.topk(k, dim=-1)  # (B, N, k)

        # Create hard mask
        hard_mask = torch.zeros_like(gate_weights)
        hard_mask.scatter_(-1, topk_idx, 1.0)  # (B, N, D)

        # STE: forward uses hard mask, backward through soft weights
        soft_mask = gate_weights * hard_mask
        ste_mask = hard_mask - soft_mask.detach() + soft_mask

        # Apply mask
        sparse_latent = latent * ste_mask

        return sparse_latent, gate_weights


class SparseLevelEncoder(nn.Module):
    def __init__(self, patch_dim: int = 768, hidden_dim: int = 256, code_dim: int = 128,
                 k_per_patch: int = 4, fourier_dim: int = 16, n_layers: int = 4,
                 attn_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.fourier_dim = fourier_dim
        self.logsnr_fourier = FourierFeatures(fourier_dim, scale=0.5)
        # Input: patch_dim + fourier_dim (logsnr features)
        self.input_proj = nn.Linear(patch_dim + fourier_dim, hidden_dim)
        self.transformer = TransformerEncoder(hidden_dim, n_layers=n_layers, attn_config=attn_config)
        self.code_proj = nn.Linear(hidden_dim, code_dim)
        self.sparsity = PerDimSparsity(code_dim, k_per_patch)
        self.fsq = ThreeBitFSQ()  # 3-bit (8 levels)

    def forward(self, x: torch.Tensor, logsnr_patches: torch.Tensor,
                grid_shape: Optional[Tuple[int, int]] = None,
                block_masks: Optional[list] = None,
                k_override: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, patch_dim) - patch features
            logsnr_patches: (B, N, 1) - per-patch logsnr values
            grid_shape: (H, W) spatial grid dimensions for attention masks
            block_masks: Optional pre-built masks to avoid inductor issues
            k_override: optional override for k (for k-annealing during training)
        Returns:
            sparse_codes: (B, N, code_dim) with ~95% zeros
            gate_weights: (B, N, code_dim) soft weights for logging
            pre_quant: (B, N, code_dim) pre-quantization values
        """
        # Encode logsnr with Fourier features
        logsnr_feat = self.logsnr_fourier(logsnr_patches)  # (B, N, fourier_dim)

        # Concatenate patch features with logsnr features
        h = torch.cat([x, logsnr_feat], dim=-1)  # (B, N, patch_dim + fourier_dim)
        h = self.input_proj(h)
        h = self.transformer(h, grid_shape=grid_shape, block_masks=block_masks)
        pre_quant = self.code_proj(h)

        # Quantize first
        codes = self.fsq(pre_quant)

        # Then apply sparsity (mask after FSQ so zeros stay zero)
        sparse_codes, gate_weights = self.sparsity(codes, k_override=k_override)

        return sparse_codes, gate_weights, pre_quant


class SparseLevelDecoder(nn.Module):
    def __init__(self, code_dim: int = 128, hidden_dim: int = 256, patch_dim: int = 768,
                 n_layers: int = 4, attn_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.input_proj = nn.Linear(code_dim, hidden_dim)
        self.transformer = TransformerDecoder(hidden_dim, n_layers=n_layers, attn_config=attn_config)
        self.output_proj = nn.Linear(hidden_dim, patch_dim)
        # LogSNR prediction head - predict per-patch logsnr
        self.logsnr_head = nn.Linear(hidden_dim, 1)

    def forward(self, codes: torch.Tensor,
                grid_shape: Optional[Tuple[int, int]] = None,
                block_masks: Optional[list] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            codes: (B, N, code_dim) - sparse quantized codes
            grid_shape: (H, W) spatial grid dimensions for attention masks
            block_masks: Optional pre-built masks to avoid inductor issues
        Returns:
            patches: (B, N, patch_dim) - reconstructed patch features
            logsnr_pred: (B, N, 1) - predicted logsnr per patch
        """
        h = self.input_proj(codes)
        h = self.transformer(h, grid_shape=grid_shape, block_masks=block_masks)
        patches = self.output_proj(h)
        logsnr_pred = self.logsnr_head(h)
        return patches, logsnr_pred


class SparsePerDimFSQAutoencoder(nn.Module):
    """
    Hierarchical Binary FSQ with per-dim sparsity and per-level logsnr handling.

    Each level encodes at a different effective SNR, allowing spatially-varying,
    level-varying logsnr fields for latent denoising objectives.

    Attention modes:
        - 'full': All layers use full O(n²) attention
        - 'sliding': All layers use sliding window attention
        - 'bigbird': Local + global register tokens (O(n * (window + global)))
        - 'gemma': 3 local + 1 global per 4-layer block (like Gemma)
        - 'gemma_bigbird': 2:2 layout (2 sliding, 2 bigbird, repeat) with registers
    """
    def __init__(
        self,
        n_levels: int = 4,
        patch_size: int = 16,
        image_size: int = 256,
        hidden_dim: int = 256,
        code_dim: int = 128,
        k_per_patch: int = 6,
        residual_scale: float = 2.0,
        fourier_dim: int = 16,
        n_layers: int = 4,
        attn_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.n_levels = n_levels
        self.patch_size = patch_size
        self.image_size = image_size
        self.n_patches = (image_size // patch_size) ** 2  # 256
        self.patch_dim = patch_size * patch_size * 3  # 768
        # Register constants as buffers to avoid Python float literals in traced code
        self.register_buffer("residual_scale", torch.tensor(residual_scale))
        self.register_buffer("one", torch.tensor(1.0))
        self.code_dim = code_dim
        self.k_per_patch = k_per_patch
        self.fourier_dim = fourier_dim
        self.n_layers = n_layers

        # Default attention config if not provided
        if attn_config is None:
            attn_config = {
                'mode': 'full',
                'window_size': 4,
                'global_layer_interval': 4,
                'n_query_heads': 8,
                'n_kv_heads': 2,
                'n_global_tokens': 4
            }

        self.attn_mode = attn_config['mode']

        self.encoders = nn.ModuleList([
            SparseLevelEncoder(self.patch_dim, hidden_dim, code_dim, k_per_patch,
                              fourier_dim, n_layers, attn_config)
            for _ in range(n_levels)
        ])
        self.decoders = nn.ModuleList([
            SparseLevelDecoder(code_dim, hidden_dim, self.patch_dim, n_layers, attn_config)
            for _ in range(n_levels)
        ])

        # Log attention configuration
        uses_regs = _uses_registers(attn_config['mode'])
        random_min_k = attn_config.get('random_min_k', 0)
        random_min_p = attn_config.get('random_min_p', 0.0)
        bigbird_layout = attn_config.get('bigbird_layout', [2, 2])
        has_random = random_min_k > 0 or random_min_p > 0
        print(f"[SparseAE] Attention: {attn_config['mode']}"
              f"{f' ({bigbird_layout[0]}L+{bigbird_layout[1]}BB)' if 'bigbird' in attn_config['mode'] else ''}, "
              f"window={attn_config['window_size']}, "
              f"registers={attn_config.get('n_global_tokens', 4) if uses_regs else 0}"
              f"{f', random=max({random_min_k}, {random_min_p:.0%}*N)' if has_random else ''}")

        # Per-level logsnr estimators: predict logsnr for residual levels based on level-0 logsnr
        # Level 0 uses the input logsnr directly, levels 1+ predict their own
        self.level_logsnr_estimators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1)
            ) for _ in range(n_levels - 1)  # Levels 1 to n_levels-1
        ])

    def build_masks(self, grid_shape: Tuple[int, int], device: torch.device) -> Tuple[list, list]:
        """
        Build attention masks for encoder and decoder transformers.

        Call this OUTSIDE torch.compile to avoid inductor bounds analysis issues.
        All encoders share the same mask, and all decoders share the same mask.

        Args:
            grid_shape: (H, W) patch grid dimensions
            device: Target device
        Returns:
            (encoder_masks, decoder_masks) - lists of BlockMask per layer
        """
        encoder_masks = self.encoders[0].transformer.build_masks(grid_shape, device)
        decoder_masks = self.decoders[0].transformer.build_masks(grid_shape, device)
        return encoder_masks, decoder_masks

    def patchify(self, images: torch.Tensor, grid_shape: Tuple[int, int]) -> torch.Tensor:
        """Patchify images using concrete grid dimensions to avoid symbolic shape issues."""
        B = images.shape[0]
        p = self.patch_size
        n_patches_h, n_patches_w = grid_shape  # Use concrete ints, not symbolic H//p
        n_patches = n_patches_h * n_patches_w
        patches = images.view(B, 3, n_patches_h, p, n_patches_w, p)
        patches = patches.permute(0, 2, 4, 3, 5, 1).contiguous()
        return patches.view(B, n_patches, self.patch_dim)

    def patchify_logsnr(self, logsnr_map: torch.Tensor, grid_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Convert spatial logsnr map to per-patch logsnr values.

        Args:
            logsnr_map: (B, 1, H, W) or (B, H, W) spatial logsnr field
            grid_shape: (GH, GW) concrete grid dimensions
        Returns:
            logsnr_patches: (B, N, 1) per-patch mean logsnr
        """
        if logsnr_map.dim() == 3:
            logsnr_map = logsnr_map.unsqueeze(1)  # (B, 1, H, W)

        B = logsnr_map.shape[0]
        p = self.patch_size
        n_patches_h, n_patches_w = grid_shape
        # Pool each patch region to get per-patch logsnr
        logsnr_patches = F.avg_pool2d(logsnr_map, kernel_size=p, stride=p)  # (B, 1, GH, GW)
        return logsnr_patches.view(B, n_patches_h * n_patches_w, 1)  # (B, N, 1)

    def unpatchify(self, patches: torch.Tensor, grid_shape: Tuple[int, int]) -> torch.Tensor:
        """Unpatchify using concrete grid dimensions to avoid symbolic shape issues."""
        B = patches.shape[0]
        p = self.patch_size
        h, w = grid_shape
        patches = patches.view(B, h, w, p, p, 3)
        patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
        return patches.view(B, 3, h * p, w * p)

    def unpatchify_logsnr(self, logsnr_patches: torch.Tensor,
                          grid_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Convert per-patch logsnr back to spatial map.

        Args:
            logsnr_patches: (B, N, 1) per-patch logsnr
            grid_shape: (H, W) grid dimensions - required to avoid math.sqrt on symbolic shapes
        Returns:
            logsnr_map: (B, 1, H, W) spatial logsnr field (upsampled)
        """
        B = logsnr_patches.shape[0]
        p = self.patch_size
        h, w = grid_shape
        logsnr_grid = logsnr_patches.view(B, h, w, 1).permute(0, 3, 1, 2)  # (B, 1, h, w)
        # Upsample to full resolution
        logsnr_map = F.interpolate(logsnr_grid, scale_factor=p, mode='nearest')
        return logsnr_map

    def forward(self, images: torch.Tensor, logsnr_map: Optional[torch.Tensor] = None,
                encoder_masks: Optional[list] = None, decoder_masks: Optional[list] = None,
                grid_shape: Optional[Tuple[int, int]] = None,
                k_override: Optional[int] = None) -> dict:
        """
        Forward pass with per-level logsnr handling.

        Args:
            images: (B, C, H, W) input images
            logsnr_map: (B, 1, H, W) spatial logsnr field. If None, uses zeros.
            encoder_masks: Optional pre-built encoder masks (built outside compile)
            decoder_masks: Optional pre-built decoder masks (built outside compile)
            grid_shape: Optional (GH, GW) grid dimensions. If None, computed from images.
                       Pass this from OUTSIDE compile to avoid symbolic shape issues.
            k_override: Optional override for k sparsity (for k-annealing during training)
        Returns:
            dict with recon, level_recons, codes, gate_weights, sparsity,
            logsnr_preds (per-level), level_logsnrs (input logsnr per level)
        """
        B = images.shape[0]
        p = self.patch_size
        if grid_shape is None:
            # Fallback - will use symbolic shapes (avoid if possible)
            H, W = images.shape[2], images.shape[3]
            grid_shape = (H // p, W // p)

        # Use concrete grid_shape for all shape-dependent operations
        n_patches = grid_shape[0] * grid_shape[1]

        patches = self.patchify(images, grid_shape)

        # Handle logsnr input - use concrete n_patches
        if logsnr_map is None:
            logsnr_patches = torch.zeros(B, n_patches, 1, device=images.device, dtype=images.dtype)
        else:
            logsnr_patches = self.patchify_logsnr(logsnr_map, grid_shape)

        level_recons = []
        codes_list = []
        all_gate_weights = []
        logsnr_preds = []
        level_logsnrs = []
        cumulative_recon = torch.zeros_like(patches)
        current_target = patches

        for level in range(self.n_levels):
            # Compute logsnr for this level
            if level == 0:
                level_logsnr = logsnr_patches
            else:
                # Predict level-specific logsnr from base logsnr
                level_logsnr = self.level_logsnr_estimators[level - 1](logsnr_patches)

            level_logsnrs.append(level_logsnr)

            if level > 0:
                residual = (current_target - cumulative_recon) * self.residual_scale
            else:
                residual = current_target

            # Pass per-level masks (each entry is a list of per-layer BlockMask)
            enc_mask = encoder_masks[level] if encoder_masks is not None else None
            dec_mask = decoder_masks[level] if decoder_masks is not None else None

            codes, gate_weights, _ = self.encoders[level](
                residual, level_logsnr, grid_shape=grid_shape, block_masks=enc_mask,
                k_override=k_override
            )
            codes_list.append(codes)
            all_gate_weights.append(gate_weights)

            decoded, logsnr_pred = self.decoders[level](
                codes, grid_shape=grid_shape, block_masks=dec_mask
            )
            logsnr_preds.append(logsnr_pred)

            if level > 0:
                decoded = decoded / self.residual_scale

            cumulative_recon = cumulative_recon + decoded
            level_recons.append(self.unpatchify(cumulative_recon, grid_shape))

        # Compute sparsity stats (all on GPU, no sync)
        # Use tensor operations to avoid Python float literals in traced code
        total_codes = codes_list[0].numel()
        nonzero_codes = sum((c != 0).sum() for c in codes_list)  # Stays on GPU as tensor
        sparsity = self.one - (nonzero_codes / (total_codes * self.n_levels))  # Returns tensor

        return {
            'recon': level_recons[-1],
            'level_recons': level_recons,
            'codes': codes_list,
            'gate_weights': all_gate_weights,
            'sparsity': sparsity,
            'logsnr_preds': logsnr_preds,  # List of (B, N, 1) per level
            'level_logsnrs': level_logsnrs,  # List of (B, N, 1) per level
            'logsnr_pred_map': self.unpatchify_logsnr(logsnr_preds[-1], grid_shape),  # Final level prediction as spatial map
            'grid_shape': grid_shape  # Pass grid shape for downstream use
        }

    def encode(self, images: torch.Tensor, logsnr_map: Optional[torch.Tensor] = None,
               grid_shape: Optional[Tuple[int, int]] = None,
               encoder_masks: Optional[list] = None,
               decoder_masks: Optional[list] = None) -> Tuple[list, list]:
        """
        Encode images to sparse binary codes.

        Args:
            images: (B, C, H, W) input images
            logsnr_map: (B, 1, H, W) spatial logsnr field. If None, uses zeros.
            grid_shape: Optional (GH, GW) grid dimensions for compiled usage.
            encoder_masks: Optional pre-built encoder masks per level.
            decoder_masks: Optional pre-built decoder masks per level.
        Returns:
            codes_list: List of (B, N, code_dim) sparse codes per level
            level_logsnrs: List of (B, N, 1) logsnr values per level
        """
        B = images.shape[0]
        p = self.patch_size
        if grid_shape is None:
            H, W = images.shape[2], images.shape[3]
            grid_shape = (H // p, W // p)

        n_patches = grid_shape[0] * grid_shape[1]
        patches = self.patchify(images, grid_shape)

        # Handle logsnr input
        if logsnr_map is None:
            logsnr_patches = torch.zeros(B, n_patches, 1, device=images.device, dtype=images.dtype)
        else:
            logsnr_patches = self.patchify_logsnr(logsnr_map, grid_shape)

        # Build masks outside the loop if not provided (avoid building inside compiled transformers)
        device = images.device
        if encoder_masks is None:
            encoder_masks = [self.encoders[i].transformer.build_masks(grid_shape, device)
                             for i in range(self.n_levels)]
        if decoder_masks is None:
            decoder_masks = [self.decoders[i].transformer.build_masks(grid_shape, device)
                             for i in range(self.n_levels)]

        codes_list = []
        level_logsnrs = []
        cumulative_recon = torch.zeros_like(patches)
        current_target = patches

        for level in range(self.n_levels):
            # Compute logsnr for this level
            if level == 0:
                level_logsnr = logsnr_patches
            else:
                level_logsnr = self.level_logsnr_estimators[level - 1](logsnr_patches)

            level_logsnrs.append(level_logsnr)

            if level > 0:
                residual = (current_target - cumulative_recon) * self.residual_scale
            else:
                residual = current_target

            codes, _, _ = self.encoders[level](residual, level_logsnr,
                                               grid_shape=grid_shape,
                                               block_masks=encoder_masks[level])
            codes_list.append(codes)

            decoded, _ = self.decoders[level](codes,
                                              grid_shape=grid_shape,
                                              block_masks=decoder_masks[level])

            if level > 0:
                decoded = decoded / self.residual_scale

            cumulative_recon = cumulative_recon + decoded

        return codes_list, level_logsnrs

    def decode(self, codes_list: list, grid_shape: Tuple[int, int],
               decoder_masks: Optional[list] = None) -> torch.Tensor:
        """
        Decode sparse codes back to image.

        Args:
            codes_list: List of (B, N, code_dim) sparse codes per level
            grid_shape: (GH, GW) grid dimensions - required to avoid symbolic shape issues.
            decoder_masks: Optional pre-built decoder masks per level.
        Returns:
            recon: (B, C, H, W) reconstructed image
        """
        # Build masks outside the loop if not provided
        device = codes_list[0].device
        if decoder_masks is None:
            decoder_masks = [self.decoders[i].transformer.build_masks(grid_shape, device)
                             for i in range(self.n_levels)]

        cumulative_recon = None

        for level, codes in enumerate(codes_list):
            decoded, _ = self.decoders[level](codes,
                                              grid_shape=grid_shape,
                                              block_masks=decoder_masks[level])

            if level > 0:
                decoded = decoded / self.residual_scale

            if cumulative_recon is None:
                cumulative_recon = decoded
            else:
                cumulative_recon = cumulative_recon + decoded

        return self.unpatchify(cumulative_recon, grid_shape)


# =============================================================================
# Interface Wrappers for Integration with Main Training Pipeline
# =============================================================================

class SparseAEPatchEmbedder(nn.Module):
    """
    Wraps SparsePerDimFSQAutoencoder to match ContextualPatchEmbedder interface.

    Supports both single images [C, H, W] and batched input [B, C, H, W].
    When used through SpanEmbedder, images are grouped by grid_shape and processed
    as batches with shared masks (efficient batch-flattening pattern).

    The main training pipeline expects:
    - .stride attribute
    - .n_attn_layers attribute (signals SpanEmbedder to use batched processing)
    - forward(x, logsnr_map, block_mask=None) -> (z, (GH, GW))
    """
    def __init__(self, ae: SparsePerDimFSQAutoencoder, embed_dim: int = 256):
        super().__init__()
        self.ae = ae
        self.stride = ae.patch_size  # Required attribute for interface compatibility
        self.embed_dim = embed_dim

        # Signal to SpanEmbedder that we have attention and need batched processing
        # This triggers _embed_batched() which groups by grid_shape
        self.n_attn_layers = ae.n_layers

        # Project concatenated level codes to embed_dim
        # Each level produces (N, code_dim), we concatenate across levels
        total_code_dim = ae.code_dim * ae.n_levels
        self.code_proj = nn.Linear(total_code_dim, embed_dim)

        # Also project the level logsnrs for output
        self.logsnr_proj = nn.Linear(ae.n_levels, 1)

        # Mask cache: grid_shape -> (encoder_masks, decoder_masks)
        # Built using the AE's own TransformerEncoder.build_masks, ensuring correct attn_config
        self._mask_cache: Dict[Tuple[int, int], Tuple[list, list]] = {}

    def _get_masks(self, grid_shape: Tuple[int, int], device: torch.device) -> Tuple[list, list]:
        """Get or build cached encoder/decoder masks for a grid shape."""
        if grid_shape not in self._mask_cache:
            encoder_masks = [self.ae.encoders[i].transformer.build_masks(grid_shape, device)
                             for i in range(self.ae.n_levels)]
            decoder_masks = [self.ae.decoders[i].transformer.build_masks(grid_shape, device)
                             for i in range(self.ae.n_levels)]
            self._mask_cache[grid_shape] = (encoder_masks, decoder_masks)
        return self._mask_cache[grid_shape]

    def _pad_and_patch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute patches for grid_shape inference (used by SpanEmbedder).

        Returns dummy tensor matching ContextualPatchEmbedder's shape structure:
        - Single: [C, GH, GW, P, P] (5D) so shape[1]=GH, shape[2]=GW
        - Batched: [B, C, GH, GW, P, P] (6D) so shape[2]=GH, shape[3]=GW
        """
        p = self.ae.patch_size
        if x.dim() == 3:
            C, H, W = x.shape
            gh, gw = H // p, W // p
            return torch.empty(C, gh, gw, p, p, device=x.device)
        else:  # Batched
            B, C, H, W = x.shape
            gh, gw = H // p, W // p
            return torch.empty(B, C, gh, gw, p, p, device=x.device)

    def forward(self, x: torch.Tensor, logsnr_map: torch.Tensor,
                block_mask=None, return_codes: bool = False):
        """
        Args:
            x: [C, H, W] single image or [B, C, H, W] batched images
            logsnr_map: [1, H, W] or [B, 1, H, W] spatial logsnr field
            block_mask: Ignored - we use our own cached masks built with correct attn_config
            return_codes: if True, also return concatenated codes for sparsity tracking
        Returns:
            z: [L, D] or [B, L, D] patch embeddings
            grid_shape: (GH, GW) grid dimensions
            codes_cat: (optional) concatenated codes if return_codes=True
        """
        # Handle both single and batched input
        single_input = x.dim() == 3
        if single_input:
            x = x.unsqueeze(0)  # [1, C, H, W]
            logsnr_map = logsnr_map.unsqueeze(0)  # [1, 1, H, W]

        B, C, H, W = x.shape
        p = self.ae.patch_size
        grid_shape = (H // p, W // p)
        device = x.device

        # Always use our own cached masks - built with correct attn_config for this AE
        # This avoids code path divergence that causes torch.compile recompilation
        encoder_masks, decoder_masks = self._get_masks(grid_shape, device)

        # Encode through sparse AE
        codes_list, level_logsnrs = self.ae.encode(x, logsnr_map,
                                                   grid_shape=grid_shape,
                                                   encoder_masks=encoder_masks,
                                                   decoder_masks=decoder_masks)

        # codes_list is list of [B, N, code_dim] per level
        # Concatenate along code dimension
        codes_cat = torch.cat(codes_list, dim=-1)  # [B, N, code_dim * n_levels]

        # Project to embed_dim
        z = self.code_proj(codes_cat)  # [B, N, embed_dim]

        # Remove batch dim if single input
        if single_input:
            z = z.squeeze(0)  # [N, embed_dim] = [L, D]
            if return_codes:
                return z, grid_shape, codes_cat.squeeze(0)
            return z, grid_shape
        else:
            if return_codes:
                return z, grid_shape, codes_cat
            return z, grid_shape


class SparseAEPatchUnembedder(nn.Module):
    """
    Wraps SparsePerDimFSQAutoencoder to match ContextualPatchUnembedder interface.

    Supports both single embeddings [L, D] and batched [B, L, D].
    When used through SpanUnembedder, embeddings are grouped by grid_shape
    and processed as batches with shared masks.

    The main training pipeline expects:
    - .n_attn_layers attribute (signals SpanUnembedder to use batched processing)
    - forward(z, shape, block_mask=None) -> [C+1, H, W] or [B, C+1, H, W]
    """
    def __init__(self, ae: SparsePerDimFSQAutoencoder, embedder: SparseAEPatchEmbedder,
                 fourier_dim: int = 16):
        super().__init__()
        self.ae = ae
        self.embedder = embedder
        self.patch_size = ae.patch_size
        self.fourier_dim = fourier_dim

        # Signal to SpanUnembedder that we have attention and need batched processing
        self.n_attn_layers = ae.n_layers

        # Project back from embed_dim to concatenated codes + fourier features for logsnr
        total_code_dim = ae.code_dim * ae.n_levels
        self.code_unproj = nn.Linear(embedder.embed_dim, total_code_dim + fourier_dim)

        # Decode fourier features to logsnr (matches ContextualPatchUnembedder)
        self.logsnr_decoder = nn.Sequential(
            nn.Linear(fourier_dim, embedder.embed_dim),
            nn.SiLU(),
            nn.Linear(embedder.embed_dim, 1)
        )
        # Initialize output layer near zero for stability
        with torch.no_grad():
            self.logsnr_decoder[-1].weight.zero_()
            self.logsnr_decoder[-1].bias.zero_()

        # Mask cache: grid_shape -> decoder_masks
        # Built using the AE's own TransformerEncoder.build_masks, ensuring correct attn_config
        self._mask_cache: Dict[Tuple[int, int], list] = {}

    def _get_masks(self, grid_shape: Tuple[int, int], device: torch.device) -> list:
        """Get or build cached decoder masks for a grid shape."""
        if grid_shape not in self._mask_cache:
            decoder_masks = [self.ae.decoders[i].transformer.build_masks(grid_shape, device)
                             for i in range(self.ae.n_levels)]
            self._mask_cache[grid_shape] = decoder_masks
        return self._mask_cache[grid_shape]

    def forward(self, z: torch.Tensor, shape: Tuple, block_mask=None) -> torch.Tensor:
        """
        Args:
            z: [L, D] single or [B, L, D] batched patch embeddings
            shape: (GH, GW) grid dimensions
            block_mask: Ignored - we use our own cached masks built with correct attn_config
        Returns:
            output: [C+1, H, W] or [B, C+1, H, W] reconstructed image with logsnr channel
        """
        # Handle both single and batched input
        single_input = z.dim() == 2
        if single_input:
            z = z.unsqueeze(0)  # [1, L, D]

        B, L, D = z.shape

        if len(shape) == 2:
            GH, GW = shape
        elif len(shape) == 1:
            GH, GW = 1, shape[0]
        else:
            GH, GW = 1, L

        if L != GH * GW:
            GH, GW = 1, L  # Fallback

        grid_shape = (GH, GW)
        device = z.device

        # Always use our own cached masks - built with correct attn_config for this AE
        # This avoids code path divergence that causes torch.compile recompilation
        decoder_masks = self._get_masks(grid_shape, device)

        # Project to codes + fourier features
        proj_out = self.code_unproj(z)  # [B, L, code_dim * n_levels + fourier_dim]
        total_code_dim = self.ae.code_dim * self.ae.n_levels
        codes_cat = proj_out[:, :, :total_code_dim]
        fourier_part = proj_out[:, :, total_code_dim:]

        # Split back to per-level codes
        codes_list = []
        code_dim = self.ae.code_dim
        for level in range(self.ae.n_levels):
            start = level * code_dim
            end = start + code_dim
            codes_list.append(codes_cat[:, :, start:end])

        # Decode through sparse AE with grid shape and masks
        recon = self.ae.decode(codes_list, grid_shape=grid_shape,
                               decoder_masks=decoder_masks)  # [B, C, H, W]

        # Predict logsnr from fourier features
        logsnr_pred = self.logsnr_decoder(fourier_part)  # [B, L, 1]
        logsnr_grid = logsnr_pred.view(B, GH, GW).unsqueeze(1)  # [B, 1, GH, GW]
        # Upsample to pixel resolution
        H, W = GH * self.patch_size, GW * self.patch_size
        logsnr_channel = F.interpolate(
            logsnr_grid, size=(H, W), mode='nearest'
        )  # [B, 1, H, W]

        result = torch.cat([recon, logsnr_channel], dim=1)  # [B, C+1, H, W]

        if single_input:
            return result.squeeze(0)  # [C+1, H, W]
        return result


class SparseAEPatchEmbedderWithLogsnr(nn.Module):
    """
    Extended embedder that also returns logsnr predictions per level.

    Supports both single images [C, H, W] and batched [B, C, H, W].
    This is useful for joint training where we want to supervise
    both reconstruction and logsnr prediction.
    """
    def __init__(self, ae: SparsePerDimFSQAutoencoder, embed_dim: int = 256):
        super().__init__()
        self.ae = ae
        self.stride = ae.patch_size
        self.embed_dim = embed_dim
        self.n_attn_layers = ae.n_layers  # Signal for batched processing

        total_code_dim = ae.code_dim * ae.n_levels
        self.code_proj = nn.Linear(total_code_dim, embed_dim)

    def _pad_and_patch(self, x: torch.Tensor) -> torch.Tensor:
        """Compute patches for grid_shape inference."""
        if x.dim() == 3:
            C, H, W = x.shape
            p = self.ae.patch_size
            return torch.empty(H // p, W // p, device=x.device)
        else:
            B, C, H, W = x.shape
            p = self.ae.patch_size
            return torch.empty(B, H // p, W // p, device=x.device)

    def forward(self, x: torch.Tensor, logsnr_map: torch.Tensor,
                block_mask=None) -> dict:
        """
        Args:
            x: [C, H, W] single or [B, C, H, W] batched images
            logsnr_map: [1, H, W] or [B, 1, H, W] spatial logsnr field
            block_mask: Optional BlockMask (built per grid_shape)
        Returns:
            dict with 'embeddings', 'grid_shape', 'ae_output' (full AE forward dict)
        """
        # Handle both single and batched input
        single_input = x.dim() == 3
        if single_input:
            x = x.unsqueeze(0)
            logsnr_map = logsnr_map.unsqueeze(0)

        B, C, H, W = x.shape
        p = self.ae.patch_size
        grid_shape = (H // p, W // p)
        device = x.device

        # Build masks if not provided
        if block_mask is None:
            encoder_masks = [self.ae.encoders[i].transformer.build_masks(grid_shape, device)
                             for i in range(self.ae.n_levels)]
            decoder_masks = [self.ae.decoders[i].transformer.build_masks(grid_shape, device)
                             for i in range(self.ae.n_levels)]
        else:
            # Replicate single mask for each layer within each level
            n_enc_layers = self.ae.encoders[0].transformer.n_layers
            n_dec_layers = self.ae.decoders[0].transformer.n_layers
            per_layer_mask_enc = [block_mask] * n_enc_layers
            per_layer_mask_dec = [block_mask] * n_dec_layers
            encoder_masks = [per_layer_mask_enc] * self.ae.n_levels
            decoder_masks = [per_layer_mask_dec] * self.ae.n_levels

        # Full forward pass for all outputs
        ae_out = self.ae(x, logsnr_map,
                         encoder_masks=encoder_masks, decoder_masks=decoder_masks,
                         grid_shape=grid_shape)

        # Extract codes and project to embeddings
        codes_cat = torch.cat(ae_out['codes'], dim=-1)
        z = self.code_proj(codes_cat)

        if single_input:
            return {
                'embeddings': z.squeeze(0),
                'grid_shape': grid_shape,
                'ae_output': ae_out,
                'recon': ae_out['recon'].squeeze(0),
                'logsnr_pred_map': ae_out['logsnr_pred_map'].squeeze(0),
                'level_logsnrs': [l.squeeze(0) for l in ae_out['level_logsnrs']],
                'sparsity': ae_out['sparsity']
            }
        else:
            return {
                'embeddings': z,
                'grid_shape': grid_shape,
                'ae_output': ae_out,
                'recon': ae_out['recon'],
                'logsnr_pred_map': ae_out['logsnr_pred_map'],
                'level_logsnrs': ae_out['level_logsnrs'],
                'sparsity': ae_out['sparsity']
            }
