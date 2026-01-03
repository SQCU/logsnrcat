"""
SwiGLU FSQ Autoencoder - Clean implementation matching reference.

NO logsnr conditioning. This is a pure image compression network.
The logsnr field exists only in the data pipeline for denoising training;
the FSQ autoencoder simply ignores it.

Key features from reference (clean_impl_swiglu_reference/train.py):
- Binary FSQ with STE (sigmoid > 0.5 -> {-1, +1})
- Level-global sparsity via learned dim_logits (same mask for all patches)
- SwiGLU neighbor decoder with 3x3 gather
- 2D RoPE in transformer attention
- Cumulative residual reconstruction with .detach()
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List

# Import shared components from model_sparse_dim
from .model_sparse_dim import BinaryFSQ, ThreeBitFSQ, PerDimSparsity

# Import transformer blocks from main codebase
from src.blocks import TransformerEncoder, TransformerDecoder


# =============================================================================
# Level-Global Sparsity (matches MultiScaleRouter from reference)
# =============================================================================

class LevelGlobalSparsity(nn.Module):
    """
    Level-global sparsity: same sparse mask for all patches at a given level.

    Each level has learned dim_logits that determine which k dimensions are active.
    This is NOT per-patch sparsity - the mask is global to the level.

    From reference MultiScaleRouter:
    - dim_logits: [n_levels, code_dim] learned logits
    - topk selection produces same mask for entire batch
    - level_values: [n_levels, code_dim] learned fixed values (2-bit quantized)
    """
    def __init__(self, code_dim: int, n_levels: int, k_per_level: int):
        super().__init__()
        self.code_dim = code_dim
        self.n_levels = n_levels
        self.k = k_per_level

        # Learned logits for dimension selection (same for all patches)
        self.dim_logits = nn.Parameter(torch.randn(n_levels, code_dim))

    def get_mask(self, level: int, device: torch.device) -> torch.Tensor:
        """
        Get the sparse mask for a level.

        Returns:
            mask: [code_dim] binary mask with k ones
        """
        _, topk_idx = self.dim_logits[level].topk(self.k)
        mask = torch.zeros(self.code_dim, device=device)
        mask.scatter_(0, topk_idx, 1.0)
        return mask

    def forward(self, codes: torch.Tensor, level: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply level-global sparsity mask to codes.

        Args:
            codes: [B, N, code_dim] quantized codes
            level: which level (0 to n_levels-1)

        Returns:
            sparse_codes: [B, N, code_dim] with (code_dim - k) dims zeroed
            mask: [code_dim] the binary mask used
        """
        device = codes.device
        mask = self.get_mask(level, device)  # [code_dim]

        # STE: forward uses hard mask, backward through sigmoid
        soft_logits = torch.sigmoid(self.dim_logits[level])
        soft_mask = soft_logits * mask
        ste_mask = mask + (soft_mask - soft_mask.detach())

        # Apply mask (broadcast over B, N)
        sparse_codes = codes * ste_mask.unsqueeze(0).unsqueeze(0)

        return sparse_codes, mask


# =============================================================================
# SwiGLU Neighbor Decoder (matches reference)
# =============================================================================

class SwiGLUNeighborHead(nn.Module):
    """
    SwiGLU gating with 3x3 neighbor gathering.

    Gathers 9 neighbors (3x3 window with reflect padding), applies SwiGLU.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_proj = nn.Linear(9 * hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(9 * hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor, grid_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            h: [B, N, hidden_dim] features after transformer
            grid_shape: (GH, GW) spatial grid dimensions

        Returns:
            out: [B, N, hidden_dim] gated output
        """
        B, N, D = h.shape
        GH, GW = grid_shape

        # Reshape to grid
        h_grid = h.view(B, GH, GW, D)

        # Reflect pad for neighbor gathering
        h_padded = F.pad(h_grid.permute(0, 3, 1, 2), (1, 1, 1, 1), mode='reflect')
        # h_padded: [B, D, GH+2, GW+2]

        # Vectorized 3x3 neighbor gathering using unfold
        # unfold extracts sliding windows: [B, D, GH, GW, 3, 3]
        neighbors = h_padded.unfold(2, 3, 1).unfold(3, 3, 1)
        # Reshape to [B, GH, GW, D, 3, 3] then flatten window dims
        neighbors = neighbors.permute(0, 2, 3, 1, 4, 5).contiguous()
        neighbors = neighbors.view(B, GH * GW, 9 * D)  # [B, N, 9*D]

        # SwiGLU
        gate = F.silu(self.gate_proj(neighbors))
        value = self.value_proj(neighbors)
        return gate * value  # [B, N, D]


# =============================================================================
# Encoder and Decoder (NO logsnr conditioning)
# =============================================================================

class SwiGLUEncoder(nn.Module):
    """
    Encoder: patches -> transformer -> code logits -> binary FSQ -> per-patch sparsity.
    NO logsnr conditioning.

    Uses PerDimSparsity: each patch independently selects its top-k dimensions
    based on content-dependent gating (like sparse_dim reference).
    """
    def __init__(
        self,
        patch_dim: int,
        hidden_dim: int,
        code_dim: int,
        k_per_patch: int = 4,
        n_layers: int = 4,
        attn_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.input_proj = nn.Linear(patch_dim, hidden_dim)
        self.transformer = TransformerEncoder(hidden_dim, n_layers=n_layers, attn_config=attn_config)
        self.code_proj = nn.Linear(hidden_dim, code_dim)
        self.fsq = BinaryFSQ()
        self.sparsity = PerDimSparsity(code_dim, k_per_patch)

    def forward(
        self,
        patches: torch.Tensor,
        grid_shape: Tuple[int, int],
        block_masks: Optional[List] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patches: [B, N, patch_dim]
            grid_shape: (GH, GW) for attention masks
            block_masks: optional pre-built masks

        Returns:
            sparse_codes: [B, N, code_dim] sparse binary codes in {-1, +1, 0}
            gate_weights: [B, N, code_dim] soft weights for logging
        """
        h = self.input_proj(patches)
        h = self.transformer(h, grid_shape=grid_shape, block_masks=block_masks)
        logits = self.code_proj(h)

        # Binary FSQ: sigmoid -> threshold -> STE -> normalize to [-1, +1]
        codes = self.fsq(logits)
        codes = codes * 2 - 1  # {0, 1} -> {-1, +1}

        # Per-patch sparsity: each patch independently selects top-k dims
        sparse_codes, gate_weights = self.sparsity(codes)

        return sparse_codes, gate_weights

    def forward_with_prequant(
        self,
        patches: torch.Tensor,
        grid_shape: Tuple[int, int],
        block_masks: Optional[List] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns pre-quantization logits for latent diffusion.

        Args:
            patches: [B, N, patch_dim]
            grid_shape: (GH, GW) for attention masks
            block_masks: optional pre-built masks

        Returns:
            sparse_codes: [B, N, code_dim] sparse binary codes in {-1, +1, 0}
            gate_weights: [B, N, code_dim] soft weights for logging
            pre_quant: [B, N, code_dim] continuous logits before FSQ quantization
        """
        h = self.input_proj(patches)
        h = self.transformer(h, grid_shape=grid_shape, block_masks=block_masks)
        logits = self.code_proj(h)

        # Binary FSQ: sigmoid -> threshold -> STE -> normalize to [-1, +1]
        codes = self.fsq(logits)
        codes = codes * 2 - 1  # {0, 1} -> {-1, +1}

        # Per-patch sparsity: each patch independently selects top-k dims
        sparse_codes, gate_weights = self.sparsity(codes)

        return sparse_codes, gate_weights, logits


class SwiGLUDecoder(nn.Module):
    """
    Decoder: codes -> transformer -> SwiGLU neighbor -> patches.
    NO logsnr prediction.
    """
    def __init__(
        self,
        code_dim: int,
        hidden_dim: int,
        patch_dim: int,
        n_layers: int = 4,
        attn_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.input_proj = nn.Linear(code_dim, hidden_dim)
        self.transformer = TransformerDecoder(hidden_dim, n_layers=n_layers, attn_config=attn_config)
        self.neighbor_head = SwiGLUNeighborHead(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, patch_dim)

    def forward(
        self,
        codes: torch.Tensor,
        grid_shape: Tuple[int, int],
        block_masks: Optional[List] = None
    ) -> torch.Tensor:
        """
        Args:
            codes: [B, N, code_dim] sparse quantized codes
            grid_shape: (GH, GW) for attention and neighbor gathering
            block_masks: optional pre-built masks

        Returns:
            patches: [B, N, patch_dim] reconstructed patches
        """
        h = self.input_proj(codes)
        h = self.transformer(h, grid_shape=grid_shape, block_masks=block_masks)
        h = self.neighbor_head(h, grid_shape)
        patches = self.output_proj(h)
        return patches


# =============================================================================
# Main Autoencoder
# =============================================================================

class SwiGLUFSQAutoencoder(nn.Module):
    """
    Hierarchical Binary FSQ Autoencoder with SwiGLU decoder.

    Clean implementation matching reference - NO logsnr conditioning.

    Architecture:
    - Per-level encoder/decoder pairs
    - Binary FSQ quantization (sigmoid > 0.5 -> {-1, +1})
    - Per-patch sparsity (each patch independently selects top-k dims)
    - Cumulative residual reconstruction with .detach()
    - SwiGLU 3x3 neighbor decoder head
    - 2D RoPE in transformer attention (via TransformerEncoder/Decoder)
    """
    def __init__(
        self,
        n_levels: int = 6,
        patch_size: int = 16,
        hidden_dim: int = 256,
        code_dim: int = 128,
        k_per_patch: int = 4,
        residual_scale: float = 2.0,
        n_layers: int = 4,
        attn_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.n_levels = n_levels
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size * 3
        self.hidden_dim = hidden_dim
        self.code_dim = code_dim
        self.k_per_patch = k_per_patch
        self.n_layers = n_layers

        # Register as buffer to avoid Python float in traced code
        self.register_buffer("residual_scale", torch.tensor(residual_scale))
        self.register_buffer("one", torch.tensor(1.0))

        # Default attention config
        if attn_config is None:
            attn_config = {
                'mode': 'sliding',
                'window_size': 2,
                'n_query_heads': 8,
                'n_kv_heads': 2,
                'n_global_tokens': 0,
                'global_layer_interval': 4,
                'bigbird_layout': [2, 2],
                'random_min_k': 0,
                'random_min_p': 0.0
            }

        # Per-level encoder/decoder (sparsity is inside each encoder now)
        self.encoders = nn.ModuleList([
            SwiGLUEncoder(self.patch_dim, hidden_dim, code_dim, k_per_patch, n_layers, attn_config)
            for _ in range(n_levels)
        ])
        self.decoders = nn.ModuleList([
            SwiGLUDecoder(code_dim, hidden_dim, self.patch_dim, n_layers, attn_config)
            for _ in range(n_levels)
        ])

        print(f"[SwiGLUFSQAutoencoder] {n_levels} levels, code_dim={code_dim}, k={k_per_patch}")
        print(f"  Attention: {attn_config['mode']}, window={attn_config.get('window_size', 'N/A')}")

    def build_masks(self, grid_shape: Tuple[int, int], device: torch.device) -> Tuple[List, List]:
        """Build attention masks for all encoder/decoder layers."""
        encoder_masks = self.encoders[0].transformer.build_masks(grid_shape, device)
        decoder_masks = self.decoders[0].transformer.build_masks(grid_shape, device)
        return encoder_masks, decoder_masks

    def patchify(self, images: torch.Tensor, grid_shape: Tuple[int, int]) -> torch.Tensor:
        """Convert images to patches."""
        B = images.shape[0]
        p = self.patch_size
        GH, GW = grid_shape
        patches = images.view(B, 3, GH, p, GW, p)
        patches = patches.permute(0, 2, 4, 3, 5, 1).contiguous()
        return patches.view(B, GH * GW, self.patch_dim)

    def unpatchify(self, patches: torch.Tensor, grid_shape: Tuple[int, int]) -> torch.Tensor:
        """Convert patches back to images."""
        B = patches.shape[0]
        p = self.patch_size
        GH, GW = grid_shape
        patches = patches.view(B, GH, GW, p, p, 3)
        patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
        return patches.view(B, 3, GH * p, GW * p)

    def forward(
        self,
        images: torch.Tensor,
        logsnr_map: Optional[torch.Tensor] = None,  # Accepted for interface compat, ignored
        encoder_masks: Optional[List] = None,
        decoder_masks: Optional[List] = None,
        grid_shape: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Forward pass - pure reconstruction.

        NOTE: logsnr_map is accepted for interface compatibility with sparse_dim variant
        but is IGNORED. This autoencoder is a pure compression network.

        Args:
            images: [B, C, H, W] input images
            logsnr_map: IGNORED - exists for interface compatibility only
            encoder_masks: optional pre-built encoder attention masks
            decoder_masks: optional pre-built decoder attention masks
            grid_shape: optional (GH, GW), computed from images if not provided

        Returns:
            Dict with:
                'recon': final reconstruction [B, C, H, W]
                'level_recons': list of cumulative reconstructions per level
                'codes': list of sparse codes per level
                'dim_masks': list of sparsity masks per level
                'sparsity': fraction of zeros in codes
        """
        # logsnr_map intentionally unused - this is a pure image compression network
        B = images.shape[0]
        p = self.patch_size

        if grid_shape is None:
            H, W = images.shape[2], images.shape[3]
            grid_shape = (H // p, W // p)

        device = images.device

        # Build masks if not provided
        if encoder_masks is None or decoder_masks is None:
            encoder_masks, decoder_masks = self.build_masks(grid_shape, device)

        patches = self.patchify(images, grid_shape)

        level_recons = []
        codes_list = []
        masks_list = []
        cumulative_recon = torch.zeros_like(patches)

        for level in range(self.n_levels):
            # Residual with .detach() - critical for independent per-level learning
            if level > 0:
                residual = (patches - cumulative_recon.detach()) * self.residual_scale
            else:
                residual = patches

            # Encode (per-patch sparsity is inside encoder now)
            sparse_codes, gate_weights = self.encoders[level](residual, grid_shape, encoder_masks)
            codes_list.append(sparse_codes)
            masks_list.append(gate_weights)  # gate_weights for logging (per-patch)

            # Decode
            decoded = self.decoders[level](sparse_codes, grid_shape, decoder_masks)

            if level > 0:
                decoded = decoded / self.residual_scale

            cumulative_recon = cumulative_recon + decoded
            level_recons.append(self.unpatchify(cumulative_recon, grid_shape))

        # Compute sparsity
        codes_stacked = torch.stack(codes_list, dim=0)
        total = codes_stacked.numel()
        nonzero = (codes_stacked != 0).sum()
        sparsity = self.one - (nonzero / total)

        return {
            'recon': level_recons[-1],
            'level_recons': level_recons,
            'codes': codes_list,
            'dim_masks': masks_list,
            'sparsity': sparsity,
            'grid_shape': grid_shape
        }

    def encode(
        self,
        images: torch.Tensor,
        logsnr_map: Optional[torch.Tensor] = None,  # Accepted for interface compat, ignored
        grid_shape: Optional[Tuple[int, int]] = None,
        encoder_masks: Optional[List] = None,
        decoder_masks: Optional[List] = None
    ) -> List[torch.Tensor]:
        """
        Encode images to sparse codes.

        NOTE: logsnr_map is accepted for interface compatibility but ignored.

        Returns:
            codes_list: list of [B, N, code_dim] sparse codes per level
        """
        # logsnr_map intentionally unused
        B = images.shape[0]
        p = self.patch_size

        if grid_shape is None:
            H, W = images.shape[2], images.shape[3]
            grid_shape = (H // p, W // p)

        device = images.device

        if encoder_masks is None or decoder_masks is None:
            encoder_masks, decoder_masks = self.build_masks(grid_shape, device)

        patches = self.patchify(images, grid_shape)

        codes_list = []
        cumulative_recon = torch.zeros_like(patches)

        for level in range(self.n_levels):
            if level > 0:
                residual = (patches - cumulative_recon.detach()) * self.residual_scale
            else:
                residual = patches

            # Encode (per-patch sparsity is inside encoder now)
            sparse_codes, _ = self.encoders[level](residual, grid_shape, encoder_masks)
            codes_list.append(sparse_codes)

            # Need to decode to compute next residual
            decoded = self.decoders[level](sparse_codes, grid_shape, decoder_masks)
            if level > 0:
                decoded = decoded / self.residual_scale
            cumulative_recon = cumulative_recon + decoded

        return codes_list

    def decode(
        self,
        codes_list: List[torch.Tensor],
        grid_shape: Tuple[int, int],
        decoder_masks: Optional[List] = None
    ) -> torch.Tensor:
        """
        Decode sparse codes to image.

        Args:
            codes_list: list of [B, N, code_dim] sparse codes per level
            grid_shape: (GH, GW) grid dimensions
            decoder_masks: optional pre-built decoder masks

        Returns:
            recon: [B, C, H, W] reconstructed image
        """
        device = codes_list[0].device

        if decoder_masks is None:
            decoder_masks = self.decoders[0].transformer.build_masks(grid_shape, device)

        cumulative_recon = None

        for level, codes in enumerate(codes_list):
            decoded = self.decoders[level](codes, grid_shape, decoder_masks)

            if level > 0:
                decoded = decoded / self.residual_scale

            if cumulative_recon is None:
                cumulative_recon = decoded
            else:
                cumulative_recon = cumulative_recon + decoded

        return self.unpatchify(cumulative_recon, grid_shape)

    def encode_with_prequant(
        self,
        images: torch.Tensor,
        logsnr_map: Optional[torch.Tensor] = None,  # Interface compat, ignored
        grid_shape: Optional[Tuple[int, int]] = None,
        encoder_masks: Optional[List] = None,
        decoder_masks: Optional[List] = None
    ) -> Tuple[List[torch.Tensor], None, List[torch.Tensor]]:
        """
        Encode images to codes AND return pre-quantization values for latent diffusion.

        The pre_quant values are the continuous logits before FSQ quantization.
        These are used as the diffusion target in latent space.

        NOTE: logsnr_map is accepted for interface compatibility but ignored.

        Args:
            images: [B, C, H, W] input images
            logsnr_map: IGNORED - for interface compatibility
            grid_shape: optional (GH, GW)
            encoder_masks: optional pre-built encoder masks
            decoder_masks: optional pre-built decoder masks

        Returns:
            codes_list: list of [B, N, code_dim] sparse codes per level
            level_logsnrs: None (no logsnr prediction in this variant)
            prequant_list: list of [B, N, code_dim] pre-quantization logits per level
        """
        B = images.shape[0]
        p = self.patch_size

        if grid_shape is None:
            H, W = images.shape[2], images.shape[3]
            grid_shape = (H // p, W // p)

        device = images.device

        if encoder_masks is None or decoder_masks is None:
            encoder_masks, decoder_masks = self.build_masks(grid_shape, device)

        patches = self.patchify(images, grid_shape)

        codes_list = []
        prequant_list = []
        cumulative_recon = torch.zeros_like(patches)

        for level in range(self.n_levels):
            if level > 0:
                residual = (patches - cumulative_recon.detach()) * self.residual_scale
            else:
                residual = patches

            # Encode with pre-quant values
            sparse_codes, _, pre_quant = self.encoders[level].forward_with_prequant(
                residual, grid_shape, encoder_masks
            )
            codes_list.append(sparse_codes)
            prequant_list.append(pre_quant)

            # Decode to compute next residual
            decoded = self.decoders[level](sparse_codes, grid_shape, decoder_masks)
            if level > 0:
                decoded = decoded / self.residual_scale
            cumulative_recon = cumulative_recon + decoded

        return codes_list, None, prequant_list

    def quantize_and_decode(
        self,
        prequant_list: List[torch.Tensor],
        grid_shape: Tuple[int, int],
        decoder_masks: Optional[List] = None
    ) -> torch.Tensor:
        """
        Quantize pre-quant values and decode to image.

        Used in latent diffusion to convert predicted pre_quant values back to images.
        Applies FSQ quantization and sparsity masking before decoding.

        Args:
            prequant_list: list of [B, N, code_dim] continuous logits per level
            grid_shape: (GH, GW) grid dimensions
            decoder_masks: optional pre-built decoder masks

        Returns:
            recon: [B, C, H, W] reconstructed image
        """
        device = prequant_list[0].device

        if decoder_masks is None:
            decoder_masks = self.decoders[0].transformer.build_masks(grid_shape, device)

        cumulative_recon = None

        for level, pre_quant in enumerate(prequant_list):
            # Apply FSQ quantization
            codes = self.encoders[level].fsq(pre_quant)
            codes = codes * 2 - 1  # {0, 1} -> {-1, +1}

            # Apply per-patch sparsity
            sparse_codes, _ = self.encoders[level].sparsity(codes)

            # Decode
            decoded = self.decoders[level](sparse_codes, grid_shape, decoder_masks)

            if level > 0:
                decoded = decoded / self.residual_scale

            if cumulative_recon is None:
                cumulative_recon = decoded
            else:
                cumulative_recon = cumulative_recon + decoded

        return self.unpatchify(cumulative_recon, grid_shape)


# =============================================================================
# Interface Wrappers for Integration with src/model.py
# =============================================================================

class SwiGLUPatchEmbedder(nn.Module):
    """
    Wrapper to match ContextualPatchEmbedder interface.

    The main model expects:
    - .stride attribute
    - .n_attn_layers attribute
    - forward(x, logsnr_map, ...) -> (z, grid_shape)

    Note: logsnr_map is IGNORED. It exists in the interface for compatibility
    with the denoising training loop, but the FSQ autoencoder doesn't use it.
    """
    def __init__(self, ae: SwiGLUFSQAutoencoder, embed_dim: int):
        super().__init__()
        self.ae = ae
        self.stride = ae.patch_size
        self.n_attn_layers = ae.n_layers
        self.embed_dim = embed_dim
        self.code_dim = ae.code_dim
        self.n_levels = ae.n_levels

        # Project concatenated codes to embed_dim (for pixel-space diffusion)
        total_code_dim = ae.code_dim * ae.n_levels
        self.code_proj = nn.Linear(total_code_dim, embed_dim)

        # Latent diffusion projections (for per-token code_dim input)
        # These are used when diffusing in latent space with flattened level tokens
        self.latent_code_proj = nn.Linear(ae.code_dim, embed_dim)
        self.logsnr_proj = nn.Linear(1, embed_dim)

        # Mask cache
        self._mask_cache: Dict[Tuple[int, int], Tuple[List, List]] = {}

    def _get_masks(self, grid_shape: Tuple[int, int], device: torch.device) -> Tuple[List, List]:
        """Get or build cached masks."""
        if grid_shape not in self._mask_cache:
            self._mask_cache[grid_shape] = self.ae.build_masks(grid_shape, device)
        return self._mask_cache[grid_shape]

    def _pad_and_patch(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patches for grid_shape computation (interface compatibility).

        Unlike ContextualPatchEmbedder, SwiGLU uses simple non-overlapping patches
        with no context window, so no padding is needed.

        Args:
            x: [C, H, W] or [B, C, H, W] input tensor
        Returns:
            patches: [C, GH, GW, P, P] or [B, C, GH, GW, P, P]
        """
        p = self.ae.patch_size
        is_batched = x.dim() == 4

        if is_batched:
            B, C, H, W = x.shape
            GH, GW = H // p, W // p
            # Reshape to [B, C, GH, P, GW, P] then permute to [B, C, GH, GW, P, P]
            patches = x.view(B, C, GH, p, GW, p).permute(0, 1, 2, 4, 3, 5)
        else:
            C, H, W = x.shape
            GH, GW = H // p, W // p
            # Reshape to [C, GH, P, GW, P] then permute to [C, GH, GW, P, P]
            patches = x.view(C, GH, p, GW, p).permute(0, 1, 3, 2, 4)

        return patches

    def forward(
        self,
        x: torch.Tensor,
        logsnr_map: torch.Tensor = None,  # IGNORED
        block_mask = None
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Args:
            x: [C, H, W] or [B, C, H, W] images
            logsnr_map: IGNORED (exists for interface compatibility)
            block_mask: IGNORED (we use our own cached masks)

        Returns:
            z: [L, D] or [B, L, D] embeddings
            grid_shape: (GH, GW)
        """
        single = x.dim() == 3
        if single:
            x = x.unsqueeze(0)

        B, C, H, W = x.shape
        p = self.ae.patch_size
        grid_shape = (H // p, W // p)
        device = x.device

        encoder_masks, decoder_masks = self._get_masks(grid_shape, device)
        codes_list = self.ae.encode(x, grid_shape=grid_shape, encoder_masks=encoder_masks, decoder_masks=decoder_masks)

        # Concatenate all level codes
        codes_cat = torch.cat(codes_list, dim=-1)  # [B, N, code_dim * n_levels]
        z = self.code_proj(codes_cat)  # [B, N, embed_dim]

        if single:
            return z.squeeze(0), grid_shape
        return z, grid_shape


class SwiGLUPatchUnembedder(nn.Module):
    """
    Wrapper to match ContextualPatchUnembedder interface.

    Note: Does NOT predict logsnr. The output is [C, H, W] not [C+1, H, W].
    If the denoising loop needs logsnr prediction, it should come from
    the main LDTformer, not from the FSQ autoencoder.
    """
    def __init__(self, ae: SwiGLUFSQAutoencoder, embedder: SwiGLUPatchEmbedder):
        super().__init__()
        self.ae = ae
        self.embedder = embedder
        self.patch_size = ae.patch_size
        self.n_attn_layers = ae.n_layers

        # Project from embed_dim back to codes (for pixel-space diffusion)
        total_code_dim = ae.code_dim * ae.n_levels
        self.code_unproj = nn.Linear(embedder.embed_dim, total_code_dim)

        # Latent diffusion projections (for per-token code_dim output)
        # These are used when diffusing in latent space with flattened level tokens
        self.latent_code_unproj = nn.Linear(embedder.embed_dim, ae.code_dim)
        self.logsnr_decoder = nn.Linear(embedder.embed_dim, 1)
        # Initialize logsnr output near zero for stability
        with torch.no_grad():
            self.logsnr_decoder.weight.zero_()
            self.logsnr_decoder.bias.zero_()

        # Mask cache
        self._mask_cache: Dict[Tuple[int, int], List] = {}

    def _get_masks(self, grid_shape: Tuple[int, int], device: torch.device) -> List:
        """Get or build cached decoder masks."""
        if grid_shape not in self._mask_cache:
            self._mask_cache[grid_shape] = self.ae.decoders[0].transformer.build_masks(grid_shape, device)
        return self._mask_cache[grid_shape]

    def forward(
        self,
        z: torch.Tensor,
        shape: Tuple,
        block_mask = None
    ) -> torch.Tensor:
        """
        Args:
            z: [L, D] or [B, L, D] embeddings
            shape: (GH, GW) grid dimensions
            block_mask: IGNORED

        Returns:
            recon: [C+1, H, W] or [B, C+1, H, W] - RGB + logsnr channel
        """
        single = z.dim() == 2
        if single:
            z = z.unsqueeze(0)

        B, L, D = z.shape
        GH, GW = shape if len(shape) == 2 else (1, L)
        grid_shape = (GH, GW)
        device = z.device
        P = self.patch_size

        # Project to codes
        codes_cat = self.code_unproj(z)  # [B, L, total_code_dim]

        # Split to per-level codes
        code_dim = self.ae.code_dim
        codes_list = list(codes_cat.split(code_dim, dim=-1))

        decoder_masks = self._get_masks(grid_shape, device)
        recon = self.ae.decode(codes_list, grid_shape, decoder_masks)  # [B, C, H, W]

        # Decode logsnr from embeddings and reshape to spatial
        logsnr_per_patch = self.logsnr_decoder(z)  # [B, L, 1]
        logsnr_per_patch = logsnr_per_patch.reshape(B, GH, GW, 1).permute(0, 3, 1, 2)  # [B, 1, GH, GW]
        # Upsample to match pixel resolution
        logsnr_spatial = logsnr_per_patch.repeat_interleave(P, dim=2).repeat_interleave(P, dim=3)  # [B, 1, H, W]

        # Concat RGB + logsnr to match expected [C+1, H, W] interface
        recon_with_logsnr = torch.cat([recon, logsnr_spatial], dim=1)  # [B, C+1, H, W]

        if single:
            return recon_with_logsnr.squeeze(0)
        return recon_with_logsnr
