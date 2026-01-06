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
# Haar Wavelet Transforms (Type A: Fixed transforms for inductive bias)
# =============================================================================

class HaarDWT2d(nn.Module):
    """
    2D Haar Discrete Wavelet Transform - fixed transform, no learnable params.

    Decomposes input into 4 subbands:
    - LL: low-low (approximation) - averages
    - LH: low-high (horizontal detail) - vertical differences
    - HL: high-low (vertical detail) - horizontal differences
    - HH: high-high (diagonal detail) - diagonal differences

    For 16x16 patch -> 4 x 8x8 coefficient blocks = 256 values (same as input).
    """
    def __init__(self):
        super().__init__()
        # Haar filter coefficients (normalized)
        self.register_buffer("sqrt2_inv", torch.tensor(0.5))  # 1/sqrt(2) * 1/sqrt(2) = 0.5

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [B, C, H, W] input image/patches

        Returns:
            LL: [B, C, H//2, W//2] approximation coefficients
            (LH, HL, HH): tuple of [B, C, H//2, W//2] detail coefficients
        """
        # Split into even/odd rows and columns
        x_ll = x[:, :, 0::2, 0::2]  # even rows, even cols
        x_lh = x[:, :, 0::2, 1::2]  # even rows, odd cols
        x_hl = x[:, :, 1::2, 0::2]  # odd rows, even cols
        x_hh = x[:, :, 1::2, 1::2]  # odd rows, odd cols

        # Haar transform: averages and differences
        LL = (x_ll + x_lh + x_hl + x_hh) * self.sqrt2_inv  # average
        LH = (x_ll - x_lh + x_hl - x_hh) * self.sqrt2_inv  # horizontal detail
        HL = (x_ll + x_lh - x_hl - x_hh) * self.sqrt2_inv  # vertical detail
        HH = (x_ll - x_lh - x_hl + x_hh) * self.sqrt2_inv  # diagonal detail

        return LL, (LH, HL, HH)


class HaarIDWT2d(nn.Module):
    """
    2D Inverse Haar Discrete Wavelet Transform - fixed transform, no learnable params.

    Reconstructs image from 4 subbands (LL, LH, HL, HH).
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("sqrt2_inv", torch.tensor(0.5))

    def forward(
        self,
        LL: torch.Tensor,
        details: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            LL: [B, C, H, W] approximation coefficients
            details: (LH, HL, HH) each [B, C, H, W] detail coefficients

        Returns:
            x: [B, C, H*2, W*2] reconstructed image
        """
        LH, HL, HH = details
        B, C, H, W = LL.shape

        # Inverse Haar: reconstruct corners
        x_ll = (LL + LH + HL + HH) * self.sqrt2_inv
        x_lh = (LL - LH + HL - HH) * self.sqrt2_inv
        x_hl = (LL + LH - HL - HH) * self.sqrt2_inv
        x_hh = (LL - LH - HL + HH) * self.sqrt2_inv

        # Vectorized interleave using stack + permute pattern
        # Goal: x[b,c,2h,2w]=x_ll, x[b,c,2h,2w+1]=x_lh, x[b,c,2h+1,2w]=x_hl, x[b,c,2h+1,2w+1]=x_hh
        # Step 1: Stack column pairs -> [B, C, H, W, 2]
        even_rows = torch.stack([x_ll, x_lh], dim=-1)  # [B, C, H, W, 2]
        odd_rows = torch.stack([x_hl, x_hh], dim=-1)   # [B, C, H, W, 2]
        # Step 2: Stack row pairs -> [B, C, 2, H, W, 2]
        x = torch.stack([even_rows, odd_rows], dim=2)
        # Step 3: Permute to interleave correctly: [B, C, 2, H, W, 2] -> [B, C, H, 2, W, 2]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        # Step 4: Reshape to final dims
        x = x.view(B, C, H * 2, W * 2)

        return x


def batch_dwt2d(patches: torch.Tensor, patch_size: int = 16, n_channels: int = 3) -> torch.Tensor:
    """
    Apply DWT to batched patches.

    Args:
        patches: [B, N, C*H*W] flattened patches
        patch_size: spatial size of patches (assumes square)
        n_channels: number of channels (inferred from patch_dim if possible)

    Returns:
        coeffs: [B, N, C*H*W] flattened coefficients (same size as input)
                Ordered as [LL, LH, HL, HH] concatenated per channel
    """
    B, N, patch_dim = patches.shape
    H = W = patch_size
    C = patch_dim // (H * W)  # Infer channels from dimensions

    # Reshape to spatial: [B*N, C, H, W]
    x = patches.view(B * N, C, H, W)

    # Apply DWT (stateless ops, no module needed)
    sqrt2_inv = 0.5
    x_ll = x[:, :, 0::2, 0::2]
    x_lh = x[:, :, 0::2, 1::2]
    x_hl = x[:, :, 1::2, 0::2]
    x_hh = x[:, :, 1::2, 1::2]

    LL = (x_ll + x_lh + x_hl + x_hh) * sqrt2_inv
    LH = (x_ll - x_lh + x_hl - x_hh) * sqrt2_inv
    HL = (x_ll + x_lh - x_hl - x_hh) * sqrt2_inv
    HH = (x_ll - x_lh - x_hl + x_hh) * sqrt2_inv

    # Flatten and concatenate: [B*N, C, H//2, W//2] * 4 -> [B*N, C*H*W]
    coeff_size = C * (H // 2) * (W // 2)
    coeffs = torch.cat([
        LL.reshape(B * N, coeff_size),
        LH.reshape(B * N, coeff_size),
        HL.reshape(B * N, coeff_size),
        HH.reshape(B * N, coeff_size)
    ], dim=-1)

    return coeffs.view(B, N, patch_dim)


def batch_idwt2d(coeffs: torch.Tensor, patch_size: int = 16, n_channels: int = 3) -> torch.Tensor:
    """
    Apply inverse DWT to batched coefficients.

    Args:
        coeffs: [B, N, C*H*W] flattened coefficients
                Ordered as [LL, LH, HL, HH] concatenated per channel
        patch_size: target spatial size
        n_channels: number of channels

    Returns:
        patches: [B, N, C*H*W] reconstructed patches
    """
    B, N, coeff_dim = coeffs.shape
    H = W = patch_size
    C = coeff_dim // (H * W)  # Infer channels
    coeff_h = coeff_w = H // 2
    coeff_size = C * coeff_h * coeff_w

    # Split into subbands
    coeffs_flat = coeffs.view(B * N, 4, coeff_size)
    LL = coeffs_flat[:, 0].view(B * N, C, coeff_h, coeff_w)
    LH = coeffs_flat[:, 1].view(B * N, C, coeff_h, coeff_w)
    HL = coeffs_flat[:, 2].view(B * N, C, coeff_h, coeff_w)
    HH = coeffs_flat[:, 3].view(B * N, C, coeff_h, coeff_w)

    # Apply inverse DWT (stateless ops)
    sqrt2_inv = 0.5
    x_ll = (LL + LH + HL + HH) * sqrt2_inv
    x_lh = (LL - LH + HL - HH) * sqrt2_inv
    x_hl = (LL + LH - HL - HH) * sqrt2_inv
    x_hh = (LL - LH - HL + HH) * sqrt2_inv

    # Vectorized interleave (no zeros allocation + slice assignment)
    # Goal: x[b,c,2h,2w]=x_ll, x[b,c,2h,2w+1]=x_lh, x[b,c,2h+1,2w]=x_hl, x[b,c,2h+1,2w+1]=x_hh
    even_rows = torch.stack([x_ll, x_lh], dim=-1)  # [B*N, C, coeff_h, coeff_w, 2]
    odd_rows = torch.stack([x_hl, x_hh], dim=-1)   # [B*N, C, coeff_h, coeff_w, 2]
    x = torch.stack([even_rows, odd_rows], dim=2)  # [B*N, C, 2, coeff_h, coeff_w, 2]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()   # [B*N, C, coeff_h, 2, coeff_w, 2]
    x = x.view(B * N, C, H, W)

    return x.view(B, N, C * H * W)


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


class LevelSparsity(nn.Module):
    """
    Per-level sparsity: learned dimension selection that's the same for all patches.

    Unlike PerDimSparsity (content-dependent per-patch), this learns a fixed set
    of k dimensions to activate for all patches at this level. More stable but
    less expressive - good default when per-patch sparsity collapses.

    Each encoder gets its own LevelSparsity instance with independent dim_logits.
    Supports k_override for k-annealing during training.
    """
    def __init__(self, code_dim: int, k: int):
        super().__init__()
        self.code_dim = code_dim
        self.k = k

        # Learned logits for dimension selection (same for all patches in this level)
        self.dim_logits = nn.Parameter(torch.randn(code_dim))

    def forward(
        self,
        codes: torch.Tensor,
        k_override: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply level-global sparsity mask to codes.

        Args:
            codes: [B, N, code_dim] quantized codes
            k_override: optional override for k (for k-annealing during training)

        Returns:
            sparse_codes: [B, N, code_dim] with (code_dim - k) dims zeroed
            soft_weights: [code_dim] sigmoid(dim_logits) for logging
        """
        device = codes.device
        k = k_override if k_override is not None else self.k

        # Get topk indices
        _, topk_idx = self.dim_logits.topk(k)

        # Create hard mask
        mask = torch.zeros(self.code_dim, device=device)
        mask.scatter_(0, topk_idx, 1.0)

        # STE: forward uses hard mask, backward through sigmoid
        soft_weights = torch.sigmoid(self.dim_logits)
        soft_mask = soft_weights * mask
        ste_mask = mask + (soft_mask - soft_mask.detach())

        # Apply mask (broadcast over B, N)
        sparse_codes = codes * ste_mask.unsqueeze(0).unsqueeze(0)

        return sparse_codes, soft_weights


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
    Encoder: patches -> transformer -> code logits -> binary FSQ -> sparsity.
    NO logsnr conditioning.

    Sparsity modes:
    - "per_level": LevelSparsity - same dims active for all patches (learned, stable)
    - "per_patch": PerDimSparsity - content-dependent per-patch selection (expressive, prone to collapse)

    Wavelet subspace routing (optional, wavelet_gating=True):
    - Input: raw patches AND DWT coefficients, each projected to hidden_dim // 2, concatenated
    - Code space: partitioned into wavelet and amplitude subspaces
    - Sparsity pattern across subspaces encodes reconstruction pathway
    """
    def __init__(
        self,
        patch_dim: int,
        hidden_dim: int,
        code_dim: int,
        k_per_patch: int = 4,
        n_layers: int = 4,
        attn_config: Optional[Dict[str, Any]] = None,
        sparsity_mode: str = "per_level",
        wavelet_gating: bool = False,
        patch_size: int = 16,
        n_wavelet_dims: Optional[int] = None
    ):
        super().__init__()
        self.wavelet_gating = wavelet_gating
        self.patch_size = patch_size
        self.code_dim = code_dim
        self.n_wavelet_dims = n_wavelet_dims or code_dim // 2
        self.n_amplitude_dims = code_dim - self.n_wavelet_dims

        # Input projection: dual path if wavelet_gating, single otherwise
        if wavelet_gating:
            self.amplitude_proj = nn.Linear(patch_dim, hidden_dim // 2)
            self.wavelet_proj = nn.Linear(patch_dim, hidden_dim // 2)
            self.input_proj = None  # Not used
        else:
            self.input_proj = nn.Linear(patch_dim, hidden_dim)
            self.amplitude_proj = None
            self.wavelet_proj = None

        self.transformer = TransformerEncoder(hidden_dim, n_layers=n_layers, attn_config=attn_config)

        # Code projection: dual if wavelet_gating (separate subspaces), single otherwise
        if wavelet_gating:
            self.wav_code_proj = nn.Linear(hidden_dim, self.n_wavelet_dims)
            self.amp_code_proj = nn.Linear(hidden_dim, self.n_amplitude_dims)
            self.code_proj = None  # Not used
        else:
            self.code_proj = nn.Linear(hidden_dim, code_dim)
            self.wav_code_proj = None
            self.amp_code_proj = None

        self.fsq = BinaryFSQ()
        self.sparsity_mode = sparsity_mode

        # Create sparsity module based on mode and wavelet_gating
        if wavelet_gating:
            if sparsity_mode == "per_level":
                self.sparsity = SubspaceSparsity(code_dim, k_per_patch, self.n_wavelet_dims)
            else:  # per_patch
                self.sparsity = SubspacePerPatchSparsity(code_dim, k_per_patch, self.n_wavelet_dims)
        else:
            if sparsity_mode == "per_level":
                self.sparsity = LevelSparsity(code_dim, k_per_patch)
            else:  # per_patch
                self.sparsity = PerDimSparsity(code_dim, k_per_patch)

    def forward(
        self,
        patches: torch.Tensor,
        grid_shape: Tuple[int, int],
        block_masks: Optional[List] = None,
        k_override: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patches: [B, N, patch_dim]
            grid_shape: (GH, GW) for attention masks
            block_masks: optional pre-built masks
            k_override: optional override for k sparsity (for k-annealing)

        Returns:
            sparse_codes: [B, N, code_dim] sparse binary codes in {-1, +1, 0}
            gate_weights: [B, N, code_dim] or [code_dim] soft weights for logging
        """
        # Input projection: dual path with concat if wavelet_gating
        if self.wavelet_gating:
            wavelet_coeffs = batch_dwt2d(patches, self.patch_size)
            h_amp = self.amplitude_proj(patches)       # [B, N, hidden_dim // 2]
            h_wav = self.wavelet_proj(wavelet_coeffs)  # [B, N, hidden_dim // 2]
            h = torch.cat([h_amp, h_wav], dim=-1)      # [B, N, hidden_dim]
        else:
            h = self.input_proj(patches)

        h = self.transformer(h, grid_shape=grid_shape, block_masks=block_masks)

        # Code projection: dual with concat if wavelet_gating
        if self.wavelet_gating:
            wav_logits = self.wav_code_proj(h)  # [B, N, n_wavelet_dims]
            amp_logits = self.amp_code_proj(h)  # [B, N, n_amplitude_dims]
            logits = torch.cat([wav_logits, amp_logits], dim=-1)  # [B, N, code_dim]
        else:
            logits = self.code_proj(h)

        # Binary FSQ: sigmoid -> threshold -> STE -> normalize to [-1, +1]
        codes = self.fsq(logits)
        codes = codes * 2 - 1  # {0, 1} -> {-1, +1}

        # Sparsity (subspace-aware if wavelet_gating)
        sparse_codes, gate_weights = self.sparsity(codes, k_override=k_override)

        return sparse_codes, gate_weights

    def forward_with_prequant(
        self,
        patches: torch.Tensor,
        grid_shape: Tuple[int, int],
        block_masks: Optional[List] = None,
        k_override: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns pre-quantization logits for latent diffusion.

        Args:
            patches: [B, N, patch_dim]
            grid_shape: (GH, GW) for attention masks
            block_masks: optional pre-built masks
            k_override: optional override for k sparsity (for k-annealing)

        Returns:
            sparse_codes: [B, N, code_dim] sparse binary codes in {-1, +1, 0}
            gate_weights: [B, N, code_dim] or [code_dim] soft weights for logging
            pre_quant: [B, N, code_dim] continuous logits before FSQ quantization
        """
        # Input projection: dual path with concat if wavelet_gating
        if self.wavelet_gating:
            wavelet_coeffs = batch_dwt2d(patches, self.patch_size)
            h_amp = self.amplitude_proj(patches)
            h_wav = self.wavelet_proj(wavelet_coeffs)
            h = torch.cat([h_amp, h_wav], dim=-1)
        else:
            h = self.input_proj(patches)

        h = self.transformer(h, grid_shape=grid_shape, block_masks=block_masks)

        # Code projection: dual with concat if wavelet_gating
        if self.wavelet_gating:
            wav_logits = self.wav_code_proj(h)
            amp_logits = self.amp_code_proj(h)
            logits = torch.cat([wav_logits, amp_logits], dim=-1)
        else:
            logits = self.code_proj(h)

        # Binary FSQ: sigmoid -> threshold -> STE -> normalize to [-1, +1]
        codes = self.fsq(logits)
        codes = codes * 2 - 1  # {0, 1} -> {-1, +1}

        # Sparsity (subspace-aware if wavelet_gating)
        sparse_codes, gate_weights = self.sparsity(codes, k_override=k_override)

        return sparse_codes, gate_weights, logits


class SwiGLUDecoder(nn.Module):
    """
    Decoder: codes -> transformer -> SwiGLU neighbor -> patches.
    NO logsnr prediction.

    Wavelet subspace routing (optional, wavelet_gating=True):
    - Codes split into wavelet and amplitude subspaces
    - Each subspace embeds to hidden_dim // 2, concatenated
    - Dual output heads: wav_head -> IDWT -> pixels, amp_head -> pixels
    - Output = wav_pixels + amp_pixels (sum of pathways)
    """
    def __init__(
        self,
        code_dim: int,
        hidden_dim: int,
        patch_dim: int,
        n_layers: int = 4,
        attn_config: Optional[Dict[str, Any]] = None,
        wavelet_gating: bool = False,
        patch_size: int = 16,
        n_wavelet_dims: Optional[int] = None
    ):
        super().__init__()
        self.wavelet_gating = wavelet_gating
        self.patch_size = patch_size
        self.patch_dim = patch_dim
        self.code_dim = code_dim
        self.n_wavelet_dims = n_wavelet_dims or code_dim // 2
        self.n_amplitude_dims = code_dim - self.n_wavelet_dims

        # Input embedding: dual path if wavelet_gating
        if wavelet_gating:
            self.wav_embed = nn.Linear(self.n_wavelet_dims, hidden_dim // 2)
            self.amp_embed = nn.Linear(self.n_amplitude_dims, hidden_dim // 2)
            self.input_proj = None  # Not used
        else:
            self.input_proj = nn.Linear(code_dim, hidden_dim)
            self.wav_embed = None
            self.amp_embed = None

        self.transformer = TransformerDecoder(hidden_dim, n_layers=n_layers, attn_config=attn_config)
        self.neighbor_head = SwiGLUNeighborHead(hidden_dim)

        # Output projection: dual if wavelet_gating
        if wavelet_gating:
            self.wav_head = nn.Linear(hidden_dim, patch_dim)  # -> IDWT -> pixels
            self.amp_head = nn.Linear(hidden_dim, patch_dim)  # -> pixels directly
            self.output_proj = None  # Not used
        else:
            self.output_proj = nn.Linear(hidden_dim, patch_dim)
            self.wav_head = None
            self.amp_head = None

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
        # Input embedding: dual path with concat if wavelet_gating
        if self.wavelet_gating:
            wav_codes = codes[..., :self.n_wavelet_dims]
            amp_codes = codes[..., self.n_wavelet_dims:]
            h_wav = self.wav_embed(wav_codes)   # [B, N, hidden_dim // 2]
            h_amp = self.amp_embed(amp_codes)   # [B, N, hidden_dim // 2]
            h = torch.cat([h_wav, h_amp], dim=-1)  # [B, N, hidden_dim]
        else:
            h = self.input_proj(codes)

        h = self.transformer(h, grid_shape=grid_shape, block_masks=block_masks)
        h = self.neighbor_head(h, grid_shape)

        # Output: dual pathway if wavelet_gating
        if self.wavelet_gating:
            wav_coeffs = self.wav_head(h)
            amp_pixels = self.amp_head(h)
            wav_pixels = batch_idwt2d(wav_coeffs, self.patch_size)
            patches = wav_pixels + amp_pixels  # Sum of pathways
        else:
            patches = self.output_proj(h)

        return patches

    def forward_with_contributions(
        self,
        codes: torch.Tensor,
        grid_shape: Tuple[int, int],
        block_masks: Optional[List] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning individual subspace contributions for visualization.
        Only meaningful when wavelet_gating=True.

        Returns:
            patches: [B, N, patch_dim] combined reconstruction
            wav_pixels: [B, N, patch_dim] wavelet pathway contribution
            amp_pixels: [B, N, patch_dim] amplitude pathway contribution
        """
        if not self.wavelet_gating:
            patches = self.forward(codes, grid_shape, block_masks)
            return patches, patches, torch.zeros_like(patches)

        wav_codes = codes[..., :self.n_wavelet_dims]
        amp_codes = codes[..., self.n_wavelet_dims:]
        h_wav = self.wav_embed(wav_codes)
        h_amp = self.amp_embed(amp_codes)
        h = torch.cat([h_wav, h_amp], dim=-1)

        h = self.transformer(h, grid_shape=grid_shape, block_masks=block_masks)
        h = self.neighbor_head(h, grid_shape)

        wav_coeffs = self.wav_head(h)
        amp_pixels = self.amp_head(h)
        wav_pixels = batch_idwt2d(wav_coeffs, self.patch_size)
        patches = wav_pixels + amp_pixels

        return patches, wav_pixels, amp_pixels

    def forward_ablated(
        self,
        codes: torch.Tensor,
        grid_shape: Tuple[int, int],
        ablate_wavelet: float = 0.0,
        ablate_amplitude: float = 0.0,
        block_masks: Optional[List] = None,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Forward with stochastic subspace ablation (dropout-style knockout).
        Only meaningful when wavelet_gating=True.

        Args:
            codes: [B, N, code_dim] sparse quantized codes
            grid_shape: (GH, GW)
            ablate_wavelet: rate [0,1] of zeroing wavelet codes
            ablate_amplitude: rate [0,1] of zeroing amplitude codes
            block_masks: optional attention masks
            deterministic: if True, zero fixed fraction; if False, Bernoulli per-element

        Returns:
            patches: [B, N, patch_dim] reconstruction with ablated subspace(s)
        """
        if not self.wavelet_gating:
            return self.forward(codes, grid_shape, block_masks)

        wav_codes = codes[..., :self.n_wavelet_dims].clone()
        amp_codes = codes[..., self.n_wavelet_dims:].clone()

        # Apply ablation
        if ablate_wavelet > 0:
            if deterministic:
                n_zero = int(self.n_wavelet_dims * ablate_wavelet)
                if n_zero > 0:
                    mask = torch.ones(self.n_wavelet_dims, device=codes.device)
                    mask[:n_zero] = 0
                    perm = torch.randperm(self.n_wavelet_dims, device=codes.device)
                    mask = mask[perm]
                    wav_codes = wav_codes * mask
            elif ablate_wavelet >= 1.0:
                wav_codes = torch.zeros_like(wav_codes)
            else:
                mask = torch.bernoulli(torch.full_like(wav_codes, 1 - ablate_wavelet))
                wav_codes = wav_codes * mask

        if ablate_amplitude > 0:
            if deterministic:
                n_zero = int(self.n_amplitude_dims * ablate_amplitude)
                if n_zero > 0:
                    mask = torch.ones(self.n_amplitude_dims, device=codes.device)
                    mask[:n_zero] = 0
                    perm = torch.randperm(self.n_amplitude_dims, device=codes.device)
                    mask = mask[perm]
                    amp_codes = amp_codes * mask
            elif ablate_amplitude >= 1.0:
                amp_codes = torch.zeros_like(amp_codes)
            else:
                mask = torch.bernoulli(torch.full_like(amp_codes, 1 - ablate_amplitude))
                amp_codes = amp_codes * mask

        h_wav = self.wav_embed(wav_codes)
        h_amp = self.amp_embed(amp_codes)
        h = torch.cat([h_wav, h_amp], dim=-1)

        h = self.transformer(h, grid_shape=grid_shape, block_masks=block_masks)
        h = self.neighbor_head(h, grid_shape)

        wav_coeffs = self.wav_head(h)
        amp_pixels = self.amp_head(h)
        wav_pixels = batch_idwt2d(wav_coeffs, self.patch_size)

        return wav_pixels + amp_pixels


# =============================================================================
# Subspace-Routed Components (Proper design: sparsity IS the router)
# =============================================================================

class SubspaceSparsity(nn.Module):
    """
    Subspace-aware sparsity with routing statistics.

    Partitions code_dim into wavelet and amplitude subspaces. The sparsity
    pattern's distribution across subspaces encodes pathway selection:
    - 6 wavelet, 0 amplitude → pure frequency-domain
    - 0 wavelet, 6 amplitude → pure spatial
    - 3 wavelet, 3 amplitude → balanced mixed

    This is a discretized rotation angle through representation space.
    """
    def __init__(self, code_dim: int, k: int, n_wavelet_dims: Optional[int] = None):
        super().__init__()
        self.code_dim = code_dim
        self.k = k
        self.n_wavelet_dims = n_wavelet_dims or code_dim // 2
        self.n_amplitude_dims = code_dim - self.n_wavelet_dims

        # Learned logits for dimension selection
        self.dim_logits = nn.Parameter(torch.randn(code_dim))

        # Pre-allocated mask buffer (avoids allocation every forward)
        self.register_buffer('_mask_buffer', torch.zeros(code_dim), persistent=False)

    def forward(
        self,
        codes: torch.Tensor,
        k_override: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply sparsity with subspace routing statistics.

        Returns:
            sparse_codes: [B, N, code_dim] with (code_dim - k) dims zeroed
            soft_weights: [code_dim] sigmoid(dim_logits)

        Note: Routing stats are stored in self.last_routing_stats for diagnostic access.
        """
        device = codes.device
        k = k_override if k_override is not None else self.k

        # Get topk indices
        _, topk_idx = self.dim_logits.topk(k)

        # Create hard mask
        mask = torch.zeros(self.code_dim, device=device)
        mask.scatter_(0, topk_idx, 1.0)

        # STE: forward uses hard mask, backward through sigmoid
        soft_weights = torch.sigmoid(self.dim_logits)
        soft_mask = soft_weights * mask
        ste_mask = mask + (soft_mask - soft_mask.detach())

        # Apply mask
        sparse_codes = codes * ste_mask.unsqueeze(0).unsqueeze(0)

        # Compute and store subspace routing statistics (diagnostic, not returned)
        wav_mask = mask[:self.n_wavelet_dims]
        amp_mask = mask[self.n_wavelet_dims:]
        wav_active = wav_mask.sum()
        amp_active = amp_mask.sum()

        # Routing entropy (prevent collapse)
        eps = 1e-7
        p_wav = wav_active / k
        p_amp = amp_active / k
        routing_entropy = -(p_wav * torch.log(p_wav + eps) + p_amp * torch.log(p_amp + eps))

        self.last_routing_stats = {
            'wav_active': wav_active,
            'amp_active': amp_active,
            'routing_entropy': routing_entropy
        }

        return sparse_codes, soft_weights


class SubspacePerPatchSparsity(nn.Module):
    """
    Per-patch subspace-aware sparsity with routing statistics.

    Like SubspaceSparsity but with content-dependent selection per patch
    (like PerDimSparsity). More expressive but prone to collapse.
    """
    def __init__(self, code_dim: int, k: int, n_wavelet_dims: Optional[int] = None):
        super().__init__()
        self.code_dim = code_dim
        self.k = k
        self.n_wavelet_dims = n_wavelet_dims or code_dim // 2
        self.n_amplitude_dims = code_dim - self.n_wavelet_dims

        # Gate predictor: from codes to gate logits (per-patch)
        self.gate_proj = nn.Linear(code_dim, code_dim)

    def forward(
        self,
        codes: torch.Tensor,
        k_override: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply per-patch sparsity with subspace routing statistics.

        Returns:
            sparse_codes: [B, N, code_dim] with (code_dim - k) dims zeroed
            gate_weights: [B, N, code_dim] soft weights

        Note: Routing stats are stored in self.last_routing_stats for diagnostic access.
        """
        B, N, D = codes.shape
        k = k_override if k_override is not None else self.k

        # Predict gate from codes (per-patch)
        gate_logits = self.gate_proj(codes)  # [B, N, code_dim]
        gate_weights = torch.sigmoid(gate_logits)

        # Top-k per patch
        _, topk_idx = gate_weights.topk(k, dim=-1)  # [B, N, k]

        # Create hard mask
        hard_mask = torch.zeros_like(gate_weights)
        hard_mask.scatter_(-1, topk_idx, 1.0)  # [B, N, code_dim]

        # STE: forward uses hard mask, backward through soft weights
        soft_mask = gate_weights * hard_mask
        ste_mask = hard_mask - soft_mask.detach() + soft_mask

        # Apply mask
        sparse_codes = codes * ste_mask

        # Compute and store subspace routing statistics (mean over batch and patches)
        wav_mask = hard_mask[..., :self.n_wavelet_dims]  # [B, N, n_wavelet_dims]
        amp_mask = hard_mask[..., self.n_wavelet_dims:]  # [B, N, n_amplitude_dims]
        wav_active = wav_mask.sum(dim=-1).mean()  # scalar
        amp_active = amp_mask.sum(dim=-1).mean()  # scalar

        # Routing entropy (prevent collapse)
        eps = 1e-7
        p_wav = wav_active / k
        p_amp = amp_active / k
        routing_entropy = -(p_wav * torch.log(p_wav + eps) + p_amp * torch.log(p_amp + eps))

        self.last_routing_stats = {
            'wav_active': wav_active,
            'amp_active': amp_active,
            'routing_entropy': routing_entropy
        }

        return sparse_codes, gate_weights


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
    - Sparsity (per_level or per_patch selectable via config)
    - Optional wavelet gating (Type A: fixed DWT/IDWT with learned gates)
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
        attn_config: Optional[Dict[str, Any]] = None,
        sparsity_mode: str = "per_level",
        wavelet_gating: bool = False,
        n_wavelet_dims: Optional[int] = None
    ):
        super().__init__()
        self.n_levels = n_levels
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size * 3
        self.hidden_dim = hidden_dim
        self.code_dim = code_dim
        self.k_per_patch = k_per_patch
        self.n_layers = n_layers
        self.sparsity_mode = sparsity_mode
        self.wavelet_gating = wavelet_gating
        self.n_wavelet_dims = n_wavelet_dims

        # Register as buffer to avoid Python float in traced code
        self.register_buffer("residual_scale", torch.tensor(residual_scale))
        self.register_buffer("one", torch.tensor(1.0))

        # Default attention config - fallback only, prefer TOML config
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

        # Per-level encoder/decoder - wavelet_gating is passed as parameter
        self.encoders = nn.ModuleList([
            SwiGLUEncoder(
                self.patch_dim, hidden_dim, code_dim, k_per_patch,
                n_layers, attn_config, sparsity_mode,
                wavelet_gating=wavelet_gating, patch_size=patch_size,
                n_wavelet_dims=n_wavelet_dims
            )
            for _ in range(n_levels)
        ])
        self.decoders = nn.ModuleList([
            SwiGLUDecoder(
                code_dim, hidden_dim, self.patch_dim, n_layers, attn_config,
                wavelet_gating=wavelet_gating, patch_size=patch_size,
                n_wavelet_dims=n_wavelet_dims
            )
            for _ in range(n_levels)
        ])

        wavelet_str = f", wavelet=True, n_wav_dims={n_wavelet_dims or code_dim // 2}" if wavelet_gating else ""
        print(f"[SwiGLUFSQAutoencoder] {n_levels} levels, code_dim={code_dim}, k={k_per_patch}, sparsity={sparsity_mode}{wavelet_str}")
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
        grid_shape: Optional[Tuple[int, int]] = None,
        k_override: Optional[int] = None
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
            k_override: optional override for k sparsity (for k-annealing during training)

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
        routing_stats_list = []  # Subspace routing stats (if wavelet_gating)
        cumulative_recon = torch.zeros_like(patches)

        for level in range(self.n_levels):
            # Residual with .detach() - critical for independent per-level learning
            if level > 0:
                residual = (patches - cumulative_recon.detach()) * self.residual_scale
            else:
                residual = patches

            # Encode - all encoders return (sparse_codes, gate_weights)
            sparse_codes, gate_weights = self.encoders[level](
                residual, grid_shape, encoder_masks, k_override=k_override
            )
            codes_list.append(sparse_codes)
            masks_list.append(gate_weights)

            # Collect routing stats from sparsity module's stored state (wavelet_gating only)
            if self.wavelet_gating:
                routing_stats_list.append(self.encoders[level].sparsity.last_routing_stats)

            # Decode - subspace-routed decoder returns just patches (sum, not blend)
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

        result = {
            'recon': level_recons[-1],
            'level_recons': level_recons,
            'codes': codes_list,
            'dim_masks': masks_list,
            'sparsity': sparsity,
            'grid_shape': grid_shape
        }

        # Add subspace routing statistics if enabled
        if self.wavelet_gating and routing_stats_list:
            # Aggregate routing stats across levels
            wav_active_mean = torch.stack([s['wav_active'] for s in routing_stats_list]).mean()
            amp_active_mean = torch.stack([s['amp_active'] for s in routing_stats_list]).mean()
            routing_entropy_mean = torch.stack([s['routing_entropy'] for s in routing_stats_list]).mean()

            result['routing_stats'] = routing_stats_list
            result['wav_active_mean'] = wav_active_mean
            result['amp_active_mean'] = amp_active_mean
            result['routing_entropy_mean'] = routing_entropy_mean

        return result

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

            # Encode with pre-quant values - all encoders return (sparse_codes, soft_weights, logits)
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

            # Apply per-patch sparsity - all sparsity modules return (sparse_codes, soft_weights)
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

    # =========================================================================
    # Visualization / Ablation Methods
    # =========================================================================

    def decode_first_k_levels(
        self,
        codes_list: List[torch.Tensor],
        grid_shape: Tuple[int, int],
        k_levels: int,
        decoder_masks: Optional[List] = None
    ) -> torch.Tensor:
        """
        Decode using only the first k levels (for residual layer visualization).

        Args:
            codes_list: list of [B, N, code_dim] sparse codes per level
            grid_shape: (GH, GW) grid dimensions
            k_levels: number of levels to use (1 to n_levels)
            decoder_masks: optional pre-built decoder masks

        Returns:
            recon: [B, C, H, W] reconstruction from first k levels only
        """
        device = codes_list[0].device

        if decoder_masks is None:
            decoder_masks = self.decoders[0].transformer.build_masks(grid_shape, device)

        cumulative_recon = None

        for level in range(min(k_levels, len(codes_list))):
            codes = codes_list[level]
            decoded = self.decoders[level](codes, grid_shape, decoder_masks)

            if level > 0:
                decoded = decoded / self.residual_scale

            if cumulative_recon is None:
                cumulative_recon = decoded
            else:
                cumulative_recon = cumulative_recon + decoded

        return self.unpatchify(cumulative_recon, grid_shape)

    def decode_with_ablation(
        self,
        codes_list: List[torch.Tensor],
        grid_shape: Tuple[int, int],
        ablate_wavelet: float = 0.0,
        ablate_amplitude: float = 0.0,
        decoder_masks: Optional[List] = None,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Decode with stochastic subspace ablation for diagnostic visualization.

        Only works when wavelet_gating=True. For standard decoders, returns normal decode.

        Args:
            codes_list: list of [B, N, code_dim] sparse codes per level
            grid_shape: (GH, GW) grid dimensions
            ablate_wavelet: ablation rate [0,1] for wavelet subspace (0=none, 1=full knockout)
            ablate_amplitude: ablation rate [0,1] for amplitude subspace
            decoder_masks: optional pre-built decoder masks
            deterministic: if True, zero fixed fraction; if False, Bernoulli per-element

        Returns:
            recon: [B, C, H, W] reconstruction with ablated subspace(s)
        """
        if not self.wavelet_gating:
            return self.decode(codes_list, grid_shape, decoder_masks)

        device = codes_list[0].device

        if decoder_masks is None:
            decoder_masks = self.decoders[0].transformer.build_masks(grid_shape, device)

        cumulative_recon = None

        for level, codes in enumerate(codes_list):
            # Use ablated forward pass
            decoded = self.decoders[level].forward_ablated(
                codes, grid_shape, ablate_wavelet, ablate_amplitude, decoder_masks, deterministic
            )

            if level > 0:
                decoded = decoded / self.residual_scale

            if cumulative_recon is None:
                cumulative_recon = decoded
            else:
                cumulative_recon = cumulative_recon + decoded

        return self.unpatchify(cumulative_recon, grid_shape)

    def decode_with_subspace_contributions(
        self,
        codes_list: List[torch.Tensor],
        grid_shape: Tuple[int, int],
        decoder_masks: Optional[List] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Decode and return individual subspace contributions per level.

        Only works when wavelet_gating=True. Returns empty dict for standard decoders.

        Args:
            codes_list: list of [B, N, code_dim] sparse codes per level
            grid_shape: (GH, GW) grid dimensions
            decoder_masks: optional pre-built decoder masks

        Returns:
            Dict with:
                'recon': [B, C, H, W] combined reconstruction
                'wav_contribution': [B, C, H, W] total wavelet pathway contribution
                'amp_contribution': [B, C, H, W] total amplitude pathway contribution
                'per_level_wav': list of [B, C, H, W] wavelet contribution per level
                'per_level_amp': list of [B, C, H, W] amplitude contribution per level
        """
        if not self.wavelet_gating:
            recon = self.decode(codes_list, grid_shape, decoder_masks)
            return {'recon': recon}

        device = codes_list[0].device

        if decoder_masks is None:
            decoder_masks = self.decoders[0].transformer.build_masks(grid_shape, device)

        cumulative_total = None
        cumulative_wav = None
        cumulative_amp = None
        per_level_wav = []
        per_level_amp = []

        for level, codes in enumerate(codes_list):
            # Get individual contributions
            decoded, wav_patches, amp_patches = self.decoders[level].forward_with_contributions(
                codes, grid_shape, decoder_masks
            )

            if level > 0:
                decoded = decoded / self.residual_scale
                wav_patches = wav_patches / self.residual_scale
                amp_patches = amp_patches / self.residual_scale

            # Unpatchify for visualization
            wav_img = self.unpatchify(wav_patches, grid_shape)
            amp_img = self.unpatchify(amp_patches, grid_shape)
            per_level_wav.append(wav_img)
            per_level_amp.append(amp_img)

            if cumulative_total is None:
                cumulative_total = decoded
                cumulative_wav = wav_patches
                cumulative_amp = amp_patches
            else:
                cumulative_total = cumulative_total + decoded
                cumulative_wav = cumulative_wav + wav_patches
                cumulative_amp = cumulative_amp + amp_patches

        return {
            'recon': self.unpatchify(cumulative_total, grid_shape),
            'wav_contribution': self.unpatchify(cumulative_wav, grid_shape),
            'amp_contribution': self.unpatchify(cumulative_amp, grid_shape),
            'per_level_wav': per_level_wav,
            'per_level_amp': per_level_amp,
        }

    def subspace_sensitivity_sweep(
        self,
        images: torch.Tensor,
        ablation_rates: Optional[List[float]] = None,
        n_trials: int = 5,
        grid_shape: Optional[Tuple[int, int]] = None,
        encoder_masks: Optional[List] = None,
        decoder_masks: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Evaluate subspace sensitivity by sweeping ablation rates.

        For each ablation rate, measures reconstruction MSE when:
        1. Only wavelet is ablated (amplitude intact)
        2. Only amplitude is ablated (wavelet intact)
        3. Both ablated at same rate

        This reveals how much each subspace contributes to reconstruction quality.

        Args:
            images: [B, C, H, W] input images
            ablation_rates: list of ablation fractions to test (default: 0 to 1 in 11 steps)
            n_trials: number of stochastic trials per rate (averaged)
            grid_shape: optional (GH, GW)
            encoder_masks: optional pre-built encoder masks
            decoder_masks: optional pre-built decoder masks

        Returns:
            Dict with:
                'ablation_rates': list of tested rates
                'mse_baseline': MSE with no ablation
                'mse_wav_ablated': list of MSE when wav ablated at each rate
                'mse_amp_ablated': list of MSE when amp ablated at each rate
                'mse_both_ablated': list of MSE when both ablated at each rate
                'd_mse_d_wav': gradient of MSE w.r.t. wavelet ablation
                'd_mse_d_amp': gradient of MSE w.r.t. amplitude ablation
        """
        if not self.wavelet_gating:
            raise ValueError("Sensitivity sweep requires wavelet_gating=True")

        if ablation_rates is None:
            ablation_rates = [i / 10.0 for i in range(11)]  # 0.0, 0.1, ..., 1.0

        B = images.shape[0]
        p = self.patch_size

        if grid_shape is None:
            H, W = images.shape[2], images.shape[3]
            grid_shape = (H // p, W // p)

        device = images.device

        if encoder_masks is None or decoder_masks is None:
            encoder_masks, decoder_masks = self.build_masks(grid_shape, device)

        # Encode once (deterministic)
        codes_list = self.encode(images, grid_shape=grid_shape,
                                 encoder_masks=encoder_masks, decoder_masks=decoder_masks)

        # Baseline MSE (no ablation)
        baseline_recon = self.decode(codes_list, grid_shape, decoder_masks)
        mse_baseline = F.mse_loss(baseline_recon, images).item()

        mse_wav_ablated = []
        mse_amp_ablated = []
        mse_both_ablated = []

        for rate in ablation_rates:
            if rate == 0.0:
                # No ablation = baseline
                mse_wav_ablated.append(mse_baseline)
                mse_amp_ablated.append(mse_baseline)
                mse_both_ablated.append(mse_baseline)
            else:
                # Average over multiple stochastic trials
                wav_mses = []
                amp_mses = []
                both_mses = []

                for _ in range(n_trials):
                    # Wavelet ablated only
                    recon_wav = self.decode_with_ablation(
                        codes_list, grid_shape, ablate_wavelet=rate, ablate_amplitude=0.0,
                        decoder_masks=decoder_masks, deterministic=False
                    )
                    wav_mses.append(F.mse_loss(recon_wav, images).item())

                    # Amplitude ablated only
                    recon_amp = self.decode_with_ablation(
                        codes_list, grid_shape, ablate_wavelet=0.0, ablate_amplitude=rate,
                        decoder_masks=decoder_masks, deterministic=False
                    )
                    amp_mses.append(F.mse_loss(recon_amp, images).item())

                    # Both ablated
                    recon_both = self.decode_with_ablation(
                        codes_list, grid_shape, ablate_wavelet=rate, ablate_amplitude=rate,
                        decoder_masks=decoder_masks, deterministic=False
                    )
                    both_mses.append(F.mse_loss(recon_both, images).item())

                mse_wav_ablated.append(sum(wav_mses) / n_trials)
                mse_amp_ablated.append(sum(amp_mses) / n_trials)
                mse_both_ablated.append(sum(both_mses) / n_trials)

        # Compute gradients (d_mse / d_ablation_rate)
        # Using central differences where possible
        def compute_gradient(mse_list, rates):
            grads = []
            for i in range(len(rates)):
                if i == 0:
                    # Forward difference
                    grad = (mse_list[1] - mse_list[0]) / (rates[1] - rates[0]) if len(rates) > 1 else 0
                elif i == len(rates) - 1:
                    # Backward difference
                    grad = (mse_list[i] - mse_list[i-1]) / (rates[i] - rates[i-1])
                else:
                    # Central difference
                    grad = (mse_list[i+1] - mse_list[i-1]) / (rates[i+1] - rates[i-1])
                grads.append(grad)
            return grads

        return {
            'ablation_rates': ablation_rates,
            'mse_baseline': mse_baseline,
            'mse_wav_ablated': mse_wav_ablated,
            'mse_amp_ablated': mse_amp_ablated,
            'mse_both_ablated': mse_both_ablated,
            'd_mse_d_wav': compute_gradient(mse_wav_ablated, ablation_rates),
            'd_mse_d_amp': compute_gradient(mse_amp_ablated, ablation_rates),
        }


# =============================================================================
# Interface Wrappers for Integration with src/model.py
# =============================================================================

class SwiGLUPatchEmbedder(nn.Module):
    """
    Wrapper to match ContextualPatchEmbedder interface for latent diffusion.

    The main model expects:
    - .stride attribute
    - .n_attn_layers attribute
    - forward(x, logsnr_map, ...) -> (z, grid_shape)

    IMPORTANT: This embedder produces N×L tokens (N patches × L levels), not N tokens.
    Each token is one level of one patch, projected via latent_code_proj (128 → embed_dim).
    The shape returned is (GH, GW, n_levels) to capture the 3D structure.

    Note: logsnr_map is used for conditioning each token.
    """
    def __init__(self, ae: SwiGLUFSQAutoencoder, embed_dim: int):
        super().__init__()
        self.ae = ae
        self.stride = ae.patch_size
        self.n_attn_layers = ae.n_layers
        self.embed_dim = embed_dim
        self.code_dim = ae.code_dim
        self.n_levels = ae.n_levels

        # Per-token projection (code_dim → embed_dim) - TRAINED in latent diffusion
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
        logsnr_map: torch.Tensor = None,
        block_masks = None
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        Encode images to latent embeddings for diffusion.

        Args:
            x: [C, H, W] or [B, C, H, W] images
            logsnr_map: [1, H, W] or [B, 1, H, W] logsnr field for conditioning
            block_masks: IGNORED (we use our own cached masks)

        Returns:
            z: [N*L, D] or [B, N*L, D] embeddings (N patches × L levels)
            shape: (GH, GW, n_levels) - 3D shape for topology
        """
        single = x.dim() == 3
        if single:
            x = x.unsqueeze(0)
            if logsnr_map is not None and logsnr_map.dim() == 3:
                logsnr_map = logsnr_map.unsqueeze(0)

        B, C, H, W = x.shape
        p = self.ae.patch_size
        GH, GW = H // p, W // p
        grid_shape = (GH, GW)
        n_patches = GH * GW
        device = x.device

        encoder_masks, decoder_masks = self._get_masks(grid_shape, device)
        codes_list = self.ae.encode(x, grid_shape=grid_shape, encoder_masks=encoder_masks, decoder_masks=decoder_masks)

        # Stack levels and flatten to N×L tokens: [B, N, L, code_dim] -> [B, N*L, code_dim]
        codes_stacked = torch.stack(codes_list, dim=2)  # [B, N, L, code_dim]
        codes_flat = codes_stacked.view(B, n_patches * self.n_levels, self.code_dim)

        # Project per-token to embed_dim
        z = self.latent_code_proj(codes_flat)  # [B, N*L, embed_dim]

        # Add logsnr conditioning if provided
        if logsnr_map is not None:
            # Pool logsnr to per-patch, expand to per-level
            logsnr_pooled = F.avg_pool2d(logsnr_map, kernel_size=p, stride=p)
            logsnr_patches = logsnr_pooled.flatten(2).transpose(1, 2)  # [B, N, 1]
            logsnr_flat = logsnr_patches.repeat(1, self.n_levels, 1)  # [B, N*L, 1]
            logsnr_features = self.logsnr_proj(logsnr_flat)  # [B, N*L, embed_dim]
            z = z + logsnr_features

        # Return 3D shape: (GH, GW, n_levels)
        shape_3d = (GH, GW, self.n_levels)

        if single:
            return z.squeeze(0), shape_3d
        return z, shape_3d


class SwiGLUPatchUnembedder(nn.Module):
    """
    Wrapper to match ContextualPatchUnembedder interface for latent diffusion.

    IMPORTANT: This unembedder expects N×L tokens (N patches × L levels), not N tokens.
    Each token is projected via latent_code_unproj (embed_dim → 128), then
    quantized and decoded through the FSQ decoder.

    Outputs [C+1, H, W] where the extra channel is predicted logsnr.
    """
    def __init__(self, ae: SwiGLUFSQAutoencoder, embedder: SwiGLUPatchEmbedder):
        super().__init__()
        self.ae = ae
        self.embedder = embedder
        self.patch_size = ae.patch_size
        self.n_attn_layers = ae.n_layers
        self.n_levels = ae.n_levels
        self.code_dim = ae.code_dim

        # Per-token projection (embed_dim → code_dim) - TRAINED in latent diffusion
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
        block_masks = None
    ) -> torch.Tensor:
        """
        Decode latent embeddings to images.

        Args:
            z: [N*L, D] or [B, N*L, D] embeddings (N patches × L levels)
            shape: (GH, GW, n_levels) - 3D shape from embedder
            block_masks: IGNORED

        Returns:
            recon: [C+1, H, W] or [B, C+1, H, W] - RGB + logsnr channel
        """
        single = z.dim() == 2
        if single:
            z = z.unsqueeze(0)

        B, total_tokens, D = z.shape
        device = z.device
        P = self.patch_size

        # Parse shape - can be (GH, GW, n_levels) or (GH, GW) for backwards compat
        if len(shape) == 3:
            GH, GW, n_levels = shape
        else:
            GH, GW = shape
            n_levels = self.n_levels

        grid_shape = (GH, GW)
        n_patches = GH * GW

        # Project each token to code_dim
        codes_flat = self.latent_code_unproj(z)  # [B, N*L, code_dim]

        # Reshape to [B, N, L, code_dim] then split to per-level list
        codes_stacked = codes_flat.view(B, n_patches, n_levels, self.code_dim)
        prequant_list = [codes_stacked[:, :, lv, :] for lv in range(n_levels)]

        # Quantize and decode through FSQ
        decoder_masks = self._get_masks(grid_shape, device)
        recon = self.ae.quantize_and_decode(prequant_list, grid_shape, decoder_masks)  # [B, C, H, W]

        # Decode logsnr: average across levels, reshape to spatial
        # z is [B, N*L, D], reshape to [B, N, L, D], mean over L
        z_reshaped = z.view(B, n_patches, n_levels, D)
        z_per_patch = z_reshaped.mean(dim=2)  # [B, N, D]
        logsnr_per_patch = self.logsnr_decoder(z_per_patch)  # [B, N, 1]
        logsnr_per_patch = logsnr_per_patch.view(B, GH, GW, 1).permute(0, 3, 1, 2)  # [B, 1, GH, GW]
        # Upsample to match pixel resolution
        logsnr_spatial = logsnr_per_patch.repeat_interleave(P, dim=2).repeat_interleave(P, dim=3)  # [B, 1, H, W]

        # Concat RGB + logsnr to match expected [C+1, H, W] interface
        recon_with_logsnr = torch.cat([recon, logsnr_spatial], dim=1)  # [B, C+1, H, W]

        if single:
            return recon_with_logsnr.squeeze(0)
        return recon_with_logsnr
