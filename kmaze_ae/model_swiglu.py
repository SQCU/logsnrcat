# kmaze_ae/model_swiglu.py - SwiGLU-style FSQ Autoencoder with 2D RoPE
"""
Port of the clean_impl_swiglu_reference to the main codebase conventions.

Key changes from model_sparse_dim.py:
- 2D RoPE position encoding in attention (split head_dim for x/y)
- Binary FSQ (1-bit) instead of 3-bit
- Level-global dimension selection instead of per-patch sparsity
- 3x3 neighborhood is sliding window attention with window_size=1.5

Uses flex_attention everywhere. Masks are pre-computed outside torch.compile.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask
from typing import Tuple, Optional, Dict, Any, List

from src.embedders import FourierFeatures
from src.blocks import SwiGLU, init_linear, init_layer_norm, _uses_registers


# =============================================================================
# 2D Rotary Position Embeddings
# =============================================================================

def get_2d_rope_freqs(
    grid_shape: Tuple[int, int],
    head_dim: int,
    device: torch.device,
    base: float = 10000.0,
    dtype: torch.dtype = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate 2D rotary position embedding cos/sin tensors.

    Split head_dim in half: first half for x-axis, second half for y-axis.
    Returns precomputed cos/sin tensors that can be cached and reused.

    Args:
        grid_shape: (H, W) grid dimensions
        head_dim: Dimension of each attention head
        device: Target device
        base: Base frequency for position encoding
        dtype: Output dtype (computed in float32 for precision, cast at end)

    Returns:
        cos: [1, 1, H*W, head_dim] cosine tensor for RoPE
        sin: [1, 1, H*W, head_dim] sine tensor for RoPE
    """
    H, W = grid_shape
    half_dim = head_dim // 2
    quarter_dim = half_dim // 2

    # Compute in float32 for precision
    inv_freq = 1.0 / (base ** (torch.arange(0, quarter_dim, device=device, dtype=torch.float32) / quarter_dim))

    # Grid positions
    pos_h = torch.arange(H, device=device).float()
    pos_w = torch.arange(W, device=device).float()

    # Outer product: [H, quarter_dim] and [W, quarter_dim]
    freqs_h = torch.outer(pos_h, inv_freq)  # [H, quarter_dim]
    freqs_w = torch.outer(pos_w, inv_freq)  # [W, quarter_dim]

    # Expand to 2D grid: [H, W, quarter_dim]
    freqs_h = freqs_h.unsqueeze(1).expand(-1, W, -1)  # [H, W, quarter_dim]
    freqs_w = freqs_w.unsqueeze(0).expand(H, -1, -1)  # [H, W, quarter_dim]

    # Flatten to sequence: [H*W, quarter_dim]
    freqs_h = freqs_h.reshape(-1, freqs_h.shape[-1])
    freqs_w = freqs_w.reshape(-1, freqs_w.shape[-1])

    # Interleave for sin/cos pattern: [seq_len, half_dim]
    freqs_h = torch.stack([freqs_h, freqs_h], dim=-1).flatten(-2)
    freqs_w = torch.stack([freqs_w, freqs_w], dim=-1).flatten(-2)

    # Concat x and y: [seq_len, head_dim]
    freqs = torch.cat([freqs_h, freqs_w], dim=-1)

    # Compute cos/sin and add batch/head dims for broadcasting
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, N, D]
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)  # [1, 1, N, D]

    # Cast to requested dtype (or keep float32 if not specified)
    if dtype is not None:
        cos = cos.to(dtype)
        sin = sin.to(dtype)

    return cos, sin


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply rotary embeddings to x using precomputed cos/sin.

    Args:
        x: [B, n_heads, seq_len, head_dim]
        cos: [1, 1, seq_len, head_dim] precomputed cosines
        sin: [1, 1, seq_len, head_dim] precomputed sines

    Returns:
        x_rot: [B, n_heads, seq_len, head_dim] with rotary encoding applied
    """
    # Rotate pairs: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
    # Use view instead of reshape to avoid copy
    x1 = x[..., ::2]   # Even indices
    x2 = x[..., 1::2]  # Odd indices
    c1 = cos[..., ::2]
    c2 = cos[..., 1::2]
    s1 = sin[..., ::2]
    s2 = sin[..., 1::2]

    # Interleave rotated results
    out = torch.empty_like(x)
    out[..., ::2] = x1 * c1 - x2 * s1
    out[..., 1::2] = x1 * s2 + x2 * c2
    return out


# =============================================================================
# Attention with 2D RoPE and flex_attention
# =============================================================================

class Attention2DRoPE(nn.Module):
    """
    GQA attention with 2D RoPE using flex_attention.

    Features:
    - Grouped-query attention (n_kv_heads < n_heads)
    - 2D rotary position embeddings
    - Post-attention sigmoid gating (query-dependent)
    - Uses flex_attention for sparse mask support
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        rope_base: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = dim // n_heads
        self.rope_base = rope_base

        assert dim % n_heads == 0
        assert n_heads % self.n_kv_heads == 0
        self.heads_per_kv = n_heads // self.n_kv_heads

        # Projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.head_dim * self.n_kv_heads, bias=False)
        self.v_proj = nn.Linear(dim, self.head_dim * self.n_kv_heads, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Post-attention gate (query-dependent, per-head)
        self.attn_gate = nn.Parameter(torch.zeros(n_heads, self.head_dim))
        self.attn_gate_bias = nn.Parameter(torch.zeros(n_heads))

        self.param_init()

    def param_init(self):
        init_linear(self.q_proj)
        init_linear(self.k_proj)
        init_linear(self.v_proj)
        init_linear(self.out_proj)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos_sin: Tuple[torch.Tensor, torch.Tensor],
        block_mask: Optional[BlockMask] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] input tensor
            rope_cos_sin: (cos, sin) tuple of [1, 1, N, head_dim] tensors (required)
            block_mask: Optional BlockMask for sparse attention

        Returns:
            [B, N, D] output tensor
        """
        B, N, D = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply 2D RoPE to Q and K (always applied - no branch)
        cos, sin = rope_cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # GQA: expand KV heads
        if self.heads_per_kv > 1:
            k = k.repeat_interleave(self.heads_per_kv, dim=1)
            v = v.repeat_interleave(self.heads_per_kv, dim=1)

        # Attention via flex_attention
        out = flex_attention(q, k, v, block_mask=block_mask)

        # Post-attention sigmoid gate (query-dependent)
        # q: [B, n_heads, N, head_dim], gate: [n_heads, head_dim]
        gate_logits = torch.einsum('bhnd,hd->bhn', q, self.attn_gate)
        gate_logits = gate_logits + self.attn_gate_bias.view(1, -1, 1)
        gate = torch.sigmoid(gate_logits).unsqueeze(-1)  # [B, n_heads, N, 1]
        out = out * gate

        # Project output
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class TransformerBlock2DRoPE(nn.Module):
    """
    Transformer block with 2D RoPE and SwiGLU FFN.

    Pre-norm architecture matching the clean_impl.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        mlp_ratio: float = 4.0,
        rope_base: float = 10000.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention2DRoPE(dim, n_heads, n_kv_heads, rope_base)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = SwiGLU(dim, hidden_dim)
        self.param_init()

    def param_init(self):
        init_layer_norm(self.norm1)
        self.attn.param_init()
        init_layer_norm(self.norm2)
        self.ffn.param_init()

    def forward(
        self,
        x: torch.Tensor,
        rope_cos_sin: Tuple[torch.Tensor, torch.Tensor],
        block_mask: Optional[BlockMask] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_cos_sin, block_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerStack2DRoPE(nn.Module):
    """
    Stack of transformer blocks with 2D RoPE.

    Manages RoPE frequency caching and mask building.
    """

    def __init__(
        self,
        dim: int,
        n_layers: int = 4,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        mlp_ratio: float = 4.0,
        rope_base: float = 10000.0,
        attn_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.head_dim = dim // n_heads
        self.rope_base = rope_base

        # Attention config
        if attn_config is None:
            attn_config = {'mode': 'sliding', 'window_size': 2}
        self.attn_config = attn_config
        self.mode = attn_config.get('mode', 'sliding')
        self.window_size = attn_config.get('window_size', 2)
        self.n_global_tokens = attn_config.get('n_global_tokens', 0)

        # Learnable register tokens for bigbird modes
        self.uses_registers = _uses_registers(self.mode)
        if self.uses_registers:
            self.register_tokens = nn.Parameter(
                torch.randn(1, self.n_global_tokens, dim) * 0.02
            )

        # Layers
        self.layers = nn.ModuleList([
            TransformerBlock2DRoPE(dim, n_heads, n_kv_heads, mlp_ratio, rope_base)
            for _ in range(n_layers)
        ])

        # Caches - stores (cos, sin) tuples
        self._rope_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._mask_cache: Dict[Tuple[int, int], List[Optional[BlockMask]]] = {}

    def _get_rope_cos_sin(
        self,
        grid_shape: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get or build cached RoPE cos/sin tensors."""
        key = (grid_shape, device, dtype)
        if key not in self._rope_cache:
            self._rope_cache[key] = get_2d_rope_freqs(
                grid_shape, self.head_dim, device, self.rope_base, dtype=dtype
            )
        return self._rope_cache[key]

    def build_masks(
        self,
        grid_shape: Tuple[int, int],
        device: torch.device
    ) -> List[Optional[BlockMask]]:
        """
        Build attention masks for all layers.

        Call OUTSIDE torch.compile to avoid inductor issues.
        """
        from src.context_manager import get_encoder_mask_for_layer

        return [
            get_encoder_mask_for_layer(grid_shape, i, self.attn_config, device)
            for i in range(self.n_layers)
        ]

    def forward(
        self,
        x: torch.Tensor,
        rope_cos_sin: Tuple[torch.Tensor, torch.Tensor],
        block_masks: List[Optional[BlockMask]]
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] input patches
            rope_cos_sin: Pre-computed (cos, sin) tuple (required)
            block_masks: Pre-built masks list, one per layer (required, use [None]*n_layers for no masking)
        """
        B, N, D = x.shape

        # Prepend register tokens for bigbird modes (branch on module constant, OK for compile)
        if self.uses_registers:
            registers = self.register_tokens.expand(B, -1, -1)
            x = torch.cat([registers, x], dim=1)
            # Pad RoPE with zeros for registers: [1, 1, n_reg, D]
            cos, sin = rope_cos_sin
            n_reg = self.n_global_tokens
            reg_zeros = torch.zeros(1, 1, n_reg, self.head_dim, device=x.device, dtype=cos.dtype)
            rope_cos_sin = (
                torch.cat([reg_zeros, cos], dim=2),
                torch.cat([reg_zeros, sin], dim=2)
            )

        # Run layers (no conditionals - masks list always provided)
        for i, layer in enumerate(self.layers):
            x = layer(x, rope_cos_sin, block_masks[i])

        # Remove register tokens (branch on module constant, OK for compile)
        if self.uses_registers:
            x = x[:, self.n_global_tokens:]

        return x


# =============================================================================
# FSQ and Sparsity
# =============================================================================

class BinaryFSQ(nn.Module):
    """Binary FSQ with STE: sigmoid -> threshold -> {-1, 1}."""

    def __init__(self):
        super().__init__()
        self.register_buffer("threshold", torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        soft = torch.sigmoid(x)
        hard = (soft > self.threshold).float()
        # STE
        hard = hard - soft.detach() + soft
        # Map {0, 1} -> {-1, 1}
        return hard * 2.0 - 1.0


class LevelGlobalSparsity(nn.Module):
    """
    Level-global dimension sparsity.

    Same dimensions are selected for all patches at a level (not per-patch).
    Uses learnable logits to determine which dimensions are active.

    This is different from PerDimSparsity which learns per-patch gates.
    """

    def __init__(self, code_dim: int = 256, k: int = 8):
        super().__init__()
        self.code_dim = code_dim
        self.k = k

        # Learnable dimension importance logits (level-global)
        self.dim_logits = nn.Parameter(torch.randn(code_dim) * 0.1)

    def forward(self, codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            codes: [B, N, code_dim] quantized codes

        Returns:
            sparse_codes: [B, N, code_dim] with only k dims active
            mask: [code_dim] binary mask indicating active dims
        """
        # Top-k dimension selection (same for all patches)
        _, topk_idx = self.dim_logits.topk(self.k)

        # Build mask (allocation is small: code_dim floats = ~1KB)
        mask = torch.zeros(self.code_dim, device=codes.device, dtype=codes.dtype)
        mask.scatter_(0, topk_idx, 1.0)

        # Apply mask
        sparse_codes = codes * mask.view(1, 1, -1)

        return sparse_codes, mask


# =============================================================================
# Encoder and Decoder
# =============================================================================

class SwiGLUEncoder(nn.Module):
    """
    Encoder with 2D RoPE transformer and binary FSQ + level-global sparsity.
    """

    def __init__(
        self,
        patch_dim: int = 768,
        hidden_dim: int = 256,
        code_dim: int = 256,
        k_active: int = 8,
        fourier_dim: int = 16,
        n_layers: int = 4,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        attn_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.fourier_dim = fourier_dim
        self.logsnr_fourier = FourierFeatures(fourier_dim, scale=0.5)

        # Input projection: patch_dim + fourier_dim -> hidden_dim
        self.input_proj = nn.Linear(patch_dim + fourier_dim, hidden_dim)

        # Transformer with 2D RoPE
        self.transformer = TransformerStack2DRoPE(
            dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            attn_config=attn_config
        )

        # Code projection
        self.code_proj = nn.Linear(hidden_dim, code_dim)

        # Quantization and sparsity
        self.fsq = BinaryFSQ()
        self.sparsity = LevelGlobalSparsity(code_dim, k_active)

    def forward(
        self,
        x: torch.Tensor,
        logsnr_patches: torch.Tensor,
        rope_cos_sin: Tuple[torch.Tensor, torch.Tensor],
        block_masks: List[Optional[BlockMask]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, patch_dim] patch features
            logsnr_patches: [B, N, 1] per-patch logsnr
            rope_cos_sin: Pre-computed (cos, sin) tuple for RoPE (required)
            block_masks: Pre-built masks list (required)

        Returns:
            sparse_codes: [B, N, code_dim] sparse quantized codes
            mask: [code_dim] active dimension mask
            pre_quant: [B, N, code_dim] pre-quantization values
        """
        # Encode logsnr
        logsnr_feat = self.logsnr_fourier(logsnr_patches)

        # Concat and project
        h = torch.cat([x, logsnr_feat], dim=-1)
        h = self.input_proj(h)

        # Transformer (no conditionals)
        h = self.transformer(h, rope_cos_sin, block_masks)

        # Code projection
        pre_quant = self.code_proj(h)

        # Quantize then sparsify
        codes = self.fsq(pre_quant)
        sparse_codes, mask = self.sparsity(codes)

        return sparse_codes, mask, pre_quant


class SwiGLUDecoder(nn.Module):
    """
    Decoder with 2D RoPE transformer.

    Uses sliding window attention with window_size=1.5 to implement
    the 3x3 neighborhood pattern from the clean_impl.
    """

    def __init__(
        self,
        code_dim: int = 256,
        hidden_dim: int = 256,
        patch_dim: int = 768,
        n_layers: int = 4,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        attn_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.input_proj = nn.Linear(code_dim, hidden_dim)

        # Transformer with 2D RoPE
        self.transformer = TransformerStack2DRoPE(
            dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            attn_config=attn_config
        )

        # Output projections
        self.output_proj = nn.Linear(hidden_dim, patch_dim)
        self.logsnr_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        codes: torch.Tensor,
        rope_cos_sin: Tuple[torch.Tensor, torch.Tensor],
        block_masks: List[Optional[BlockMask]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            codes: [B, N, code_dim] sparse codes
            rope_cos_sin: Pre-computed (cos, sin) tuple for RoPE (required)
            block_masks: Pre-built masks list (required)

        Returns:
            patches: [B, N, patch_dim] reconstructed patches
            logsnr_pred: [B, N, 1] predicted logsnr
        """
        h = self.input_proj(codes)
        h = self.transformer(h, rope_cos_sin, block_masks)
        patches = self.output_proj(h)
        logsnr_pred = self.logsnr_head(h)
        return patches, logsnr_pred


# =============================================================================
# Full Autoencoder
# =============================================================================

class SwiGLUFSQAutoencoder(nn.Module):
    """
    Multi-scale FSQ Autoencoder with SwiGLU-style attention and 2D RoPE.

    Key features:
    - Binary FSQ (1-bit per active dimension)
    - Level-global dimension sparsity (same dims active for all patches in level)
    - 2D RoPE position encoding
    - Sliding window attention (3x3 neighborhood via window_size=1.5)
    - Hierarchical residual refinement
    """

    def __init__(
        self,
        n_levels: int = 6,
        patch_size: int = 16,
        image_size: int = 256,
        hidden_dim: int = 128,
        code_dim: int = 256,
        k_active: int = 8,
        residual_scale: float = 2.0,
        fourier_dim: int = 16,
        n_layers: int = 1,
        n_heads: int = 4,
        n_kv_heads: Optional[int] = None,
        attn_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.n_levels = n_levels
        self.patch_size = patch_size
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.code_dim = code_dim
        self.k_active = k_active
        self.k_per_patch = k_active  # Alias for compatibility with training code
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads

        self.n_patches = (image_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * 3

        # Register constants as buffers
        self.register_buffer("residual_scale", torch.tensor(residual_scale))
        self.register_buffer("one", torch.tensor(1.0))

        # Default attention config: 3x3 neighborhood via sliding window
        if attn_config is None:
            attn_config = {
                'mode': 'sliding',
                'window_size': 2,  # Euclidean dist² ≤ 4 covers 3x3
                'n_global_tokens': 0,
                'n_query_heads': n_heads,
                'n_kv_heads': n_kv_heads or n_heads
            }
        self.attn_config = attn_config

        # Encoders and decoders (one per level)
        self.encoders = nn.ModuleList([
            SwiGLUEncoder(
                patch_dim=self.patch_dim,
                hidden_dim=hidden_dim,
                code_dim=code_dim,
                k_active=k_active,
                fourier_dim=fourier_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                attn_config=attn_config
            )
            for _ in range(n_levels)
        ])

        self.decoders = nn.ModuleList([
            SwiGLUDecoder(
                code_dim=code_dim,
                hidden_dim=hidden_dim,
                patch_dim=self.patch_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                attn_config=attn_config
            )
            for _ in range(n_levels)
        ])

        # Level-specific logsnr estimators
        self.level_logsnr_estimators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1)
            )
            for _ in range(n_levels - 1)
        ])

        # Centralized RoPE cache (shared across all encoders/decoders)
        self.head_dim = hidden_dim // n_heads
        self.rope_base = 10000.0
        self._rope_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

        print(f"[SwiGLUFSQAutoencoder] {n_levels} levels, code_dim={code_dim}, k={k_active}")
        print(f"  Attention: {attn_config.get('mode', 'local')}, window={attn_config.get('window_size', 1.5)}")

    def _get_rope_cos_sin(
        self,
        grid_shape: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get or compute centralized (cos, sin) tensors for RoPE."""
        key = (grid_shape, device, dtype)
        if key not in self._rope_cache:
            self._rope_cache[key] = get_2d_rope_freqs(
                grid_shape, self.head_dim, device, self.rope_base, dtype=dtype
            )
        return self._rope_cache[key]

    def build_masks(
        self,
        grid_shape: Tuple[int, int],
        device: torch.device
    ) -> Tuple[List[List], List[List]]:
        """
        Build masks for all encoders and decoders.

        Call OUTSIDE torch.compile.

        Returns:
            (encoder_masks, decoder_masks) - each is [n_levels][n_layers] list
        """
        encoder_masks = [
            self.encoders[i].transformer.build_masks(grid_shape, device)
            for i in range(self.n_levels)
        ]
        decoder_masks = [
            self.decoders[i].transformer.build_masks(grid_shape, device)
            for i in range(self.n_levels)
        ]
        return encoder_masks, decoder_masks

    def patchify(self, images: torch.Tensor, grid_shape: Tuple[int, int]) -> torch.Tensor:
        """Convert images to patches."""
        B = images.shape[0]
        p = self.patch_size
        n_h, n_w = grid_shape
        patches = images.view(B, 3, n_h, p, n_w, p)
        patches = patches.permute(0, 2, 4, 3, 5, 1).contiguous()
        return patches.view(B, n_h * n_w, self.patch_dim)

    def patchify_logsnr(
        self,
        logsnr_map: torch.Tensor,
        grid_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """Convert spatial logsnr to per-patch values."""
        if logsnr_map.dim() == 3:
            logsnr_map = logsnr_map.unsqueeze(1)
        B = logsnr_map.shape[0]
        p = self.patch_size
        n_h, n_w = grid_shape
        logsnr_patches = F.avg_pool2d(logsnr_map, kernel_size=p, stride=p)
        return logsnr_patches.view(B, n_h * n_w, 1)

    def unpatchify(self, patches: torch.Tensor, grid_shape: Tuple[int, int]) -> torch.Tensor:
        """Convert patches back to image."""
        B = patches.shape[0]
        p = self.patch_size
        h, w = grid_shape
        patches = patches.view(B, h, w, p, p, 3)
        patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
        return patches.view(B, 3, h * p, w * p)

    def unpatchify_logsnr(
        self,
        logsnr_patches: torch.Tensor,
        grid_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """Convert per-patch logsnr to spatial map."""
        B = logsnr_patches.shape[0]
        p = self.patch_size
        h, w = grid_shape
        logsnr_grid = logsnr_patches.view(B, h, w, 1).permute(0, 3, 1, 2)
        return F.interpolate(logsnr_grid, scale_factor=p, mode='nearest')

    def forward(
        self,
        images: torch.Tensor,
        logsnr_map: Optional[torch.Tensor] = None,
        encoder_masks: Optional[List[List[Optional[BlockMask]]]] = None,
        decoder_masks: Optional[List[List[Optional[BlockMask]]]] = None,
        rope_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        grid_shape: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Forward pass. Accepts optional params for convenience (inference path).

        NOTE: This method is NOT inside torch.compile - only the inner .transformer
        modules are compiled. Optional handling here is safe.

        Args:
            images: [B, C, H, W] input images
            logsnr_map: [B, 1, H, W] spatial logsnr field (default: zeros)
            encoder_masks: Pre-built encoder masks per level (default: build on demand)
            decoder_masks: Pre-built decoder masks per level (default: build on demand)
            rope_cos_sin: Pre-computed (cos, sin) tuple (default: compute on demand)
            grid_shape: (GH, GW) grid dimensions (default: infer from images)

        Returns:
            Dict with recon, level_recons, codes, masks, sparsity, logsnr_preds
        """
        B = images.shape[0]
        p = self.patch_size
        device = images.device

        # Infer grid_shape if not provided
        if grid_shape is None:
            H, W = images.shape[2], images.shape[3]
            grid_shape = (H // p, W // p)

        n_patches = grid_shape[0] * grid_shape[1]
        patches = self.patchify(images, grid_shape)

        # Handle logsnr
        if logsnr_map is None:
            logsnr_patches = torch.zeros(B, n_patches, 1, device=device, dtype=images.dtype)
        else:
            logsnr_patches = self.patchify_logsnr(logsnr_map, grid_shape)

        # Build masks if not provided
        if encoder_masks is None or decoder_masks is None:
            enc_masks, dec_masks = self.build_masks(grid_shape, device)
            encoder_masks = encoder_masks or enc_masks
            decoder_masks = decoder_masks or dec_masks

        # Compute RoPE if not provided
        # Use model weight dtype (bf16 if model was cast), not input dtype (often float32)
        if rope_cos_sin is None:
            model_dtype = self.encoders[0].input_proj.weight.dtype
            rope_cos_sin = self._get_rope_cos_sin(grid_shape, device, model_dtype)

        # Hierarchical encoding
        level_recons = []
        codes_list = []
        masks_list = []
        logsnr_preds = []
        level_logsnrs = []
        cumulative_recon = torch.zeros_like(patches)
        current_target = patches

        for level in range(self.n_levels):
            # Level-specific logsnr (branches on level index are OK - unrolled by compiler)
            if level == 0:
                level_logsnr = logsnr_patches
            else:
                level_logsnr = self.level_logsnr_estimators[level - 1](logsnr_patches)
            level_logsnrs.append(level_logsnr)

            # Residual
            if level > 0:
                residual = (current_target - cumulative_recon) * self.residual_scale
            else:
                residual = current_target

            # Encode (no conditionals - all params required)
            codes, mask, _ = self.encoders[level](
                residual, level_logsnr, rope_cos_sin, encoder_masks[level]
            )
            codes_list.append(codes)
            masks_list.append(mask)

            # Decode (no conditionals - all params required)
            decoded, logsnr_pred = self.decoders[level](
                codes, rope_cos_sin, decoder_masks[level]
            )
            logsnr_preds.append(logsnr_pred)

            if level > 0:
                decoded = decoded / self.residual_scale

            cumulative_recon = cumulative_recon + decoded
            level_recons.append(self.unpatchify(cumulative_recon, grid_shape))

        # Sparsity stats
        total_codes = codes_list[0].numel()
        nonzero_codes = sum((c != 0).sum() for c in codes_list)
        sparsity = self.one - (nonzero_codes / (total_codes * self.n_levels))

        return {
            'recon': level_recons[-1],
            'level_recons': level_recons,
            'codes': codes_list,
            'dim_masks': masks_list,
            'sparsity': sparsity,
            'logsnr_preds': logsnr_preds,
            'level_logsnrs': level_logsnrs,
            'logsnr_pred_map': self.unpatchify_logsnr(logsnr_preds[-1], grid_shape),
            'grid_shape': grid_shape
        }

    def encode(
        self,
        images: torch.Tensor,
        logsnr_map: Optional[torch.Tensor] = None,
        grid_shape: Optional[Tuple[int, int]] = None,
        encoder_masks: Optional[List[List]] = None,
        decoder_masks: Optional[List[List]] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Encode images to sparse codes (inference path, builds masks on demand)."""
        B = images.shape[0]
        p = self.patch_size
        device = images.device

        if grid_shape is None:
            H, W = images.shape[2], images.shape[3]
            grid_shape = (H // p, W // p)

        n_patches = grid_shape[0] * grid_shape[1]
        patches = self.patchify(images, grid_shape)

        if logsnr_map is None:
            logsnr_patches = torch.zeros(B, n_patches, 1, device=device, dtype=images.dtype)
        else:
            logsnr_patches = self.patchify_logsnr(logsnr_map, grid_shape)

        # Build masks if not provided (inference path only)
        if encoder_masks is None:
            encoder_masks = [
                self.encoders[i].transformer.build_masks(grid_shape, device)
                for i in range(self.n_levels)
            ]
        if decoder_masks is None:
            decoder_masks = [
                self.decoders[i].transformer.build_masks(grid_shape, device)
                for i in range(self.n_levels)
            ]

        # Compute RoPE - use model weight dtype, not input dtype
        model_dtype = self.encoders[0].input_proj.weight.dtype
        rope_cos_sin = self._get_rope_cos_sin(grid_shape, device, model_dtype)

        codes_list = []
        level_logsnrs = []
        cumulative_recon = torch.zeros_like(patches)
        current_target = patches

        for level in range(self.n_levels):
            if level == 0:
                level_logsnr = logsnr_patches
            else:
                level_logsnr = self.level_logsnr_estimators[level - 1](logsnr_patches)
            level_logsnrs.append(level_logsnr)

            if level > 0:
                residual = (current_target - cumulative_recon) * self.residual_scale
            else:
                residual = current_target

            # Positional args to match new signature
            codes, _, _ = self.encoders[level](
                residual, level_logsnr, rope_cos_sin, encoder_masks[level]
            )
            codes_list.append(codes)

            decoded, _ = self.decoders[level](
                codes, rope_cos_sin, decoder_masks[level]
            )

            if level > 0:
                decoded = decoded / self.residual_scale

            cumulative_recon = cumulative_recon + decoded

        return codes_list, level_logsnrs

    def decode(
        self,
        codes_list: List[torch.Tensor],
        grid_shape: Tuple[int, int],
        decoder_masks: Optional[List[List]] = None
    ) -> torch.Tensor:
        """Decode codes to image (inference path, builds masks on demand)."""
        device = codes_list[0].device

        if decoder_masks is None:
            decoder_masks = [
                self.decoders[i].transformer.build_masks(grid_shape, device)
                for i in range(self.n_levels)
            ]

        # Compute RoPE - use model weight dtype for consistency
        model_dtype = self.decoders[0].input_proj.weight.dtype
        rope_cos_sin = self._get_rope_cos_sin(grid_shape, device, model_dtype)

        cumulative_recon = None

        for level, codes in enumerate(codes_list):
            # Positional args to match new signature
            decoded, _ = self.decoders[level](
                codes, rope_cos_sin, decoder_masks[level]
            )

            if level > 0:
                decoded = decoded / self.residual_scale

            if cumulative_recon is None:
                cumulative_recon = decoded
            else:
                cumulative_recon = cumulative_recon + decoded

        return self.unpatchify(cumulative_recon, grid_shape)


# =============================================================================
# Interface Wrappers (matching SparseAEPatchEmbedder/Unembedder)
# =============================================================================

class SwiGLUPatchEmbedder(nn.Module):
    """
    Wraps SwiGLUFSQAutoencoder to match ContextualPatchEmbedder interface.
    """

    def __init__(self, ae: SwiGLUFSQAutoencoder, embed_dim: int = 256):
        super().__init__()
        self.ae = ae
        self.stride = ae.patch_size
        self.embed_dim = embed_dim
        self.n_attn_layers = ae.n_layers

        # Project concatenated codes to embed_dim
        total_code_dim = ae.code_dim * ae.n_levels
        self.code_proj = nn.Linear(total_code_dim, embed_dim)

        # Project level logsnrs to single logsnr (required by train.py)
        self.logsnr_proj = nn.Linear(ae.n_levels, 1)

        # Mask cache
        self._mask_cache: Dict[Tuple[int, int], Tuple[List, List]] = {}

    def _get_masks(
        self,
        grid_shape: Tuple[int, int],
        device: torch.device
    ) -> Tuple[List, List]:
        """Get or build cached masks."""
        if grid_shape not in self._mask_cache:
            encoder_masks, decoder_masks = self.ae.build_masks(grid_shape, device)
            self._mask_cache[grid_shape] = (encoder_masks, decoder_masks)
        return self._mask_cache[grid_shape]

    def _pad_and_patch(self, x: torch.Tensor) -> torch.Tensor:
        """For grid_shape inference by SpanEmbedder."""
        p = self.ae.patch_size
        if x.dim() == 3:
            C, H, W = x.shape
            gh, gw = H // p, W // p
            return torch.empty(C, gh, gw, p, p, device=x.device)
        else:
            B, C, H, W = x.shape
            gh, gw = H // p, W // p
            return torch.empty(B, C, gh, gw, p, p, device=x.device)

    def forward(
        self,
        x: torch.Tensor,
        logsnr_map: torch.Tensor,
        block_mask=None,
        return_codes: bool = False
    ):
        """
        Args:
            x: [C, H, W] or [B, C, H, W]
            logsnr_map: [1, H, W] or [B, 1, H, W]
            block_mask: Ignored (we use cached masks)
            return_codes: If True, also return codes

        Returns:
            z: [L, D] or [B, L, D]
            grid_shape: (GH, GW)
        """
        single_input = x.dim() == 3
        if single_input:
            x = x.unsqueeze(0)
            logsnr_map = logsnr_map.unsqueeze(0)

        B, C, H, W = x.shape
        p = self.ae.patch_size
        grid_shape = (H // p, W // p)
        device = x.device

        encoder_masks, decoder_masks = self._get_masks(grid_shape, device)

        codes_list, _ = self.ae.encode(
            x, logsnr_map,
            grid_shape=grid_shape,
            encoder_masks=encoder_masks,
            decoder_masks=decoder_masks
        )

        codes_cat = torch.cat(codes_list, dim=-1)
        z = self.code_proj(codes_cat)

        if single_input:
            z = z.squeeze(0)
            if return_codes:
                return z, grid_shape, codes_cat.squeeze(0)
            return z, grid_shape
        else:
            if return_codes:
                return z, grid_shape, codes_cat
            return z, grid_shape


class SwiGLUPatchUnembedder(nn.Module):
    """
    Wraps SwiGLUFSQAutoencoder to match ContextualPatchUnembedder interface.
    """

    def __init__(
        self,
        ae: SwiGLUFSQAutoencoder,
        embedder: SwiGLUPatchEmbedder,
        fourier_dim: int = 16
    ):
        super().__init__()
        self.ae = ae
        self.embedder = embedder
        self.patch_size = ae.patch_size
        self.fourier_dim = fourier_dim
        self.n_attn_layers = ae.n_layers

        total_code_dim = ae.code_dim * ae.n_levels
        self.code_unproj = nn.Linear(embedder.embed_dim, total_code_dim + fourier_dim)

        self.logsnr_decoder = nn.Sequential(
            nn.Linear(fourier_dim, embedder.embed_dim),
            nn.SiLU(),
            nn.Linear(embedder.embed_dim, 1)
        )
        with torch.no_grad():
            self.logsnr_decoder[-1].weight.zero_()
            self.logsnr_decoder[-1].bias.zero_()

        self._mask_cache: Dict[Tuple[int, int], List] = {}

    def _get_masks(self, grid_shape: Tuple[int, int], device: torch.device) -> List:
        """Get or build cached decoder masks."""
        if grid_shape not in self._mask_cache:
            _, decoder_masks = self.ae.build_masks(grid_shape, device)
            self._mask_cache[grid_shape] = decoder_masks
        return self._mask_cache[grid_shape]

    def forward(
        self,
        z: torch.Tensor,
        shape: Tuple,
        block_mask=None
    ) -> torch.Tensor:
        """
        Args:
            z: [L, D] or [B, L, D]
            shape: (GH, GW)
            block_mask: Ignored

        Returns:
            [C+1, H, W] or [B, C+1, H, W]
        """
        single_input = z.dim() == 2
        if single_input:
            z = z.unsqueeze(0)

        B, L, D = z.shape

        if len(shape) == 2:
            GH, GW = shape
        else:
            GH, GW = 1, L

        if L != GH * GW:
            GH, GW = 1, L

        grid_shape = (GH, GW)
        device = z.device

        decoder_masks = self._get_masks(grid_shape, device)

        # Unproject
        proj_out = self.code_unproj(z)
        total_code_dim = self.ae.code_dim * self.ae.n_levels
        codes_cat = proj_out[:, :, :total_code_dim]
        fourier_part = proj_out[:, :, total_code_dim:]

        # Split codes
        codes_list = []
        code_dim = self.ae.code_dim
        for level in range(self.ae.n_levels):
            start = level * code_dim
            end = start + code_dim
            codes_list.append(codes_cat[:, :, start:end])

        # Decode
        recon = self.ae.decode(codes_list, grid_shape=grid_shape, decoder_masks=decoder_masks)

        # LogSNR prediction
        logsnr_pred = self.logsnr_decoder(fourier_part)
        logsnr_grid = logsnr_pred.view(B, GH, GW).unsqueeze(1)
        H, W = GH * self.patch_size, GW * self.patch_size
        logsnr_channel = F.interpolate(logsnr_grid, size=(H, W), mode='nearest')

        result = torch.cat([recon, logsnr_channel], dim=1)

        if single_input:
            return result.squeeze(0)
        return result
