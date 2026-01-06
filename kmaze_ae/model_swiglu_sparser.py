"""
SwiGLU FSQ Autoencoder - Weight-Shared MoE Variant

Key differences from model_swiglu.py:
- Single shared encoder/decoder applied to ALL residual levels (not n_levels copies)
- SigmoidMoE replaces SwiGLU in transformer blocks for more expressivity
- NO level embeddings - MoE router learns level behavior from input statistics
- Dramatically fewer parameters with more expressive routing

The insight: residual levels are the same task at different granularities.
Different levels have different input distributions (full content vs residuals),
so the MoE router naturally learns to route differently without explicit labels.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List

from .model_sparse_dim import BinaryFSQ, ThreeBitFSQ, PerDimSparsity
from .model_swiglu import (
    HaarDWT2d, HaarIDWT2d, batch_dwt2d, batch_idwt2d,
    LevelSparsity, SubspaceSparsity, SubspacePerPatchSparsity,
    SwiGLUNeighborHead
)
from src.blocks import EncoderAttention, SigmoidMoE, init_linear, init_layer_norm


# =============================================================================
# MoE Transformer Blocks (replaces SwiGLU with SigmoidMoE)
# =============================================================================

class MoEEncoderBlock(nn.Module):
    """
    Transformer block with SigmoidMoE instead of SwiGLU.

    Returns (output, aux_loss) where aux_loss is the MoE load balancing loss.
    """
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        num_experts: int = 16,
        num_active: int = 3,
        jitter_noise: float = 0.1
    ):
        super().__init__()
        self.dim = dim

        # Attention
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.attn = EncoderAttention(dim, n_heads, n_kv_heads)
        self.gate_proj = nn.Linear(dim, dim, bias=False)

        # MoE MLP (replaces SwiGLU)
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)
        hidden_dim = int(dim * 4)  # Standard 4x expansion
        self.mlp = SigmoidMoE(
            dim, hidden_dim,
            num_experts=num_experts,
            num_active=num_active,
            jitter_noise=jitter_noise
        )
        self.param_init()

    def param_init(self):
        init_layer_norm(self.norm1)
        self.attn.param_init()
        init_linear(self.gate_proj)
        init_layer_norm(self.norm2)
        self.mlp.param_init()

    def forward(self, x: torch.Tensor, block_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (output, aux_loss)."""
        # Attention with gated residual
        h = self.attn(self.norm1(x), block_mask=block_mask)
        gate = torch.sigmoid(self.gate_proj(h))
        x = x + h * gate

        # MoE MLP
        h_mlp, aux_loss = self.mlp(self.norm2(x))
        x = x + h_mlp

        return x, aux_loss


class MoETransformerEncoder(nn.Module):
    """
    Transformer encoder stack with SigmoidMoE blocks.

    Supports same attention modes as TransformerEncoder (full, sliding, bigbird, gemma_bigbird).
    """
    def __init__(
        self,
        dim: int = 256,
        n_layers: int = 4,
        attn_config: Optional[Dict[str, Any]] = None,
        num_experts: int = 16,
        num_active: int = 3,
        jitter_noise: float = 0.1
    ):
        super().__init__()
        self.n_layers = n_layers

        if attn_config is None:
            attn_config = {
                'mode': 'sliding', 'window_size': 2,
                'n_query_heads': 8, 'n_kv_heads': 2,
                'n_global_tokens': 0, 'global_layer_interval': 4,
                'bigbird_layout': [2, 2]
            }

        self.attn_config = attn_config
        self.mode = attn_config['mode']
        n_heads = attn_config['n_query_heads']
        n_kv_heads = attn_config['n_kv_heads']
        self.n_global_tokens = attn_config['n_global_tokens']

        # Register tokens for bigbird modes
        self.uses_registers = self.mode in ('bigbird', 'gemma_bigbird')
        if self.uses_registers:
            self.register_tokens = nn.Parameter(torch.randn(1, self.n_global_tokens, dim) * 0.02)

        # MoE transformer layers
        self.layers = nn.ModuleList([
            MoEEncoderBlock(
                dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
                num_experts=num_experts, num_active=num_active,
                jitter_noise=jitter_noise
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        grid_shape: Optional[Tuple[int, int]] = None,
        block_masks: Optional[list] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (output, total_aux_loss).
        """
        from src.context_manager import get_encoder_mask_for_layer

        B, N, D = x.shape

        if grid_shape is None:
            side = int(N ** 0.5)
            grid_shape = (side, side)

        # Prepend registers for bigbird modes
        if self.uses_registers:
            registers = self.register_tokens.expand(B, -1, -1)
            x = torch.cat([registers, x], dim=1)

        device = x.device
        total_aux_loss = torch.tensor(0.0, device=device)

        for i, layer in enumerate(self.layers):
            if block_masks is not None:
                mask = block_masks[i]
            else:
                mask = get_encoder_mask_for_layer(grid_shape, i, self.attn_config, device)
            x, aux_loss = layer(x, block_mask=mask)
            total_aux_loss = total_aux_loss + aux_loss

        # Remove registers
        if self.uses_registers:
            x = x[:, self.n_global_tokens:]

        return x, total_aux_loss

    def build_masks(self, grid_shape: Tuple[int, int], device: torch.device) -> list:
        from src.context_manager import get_encoder_mask_for_layer
        return [
            get_encoder_mask_for_layer(grid_shape, i, self.attn_config, device)
            for i in range(self.n_layers)
        ]


class MoETransformerDecoder(nn.Module):
    """
    Transformer decoder stack with SigmoidMoE blocks.
    """
    def __init__(
        self,
        dim: int = 256,
        n_layers: int = 4,
        attn_config: Optional[Dict[str, Any]] = None,
        num_experts: int = 16,
        num_active: int = 3,
        jitter_noise: float = 0.1
    ):
        super().__init__()
        self.n_layers = n_layers

        if attn_config is None:
            attn_config = {
                'mode': 'sliding', 'window_size': 2,
                'n_query_heads': 8, 'n_kv_heads': 2,
                'n_global_tokens': 0, 'global_layer_interval': 4,
                'bigbird_layout': [2, 2]
            }

        self.attn_config = attn_config
        self.mode = attn_config['mode']
        n_heads = attn_config['n_query_heads']
        n_kv_heads = attn_config['n_kv_heads']
        self.n_global_tokens = attn_config['n_global_tokens']

        # Register tokens for bigbird modes
        self.uses_registers = self.mode in ('bigbird', 'gemma_bigbird')
        if self.uses_registers:
            self.register_tokens = nn.Parameter(torch.randn(1, self.n_global_tokens, dim) * 0.02)

        # MoE transformer layers
        self.layers = nn.ModuleList([
            MoEEncoderBlock(
                dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
                num_experts=num_experts, num_active=num_active,
                jitter_noise=jitter_noise
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        grid_shape: Optional[Tuple[int, int]] = None,
        block_masks: Optional[list] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (output, total_aux_loss)."""
        from src.context_manager import get_encoder_mask_for_layer

        B, N, D = x.shape

        if grid_shape is None:
            side = int(N ** 0.5)
            grid_shape = (side, side)

        if self.uses_registers:
            registers = self.register_tokens.expand(B, -1, -1)
            x = torch.cat([registers, x], dim=1)

        device = x.device
        total_aux_loss = torch.tensor(0.0, device=device)

        for i, layer in enumerate(self.layers):
            if block_masks is not None:
                mask = block_masks[i]
            else:
                mask = get_encoder_mask_for_layer(grid_shape, i + 100, self.attn_config, device)
            x, aux_loss = layer(x, block_mask=mask)
            total_aux_loss = total_aux_loss + aux_loss

        if self.uses_registers:
            x = x[:, self.n_global_tokens:]

        return x, total_aux_loss

    def build_masks(self, grid_shape: Tuple[int, int], device: torch.device) -> list:
        from src.context_manager import get_encoder_mask_for_layer
        return [
            get_encoder_mask_for_layer(grid_shape, i + 100, self.attn_config, device)
            for i in range(self.n_layers)
        ]


# =============================================================================
# Weight-Shared Encoder/Decoder (MoE learns level behavior from input stats)
# =============================================================================

class SharedMoEEncoder(nn.Module):
    """
    Single encoder applied to all residual levels.

    NO level embeddings - the MoE router learns level-appropriate behavior
    from input statistics alone. Different residual levels have fundamentally
    different distributions (level 0 = full content, level N = fine residuals),
    so the sigmoid router naturally learns to route differently.

    This is more robust than explicit level conditioning because:
    1. Input statistics already encode "which level" implicitly
    2. Router learns task structure, not arbitrary level labels
    3. Generalizes to different n_levels without retraining embeddings
    """
    def __init__(
        self,
        patch_dim: int,
        hidden_dim: int,
        code_dim: int,
        n_levels: int,
        k_per_patch: int = 4,
        n_layers: int = 4,
        attn_config: Optional[Dict[str, Any]] = None,
        sparsity_mode: str = "per_level",
        wavelet_gating: bool = False,
        patch_size: int = 16,
        n_wavelet_dims: Optional[int] = None,
        num_experts: int = 16,
        num_active: int = 3,
        jitter_noise: float = 0.1
    ):
        super().__init__()
        self.n_levels = n_levels
        self.wavelet_gating = wavelet_gating
        self.patch_size = patch_size
        self.code_dim = code_dim
        self.n_wavelet_dims = n_wavelet_dims or code_dim // 2
        self.n_amplitude_dims = code_dim - self.n_wavelet_dims

        # Input projection
        if wavelet_gating:
            self.amplitude_proj = nn.Linear(patch_dim, hidden_dim // 2)
            self.wavelet_proj = nn.Linear(patch_dim, hidden_dim // 2)
            self.input_proj = None
        else:
            self.input_proj = nn.Linear(patch_dim, hidden_dim)
            self.amplitude_proj = None
            self.wavelet_proj = None

        # Single shared MoE transformer
        self.transformer = MoETransformerEncoder(
            hidden_dim, n_layers=n_layers, attn_config=attn_config,
            num_experts=num_experts, num_active=num_active, jitter_noise=jitter_noise
        )

        # Code projection
        if wavelet_gating:
            self.wav_code_proj = nn.Linear(hidden_dim, self.n_wavelet_dims)
            self.amp_code_proj = nn.Linear(hidden_dim, self.n_amplitude_dims)
            self.code_proj = None
        else:
            self.code_proj = nn.Linear(hidden_dim, code_dim)
            self.wav_code_proj = None
            self.amp_code_proj = None

        self.fsq = BinaryFSQ()
        self.sparsity_mode = sparsity_mode

        # Per-level sparsity (dim_logits are level-specific even with shared backbone)
        if wavelet_gating:
            if sparsity_mode == "per_level":
                self.sparsity_modules = nn.ModuleList([
                    SubspaceSparsity(code_dim, k_per_patch, self.n_wavelet_dims)
                    for _ in range(n_levels)
                ])
            else:
                self.sparsity_modules = nn.ModuleList([
                    SubspacePerPatchSparsity(code_dim, k_per_patch, self.n_wavelet_dims)
                    for _ in range(n_levels)
                ])
        else:
            if sparsity_mode == "per_level":
                self.sparsity_modules = nn.ModuleList([
                    LevelSparsity(code_dim, k_per_patch)
                    for _ in range(n_levels)
                ])
            else:
                self.sparsity_modules = nn.ModuleList([
                    PerDimSparsity(code_dim, k_per_patch)
                    for _ in range(n_levels)
                ])

    def forward(
        self,
        patches: torch.Tensor,
        level: int,
        grid_shape: Tuple[int, int],
        block_masks: Optional[List] = None,
        k_override: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode patches at a specific residual level.

        Returns:
            sparse_codes: [B, N, code_dim]
            gate_weights: sparsity weights for logging
            aux_loss: MoE load balancing loss
        """
        B, N, _ = patches.shape
        device = patches.device

        # Input projection
        if self.wavelet_gating:
            wavelet_coeffs = batch_dwt2d(patches, self.patch_size)
            h_amp = self.amplitude_proj(patches)
            h_wav = self.wavelet_proj(wavelet_coeffs)
            h = torch.cat([h_amp, h_wav], dim=-1)
        else:
            h = self.input_proj(patches)

        # No level conditioning - MoE router learns from input statistics
        # Shared transformer (returns aux_loss)
        h, aux_loss = self.transformer(h, grid_shape=grid_shape, block_masks=block_masks)

        # Code projection
        if self.wavelet_gating:
            wav_logits = self.wav_code_proj(h)
            amp_logits = self.amp_code_proj(h)
            logits = torch.cat([wav_logits, amp_logits], dim=-1)
        else:
            logits = self.code_proj(h)

        # Binary FSQ
        codes = self.fsq(logits)
        codes = codes * 2 - 1  # {0, 1} -> {-1, +1}

        # Level-specific sparsity
        sparse_codes, gate_weights = self.sparsity_modules[level](codes, k_override=k_override)

        return sparse_codes, gate_weights, aux_loss

    def forward_with_prequant(
        self,
        patches: torch.Tensor,
        level: int,
        grid_shape: Tuple[int, int],
        block_masks: Optional[List] = None,
        k_override: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (sparse_codes, gate_weights, pre_quant, aux_loss)."""
        B, N, _ = patches.shape
        device = patches.device

        if self.wavelet_gating:
            wavelet_coeffs = batch_dwt2d(patches, self.patch_size)
            h_amp = self.amplitude_proj(patches)
            h_wav = self.wavelet_proj(wavelet_coeffs)
            h = torch.cat([h_amp, h_wav], dim=-1)
        else:
            h = self.input_proj(patches)

        # No level conditioning - MoE router learns from input statistics
        h, aux_loss = self.transformer(h, grid_shape=grid_shape, block_masks=block_masks)

        if self.wavelet_gating:
            wav_logits = self.wav_code_proj(h)
            amp_logits = self.amp_code_proj(h)
            logits = torch.cat([wav_logits, amp_logits], dim=-1)
        else:
            logits = self.code_proj(h)

        codes = self.fsq(logits)
        codes = codes * 2 - 1

        sparse_codes, gate_weights = self.sparsity_modules[level](codes, k_override=k_override)

        return sparse_codes, gate_weights, logits, aux_loss


class SharedMoEDecoder(nn.Module):
    """
    Single decoder applied to all residual levels.

    NO level embeddings - MoE router learns level-appropriate behavior from
    the code statistics. Codes from different levels have different sparsity
    patterns and value distributions, providing implicit level information.
    """
    def __init__(
        self,
        code_dim: int,
        hidden_dim: int,
        patch_dim: int,
        n_levels: int,
        n_layers: int = 4,
        attn_config: Optional[Dict[str, Any]] = None,
        wavelet_gating: bool = False,
        patch_size: int = 16,
        n_wavelet_dims: Optional[int] = None,
        num_experts: int = 16,
        num_active: int = 3,
        jitter_noise: float = 0.1
    ):
        super().__init__()
        self.n_levels = n_levels
        self.wavelet_gating = wavelet_gating
        self.patch_size = patch_size
        self.patch_dim = patch_dim
        self.code_dim = code_dim
        self.n_wavelet_dims = n_wavelet_dims or code_dim // 2
        self.n_amplitude_dims = code_dim - self.n_wavelet_dims

        # Input embedding
        if wavelet_gating:
            self.wav_embed = nn.Linear(self.n_wavelet_dims, hidden_dim // 2)
            self.amp_embed = nn.Linear(self.n_amplitude_dims, hidden_dim // 2)
            self.input_proj = None
        else:
            self.input_proj = nn.Linear(code_dim, hidden_dim)
            self.wav_embed = None
            self.amp_embed = None

        # Single shared MoE transformer
        self.transformer = MoETransformerDecoder(
            hidden_dim, n_layers=n_layers, attn_config=attn_config,
            num_experts=num_experts, num_active=num_active, jitter_noise=jitter_noise
        )
        self.neighbor_head = SwiGLUNeighborHead(hidden_dim)

        # Output projection
        if wavelet_gating:
            self.wav_head = nn.Linear(hidden_dim, patch_dim)
            self.amp_head = nn.Linear(hidden_dim, patch_dim)
            self.output_proj = None
        else:
            self.output_proj = nn.Linear(hidden_dim, patch_dim)
            self.wav_head = None
            self.amp_head = None

    def forward(
        self,
        codes: torch.Tensor,
        level: int,
        grid_shape: Tuple[int, int],
        block_masks: Optional[List] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode codes at a specific residual level.

        Returns:
            patches: [B, N, patch_dim]
            aux_loss: MoE load balancing loss
        """
        # Input embedding
        if self.wavelet_gating:
            wav_codes = codes[..., :self.n_wavelet_dims]
            amp_codes = codes[..., self.n_wavelet_dims:]
            h_wav = self.wav_embed(wav_codes)
            h_amp = self.amp_embed(amp_codes)
            h = torch.cat([h_wav, h_amp], dim=-1)
        else:
            h = self.input_proj(codes)

        # No level conditioning - MoE router learns from code statistics
        # Shared transformer
        h, aux_loss = self.transformer(h, grid_shape=grid_shape, block_masks=block_masks)
        h = self.neighbor_head(h, grid_shape)

        # Output
        if self.wavelet_gating:
            wav_coeffs = self.wav_head(h)
            amp_pixels = self.amp_head(h)
            wav_pixels = batch_idwt2d(wav_coeffs, self.patch_size)
            patches = wav_pixels + amp_pixels
        else:
            patches = self.output_proj(h)

        return patches, aux_loss

    def forward_ablated(
        self,
        codes: torch.Tensor,
        level: int,
        grid_shape: Tuple[int, int],
        ablate_wavelet: float = 0.0,
        ablate_amplitude: float = 0.0,
        block_masks: Optional[List] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode with stochastic subspace ablation for interpretability.

        Args:
            codes: [B, N, code_dim] sparse codes
            level: residual level index
            grid_shape: (GH, GW) grid dimensions
            ablate_wavelet: ablation rate [0,1] for wavelet subspace
            ablate_amplitude: ablation rate [0,1] for amplitude subspace
            block_masks: optional pre-built decoder masks
            deterministic: if True, zero fixed fraction; if False, Bernoulli per-element

        Returns:
            patches: [B, N, patch_dim]
            aux_loss: MoE load balancing loss
        """
        if not self.wavelet_gating:
            return self.forward(codes, level, grid_shape, block_masks)

        # Split codes into subspaces
        wav_codes = codes[..., :self.n_wavelet_dims]
        amp_codes = codes[..., self.n_wavelet_dims:]

        # Apply ablation
        if ablate_wavelet > 0:
            if deterministic:
                # Zero out fixed fraction of dims
                n_ablate = int(self.n_wavelet_dims * ablate_wavelet)
                wav_codes = wav_codes.clone()
                wav_codes[..., :n_ablate] = 0
            else:
                # Bernoulli per-element
                mask = torch.rand_like(wav_codes) > ablate_wavelet
                wav_codes = wav_codes * mask

        if ablate_amplitude > 0:
            if deterministic:
                n_ablate = int(self.n_amplitude_dims * ablate_amplitude)
                amp_codes = amp_codes.clone()
                amp_codes[..., :n_ablate] = 0
            else:
                mask = torch.rand_like(amp_codes) > ablate_amplitude
                amp_codes = amp_codes * mask

        # Embed ablated codes
        h_wav = self.wav_embed(wav_codes)
        h_amp = self.amp_embed(amp_codes)
        h = torch.cat([h_wav, h_amp], dim=-1)

        # Shared transformer
        h, aux_loss = self.transformer(h, grid_shape=grid_shape, block_masks=block_masks)
        h = self.neighbor_head(h, grid_shape)

        # Output
        wav_coeffs = self.wav_head(h)
        amp_pixels = self.amp_head(h)
        wav_pixels = batch_idwt2d(wav_coeffs, self.patch_size)
        patches = wav_pixels + amp_pixels

        return patches, aux_loss


# =============================================================================
# Main Autoencoder
# =============================================================================

class SwiGLUMoEAutoencoder(nn.Module):
    """
    Weight-Shared MoE Autoencoder.

    Key differences from SwiGLUFSQAutoencoder:
    - Single shared encoder/decoder for ALL residual levels
    - SigmoidMoE provides expressivity through expert routing
    - NO level embeddings - router learns from input statistics
    - ~7x fewer transformer parameters (1 encoder/decoder vs n_levels)

    The MoE routing naturally learns level-specific behavior because
    different residual levels have fundamentally different distributions:
    - Level 0: Full image patches (high variance, structured)
    - Level N: Fine residuals (low variance, detail patterns)

    The router sees these statistical differences and routes accordingly.
    No explicit level signal needed - that would be redundant and a crutch.
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
        n_wavelet_dims: Optional[int] = None,
        num_experts: int = 16,
        num_active: int = 3,
        jitter_noise: float = 0.1
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
        self.num_experts = num_experts
        self.num_active = num_active

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

        # Single shared encoder/decoder (weight-shared across levels)
        self.encoder = SharedMoEEncoder(
            self.patch_dim, hidden_dim, code_dim, n_levels, k_per_patch,
            n_layers, attn_config, sparsity_mode,
            wavelet_gating=wavelet_gating, patch_size=patch_size,
            n_wavelet_dims=n_wavelet_dims,
            num_experts=num_experts, num_active=num_active, jitter_noise=jitter_noise
        )
        self.decoder = SharedMoEDecoder(
            code_dim, hidden_dim, self.patch_dim, n_levels, n_layers, attn_config,
            wavelet_gating=wavelet_gating, patch_size=patch_size,
            n_wavelet_dims=n_wavelet_dims,
            num_experts=num_experts, num_active=num_active, jitter_noise=jitter_noise
        )

        wavelet_str = f", wavelet=True, n_wav_dims={n_wavelet_dims or code_dim // 2}" if wavelet_gating else ""
        print(f"[SwiGLUMoEAutoencoder] {n_levels} levels (SHARED), code_dim={code_dim}, k={k_per_patch}, sparsity={sparsity_mode}{wavelet_str}")
        print(f"  MoE: {num_experts} experts, {num_active} active per token")
        print(f"  Attention: {attn_config['mode']}, window={attn_config.get('window_size', 'N/A')}")

    def build_masks(self, grid_shape: Tuple[int, int], device: torch.device) -> Tuple[List, List]:
        """Build attention masks for encoder and decoder."""
        encoder_masks = self.encoder.transformer.build_masks(grid_shape, device)
        decoder_masks = self.decoder.transformer.build_masks(grid_shape, device)
        return encoder_masks, decoder_masks

    def patchify(self, images: torch.Tensor, grid_shape: Tuple[int, int]) -> torch.Tensor:
        B = images.shape[0]
        p = self.patch_size
        GH, GW = grid_shape
        patches = images.view(B, 3, GH, p, GW, p)
        patches = patches.permute(0, 2, 4, 3, 5, 1).contiguous()
        return patches.view(B, GH * GW, self.patch_dim)

    def unpatchify(self, patches: torch.Tensor, grid_shape: Tuple[int, int]) -> torch.Tensor:
        B = patches.shape[0]
        p = self.patch_size
        GH, GW = grid_shape
        patches = patches.view(B, GH, GW, p, p, 3)
        patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
        return patches.view(B, 3, GH * p, GW * p)

    def forward(
        self,
        images: torch.Tensor,
        logsnr_map: Optional[torch.Tensor] = None,
        encoder_masks: Optional[List] = None,
        decoder_masks: Optional[List] = None,
        grid_shape: Optional[Tuple[int, int]] = None,
        k_override: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Forward pass with weight-shared encoder/decoder.

        Returns same interface as SwiGLUFSQAutoencoder plus 'aux_loss'.
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

        level_recons = []
        codes_list = []
        masks_list = []
        routing_stats_list = []
        cumulative_recon = torch.zeros_like(patches)
        total_aux_loss = torch.tensor(0.0, device=device)

        for level in range(self.n_levels):
            # Residual with .detach()
            if level > 0:
                residual = (patches - cumulative_recon.detach()) * self.residual_scale
            else:
                residual = patches

            # Encode (shared encoder with level conditioning)
            sparse_codes, gate_weights, enc_aux = self.encoder(
                residual, level, grid_shape, encoder_masks, k_override=k_override
            )
            codes_list.append(sparse_codes)
            masks_list.append(gate_weights)
            total_aux_loss = total_aux_loss + enc_aux

            # Collect routing stats
            if self.wavelet_gating:
                routing_stats_list.append(self.encoder.sparsity_modules[level].last_routing_stats)

            # Decode (shared decoder with level conditioning)
            decoded, dec_aux = self.decoder(sparse_codes, level, grid_shape, decoder_masks)
            total_aux_loss = total_aux_loss + dec_aux

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
            'grid_shape': grid_shape,
            'aux_loss': total_aux_loss,
        }

        if self.wavelet_gating and routing_stats_list:
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
        logsnr_map: Optional[torch.Tensor] = None,
        grid_shape: Optional[Tuple[int, int]] = None,
        encoder_masks: Optional[List] = None,
        decoder_masks: Optional[List] = None
    ) -> List[torch.Tensor]:
        """Encode images to sparse codes (ignores aux_loss)."""
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

            sparse_codes, _, _ = self.encoder(residual, level, grid_shape, encoder_masks)
            codes_list.append(sparse_codes)

            decoded, _ = self.decoder(sparse_codes, level, grid_shape, decoder_masks)
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
        """Decode sparse codes to image (ignores aux_loss)."""
        device = codes_list[0].device

        if decoder_masks is None:
            decoder_masks = self.decoder.transformer.build_masks(grid_shape, device)

        cumulative_recon = None

        for level, codes in enumerate(codes_list):
            decoded, _ = self.decoder(codes, level, grid_shape, decoder_masks)

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
            decoder_masks = self.decoder.transformer.build_masks(grid_shape, device)

        cumulative_recon = None

        for level, codes in enumerate(codes_list):
            # Use ablated forward pass
            decoded, _ = self.decoder.forward_ablated(
                codes, level, grid_shape, ablate_wavelet, ablate_amplitude,
                decoder_masks, deterministic
            )

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
        logsnr_map: Optional[torch.Tensor] = None,
        grid_shape: Optional[Tuple[int, int]] = None,
        encoder_masks: Optional[List] = None,
        decoder_masks: Optional[List] = None
    ) -> Tuple[List[torch.Tensor], None, List[torch.Tensor]]:
        """Encode with pre-quantization values for latent diffusion."""
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

            sparse_codes, _, pre_quant, _ = self.encoder.forward_with_prequant(
                residual, level, grid_shape, encoder_masks
            )
            codes_list.append(sparse_codes)
            prequant_list.append(pre_quant)

            decoded, _ = self.decoder(sparse_codes, level, grid_shape, decoder_masks)
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
        """Quantize pre-quant values and decode."""
        device = prequant_list[0].device

        if decoder_masks is None:
            decoder_masks = self.decoder.transformer.build_masks(grid_shape, device)

        cumulative_recon = None

        for level, pre_quant in enumerate(prequant_list):
            codes = self.encoder.fsq(pre_quant)
            codes = codes * 2 - 1
            sparse_codes, _ = self.encoder.sparsity_modules[level](codes)

            decoded, _ = self.decoder(sparse_codes, level, grid_shape, decoder_masks)

            if level > 0:
                decoded = decoded / self.residual_scale

            if cumulative_recon is None:
                cumulative_recon = decoded
            else:
                cumulative_recon = cumulative_recon + decoded

        return self.unpatchify(cumulative_recon, grid_shape)
