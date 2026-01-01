# src/model.py - LDTformer Transformer Models
"""
Main transformer architecture for multimodal (text + image) processing.

This file contains the core LDTformer models. Supporting modules have been
extracted to separate files for maintainability:
- blocks.py: SwiGLU, SigmoidMoE, MLPResBlock, init helpers
- rope.py: HouseholderOrthogonal, RnRoPE
- embedders.py: FourierFeatures, ContextualPatchEmbedder/Unembedder
- paging.py: PageTable, update_kv_cache
- context_manager.py: ContextBlock, Span, SpanEmbedder/Unembedder, topology, masks
"""

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention
from typing import Tuple, List, Dict, Optional

# Import from extracted modules
from .blocks import (
    SwiGLU,
    SigmoidMoE,
    MLPResBlock,
    EncoderAttention,
    EncoderBlock,
    init_linear,
    init_layer_norm,
    propagate_param_init,
)

from .rope import (
    HouseholderOrthogonal,
    RnRoPE,
)

from .embedders import (
    FourierFeatures,
    FourierScaleDecoder,
    ContextualPatchEmbedder,
    ContextualPatchUnembedder,
)

from .paging import (
    PageTable,
    update_kv_cache,
)

from .context_manager import (
    ContextBlock,
    Span,
    SpanEmbedder,
    SpanUnembedder,
    generate_content_hash_stream,
    render_topology_embeddings,
    build_dual_masks,
    materialize_mask_for_analysis,
    build_encoder_mask,
    get_encoder_mask_for_layer,
)

# Re-export everything for backwards compatibility
__all__ = [
    # blocks
    'SwiGLU', 'SigmoidMoE', 'MLPResBlock',
    'EncoderAttention', 'EncoderBlock',
    'init_linear', 'init_layer_norm', 'propagate_param_init',
    # rope
    'HouseholderOrthogonal', 'RnRoPE',
    # embedders
    'FourierFeatures', 'FourierScaleDecoder',
    'ContextualPatchEmbedder', 'ContextualPatchUnembedder',
    # paging
    'PageTable', 'update_kv_cache',
    # context_manager
    'ContextBlock', 'Span', 'SpanEmbedder', 'SpanUnembedder',
    'generate_content_hash_stream', 'render_topology_embeddings',
    'build_dual_masks', 'materialize_mask_for_analysis',
    'build_encoder_mask', 'get_encoder_mask_for_layer',
    # This file
    'LDTformerAttentionKVC', 'LDTformerAttentionZC',
    'LDTformerBlockKVC', 'LDTformerBlockZC',
    'coolerLDTformerKVC', 'coolerLDTformerZC',
]


# =========================================================
# ATTENTION LAYERS
# =========================================================

class LDTformerAttentionKVC(nn.Module):
    """
    Attention layer with KV cache support for inference.

    Uses RnRoPE for position encoding and writes to persistent KV cache.
    """
    def __init__(self, dim: int, num_heads: int, topo_dim: int, is_global=False, rope_base: float = 500.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rope = RnRoPE(self.head_dim, topo_dim, rope_base=rope_base)
        self.param_init()

    def param_init(self):
        init_linear(self.qkv)
        init_linear(self.proj)
        self.rope.param_init()

    def forward(
        self,
        x: torch.Tensor,           # [B, L, D] - ACTIVE tokens only
        topo_active: torch.Tensor, # [B, L_active, Topo_Dim] - GLOBAL COORDS
        k_cache: torch.Tensor,     # [1, H, Capacity, D] - FULL heap
        v_cache: torch.Tensor,     # [1, H, Capacity, D] - FULL heap
        slot_mapping: torch.Tensor,
        block_mask: object,
        scale: float = 1.0
    ):
        """
        Stateless Attention with KV cache:
        1. Projects Inputs
        2. Applies RoPE using Topology
        3. Commits New Data to Paged Heap
        4. Attends over Paged Heap using Physical Mask
        """
        B, L, D = x.shape

        # 1. Compute Q, K, V for NEW/ACTIVE tokens
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        # Apply RoPE using GLOBAL coordinates
        q, k = self.rope(q, k, topo_active, scale=scale)

        # 3. Cache Write (Side Effect)
        k_write = k.transpose(1, 2).reshape(B * L, self.num_heads, 1, self.head_dim).clone()
        v_write = v.transpose(1, 2).reshape(B * L, self.num_heads, 1, self.head_dim).clone()

        update_kv_cache(k_write, v_write, k_cache, v_cache, slot_mapping)

        # Attention uses HEAP topology (via the mask)
        out = flex_attention(q, k_cache, v_cache, block_mask=block_mask)

        # 5. Projection
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


class LDTformerAttentionZC(nn.Module):
    """
    Zero-cache attention layer for training.

    Uses RnRoPE for position encoding without persistent KV cache.
    """
    def __init__(self, dim: int, num_heads: int, topo_dim: int, rope_base: float = 500.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rope = RnRoPE(self.head_dim, topo_dim, rope_base=rope_base)
        self.param_init()

    def param_init(self):
        init_linear(self.qkv)
        init_linear(self.proj)
        self.rope.param_init()

    def forward(
        self,
        x: torch.Tensor,           # [B, L, D]
        topo_active: torch.Tensor, # [B, L, Topo_Dim]
        slot_mapping: torch.Tensor,
        block_mask: object,
        scale: float = 1.0
    ):
        """Zero-cache attention for training."""
        B, L, D = x.shape

        # 1. Compute Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        # Apply RoPE using GLOBAL coordinates
        q, k = self.rope(q, k, topo_active, scale=scale)

        # Attention
        out = flex_attention(q, k, v, block_mask=block_mask)

        # Projection
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


# =========================================================
# TRANSFORMER BLOCKS
# =========================================================

class LDTformerBlockKVC(nn.Module):
    """Transformer block with KV cache support."""
    def __init__(self, dim: int, num_heads: int, topo_dim: int, mlp_ratio: float = 4.0,
                 is_global=False, num_experts=8, num_active=3, jitter_noise: float = 0.1,
                 rope_base: float = 500.0):
        super().__init__()
        self.is_global = is_global
        self.rope_base = rope_base * (100 ** is_global)
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.attn = LDTformerAttentionKVC(dim, num_heads, topo_dim, rope_base=self.rope_base)
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)

        hidden_dim = int(dim * mlp_ratio)
        self.moe = SigmoidMoE(dim, hidden_dim, num_experts=num_experts,
                              num_active=num_active, jitter_noise=jitter_noise)
        self.gate_proj = nn.Linear(dim, dim)
        self.param_init()

    def param_init(self):
        init_layer_norm(self.norm1)
        self.attn.param_init()
        init_layer_norm(self.norm2)
        self.moe.param_init()
        init_linear(self.gate_proj)

    def forward(self, x, topo, k_cache, v_cache, slots, mask, scale: float = 1.0):
        # Attention Sub-block
        h = self.norm1(x)
        h = self.attn(h, topo, k_cache, v_cache, slots, mask, scale=scale)
        gh = torch.sigmoid(self.gate_proj(h))
        x = x + (h * gh)

        # MoE Sub-block
        h_moe, aux_loss = self.moe(self.norm2(x))
        x = x + h_moe

        return x, aux_loss


class LDTformerBlockZC(nn.Module):
    """Transformer block for zero-cache training."""
    def __init__(self, dim: int, num_heads: int, topo_dim: int, mlp_ratio: float = 4.0,
                 is_global=False, num_experts=8, num_active=3, jitter_noise: float = 0.1,
                 rope_base: float = 500.0):
        super().__init__()
        self.is_global = is_global
        self.rope_base = rope_base * (100 ** is_global)
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.attn = LDTformerAttentionZC(dim, num_heads, topo_dim, rope_base=self.rope_base)
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)

        hidden_dim = int(dim * mlp_ratio)
        self.moe = SigmoidMoE(dim, hidden_dim, num_experts=num_experts,
                              num_active=num_active, jitter_noise=jitter_noise)
        self.gate_proj = nn.Linear(dim, dim)
        self.param_init()

    def param_init(self):
        init_layer_norm(self.norm1)
        self.attn.param_init()
        init_layer_norm(self.norm2)
        self.moe.param_init()
        init_linear(self.gate_proj)

    def forward(self, x, topo, slots, mask, scale: float = 1.0):
        # Attention Sub-block
        h = self.norm1(x)
        h = self.attn(h, topo, slots, mask, scale=scale)
        gh = torch.sigmoid(self.gate_proj(h))
        x = x + (h * gh)

        # MoE Sub-block
        h_moe, aux_loss = self.moe(self.norm2(x))
        x = x + h_moe

        return x, aux_loss


# =========================================================
# FULL MODELS
# =========================================================

class coolerLDTformerKVC(nn.Module):
    """Full LDTformer model with KV cache support for inference."""
    def __init__(self, dim=256, depth=8, num_heads=8, topo_dim=4, mlp_depth=1,
                 vocab_size=65536, global_layer_interval=4, num_experts=8, num_active=3,
                 rope_base: int = 500, mlp_ratio: float = 4.0, jitter_noise: float = 0.1,
                 context_size: int = 4, stride: int = 2, fourier_dim: int = 16,
                 window_size: float = 10.0):
        super().__init__()

        self.global_layer_interval = global_layer_interval
        self.window_size = window_size
        self.text_embed = nn.Embedding(vocab_size, dim)
        self.patch_embedder = ContextualPatchEmbedder(
            input_channels=3, embed_dim=dim, context_size=context_size,
            stride=stride, fourier_dim=fourier_dim, mlp_depth=mlp_depth
        )

        self.layers = nn.ModuleList([
            LDTformerBlockKVC(dim, num_heads, topo_dim,
                              is_global=((i+1) % global_layer_interval == 0),
                              num_experts=num_experts, num_active=num_active, rope_base=rope_base)
            for i in range(depth)
        ])

        self.text_head = nn.Linear(dim, vocab_size)
        self.patch_unembedder = ContextualPatchUnembedder(
            output_channels=3, embed_dim=dim, patch_size=stride, mlp_depth=mlp_depth
        )

        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.param_init()

    def param_init(self):
        torch.nn.init.normal_(self.text_embed.weight, mean=0.0, std=0.02)
        init_linear(self.text_head)
        init_layer_norm(self.final_norm)
        self.patch_embedder.param_init()
        self.patch_unembedder.param_init()
        for layer in self.layers:
            layer.param_init()

    def forward(
        self,
        z: torch.Tensor,
        topo_embeds: torch.Tensor,
        k_caches: list,
        v_caches: list,
        slot_mapping: torch.Tensor,
        block_masks: Tuple[object, object],
        scale: float = 1.0
    ) -> Tuple[torch.Tensor, float]:
        """Pure transformer pass."""
        mask_local, mask_global = block_masks
        x = z
        total_aux = 0.0

        for i, layer in enumerate(self.layers):
            block_mask = mask_global if layer.is_global else mask_local
            x, aux = layer(x, topo_embeds, k_caches[i], v_caches[i],
                           slot_mapping, block_mask, scale=scale)
            total_aux += aux

        x = self.final_norm(x)
        return x, total_aux

    def dump(self) -> Dict[str, torch.Tensor]:
        """Return reference to parameters (no move)."""
        return {k: v.clone() for k, v in self.state_dict().items()}

    def flush(self):
        """Zero out all parameters and gradients."""
        for p in self.parameters():
            p.data.zero_()
            if p.grad is not None:
                p.grad.zero_()

    def param_load(self, state_dict):
        """Load a specific parameter set."""
        self.load_state_dict(state_dict)


class coolerLDTformerZC(nn.Module):
    """Full LDTformer model for zero-cache training."""
    def __init__(self, dim=256, depth=8, num_heads=8, topo_dim=4, mlp_depth=1,
                 vocab_size=65536, global_layer_interval=4, num_experts=8, num_active=3,
                 rope_base: int = 500, mlp_ratio: float = 4.0, jitter_noise: float = 0.1,
                 context_size: int = 4, stride: int = 2, fourier_dim: int = 16,
                 window_size: float = 10.0, sparse_ae_config: dict = None):
        super().__init__()

        self.global_layer_interval = global_layer_interval
        self.window_size = window_size
        self.text_embed = nn.Embedding(vocab_size, dim)

        # Build patch embedder/unembedder based on config
        # When sparse_ae_config is provided, use SparseAE components
        # Otherwise use standard ContextualPatchEmbedder/Unembedder
        self.uses_sparse_ae = sparse_ae_config is not None and sparse_ae_config.get('enabled', False)

        if self.uses_sparse_ae:
            ae_type = sparse_ae_config.get('ae_type', 'sparse_dim')

            attn_cfg = sparse_ae_config.get('attention', {
                'mode': 'full', 'window_size': 4, 'global_layer_interval': 4,
                'n_query_heads': 8, 'n_kv_heads': 2, 'n_global_tokens': 4
            })

            if ae_type == 'swiglu':
                # SwiGLU variant: binary FSQ, level-global sparsity, 2D RoPE
                from kmaze_ae.model_swiglu import (
                    SwiGLUFSQAutoencoder,
                    SwiGLUPatchEmbedder,
                    SwiGLUPatchUnembedder
                )

                # For swiglu, default to sliding window (3x3 neighborhood)
                if 'mode' not in attn_cfg:
                    attn_cfg['mode'] = 'sliding'
                if 'window_size' not in attn_cfg:
                    attn_cfg['window_size'] = 2  # Euclidean dist² ≤ 4 covers 3x3

                self.sparse_ae = SwiGLUFSQAutoencoder(
                    n_levels=sparse_ae_config.get('n_levels', 6),
                    patch_size=sparse_ae_config.get('patch_size', 16),
                    image_size=256,  # Dynamic per batch
                    hidden_dim=sparse_ae_config.get('hidden_dim', 128),
                    code_dim=sparse_ae_config.get('code_dim', 256),
                    k_active=sparse_ae_config.get('k_per_patch', 8),
                    residual_scale=sparse_ae_config.get('residual_scale', 2.0),
                    fourier_dim=sparse_ae_config.get('fourier_dim', 16),
                    n_layers=sparse_ae_config.get('n_layers', 1),
                    n_heads=attn_cfg.get('n_query_heads', 4),
                    n_kv_heads=attn_cfg.get('n_kv_heads', 2),
                    attn_config=attn_cfg
                )

                self.patch_embedder = SwiGLUPatchEmbedder(self.sparse_ae, embed_dim=dim)
                self.patch_unembedder = SwiGLUPatchUnembedder(
                    self.sparse_ae, self.patch_embedder,
                    fourier_dim=sparse_ae_config.get('fourier_dim', 16)
                )

                print(f"[Model] Using SwiGLU AE: {sparse_ae_config.get('n_levels', 6)} levels, "
                      f"code_dim={sparse_ae_config.get('code_dim', 256)}, "
                      f"k={sparse_ae_config.get('k_per_patch', 8)}, "
                      f"attn={attn_cfg.get('mode', 'local')}")

            else:
                # Default: sparse_dim variant (3-bit, per-patch sparsity)
                from kmaze_ae.model_sparse_dim import (
                    SparsePerDimFSQAutoencoder,
                    SparseAEPatchEmbedder,
                    SparseAEPatchUnembedder
                )

                self.sparse_ae = SparsePerDimFSQAutoencoder(
                    n_levels=sparse_ae_config.get('n_levels', 6),
                    patch_size=sparse_ae_config.get('patch_size', 16),
                    image_size=256,  # Dynamic per batch
                    hidden_dim=sparse_ae_config.get('hidden_dim', 256),
                    code_dim=sparse_ae_config.get('code_dim', 128),
                    k_per_patch=sparse_ae_config.get('k_per_patch', 6),
                    residual_scale=sparse_ae_config.get('residual_scale', 2.0),
                    fourier_dim=sparse_ae_config.get('fourier_dim', 16),
                    n_layers=sparse_ae_config.get('n_layers', 4),
                    attn_config=attn_cfg
                )

                self.patch_embedder = SparseAEPatchEmbedder(self.sparse_ae, embed_dim=dim)
                self.patch_unembedder = SparseAEPatchUnembedder(
                    self.sparse_ae, self.patch_embedder,
                    fourier_dim=sparse_ae_config.get('fourier_dim', 16)
                )

                print(f"[Model] Using SparseAE: {sparse_ae_config.get('n_levels', 6)} levels, "
                      f"code_dim={sparse_ae_config.get('code_dim', 128)}, "
                      f"k={sparse_ae_config.get('k_per_patch', 6)}")
        else:
            self.sparse_ae = None
            self.patch_embedder = ContextualPatchEmbedder(
                input_channels=3, embed_dim=dim, context_size=context_size,
                stride=stride, fourier_dim=fourier_dim, mlp_depth=mlp_depth
            )
            self.patch_unembedder = ContextualPatchUnembedder(
                output_channels=3, embed_dim=dim, patch_size=stride, mlp_depth=mlp_depth
            )

        self.layers = nn.ModuleList([
            LDTformerBlockZC(dim, num_heads, topo_dim, mlp_ratio=mlp_ratio,
                             is_global=((i+1) % global_layer_interval == 0),
                             num_experts=num_experts, num_active=num_active,
                             jitter_noise=jitter_noise, rope_base=rope_base)
            for i in range(depth)
        ])

        self.text_head = nn.Linear(dim, vocab_size)

        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.param_init()

    def param_init(self):
        torch.nn.init.normal_(self.text_embed.weight, mean=0.0, std=0.02)
        init_linear(self.text_head)
        init_layer_norm(self.final_norm)
        # Only call param_init if the embedder has it (standard embedders do, sparse AE doesn't)
        if hasattr(self.patch_embedder, 'param_init'):
            self.patch_embedder.param_init()
        if hasattr(self.patch_unembedder, 'param_init'):
            self.patch_unembedder.param_init()
        for layer in self.layers:
            layer.param_init()

    def forward(
        self,
        z: torch.Tensor,
        topo_embeds: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_masks: Tuple[object, object],
        scale: float = 1.0
    ) -> Tuple[torch.Tensor, float]:
        """Pure transformer pass."""
        mask_local, mask_global = block_masks
        x = z
        total_aux = 0.0

        for i, layer in enumerate(self.layers):
            block_mask = mask_global if layer.is_global else mask_local
            x, aux = layer(x, topo_embeds, slot_mapping, block_mask, scale=scale)
            total_aux += aux

        x = self.final_norm(x)
        return x, total_aux

    def dump(self) -> Dict[str, torch.Tensor]:
        """Return reference to parameters (no move)."""
        return {k: v.clone() for k, v in self.state_dict().items()}

    def flush(self):
        """Zero out all parameters and gradients."""
        for p in self.parameters():
            p.data.zero_()
            if p.grad is not None:
                p.grad.zero_()

    def param_load(self, state_dict):
        """Load a specific parameter set."""
        self.load_state_dict(state_dict)
