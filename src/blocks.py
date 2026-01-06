# src/blocks.py - Basic building blocks and initialization helpers
"""
Core neural network building blocks used across the codebase.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, BlockMask
from typing import Optional, Dict, Any, Tuple

# Triton for sparse MoE kernels - JIT compiles to PTX, works cross-platform
import triton
import triton.language as tl


# === Initialization Helpers ===

def init_linear(m: nn.Linear, std=0.02):
    """Initialize a linear layer with Xavier uniform weights and zero bias."""
    if hasattr(m, 'weight'):
        torch.nn.init.xavier_uniform_(m.weight)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.zeros_(m.bias)


def init_layer_norm(m):
    """Initialize a layer norm with ones for weight and zeros for bias."""
    if hasattr(m, 'weight') and m.weight is not None:
        nn.init.ones_(m.weight)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.zeros_(m.bias)


def propagate_param_init(module):
    """
    Recursively calls param_init() on all submodules that define it.
    """
    if hasattr(module, 'param_init'):
        module.param_init()

    for child in module.children():
        propagate_param_init(child)


# === FFN Blocks ===

class SwiGLU(nn.Module):
    """SwiGLU feedforward block: SiLU-gated linear unit."""
    def __init__(self, dim, hidden_dim, bias=False):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.param_init()

    def param_init(self):
        init_linear(self.w12)
        init_linear(self.w3)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class SigmoidMoE(nn.Module):
    """
    Sigmoid-gated Mixture of Experts with sparse computation via grouped_gemm.

    Uses permute/gmm/unpermute pattern: sort tokens by expert assignment, run each
    expert on its contiguous slice via grouped GEMM, scatter results back.
    This is O(k*N) compute for k active experts, not O(E*N).
    """
    def __init__(self, dim, hidden_dim, num_experts=8, num_active=2, jitter_noise=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.num_active = num_active
        self.jitter_noise = jitter_noise
        self.hidden_dim = hidden_dim
        self.dim = dim

        self.router = nn.Linear(dim, num_experts)

        # Stacked expert weights for grouped GEMM
        # w1: [E, 2*H, D] for gate+up projection
        # w2: [E, D, H] for down projection
        self.w1 = nn.Parameter(torch.empty(num_experts, 2 * hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))

        self.param_init()

    def param_init(self):
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        for e in range(self.num_experts):
            nn.init.xavier_uniform_(self.w1.data[e])
            nn.init.xavier_uniform_(self.w2.data[e])

    def forward(self, x):
        B, L, D = x.shape
        N = B * L
        K = self.num_active
        E = self.num_experts

        # 1. Routing
        router_logits = self.router(x)  # [B, L, E]
        if self.training and self.jitter_noise > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.jitter_noise

        scores = torch.sigmoid(router_logits)  # [B, L, E]
        top_k_scores, top_k_indices = torch.topk(scores, K, dim=-1)  # [B, L, K]

        # Normalize weights
        denom = top_k_scores.sum(dim=-1, keepdim=True) + 1e-6
        router_weights = (top_k_scores / denom).to(x.dtype)  # [B, L, K]

        # 2. Flatten for grouped_gemm ops
        x_flat = x.view(N, D).contiguous()
        indices_flat = top_k_indices.view(N, K).to(torch.int32).contiguous()
        weights_flat = router_weights.view(N, K).contiguous()

        # 3. Permute: sort tokens by expert assignment
        # permuted_x: [N*K, D] - each token appears K times, grouped by expert
        permuted_x, row_id_map = grouped_gemm.ops.permute(x_flat, indices_flat)

        # 4. Compute batch_sizes: tokens per expert
        expert_counts = torch.zeros(E, dtype=torch.int64, device=x.device)
        expert_counts.scatter_add_(0, indices_flat.view(-1).long(),
                                   torch.ones(N * K, dtype=torch.int64, device=x.device))

        # 5. Expert FFN via grouped GEMM
        # x @ w1.T -> [N*K, 2*H]
        h = grouped_gemm.ops.gmm(permuted_x, self.w1, expert_counts, trans_b=True)

        # SwiGLU activation
        h1, h2 = h.chunk(2, dim=-1)
        h = F.silu(h1) * h2  # [N*K, H]

        # h @ w2.T -> [N*K, D]
        expert_out = grouped_gemm.ops.gmm(h, self.w2, expert_counts, trans_b=True)

        # 6. Unpermute: scatter back with weights
        out_flat = grouped_gemm.ops.unpermute(expert_out, row_id_map, weights_flat)

        aux_loss = 1e-2 * (router_logits ** 2).mean()
        return out_flat.view(B, L, D), aux_loss


class MLPResBlock(nn.Module):
    """MLP residual block with RMSNorm and SwiGLU."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.RMSNorm(dim, elementwise_affine=False)
        self.net = SwiGLU(dim, dim*2)
        self.param_init()

    def param_init(self):
        init_layer_norm(self.norm)
        self.net.param_init()

    def forward(self, x):
        return x + self.net(self.norm(x))


# === Encoder Blocks (GQA-compatible, no RoPE) ===

class EncoderAttention(nn.Module):
    """
    Flexible attention layer with GQA support.

    Supports grouped-query attention where n_kv_heads < n_heads.
    No position encoding (RoPE) - positions are handled via external masks.
    Suitable for encoder-only models like autoencoders.

    Args:
        dim: Model dimension
        n_heads: Number of query heads
        n_kv_heads: Number of key/value heads (defaults to n_heads for MHA)
    """
    def __init__(self, dim: int, n_heads: int = 8, n_kv_heads: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = dim // n_heads

        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"
        assert n_heads % self.n_kv_heads == 0, f"n_heads {n_heads} must be divisible by n_kv_heads {self.n_kv_heads}"

        self.heads_per_kv = n_heads // self.n_kv_heads

        # Separate projections for GQA
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.head_dim * self.n_kv_heads, bias=False)
        self.v_proj = nn.Linear(dim, self.head_dim * self.n_kv_heads, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.param_init()

    def param_init(self):
        init_linear(self.q_proj)
        init_linear(self.k_proj)
        init_linear(self.v_proj)
        init_linear(self.out_proj)

    def forward(self, x: torch.Tensor, block_mask: Optional[BlockMask] = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] input tensor
            block_mask: Optional BlockMask for sparse attention
        Returns:
            [B, N, D] output tensor
        """
        B, N, D = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # GQA: expand KV heads to match query heads
        if self.heads_per_kv > 1:
            k = k.repeat_interleave(self.heads_per_kv, dim=1)
            v = v.repeat_interleave(self.heads_per_kv, dim=1)

        # Apply attention
        out = flex_attention(q, k, v, block_mask=block_mask)

        # Project output
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class EncoderBlock(nn.Module):
    """
    Configurable transformer block for encoder-only models.

    Features:
    - GQA support via n_kv_heads
    - Gated attention residual (matches main transformer)
    - Choice of MLP (simple or MoE)
    - No position encoding (handled via external masks)

    Args:
        dim: Model dimension
        n_heads: Number of query heads
        n_kv_heads: Number of key/value heads (defaults to n_heads for MHA)
        mlp_ratio: MLP hidden dimension multiplier
        use_moe: Whether to use mixture of experts for MLP
        num_experts: Number of experts (if use_moe=True)
        num_active: Number of active experts per token (if use_moe=True)
        jitter_noise: Router jitter noise (if use_moe=True)
    """
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        mlp_ratio: float = 4.0,
        use_moe: bool = False,
        num_experts: int = 8,
        num_active: int = 2,
        jitter_noise: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.use_moe = use_moe

        # Attention
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.attn = EncoderAttention(dim, n_heads, n_kv_heads)
        self.gate_proj = nn.Linear(dim, dim, bias=False)

        # MLP
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)
        hidden_dim = int(dim * mlp_ratio)

        if use_moe:
            self.mlp = SigmoidMoE(dim, hidden_dim, num_experts=num_experts,
                                  num_active=num_active, jitter_noise=jitter_noise)
        else:
            self.mlp = SwiGLU(dim, hidden_dim)

        self.param_init()

    def param_init(self):
        init_layer_norm(self.norm1)
        self.attn.param_init()
        init_linear(self.gate_proj)
        init_layer_norm(self.norm2)
        if hasattr(self.mlp, 'param_init'):
            self.mlp.param_init()

    def forward(self, x: torch.Tensor, block_mask: Optional[BlockMask] = None):
        """
        Args:
            x: [B, N, D] input tensor
            block_mask: Optional BlockMask for sparse attention
        Returns:
            If use_moe: (output, aux_loss)
            Otherwise: output
        """
        # Attention with gated residual
        h = self.attn(self.norm1(x), block_mask=block_mask)
        gate = torch.sigmoid(self.gate_proj(h))
        x = x + h * gate

        # MLP
        if self.use_moe:
            h_mlp, aux_loss = self.mlp(self.norm2(x))
            x = x + h_mlp
            return x, aux_loss
        else:
            x = x + self.mlp(self.norm2(x))
            return x


# === Transformer Stacks ===
# Import here to avoid circular dependency (context_manager may import from blocks)
from .context_manager import get_encoder_mask_for_layer


def _uses_registers(mode: str) -> bool:
    """Check if mode uses register tokens (bigbird or gemma_bigbird)."""
    return mode in ('bigbird', 'gemma_bigbird')


class TransformerEncoder(nn.Module):
    """Transformer encoder stack using EncoderBlock with configurable attention.

    Supports multiple attention patterns:
        - 'full': All layers use full attention (no mask)
        - 'sliding': All layers use sliding window
        - 'bigbird': All layers use BigBird (local + global register tokens)
        - 'gemma': Alternating local/global (every Nth layer is global)
        - 'gemma_bigbird': Alternating sliding/bigbird (configurable layout)

    Uses EncoderBlock from src.blocks which provides:
    - GQA (grouped query attention) via n_kv_heads
    - Gated attention residuals
    - SwiGLU MLP (or optional MoE)

    For bigbird/gemma_bigbird modes, learnable register tokens are prepended
    to the sequence. These act as global memory that all positions can attend to.
    """

    def __init__(self, dim: int = 256, n_layers: int = 4,
                 attn_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.n_layers = n_layers

        # Default config
        if attn_config is None:
            attn_config = {'mode': 'full', 'window_size': 4.0, 'global_layer_interval': 4,
                          'n_query_heads': 8, 'n_kv_heads': 2, 'n_global_tokens': 4}

        # Store config for mask building in forward()
        self.attn_config = attn_config
        self.mode = attn_config['mode']
        n_heads = attn_config['n_query_heads']
        n_kv_heads = attn_config['n_kv_heads']
        self.n_global_tokens = attn_config['n_global_tokens']

        # Learnable register tokens for bigbird modes
        self.uses_registers = _uses_registers(self.mode)
        if self.uses_registers:
            self.register_tokens = nn.Parameter(torch.randn(1, self.n_global_tokens, dim) * 0.02)

        # Create layers using shared EncoderBlock
        self.layers = nn.ModuleList([
            EncoderBlock(dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, grid_shape: Optional[Tuple[int, int]] = None,
                block_masks: Optional[list] = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] input patches
            grid_shape: (H, W) spatial grid dimensions. Required for non-full attention modes.
                        If None, infers square grid from N.
            block_masks: Optional list of pre-built masks (one per layer). If provided,
                        these are used instead of building masks dynamically. This avoids
                        inductor bounds analysis issues when compiled with dynamic shapes.
        """
        B, N, D = x.shape

        # Infer grid shape if not provided
        if grid_shape is None:
            side = int(N ** 0.5)
            grid_shape = (side, side)

        # Prepend register tokens for bigbird modes
        if self.uses_registers:
            registers = self.register_tokens.expand(B, -1, -1)  # [B, n_global, D]
            x = torch.cat([registers, x], dim=1)  # [B, n_global + N, D]

        # Use pre-built masks if provided, otherwise build dynamically
        device = x.device
        for i, layer in enumerate(self.layers):
            if block_masks is not None:
                mask = block_masks[i]
            else:
                mask = get_encoder_mask_for_layer(grid_shape, i, self.attn_config, device)
            x = layer(x, block_mask=mask)

        # Remove register tokens before returning
        if self.uses_registers:
            x = x[:, self.n_global_tokens:]  # [B, N, D]

        return x

    def build_masks(self, grid_shape: Tuple[int, int], device: torch.device) -> list:
        """
        Build attention masks for all layers.

        Call this OUTSIDE torch.compile to avoid inductor bounds analysis issues.

        Args:
            grid_shape: (H, W) spatial grid dimensions
            device: Target device
        Returns:
            List of BlockMask (one per layer)
        """
        return [
            get_encoder_mask_for_layer(grid_shape, i, self.attn_config, device)
            for i in range(self.n_layers)
        ]


class TransformerDecoder(nn.Module):
    """Transformer decoder stack using EncoderBlock with configurable attention.

    Same attention modes as TransformerEncoder. Uses layer offset of 100
    to ensure different mask patterns than encoder layers.
    """

    def __init__(self, dim: int = 256, n_layers: int = 4,
                 attn_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.n_layers = n_layers

        # Default config
        if attn_config is None:
            attn_config = {'mode': 'full', 'window_size': 4.0, 'global_layer_interval': 4,
                          'n_query_heads': 8, 'n_kv_heads': 2, 'n_global_tokens': 4}

        # Store config for mask building in forward()
        self.attn_config = attn_config
        self.mode = attn_config['mode']
        n_heads = attn_config['n_query_heads']
        n_kv_heads = attn_config['n_kv_heads']
        self.n_global_tokens = attn_config['n_global_tokens']

        # Learnable register tokens for bigbird modes
        self.uses_registers = _uses_registers(self.mode)
        if self.uses_registers:
            self.register_tokens = nn.Parameter(torch.randn(1, self.n_global_tokens, dim) * 0.02)

        # Create layers using shared EncoderBlock
        self.layers = nn.ModuleList([
            EncoderBlock(dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, grid_shape: Optional[Tuple[int, int]] = None,
                block_masks: Optional[list] = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] input codes
            grid_shape: (H, W) spatial grid dimensions. Required for non-full attention modes.
                        If None, infers square grid from N.
            block_masks: Optional list of pre-built masks (one per layer). If provided,
                        these are used instead of building masks dynamically.
        """
        B, N, D = x.shape

        # Infer grid shape if not provided
        if grid_shape is None:
            side = int(N ** 0.5)
            grid_shape = (side, side)

        # Prepend register tokens for bigbird modes
        if self.uses_registers:
            registers = self.register_tokens.expand(B, -1, -1)
            x = torch.cat([registers, x], dim=1)

        # Use pre-built masks if provided, otherwise build dynamically
        device = x.device
        for i, layer in enumerate(self.layers):
            if block_masks is not None:
                mask = block_masks[i]
            else:
                mask = get_encoder_mask_for_layer(grid_shape, i + 100, self.attn_config, device)
            x = layer(x, block_mask=mask)

        # Remove register tokens before returning
        if self.uses_registers:
            x = x[:, self.n_global_tokens:]  # [B, N, D]

        return x

    def build_masks(self, grid_shape: Tuple[int, int], device: torch.device) -> list:
        """
        Build attention masks for all layers.

        Call this OUTSIDE torch.compile to avoid inductor bounds analysis issues.

        Args:
            grid_shape: (H, W) spatial grid dimensions
            device: Target device
        Returns:
            List of BlockMask (one per layer)
        """
        return [
            get_encoder_mask_for_layer(grid_shape, i + 100, self.attn_config, device)
            for i in range(self.n_layers)
        ]
