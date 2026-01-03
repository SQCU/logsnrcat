# src/rope.py - Rotary Position Embeddings for N-dimensional spaces
"""
Implements RnRoPE (N-dimensional Rotary Position Embeddings) using
Householder reflections for orthogonal transformations.

Reference: https://arxiv.org/abs/2504.06308
"Rethinking RoPE: A Mathematical Blueprint for N-dimensional Rotary Positional Embedding"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HouseholderOrthogonal(nn.Module):
    """
    Parametrized Orthogonal Matrix via product of Householder reflections.
    Used to project N-dimensional spatial coordinates into the rotation subspace.
    """
    def __init__(self, dim, num_reflections=4):
        super().__init__()
        self.dim = dim
        self.num_reflections = num_reflections
        self.vs = nn.Parameter(torch.empty(num_reflections, dim))
        self.param_init()

    def param_init(self):
        # Initialize vectors with small random noise
        nn.init.normal_(self.vs, mean=0.0, std=0.02)

    def get_matrix(self):
        # Start with Identity
        Q = torch.eye(self.dim, device=self.vs.device, dtype=self.vs.dtype)
        # Iteratively apply reflections: H = I - 2vv^T / ||v||^2
        for i in range(self.vs.shape[0]):
            v = self.vs[i].unsqueeze(1)
            v_norm_sq = torch.sum(v ** 2) + 1e-8
            # Q_new = (I - 2vv'/v'v) Q_old = Q_old - (2/v'v) v (v' Q_old)
            term = (2.0 / v_norm_sq) * v @ (v.t() @ Q)
            Q = Q - term
        return Q

    def forward(self, x, inverse=False):
        Q = self.get_matrix()
        return x @ Q.t() if inverse else x @ Q


class RnRoPE(nn.Module):
    """
    N-dimensional Rotary Position Embedding.

    Extends standard RoPE to handle multi-dimensional topology coordinates
    (e.g., temporal highway + 2D spatial grid for images).

    Args:
        head_dim: Dimension of each attention head
        topo_dim: Number of topology dimensions (e.g., 3 for highway + 2D spatial)
        rope_base: Base frequency for position encoding
    """
    def __init__(self, head_dim: int, topo_dim: int, rope_base: float = 500.0):
        super().__init__()
        self.head_dim = head_dim
        self.topo_dim = topo_dim
        self.freq_dim = head_dim // 2

        # Householder rotation for latent space projection
        self.orthogonal = HouseholderOrthogonal(head_dim, num_reflections=head_dim//2)

        # Calculate how many frequency bands each topology dimension gets.
        # e.g., Head=64 -> Freq=32. Topo=3 -> 10 bands per dim.
        self.features_per_subspace = self.freq_dim // topo_dim

        self.register_buffer(
            'inv_freq',
            1.0 / (rope_base ** (torch.arange(0, self.features_per_subspace).float() / self.features_per_subspace))
        )

        self.param_init()

    def param_init(self):
        self.orthogonal.param_init()

    def forward(self, q: torch.Tensor, k: torch.Tensor, topo_embeds: torch.Tensor, scale: float = 1.0):
        """
        Apply rotary position embeddings based on topology coordinates.

        Args:
            q, k: [B, H, L, D] query and key tensors
            topo_embeds: [B, L, Topo_Dim] topology coordinates
            scale: Scaling factor for context length generalization (inv_freq / scale)

        Returns:
            q_rot, k_rot: Position-encoded query and key tensors
        """
        B, H, L, D = q.shape

        # 1. Rotate into frequency-friendly space
        # Collapse B, H, L for efficient matmul
        # Note: .contiguous() after reshape prevents stride accumulation that can
        # overflow 32-bit C long on Windows when compiled with dynamic shapes
        q = self.orthogonal(q.reshape(B*H*L, D), inverse=True).reshape(B, H, L, D).contiguous()
        k = self.orthogonal(k.reshape(B*H*L, D), inverse=True).reshape(B, H, L, D).contiguous()

        # 2. Vectorized Frequency Computation
        # Slice inputs to supported dimensions (handles implicit truncation if input has extra dims)
        t_embeds = topo_embeds[..., :self.topo_dim]  # [B, L, Topo_Dim]

        # Scale frequencies (Context Generalization)
        inv_freq_scaled = self.inv_freq / scale  # [Subspace_Dim]

        # Compute phases: Outer Product
        # [B, L, Topo, 1] * [1, 1, 1, Subspace] -> [B, L, Topo, Subspace]
        freqs = t_embeds.unsqueeze(-1) * inv_freq_scaled.view(1, 1, 1, -1)

        # Flatten to single frequency vector: [B, L, Topo * Subspace]
        full_freqs = freqs.view(B, L, -1)

        # 3. Pad to match freq_dim (head_dim // 2)
        # We prefer padding over branching. If Topo*Subspace < Freq_Dim, we pad zeros.
        # (Zero freq = No rotation = Identity for those dimensions, which is safe).
        curr_dim = full_freqs.shape[-1]
        if curr_dim < self.freq_dim:
            full_freqs = F.pad(full_freqs, (0, self.freq_dim - curr_dim))

        # 4. Create Rotation Matrices
        # [B, L, freq_dim] -> [B, 1, L, freq_dim] -> [B, 1, L, head_dim]
        # Duplicate for real/imaginary parts
        # Use expand + contiguous instead of repeat to avoid stride issues with dynamic compile
        cos_base = full_freqs.cos().unsqueeze(1)  # [B, 1, L, freq_dim]
        sin_base = full_freqs.sin().unsqueeze(1)
        cos = torch.cat([cos_base, cos_base], dim=-1)[..., :D].contiguous()
        sin = torch.cat([sin_base, sin_base], dim=-1)[..., :D].contiguous()

        # 5. Apply RoPE (Standard Rotate Half)
        def rotate_half(x):
            x1, x2 = x[..., :D//2], x[..., D//2:]
            return torch.cat([-x2, x1], dim=-1)

        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)

        # 6. Rotate back to original basis
        q_out = self.orthogonal(q_rot.reshape(B*H*L, D), inverse=False).reshape(B, H, L, D).contiguous()
        k_out = self.orthogonal(k_rot.reshape(B*H*L, D), inverse=False).reshape(B, H, L, D).contiguous()

        return q_out, k_out
