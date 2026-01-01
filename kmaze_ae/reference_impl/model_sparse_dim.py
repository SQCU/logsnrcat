import torch
import torch.nn as nn
import torch.nn.functional as F


class GQATransformerLayer(nn.Module):
    def __init__(self, dim: int, n_query_heads: int = 8, n_kv_heads: int = 2):
        super().__init__()
        self.dim = dim
        self.n_query_heads = n_query_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_query_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, self.head_dim * n_kv_heads)
        self.v_proj = nn.Linear(dim, self.head_dim * n_kv_heads)
        self.out_proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        normed = self.norm1(x)

        q = self.q_proj(normed).view(B, N, self.n_query_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)

        heads_per_kv = self.n_query_heads // self.n_kv_heads
        k = k.repeat_interleave(heads_per_kv, dim=1)
        v = v.repeat_interleave(heads_per_kv, dim=1)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, D)
        x = x + self.out_proj(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim: int = 256, n_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([GQATransformerLayer(dim) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim: int = 256, n_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([GQATransformerLayer(dim) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class BinaryFSQ(nn.Module):
    """Binary FSQ with STE."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        soft = torch.sigmoid(x)
        hard = (soft > 0.5).float()
        return hard - soft.detach() + soft


class ThreeBitFSQ(nn.Module):
    """3-bit FSQ (8 levels: 0-7) with STE."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sigmoid to [0, 1], scale to [0, 7]
        soft = torch.sigmoid(x) * 7.0
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

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent: (B, N, code_dim) - pre-quantization latent
        Returns:
            sparse_latent: (B, N, code_dim) with ~95% zeros
            gate_weights: (B, N, code_dim) soft weights for logging
        """
        B, N, D = latent.shape

        # Predict gate from latent
        gate_logits = self.gate_proj(latent)  # (B, N, D)
        gate_weights = torch.sigmoid(gate_logits)  # (B, N, D)

        # Top-k per patch
        _, topk_idx = gate_weights.topk(self.k, dim=-1)  # (B, N, k)

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
    def __init__(self, patch_dim: int = 768, hidden_dim: int = 256, code_dim: int = 128, k_per_patch: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(patch_dim, hidden_dim)
        self.transformer = TransformerEncoder(hidden_dim, n_layers=4)
        self.code_proj = nn.Linear(hidden_dim, code_dim)
        self.sparsity = PerDimSparsity(code_dim, k_per_patch)
        self.fsq = ThreeBitFSQ()  # 3-bit (8 levels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (sparse_codes, gate_weights, pre_sparse)"""
        h = self.input_proj(x)
        h = self.transformer(h)
        pre_quant = self.code_proj(h)

        # Quantize first
        codes = self.fsq(pre_quant)

        # Then apply sparsity (mask after FSQ so zeros stay zero)
        sparse_codes, gate_weights = self.sparsity(codes)

        return sparse_codes, gate_weights, pre_quant


class SparseLevelDecoder(nn.Module):
    def __init__(self, code_dim: int = 128, hidden_dim: int = 256, patch_dim: int = 768):
        super().__init__()
        self.input_proj = nn.Linear(code_dim, hidden_dim)
        self.transformer = TransformerDecoder(hidden_dim, n_layers=4)
        self.output_proj = nn.Linear(hidden_dim, patch_dim)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(codes)
        h = self.transformer(h)
        return self.output_proj(h)


class SparsePerDimFSQAutoencoder(nn.Module):
    """
    Hierarchical Binary FSQ with per-dim sparsity.

    128 dims, 95% sparsity (keep 6 dims per patch), 4 levels.
    """
    def __init__(
        self,
        n_levels: int = 4,
        patch_size: int = 16,
        image_size: int = 256,
        hidden_dim: int = 256,
        code_dim: int = 128,
        k_per_patch: int = 6,
        residual_scale: float = 2.0
    ):
        super().__init__()
        self.n_levels = n_levels
        self.patch_size = patch_size
        self.image_size = image_size
        self.n_patches = (image_size // patch_size) ** 2  # 256
        self.patch_dim = patch_size * patch_size * 3  # 768
        self.residual_scale = residual_scale
        self.code_dim = code_dim
        self.k_per_patch = k_per_patch

        self.encoders = nn.ModuleList([
            SparseLevelEncoder(self.patch_dim, hidden_dim, code_dim, k_per_patch)
            for _ in range(n_levels)
        ])
        self.decoders = nn.ModuleList([
            SparseLevelDecoder(code_dim, hidden_dim, self.patch_dim)
            for _ in range(n_levels)
        ])

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        B, C, H, W = images.shape
        p = self.patch_size
        patches = images.view(B, C, H // p, p, W // p, p)
        patches = patches.permute(0, 2, 4, 3, 5, 1).contiguous()
        return patches.view(B, self.n_patches, self.patch_dim)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        B = patches.shape[0]
        p = self.patch_size
        h = w = self.image_size // p
        patches = patches.view(B, h, w, p, p, 3)
        patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
        return patches.view(B, 3, self.image_size, self.image_size)

    def forward(self, images: torch.Tensor) -> dict:
        patches = self.patchify(images)

        level_recons = []
        codes_list = []
        all_gate_weights = []
        cumulative_recon = torch.zeros_like(patches)
        current_target = patches

        for level in range(self.n_levels):
            if level > 0:
                residual = (current_target - cumulative_recon) * self.residual_scale
            else:
                residual = current_target

            codes, gate_weights, _ = self.encoders[level](residual)
            codes_list.append(codes)
            all_gate_weights.append(gate_weights)

            decoded = self.decoders[level](codes)

            if level > 0:
                decoded = decoded / self.residual_scale

            cumulative_recon = cumulative_recon + decoded
            level_recons.append(self.unpatchify(cumulative_recon))

        # Compute sparsity stats
        # codes are (B, 256, 128), count non-zeros
        total_codes = codes_list[0].numel()
        nonzero_codes = sum((c != 0).sum().item() for c in codes_list)
        sparsity = 1.0 - (nonzero_codes / (total_codes * self.n_levels))

        return {
            'recon': level_recons[-1],
            'level_recons': level_recons,
            'codes': codes_list,
            'gate_weights': all_gate_weights,
            'sparsity': sparsity
        }

    def encode(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Encode images to sparse binary codes."""
        patches = self.patchify(images)

        codes_list = []
        cumulative_recon = torch.zeros_like(patches)
        current_target = patches

        for level in range(self.n_levels):
            if level > 0:
                residual = (current_target - cumulative_recon) * self.residual_scale
            else:
                residual = current_target

            codes, _, _ = self.encoders[level](residual)
            codes_list.append(codes)

            decoded = self.decoders[level](codes)

            if level > 0:
                decoded = decoded / self.residual_scale

            cumulative_recon = cumulative_recon + decoded

        return codes_list
