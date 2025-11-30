"""
diffusion_checkerboard_rn_rope.py

Architecture: Gemma-style DiT (3:1 Hybrid Window/Global Attention).
Positional: Rn RoPE (Householder Parameterization).
Mechanism: Gated Attention + FlexAttention.
Dataset: Aliased Rotated Checkerboards.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import numpy as np
import matplotlib.pyplot as plt
import math

# ============================================================================
# 1. Householder Orthogonal Matrix (Lie Group Theory)
# ============================================================================

class HouseholderOrthogonal(nn.Module):
    """
    Parameterizes a learnable Orthogonal Matrix Q using Householder reflections.
    H = I - 2 * (v v^T) / (v^T v)
    Q = H_1 * H_2 * ... * H_k
    """
    def __init__(self, dim, num_reflections=4):
        super().__init__()
        self.dim = dim
        self.vs = nn.Parameter(torch.randn(num_reflections, dim) * 0.02)
        
    def get_matrix(self):
        Q = torch.eye(self.dim, device=self.vs.device)
        for i in range(self.vs.shape[0]):
            v = self.vs[i].unsqueeze(1) # [dim, 1]
            v_norm_sq = torch.sum(v ** 2) + 1e-8
            term = (2.0 / v_norm_sq) * v @ (v.t() @ Q)
            Q = Q - term
        return Q

    def forward(self, x, inverse=False):
        Q = self.get_matrix()
        if inverse:
            return x @ Q.t()
        else:
            return x @ Q

# ============================================================================
# 2. The Spatial Buffer & Rn RoPE
# ============================================================================

class SpatialBuffer(nn.Module):
    def __init__(self, grid_size=8, head_dim=64, device='cuda'):
        super().__init__()
        self.grid_size = grid_size
        
        # 1. Coordinates
        y = torch.arange(grid_size, device=device)
        x = torch.arange(grid_size, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        
        # [64, 2]
        self.coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1).float()
        
        # 2. Distance Matrix (for SWA Mask)
        d = torch.cdist(self.coords, self.coords, p=2)
        self.register_buffer('dist_matrix', d)
        
        # 3. RoPE Frequencies
        self.dim_half = head_dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim_half, 2, device=device).float() / self.dim_half))
        self.register_buffer('inv_freq', inv_freq)

    def get_standard_rope_cos_sin(self, B):
        y_pos = self.coords[:, 0]
        x_pos = self.coords[:, 1]
        
        freqs_y = torch.einsum('i, j -> i j', y_pos, self.inv_freq)
        freqs_x = torch.einsum('i, j -> i j', x_pos, self.inv_freq)
        
        freqs = torch.cat([freqs_y, freqs_x], dim=-1) # [64, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1) # [64, head_dim]
        
        return emb.cos()[None], emb.sin()[None]

class RnRoPE(nn.Module):
    def __init__(self, head_dim, spatial_buffer):
        super().__init__()
        self.buffer = spatial_buffer
        self.orthogonal = HouseholderOrthogonal(head_dim, num_reflections=head_dim//2)
        
    def forward(self, q, k):
        # 1. Rotate into interaction basis (Apply Q.T)
        q = self.orthogonal(q, inverse=True)
        k = self.orthogonal(k, inverse=True)
        
        # 2. Apply Standard RoPE
        B = q.shape[0]
        cos, sin = self.buffer.get_standard_rope_cos_sin(B)
        cos = cos.unsqueeze(1) 
        sin = sin.unsqueeze(1)
        
        q_rot, k_rot = self.apply_standard_rotary(q, k, cos, sin)
        
        # 3. Rotate back (Apply Q)
        q_final = self.orthogonal(q_rot, inverse=False)
        k_final = self.orthogonal(k_rot, inverse=False)
        return q_final, k_final

    def apply_standard_rotary(self, q, k, cos, sin):
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        rotate_q = torch.cat((-q2, q1), dim=-1)
        rotate_k = torch.cat((-k2, k1), dim=-1)
        q_out = (q * cos) + (rotate_q * sin)
        k_out = (k * cos) + (rotate_k * sin)
        return q_out, k_out

# ============================================================================
# 3. Masks (FIXED)
# ============================================================================

def get_sliding_window_mask(dist_matrix, radius=2.5):
    """
    Robust sliding window mask that handles FlexAttention block padding safely.
    """
    Q_LEN, KV_LEN = dist_matrix.shape
    valid_mask = dist_matrix < radius # [64, 64] bool
    
    def mask_mod(b, h, q_idx, kv_idx):
        # CLAMP INDICES to prevent CUDA Asserts when FlexAttention pads blocks
        q_safe = q_idx.clamp(0, Q_LEN - 1)
        k_safe = kv_idx.clamp(0, KV_LEN - 1)
        
        # Index into the captured tensor
        val = valid_mask[q_safe, k_safe]
        
        # Logical mask for OOB threads
        in_bounds = (q_idx < Q_LEN) & (kv_idx < KV_LEN)
        
        return val & in_bounds

    return create_block_mask(mask_mod, B=1, H=1, Q_LEN=Q_LEN, KV_LEN=KV_LEN)

def get_global_mask():
    def mask_mod(b, h, q_idx, kv_idx):
        # MUST RETURN TENSOR, NOT BOOL
        return q_idx > -1 
    return create_block_mask(mask_mod, B=1, H=1, Q_LEN=64, KV_LEN=64)

class GatedAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, spatial_buffer, is_global=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.is_global = is_global
        
        self.norm1 = nn.RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.rn_rope = RnRoPE(self.head_dim, spatial_buffer)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, dim)
        
        self.norm2 = nn.RMSNorm(dim)
        self.mlp_gate = nn.Linear(dim, dim * 4 * 2) 
        self.mlp_out = nn.Linear(dim * 4, dim)

    def forward(self, x, block_mask):
        B, S, D = x.shape
        resid = x
        x_norm = self.norm1(x)
        
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        q, k = self.rn_rope(q, k)
        
        attn_out = flex_attention(q, k, v, block_mask=block_mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        attn_out = self.out_proj(attn_out)
        
        gate = torch.sigmoid(self.gate_proj(attn_out))
        attn_out = attn_out * gate
        x = resid + attn_out
        
        resid = x
        x_norm = self.norm2(x)
        mlp_raw = self.mlp_gate(x_norm)
        g, val = mlp_raw.chunk(2, dim=-1)
        mlp_act = val * F.gelu(g)
        mlp_out = self.mlp_out(mlp_act)
        x = resid + mlp_out
        return x

# ============================================================================
# 4. The Transformer (3:1 Ratio)
# ============================================================================

class HybridGemmaDiT(nn.Module):
    def __init__(self, mode='factorized', embed_dim=256, depth=8):
        super().__init__()
        self.mode = mode
        
        self.patch_in = nn.Linear(20, embed_dim)
        self.spin_encoder = FourierFeatures(num_bands=4)
        
        self.spatial = SpatialBuffer(grid_size=8, head_dim=embed_dim//4, device='cuda')
        self.mask_local = get_sliding_window_mask(self.spatial.dist_matrix, radius=2.5)
        self.mask_global = get_global_mask()
        
        self.layers = nn.ModuleList()
        for i in range(depth):
            is_global = ((i + 1) % 4 == 0)
            self.layers.append(
                GatedAttentionBlock(
                    dim=embed_dim, 
                    num_heads=4, 
                    spatial_buffer=self.spatial,
                    is_global=is_global
                )
            )
            
        self.norm_final = nn.RMSNorm(embed_dim)
        if mode == 'factorized':
            self.head = nn.Linear(embed_dim, embed_dim + 1)
            self.patch_out = nn.Linear(embed_dim, 12)
        else:
            self.head = nn.Linear(embed_dim, embed_dim)
            self.patch_out = nn.Linear(embed_dim, 12)

    def forward(self, z_t, logsnr):
        B = z_t.shape[0]
        patches = z_t.unfold(2, 2, 2).unfold(3, 2, 2).permute(0, 2, 3, 1, 4, 5).reshape(B, 64, 12)
        spins = self.spin_encoder(logsnr).unsqueeze(1).expand(-1, 64, -1)
        x = self.patch_in(torch.cat([patches, spins], dim=-1))
        
        for layer in self.layers:
            mask = self.mask_global if layer.is_global else self.mask_local
            x = layer(x, mask)
            
        x = self.norm_final(x)
        out = self.head(x)
        
        if self.mode == 'factorized':
            z_pred = self.patch_out(out[:, :, :-1])
            l_pred = out[:, :, -1].mean(dim=1)
        else:
            z_pred = self.patch_out(out)
            l_pred = None
            
        z_pred = z_pred.view(B, 8, 8, 3, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, 3, 16, 16)
        return z_pred, l_pred

# ============================================================================
# 5. Helpers
# ============================================================================

class FourierFeatures(nn.Module):
    def __init__(self, num_bands=4, max_range=40.0):
        super().__init__()
        base_freq = 2 * np.pi / max_range
        freqs = base_freq * (2.0 ** torch.arange(num_bands))
        self.register_buffer('freqs', freqs)
    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(-1)
        args = x * self.freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

def get_schedule(t): return 20.0 - 40.0 * t
def get_alpha_sigma(logsnr):
    return torch.sqrt(torch.sigmoid(logsnr)), torch.sqrt(torch.sigmoid(-logsnr))

def generate_rotated_checkerboards(B, device='cuda'):
    linspace = torch.linspace(-8, 8, 16, device=device)
    y, x = torch.meshgrid(linspace, linspace, indexing='ij')
    x_flat = x.flatten().unsqueeze(0).expand(B, -1)
    y_flat = y.flatten().unsqueeze(0).expand(B, -1)
    
    theta = torch.rand(B, 1, device=device) * 2 * math.pi
    cos_t = torch.cos(theta); sin_t = torch.sin(theta)
    
    x_rot = x_flat * cos_t + y_flat * sin_t
    y_rot = -x_flat * sin_t + y_flat * cos_t
    
    scale = 4.0
    x_idx = torch.floor(x_rot / scale + 0.01)
    y_idx = torch.floor(y_rot / scale + 0.01)
    
    pat = ((x_idx + y_idx) % 2).view(B, 16, 16)
    
    c1 = torch.rand(B, 3, 1, 1, device=device)
    c2 = torch.rand(B, 3, 1, 1, device=device)
    mask = pat.unsqueeze(1)
    return c1 * (1 - mask) + c2 * mask

# ============================================================================
# 6. Execution
# ============================================================================

def run_experiment():
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    
    # Must compile for FlexAttention
    model = HybridGemmaDiT('factorized', depth=8).to(device)
    model = torch.compile(model)
    
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    
    print("Training Hybrid Gemma (3:1 SWA/Global) with Rn-RoPE & Gated Attn...")
    
    for i in range(1001):
        opt.zero_grad()
        
        x0 = generate_rotated_checkerboards(2048, device)
        t = torch.rand(2048, device=device).clamp(0.001, 0.999)
        logsnr = get_schedule(t)
        alpha, sigma = get_alpha_sigma(logsnr)
        
        eps = torch.randn_like(x0)
        z_t = x0 * alpha.view(-1,1,1,1) + eps * sigma.view(-1,1,1,1)
        
        raw, l_pred = model(z_t, logsnr)
        
        sigma_p = torch.sqrt(torch.sigmoid(-l_pred)).view(-1, 1, 1, 1)
        loss = F.mse_loss(raw * sigma_p, eps)
        
        loss.backward()
        opt.step()
        
        if i % 100 == 0:
            print(f"Step {i}: Loss {loss.item():.5f}")
            
    print("Sampling...")
    model.eval()
    with torch.no_grad():
        z = torch.randn(8, 3, 16, 16, device=device)
        ts = torch.linspace(1.0, 0.001, 50, device=device)
        
        for i in range(len(ts)-1):
            t_cur = ts[i]
            logsnr = get_schedule(torch.full((8,), t_cur, device=device))
            raw, l_pred = model(z, logsnr)
            
            sigma_p = torch.sqrt(torch.sigmoid(-l_pred)).view(-1,1,1,1)
            eps = raw * sigma_p
            
            logsnr_next = get_schedule(torch.full((8,), ts[i+1], device=device))
            _, sigma_c = get_alpha_sigma(logsnr)
            _, sigma_n = get_alpha_sigma(logsnr_next)
            
            z = z + (sigma_n.view(-1,1,1,1) - sigma_c.view(-1,1,1,1)) * eps
            
    z = z.clamp(0, 1).cpu()
    fig, axes = plt.subplots(1, 8, figsize=(16, 2))
    for i in range(8):
        axes[i].imshow(z[i].permute(1, 2, 0))
        axes[i].axis('off')
    plt.show()

if __name__ == "__main__":
    run_experiment()