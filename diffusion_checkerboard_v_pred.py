"""
diffusion_checkerboard_v_pred.py

Change Log:
1. Target: V-Prediction (alpha * eps - sigma * x0).
2. Sampler: Euler Ancestral for V.
3. Architecture: Unchanged (Hybrid Gemma DiT + RnRoPE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import numpy as np
import matplotlib.pyplot as plt
import math

# [Insert HouseholderOrthogonal, SpatialBuffer, RnRoPE, get_sliding_window_mask, 
#  get_global_mask, GatedAttentionBlock, HybridGemmaDiT, FourierFeatures 
#  classes from previous script EXACTLY AS IS]

# ... [Copying definitions to ensure standalone execution] ...

class HouseholderOrthogonal(nn.Module):
    def __init__(self, dim, num_reflections=4):
        super().__init__()
        self.dim = dim
        self.vs = nn.Parameter(torch.randn(num_reflections, dim) * 0.02)
    def get_matrix(self):
        Q = torch.eye(self.dim, device=self.vs.device)
        for i in range(self.vs.shape[0]):
            v = self.vs[i].unsqueeze(1)
            v_norm_sq = torch.sum(v ** 2) + 1e-8
            term = (2.0 / v_norm_sq) * v @ (v.t() @ Q)
            Q = Q - term
        return Q
    def forward(self, x, inverse=False):
        Q = self.get_matrix()
        return x @ Q.t() if inverse else x @ Q

class SpatialBuffer(nn.Module):
    def __init__(self, grid_size=8, head_dim=64, device='cuda'):
        super().__init__()
        self.grid_size = grid_size
        y = torch.arange(grid_size, device=device)
        x = torch.arange(grid_size, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        self.coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1).float()
        d = torch.cdist(self.coords, self.coords, p=2)
        self.register_buffer('dist_matrix', d)
        self.dim_half = head_dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim_half, 2, device=device).float() / self.dim_half))
        self.register_buffer('inv_freq', inv_freq)
    def get_standard_rope_cos_sin(self, B):
        y_pos = self.coords[:, 0]; x_pos = self.coords[:, 1]
        freqs_y = torch.einsum('i, j -> i j', y_pos, self.inv_freq)
        freqs_x = torch.einsum('i, j -> i j', x_pos, self.inv_freq)
        freqs = torch.cat([freqs_y, freqs_x], dim=-1)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None], emb.sin()[None]

class RnRoPE(nn.Module):
    def __init__(self, head_dim, spatial_buffer):
        super().__init__()
        self.buffer = spatial_buffer
        self.orthogonal = HouseholderOrthogonal(head_dim, num_reflections=head_dim//2)
    def forward(self, q, k):
        q = self.orthogonal(q, inverse=True)
        k = self.orthogonal(k, inverse=True)
        B = q.shape[0]
        cos, sin = self.buffer.get_standard_rope_cos_sin(B)
        cos = cos.unsqueeze(1); sin = sin.unsqueeze(1)
        q_rot, k_rot = self.apply_standard_rotary(q, k, cos, sin)
        q_final = self.orthogonal(q_rot, inverse=False)
        k_final = self.orthogonal(k_rot, inverse=False)
        return q_final, k_final
    def apply_standard_rotary(self, q, k, cos, sin):
        q1, q2 = q.chunk(2, dim=-1); k1, k2 = k.chunk(2, dim=-1)
        rotate_q = torch.cat((-q2, q1), dim=-1); rotate_k = torch.cat((-k2, k1), dim=-1)
        return (q * cos) + (rotate_q * sin), (k * cos) + (rotate_k * sin)

def get_sliding_window_mask(dist_matrix, radius=2.5):
    Q_LEN, KV_LEN = dist_matrix.shape
    valid_mask = dist_matrix < radius
    def mask_mod(b, h, q_idx, kv_idx):
        q_safe = q_idx.clamp(0, Q_LEN - 1)
        k_safe = kv_idx.clamp(0, KV_LEN - 1)
        val = valid_mask[q_safe, k_safe]
        in_bounds = (q_idx < Q_LEN) & (kv_idx < KV_LEN)
        return val & in_bounds
    return create_block_mask(mask_mod, B=1, H=1, Q_LEN=Q_LEN, KV_LEN=KV_LEN)

def get_global_mask():
    def mask_mod(b, h, q_idx, kv_idx):
        return q_idx > -1 
    return create_block_mask(mask_mod, B=1, H=1, Q_LEN=64, KV_LEN=64)

class GatedAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, spatial_buffer, is_global=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.is_global = is_global
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.rn_rope = RnRoPE(self.head_dim, spatial_buffer)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, dim)
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)
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

class HybridGemmaDiT(nn.Module):
    def __init__(self, mode='v_pred', embed_dim=256, depth=8):
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
            self.layers.append(GatedAttentionBlock(dim=embed_dim, num_heads=4, spatial_buffer=self.spatial, is_global=is_global))
        self.norm_final = nn.RMSNorm(embed_dim)
        
        # Output: V prediction (12 dim per patch)
        # No lambda head needed for standard V-pred, but we keep structure
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
        
        z_pred = self.patch_out(out)
        z_pred = z_pred.view(B, 8, 8, 3, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, 3, 16, 16)
        return z_pred

def get_schedule(t): return 20.0 - 40.0 * t
def get_alpha_sigma(logsnr):
    return torch.sqrt(torch.sigmoid(logsnr)), torch.sqrt(torch.sigmoid(-logsnr))

def generate_rotated_checkerboards(B, device='cuda'):
    # Verified Ground Truth Generator
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

def run_experiment():
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    
    # Init V-Prediction Model
    model = HybridGemmaDiT('v_pred', depth=4).to(device)
    model = torch.compile(model)
    # Separate Householder learning rate
    householder_params = [p for n,p in model.named_parameters() if 'orthogonal.vs' in n]
    other_params = [p for n,p in model.named_parameters() if 'orthogonal.vs' not in n]
    # Option B: Don't schedule Householder
    opt_main = torch.optim.AdamW(other_params, lr=5e-4, weight_decay=0.1)
    opt_house = torch.optim.AdamW(householder_params, lr=0.1, weight_decay=0.0)
    # Learning rate schedule with warmup + decay
    from torch.optim.lr_scheduler import OneCycleLR
    scheduler_main = OneCycleLR(opt_main, max_lr=1e-3, total_steps=10001, 
                        pct_start=0.1, div_factor=10, final_div_factor=100)
    scheduler_house = OneCycleLR(opt_house, max_lr=1e-1, total_steps=10001, 
                        pct_start=0.1, div_factor=10, final_div_factor=100)
        
    BATCH_SIZE = 1024
    print(f"Training Hybrid Gemma (V-Prediction) | Batch Size: {BATCH_SIZE}...")
    
    for i in range(10001):
        opt_main.zero_grad()
        opt_house.zero_grad()
        
        x0 = generate_rotated_checkerboards(BATCH_SIZE, device)
        t = torch.rand(BATCH_SIZE, device=device).clamp(0.001, 0.999)
        logsnr = get_schedule(t)
        alpha, sigma = get_alpha_sigma(logsnr)
        
        eps = torch.randn_like(x0)
        z_t = x0 * alpha.view(-1,1,1,1) + eps * sigma.view(-1,1,1,1)
        
        # Calculate V-Target
        v_true = alpha.view(-1,1,1,1) * eps - sigma.view(-1,1,1,1) * x0
        
        # Forward
        v_pred = model(z_t, logsnr)
        
        # Loss
        loss = F.mse_loss(v_pred, v_true)
        
        loss.backward()
        opt_main.step()
        opt_house.step()
        scheduler_main.step()
        scheduler_house.step()
        
        if i % 100 == 0:
            print(f"Step {i}: Loss {loss.item():.5f}")
            # Add after loss.backward():
            house_grad_norms = [p.grad.norm().item() for n,p in model.named_parameters() 
                                if 'orthogonal.vs' in n]
            print(f"Householder gradient norms: {house_grad_norms}")
            #print(f"Current LRs: {[group['lr'] for group in opt.param_groups]}")
            #two opts this time
            with torch.no_grad():
                for i, layer in enumerate(model.layers):
                    Q = layer.rn_rope.orthogonal.get_matrix()
                    
                    # Singular values tell you dimensionality
                    U, S, V = torch.svd(Q)
                    print(f"Layer {i} singular values (top 10): {S[:10].cpu().numpy()}")
                    
                    # Reflection vector norms tell you which are active
                    v_norms = torch.norm(layer.rn_rope.orthogonal.vs, dim=1)
                    print(f"Layer {i} reflection norms: {v_norms.cpu().numpy()}")
                    print(f"  Active (>0.5): {(v_norms > 0.5).sum().item()}/{len(v_norms)}")
                    print()
            
    print("Sampling with V-Prediction...")
    model.eval()
    with torch.no_grad():
        z = torch.randn(8, 3, 16, 16, device=device)
        ts = torch.linspace(1.0, 0.001, 50, device=device)
        
        for i in range(len(ts)-1):
            t_cur = ts[i]
            logsnr = get_schedule(torch.full((8,), t_cur, device=device))
            v_pred = model(z, logsnr)
            
            # Derive x0 and eps from v_pred
            alpha, sigma = get_alpha_sigma(logsnr)
            alpha = alpha.view(-1,1,1,1)
            sigma = sigma.view(-1,1,1,1)
            
            x0_pred = alpha * z - sigma * v_pred
            eps_pred = sigma * z + alpha * v_pred
            
            # Euler Step
            logsnr_next = get_schedule(torch.full((8,), ts[i+1], device=device))
            alpha_next, sigma_next = get_alpha_sigma(logsnr_next)
            
            # z_next = alpha_next * x0 + sigma_next * eps
            # This is the standard DDIM/Euler transfer
            z = alpha_next.view(-1,1,1,1) * x0_pred + sigma_next.view(-1,1,1,1) * eps_pred
            
    z = z.clamp(0, 1).cpu()
    
    # Ground Truth for comparison
    gt = generate_rotated_checkerboards(8, device).cpu()
    
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        # GT
        axes[0, i].imshow(gt[i].permute(1, 2, 0))
        axes[0, i].axis('off')
        if i==0: axes[0,i].set_title("Ground Truth (Random)")
        
        # Samples
        axes[1, i].imshow(z[i].permute(1, 2, 0))
        axes[1, i].axis('off')
        if i==0: axes[1,i].set_title("V-Prediction Samples")
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()