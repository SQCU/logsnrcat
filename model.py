#model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from diffusion_utils import FourierFeatures

# --- Geometry & RoPE ---
#thanks and complaints all go to:
""" https://arxiv.org/abs/2504.06308
Rethinking RoPE: A Mathematical Blueprint for
N-dimensional Rotary Positional Embedding
Haiping Liu Lijing Lin Jingyuan Sun Zhegong Shangguan
Mauricio A. Alvarez Hongpeng Zhou∗
University of Manchester
* Corresponding author: hongpeng.zhou@manchester.ac.uk
"""
#and valued contributor
"""
gemini 3 pro preview: aistudio.google.com
"""
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

class FourierScaleDecoder(nn.Module):
    """
    Decodes the 'Scale' (Magnitude) directly from the Physics (Fourier Features).
    This implements the 'Adaptive Layer Norm' behavior at the output.
    
    scale = exp(MLP(fourier_features))
    """
    def __init__(self, fourier_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize to output 0, so exp(0) = 1.0 (Unit Scale start)
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def forward(self, fourier_features):
        # fourier_features: [B, F_Dim] -> scale: [B, Out_Dim]
        log_scale = self.net(fourier_features)
        return torch.exp(log_scale)


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

    def get_rope(self, B):
        y_pos, x_pos = self.coords[:, 0], self.coords[:, 1]
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
        cos, sin = self.buffer.get_rope(B)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        
        # Standard Rotary Apply
        q1, q2 = q.chunk(2, dim=-1); k1, k2 = k.chunk(2, dim=-1)
        rq = torch.cat((-q2, q1), dim=-1); rk = torch.cat((-k2, k1), dim=-1)
        q_rot = (q * cos) + (rq * sin)
        k_rot = (k * cos) + (rk * sin)
        
        q_final = self.orthogonal(q_rot, inverse=False)
        k_final = self.orthogonal(k_rot, inverse=False)
        return q_final, k_final

# --- Masks ---

def get_sliding_window_mask(dist_matrix, radius=2.5):
    Q_LEN, KV_LEN = dist_matrix.shape
    valid_mask = dist_matrix < radius
    def mask_mod(b, h, q_idx, kv_idx):
        q_safe = q_idx.clamp(0, Q_LEN - 1)
        k_safe = kv_idx.clamp(0, KV_LEN - 1)
        return valid_mask[q_safe, k_safe] & (q_idx < Q_LEN) & (kv_idx < KV_LEN)
    return create_block_mask(mask_mod, B=1, H=1, Q_LEN=Q_LEN, KV_LEN=KV_LEN)

def get_global_mask():
    def mask_mod(b, h, q_idx, kv_idx):
        return q_idx > -1
    return create_block_mask(mask_mod, B=1, H=1, Q_LEN=64, KV_LEN=64)

# --- Network ---

class GatedAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, spatial_buffer, is_global=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.is_global = is_global
        
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.rn_rope = RnRoPE(self.head_dim, spatial_buffer)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, dim)
        
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)
        self.mlp_gate = nn.Linear(dim, dim * 8)
        self.mlp_out = nn.Linear(dim * 4, dim)

    def forward(self, x, block_mask):
        B, S, D = x.shape
        resid = x
        x = self.norm1(x)
        
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        q, k = self.rn_rope(q, k)
        
        attn = flex_attention(q, k, v, block_mask=block_mask)
        attn = attn.transpose(1, 2).contiguous().view(B, S, D)
        attn = self.out_proj(attn)
        
        gate = torch.sigmoid(self.gate_proj(attn))
        x = resid + (attn * gate)
        
        resid = x
        g, val = self.mlp_gate(self.norm2(x)).chunk(2, dim=-1)
        x = resid + self.mlp_out(val * F.gelu(g))
        return x



class HybridGemmaDiT(nn.Module):
    def __init__(self, mode='factorized', embed_dim=256, depth=8):
        super().__init__()
        self.mode = mode
        
        self.patch_in = nn.Linear(20, embed_dim)
        
        # Fourier Features used for Input AND Output Scale Decoding
        self.fourier_dim = 8 # 4 bands * 2
        self.spin_encoder = FourierFeatures(num_bands=4)
        
        self.spatial = SpatialBuffer(grid_size=8, head_dim=embed_dim//4)
        self.mask_local = get_sliding_window_mask(self.spatial.dist_matrix, 2.5)
        self.mask_global = get_global_mask()
        
        self.layers = nn.ModuleList([
            GatedAttentionBlock(embed_dim, 4, self.spatial, is_global=((i+1)%4==0))
            for i in range(depth)
        ])
        
        self.norm_final = nn.RMSNorm(embed_dim, elementwise_affine=False)
        
        # Output Config
        self.patch_dim = 12
        self.output_head = nn.Linear(embed_dim, self.patch_dim)
        
        if mode == 'factorized':
            # Factorized Mode: Output = Direction * Scale
            # Scale is synthesized directly from Fourier features
            self.scale_decoder = FourierScaleDecoder(
                fourier_dim=self.fourier_dim,
                hidden_dim=embed_dim,
                output_dim=self.patch_dim
            )
            # Passthrough Lambda head (Scalar prediction)
            self.lambda_head = nn.Linear(embed_dim, 1)
        else:
            self.scale_decoder = None
            self.lambda_head = None

    def forward(self, z_t, logsnr):
        B = z_t.shape[0]
        patches = z_t.unfold(2, 2, 2).unfold(3, 2, 2).permute(0, 2, 3, 1, 4, 5).reshape(B, 64, 12)
        
        # 1. Encode Physics (Spins)
        spins = self.spin_encoder(logsnr) # [B, 8]
        spins_expanded = spins.unsqueeze(1).expand(-1, 64, -1)
        
        # 2. Input Embedding
        x = self.patch_in(torch.cat([patches, spins_expanded], dim=-1))
        
        # 3. Transformer
        for layer in self.layers:
            mask = self.mask_global if layer.is_global else self.mask_local
            x = layer(x, mask)
            
        x = self.norm_final(x)
        
        # 4. Decode
        # Predict Direction (Normalized Pattern)
        z_pred = self.output_head(x)
        l_pred = None
        
        if self.mode == 'factorized':
            # Decode Scale from Physics (Fourier Features)
            scale = self.scale_decoder(spins) # [B, 12]
            scale = scale.unsqueeze(1) # Broadcast over tokens [B, 1, 12]
            
            # Apply Scale (Modulation)
            z_pred = z_pred * scale
            
            # Predict Lambda (Passthrough)
            l_pred = self.lambda_head(x.mean(dim=1)).squeeze(-1)
            
        return z_pred.view(B, 8, 8, 3, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, 3, 16, 16), l_pred