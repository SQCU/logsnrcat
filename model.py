import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from diffusion_utils import FourierFeatures
import math

# --- 0. Geometry & RoPE ---
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

# --- 1. Top-Level Metric Definition ---

def euclid_rpow_n(shape, device='cuda'):
    """
    Generates N-dimensional coordinates for a grid of given shape.
    Returns: [Product(shape), Rank(shape)] tensor.
    Example: shape=(16, 16) -> [256, 2] tensor of (y, x) coords.
    """
    coords = [torch.arange(s, device=device, dtype=torch.float32) for s in shape]
    mesh = torch.meshgrid(*coords, indexing='ij')
    # Stack and flatten: [D, N] -> [N, D]
    stacked = torch.stack(mesh, dim=-1).reshape(-1, len(shape))
    return stacked

from collections import namedtuple
# Container for the geometry state of a specific forward pass
# Container for the geometry state of a specific forward pass
GeometryState = namedtuple('GeometryState', [
    'rope_pos',       # [N, RoPE_Dims]
    'span_ids',       # [N]
    'is_causal',      # [N]
    'total_len',      # int
    'scale',          # float
    'inv_freq'        # [D_sub]
])

class MultiSpanAllocator(nn.Module):
    def __init__(self, max_tokens=16384, head_dim=64, max_spatial_dims=2, base_ref_len=64.0, device='cuda'):
        super().__init__()
        self.head_dim = head_dim
        self.max_spatial_dims = max_spatial_dims
        self.base_ref_len = base_ref_len
        self.num_rope_dims = 1 + max_spatial_dims
        self.max_tokens = max_tokens
        
        # Frequency Generation
        self.freq_dim = head_dim // 2
        self.dim_per_subspace = self.freq_dim // self.num_rope_dims
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.dim_per_subspace, 2, device=device).float() / self.dim_per_subspace))
        self.register_buffer('inv_freq', inv_freq)

        # FIXED: Register a STATIC buffer for coordinates. 
        # Dynamo loves Buffers. It hates local variables.
        # We pre-fill this with a default grid pattern to be safe, 
        # but in practice we'd write to it or index it carefully.
        # For simplicity in this fix, we will just use 1D indices for the buffer 
        # and assume the mask kernel can read this global state.
        self.register_buffer('shared_coords', torch.zeros(max_tokens, max_spatial_dims, device=device))
        
        # We also need buffers for span_ids and is_causal if we want to capture them in mask_mod without lifting errors
        self.register_buffer('shared_span_ids', torch.zeros(max_tokens, dtype=torch.int32, device=device))
        self.register_buffer('shared_is_causal', torch.zeros(max_tokens, dtype=torch.bool, device=device))
        # ADD THIS: Buffer for radius_sq so mask_mod can capture it
        self.register_buffer('current_radius_sq', torch.tensor(6.25, device=device))  # default: 2.5^2

    def update_buffers(self, spans):
        """
        Updates the internal buffers with the new span topology.
        This runs in standard eager mode before the compiled block, or is traced as tensor updates.
        """
        device = self.inv_freq.device
        ptr = 0
        total_len = 0
        
        # We build temporary lists/tensors then copy to buffer
        # This update is "stateful" but necessary for flex_attention capture
        for i, span in enumerate(spans):
            length = span['len']
            total_len += length
            if ptr + length > self.max_tokens:
                raise ValueError(f"Sequence length {ptr+length} exceeds buffer size {self.max_tokens}")
                
            # Update Span IDs
            self.shared_span_ids[ptr:ptr+length] = i
            
            if span.get('causal', True):
                self.shared_is_causal[ptr:ptr+length] = True
                # For text, spatial coords are 0 (or just linear 1D map)
                self.shared_coords[ptr:ptr+length, :] = 0.0
            else:
                self.shared_is_causal[ptr:ptr+length] = False
                # Handle Geometry
                if 'shape' in span:
                    # Generate coords on fly
                    # Note: We do this computation every step. In a real efficient rig, 
                    # we would cache these patterns.
                    coords = self._generate_grid(span['shape'], device)
                    D = min(coords.shape[1], self.max_spatial_dims)
                    self.shared_coords[ptr:ptr+length, :D] = coords[:, :D]
                    
            ptr += length
        return total_len

    def _generate_grid(self, shape, device):
        coords = [torch.arange(s, device=device, dtype=torch.float32) for s in shape]
        mesh = torch.meshgrid(*coords, indexing='ij')
        return torch.stack(mesh, dim=-1).reshape(-1, len(shape))

    def compute_geometry(self, spans):
        """
        Returns the RoPE positions (activations) but relies on buffers for Masking.
        """
        total_len = self.update_buffers(spans)
        
        # Generate RoPE positions (activations are fine for RoPE, just not for Mask compilation)
        # We can reuse the buffers we just filled to generate RoPE pos
        rope_pos = torch.zeros(total_len, self.num_rope_dims, device=self.inv_freq.device)
        
        # Dim 0: Time/Sequence
        rope_pos[:, 0] = torch.arange(total_len, device=self.inv_freq.device)
        
        # Dim 1+: Space (Read from the buffer we just updated)
        # Note: We read the slice corresponding to valid data
        active_coords = self.shared_coords[:total_len]
        D = active_coords.shape[1]
        rope_pos[:, 1:1+D] = active_coords
        
        scale = total_len / self.base_ref_len
        
        return GeometryState(rope_pos, None, None, total_len, scale, self.inv_freq)

    def get_mask(self, total_len, radius=2.5):
        """
        Mask Mod captures 'self' (the module), allowing access to registered buffers.
        """
        # UPDATE the buffer instead of creating a local variable
        self.current_radius_sq.fill_(float(radius) ** 2)
        
        def mask_mod(b, h, q_idx, kv_idx):
            # A. Block Causality
            q_span = self.shared_span_ids[q_idx]
            k_span = self.shared_span_ids[kv_idx]
            is_history = q_span > k_span
            
            # B. Intra-Span Logic
            is_same_span = q_span == k_span
            
            # C. Modality Causality
            is_causal_q = self.shared_is_causal[q_idx]
            is_valid_time = (~is_causal_q) | (q_idx >= kv_idx)
            
            # D. Geometric Sparsity - READ from buffer instead of closure
            q_c = self.shared_coords[q_idx]
            k_c = self.shared_coords[kv_idx]
            dist_sq = ((q_c - k_c) ** 2).sum(dim=-1)
            is_valid_space = dist_sq < self.current_radius_sq  # <-- NOW A BUFFER
            
            return is_history | (is_same_span & is_valid_time & is_valid_space)

        return create_block_mask(mask_mod, B=1, H=1, Q_LEN=total_len, KV_LEN=total_len)

    def get_global_mask(self, total_len):
        def mask_mod(b, h, q_idx, kv_idx):
            q_span = self.shared_span_ids[q_idx]
            k_span = self.shared_span_ids[kv_idx]
            is_history = q_span > k_span
            is_same_span = q_span == k_span
            is_causal_q = self.shared_is_causal[q_idx]
            is_valid_time = (~is_causal_q) | (q_idx >= kv_idx)
            return is_history | (is_same_span & is_valid_time)
            
        return create_block_mask(mask_mod, B=1, H=1, Q_LEN=total_len, KV_LEN=total_len)

# --- RnRoPE ---

class RnRoPE(nn.Module):
    def __init__(self, head_dim, num_rope_dims):
        super().__init__()
        self.head_dim = head_dim
        self.num_rope_dims = num_rope_dims
        self.orthogonal = HouseholderOrthogonal(head_dim, num_reflections=head_dim//2)
        self.freq_dim = head_dim // 2
        
    def forward(self, q, k, geo_state):
        q = self.orthogonal(q, inverse=True)
        k = self.orthogonal(k, inverse=True)
        
        inv_freq_scaled = geo_state.inv_freq / geo_state.scale
        rope_pos = geo_state.rope_pos
        
        freqs_list = []
        for d in range(self.num_rope_dims):
            vals = rope_pos[:, d]
            f = torch.outer(vals, inv_freq_scaled)
            freqs_list.append(f)
            
        full_freqs = torch.cat(freqs_list, dim=-1)
        
        if full_freqs.shape[-1] > self.freq_dim:
            full_freqs = full_freqs[:, :self.freq_dim]
        elif full_freqs.shape[-1] < self.freq_dim:
            pad = torch.zeros(full_freqs.shape[0], self.freq_dim - full_freqs.shape[-1], device=q.device)
            full_freqs = torch.cat([full_freqs, pad], dim=-1)
            
        emb = torch.cat((full_freqs, full_freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(1)
        sin = emb.sin().unsqueeze(0).unsqueeze(1)
        
        def apply_rot(x, c, s):
            x1, x2 = x.chunk(2, dim=-1)
            return (x * c) + (torch.cat((-x2, x1), dim=-1) * s)
            
        q_rot = apply_rot(q, cos, sin)
        k_rot = apply_rot(k, cos, sin)
        
        return self.orthogonal(q_rot, inverse=False), self.orthogonal(k_rot, inverse=False)

# --- FFN & Blocks ---

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, bias=False):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        
    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)

class SigmoidMoE(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts=8, num_active=2, jitter_noise=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.num_active = num_active
        self.jitter_noise = jitter_noise
        self.router = nn.Linear(dim, num_experts)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        self.experts = nn.ModuleList([SwiGLU(dim, hidden_dim) for _ in range(num_experts)])
        
    def forward(self, x):
        router_logits = self.router(x)
        if self.training and self.jitter_noise > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.jitter_noise
            
        scores = torch.sigmoid(router_logits)
        top_k_scores, top_k_indices = torch.topk(scores, self.num_active, dim=-1)
        router_weights = top_k_scores / (top_k_scores.sum(dim=-1, keepdim=True) + 1e-6)
        
        out = torch.zeros_like(x)
        for i in range(self.num_active):
            idx = top_k_indices[:, :, i]
            weight = router_weights[:, :, i:i+1]
            for e in range(self.num_experts):
                mask = (idx == e)
                if mask.any():
                    out = out + (self.experts[e](x) * weight * mask.unsqueeze(-1))
        
        aux_loss = 1e-2 * (router_logits ** 2).mean()
        return out, aux_loss

class FourierScaleDecoder(nn.Module):
    def __init__(self, fourier_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def forward(self, f):
        return torch.exp(self.net(f))

class GatedAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, spatial_allocator, is_global=False, ffn_type='swiglu'):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.is_global = is_global
        self.spatial = spatial_allocator
        
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.rn_rope = RnRoPE(self.head_dim, spatial_allocator.num_rope_dims)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, dim)
        
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)
        self.ffn_type = ffn_type
        ffn_hidden = dim * 4
        
        if ffn_type == 'swiglu':
            self.ffn = SwiGLU(dim, ffn_hidden)
        elif ffn_type == 'moe':
            self.ffn = SigmoidMoE(dim, ffn_hidden, num_experts=8, num_active=3)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

    def forward(self, x, block_mask, geo_state):
        B, S, D = x.shape
        resid = x
        x_norm = self.norm1(x)
        
        q, k, v = self.qkv(x_norm).chunk(3, dim=-1)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        q, k = self.rn_rope(q, k, geo_state)

        attn = flex_attention(q, k, v, block_mask=block_mask)
        attn = attn.transpose(1, 2).contiguous().view(B, S, D)
        attn = self.out_proj(attn)
        
        gate_attn = torch.sigmoid(self.gate_proj(attn))
        x = resid + (attn * gate_attn)
        
        resid = x
        x_norm = self.norm2(x)
        
        aux_loss = 0.0
        if self.ffn_type == 'moe':
            ffn_out, aux_loss = self.ffn(x_norm)
        else:
            ffn_out = self.ffn(x_norm)
            
        x = resid + ffn_out
        return x, aux_loss

class MLPResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.RMSNorm(dim, elementwise_affine=False)
        self.net = SwiGLU(dim, dim*2)
    def forward(self, x):
        return x + self.net(self.norm(x))

class ContextualPatchEmbedder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, embed_dim=256, context_size=4, mlp_depth=1):
        super().__init__()
        self.context_size = context_size
        self.stride = 2
        self.padding = 1 
        patch_flat_dim = (context_size ** 2) * input_dim
        self.input_proj = nn.Linear(patch_flat_dim + 8, hidden_dim)
        self.res_blocks = nn.Sequential(*[MLPResBlock(hidden_dim) for _ in range(max(0, mlp_depth - 1))])
        self.output_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, embed_dim))

    def forward(self, x, spins):
        x_pad = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        patches = x_pad.unfold(2, self.context_size, self.stride).unfold(3, self.context_size, self.stride)
        B, C, GH, GW, K, _ = patches.shape
        num_tokens = GH * GW
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, num_tokens, -1)
        spins_expanded = spins.unsqueeze(1).expand(-1, num_tokens, -1)
        x = self.input_proj(torch.cat([patches, spins_expanded], dim=-1))
        x = self.res_blocks(x)
        return self.output_proj(x)

class NonLinearOutputHead(nn.Module):
    def __init__(self, embed_dim, output_dim=12, mlp_depth=1): 
        super().__init__()
        self.res_blocks = nn.Sequential(*[MLPResBlock(embed_dim) for _ in range(max(0, mlp_depth - 1))])
        self.output_proj = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, output_dim))
    def forward(self, x):
        x = self.res_blocks(x)
        return self.output_proj(x)

class HybridGemmaDiT(nn.Module):
    def __init__(self, mode='factorized', embed_dim=256, depth=8, ffn_type='swiglu', interface_depth=1):
        super().__init__()
        self.mode = mode
        self.patch_in = ContextualPatchEmbedder(input_dim=3, hidden_dim=embed_dim, embed_dim=embed_dim, context_size=4, mlp_depth=interface_depth)
        self.fourier_dim = 8
        self.spin_encoder = FourierFeatures(num_bands=4)
        self.spatial = MultiSpanAllocator(head_dim=embed_dim//4, max_spatial_dims=2)
        # Cache for BlockMasks to prevent expensive re-tracing during forward
        self.mask_cache = {} 
        
        self.layers = nn.ModuleList([
            GatedAttentionBlock(embed_dim, 4, self.spatial, is_global=((i+1)%4==0), ffn_type=ffn_type)
            for i in range(depth)
        ])
        self.norm_final = nn.RMSNorm(embed_dim, elementwise_affine=False)
        self.patch_dim = 12
        self.output_head = NonLinearOutputHead(embed_dim, self.patch_dim, mlp_depth=interface_depth)
        
        if mode == 'factorized':
            self.scale_decoder = FourierScaleDecoder(self.fourier_dim, embed_dim, self.patch_dim)
            self.lambda_head = nn.Linear(embed_dim, 1)
        else:
            self.scale_decoder = None; self.lambda_head = None

    def get_masks(self, grid_size):
        # Back-compat support using the new allocator
        spans = [{'len': grid_size*grid_size, 'shape': (grid_size, grid_size), 'causal': False}]
        geo_state = self.spatial.compute_geometry(spans)
        
        eff_res = math.sqrt(geo_state.total_len)
        base_res = 8.0; base_radius = 2.5; log_scale = 1.5
        radius = base_radius if eff_res <= base_res else base_radius + log_scale * math.log2(eff_res / base_res)
            
        mask_local = self.spatial.get_mask(geo_state, radius=radius)
        mask_global = self.spatial.get_global_mask(geo_state)
        return mask_local, mask_global, geo_state

    def forward(self, z_t, logsnr, spans):
        B, C, H, W = z_t.shape
        grid_h, grid_w = H // 2, W // 2
        
        # 1. Compute Geometry
        geo_state = self.spatial.compute_geometry(spans)
        
        # 2. Mask Logic with Caching
        total_len = geo_state.total_len
        base_len = 64.0; base_radius = 2.5; log_scale = 0.75
        
        # Ensure total_len is int for math ops
        if isinstance(total_len, torch.Tensor):
            total_len_val = int(total_len.item())
        else:
            total_len_val = total_len
            
        if total_len_val <= base_len:
            radius = float(base_radius)
        else:
            radius = float(base_radius + log_scale * math.log2(total_len_val / base_len))
        
        cache_key = (total_len_val, round(radius, 4))
        
        if cache_key in self.mask_cache:
            mask_local, mask_global = self.mask_cache[cache_key]
        else:
            mask_local = self.spatial.get_mask(total_len_val, radius=radius)
            mask_global = self.spatial.get_global_mask(total_len_val)
            self.mask_cache[cache_key] = (mask_local, mask_global)
        
        spins = self.spin_encoder(logsnr)
        x = self.patch_in(z_t, spins) 
        
        total_aux_loss = 0.0
        for layer in self.layers:
            mask = mask_global if layer.is_global else mask_local
            x, al = layer(x, mask, geo_state)
            total_aux_loss += al
            
        x = self.norm_final(x)
        z_pred = self.output_head(x)
        
        l_pred = None
        if self.mode == 'factorized':
            scale = self.scale_decoder(spins).unsqueeze(1)
            z_pred = z_pred * scale
            l_pred = self.lambda_head(x.mean(dim=1)).squeeze(-1)
            
        return z_pred.view(B, grid_h, grid_w, 3, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W), l_pred, total_aux_loss

        """
        Forward pass with flexible topology.

        Args:
            z_t (Tensor): Input latents. Currently [B, C, H, W] for compatibility, 
                            or [B, N, D] for flat inputs.
            logsnr (Tensor): Noise levels [B].
            spans (List[Dict]): Metadata defining the sequence topology. 
                                The sum of 'len' in spans must match N.
                
                Format per span dict:
                {
                    # REQUIRED
                    'len': int,       # Number of tokens in this span
                    
                    # OPTIONAL (Topology)
                    'causal': bool,   # True = Text/Time (Autoregressive mask, Integer RoPE steps).
                                        # False = Latent/Space (Bidirectional mask, Fractional RoPE steps).
                                        # Defaults to True.
                    
                    # OPTIONAL (Geometry - Only used if causal=False)
                    'shape': tuple,   # (H, W) tuple for 2D/3D grids. 
                                        # Generates Euclidean coordinates automatically.
                    'coords': Tensor, # [Len, D] Tensor of explicit coordinates.
                                        # Use for graphs or manifolds. Overrides 'shape'.
                }

                Example (Image Captioning):
                [
                    {'len': 16, 'causal': True},                 # "Prefix: "
                    {'len': 256, 'causal': False, 'shape':(16,16)}, # [Image]
                    {'len': 32, 'causal': True}                  # " description..."
                ]
        """
