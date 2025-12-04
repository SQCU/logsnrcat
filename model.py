import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from diffusion_utils import FourierFeatures

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

def euclidean_sq(q_params, k_params):
    """
    Canonical Squared Euclidean Distance.
    Used by the attention kernel for intra-span geometric masking.
    q_params, k_params: [..., D] tensors of metric coefficients.
    """
    return torch.sum((q_params - k_params) ** 2, dim=-1)

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
GeometryState = namedtuple('GeometryState', [
    'rope_pos',       # [N, RoPE_Dims] - For RnRoPE (Causal + Spatial)
    'metric_coeffs',  # [N, Geo_Dims]  - For mask metric
    'span_ids',       # [N]            - For mask block-causality
    'is_causal',      # [N]            - For mask modality-causality
    'scale',          # float          - Reference Scaling Factor
    'inv_freq'        # [D_sub]        - Base frequencies
])


class MultiSpanAllocator(nn.Module):
    def __init__(self, max_tokens=262144, head_dim=64, max_spatial_dims=2, base_ref_len=64.0, device='cuda'):
        super().__init__()
        self.head_dim = head_dim
        self.max_spatial_dims = max_spatial_dims
        self.base_ref_len = base_ref_len
        
        # Dimensions: 1 Causal + K Spatial
        self.num_rope_dims = 1 + max_spatial_dims
        self.dim_per_subspace = head_dim // self.num_rope_dims
        
        # Single Frequency Basis (Reference Scaling)
        # Shared across all dimensions to ensure isotropy
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.dim_per_subspace, 2, device=device).float() / self.dim_per_subspace))
        self.register_buffer('inv_freq', inv_freq)

    def compute_geometry(self, spans):
        """
        Constructs the GeometryState.
        spans: List of dicts, e.g. [{'len': 50, 'type': 'text'}, {'len': 256, 'shape': (16,16)}]
        """
        device = self.inv_freq.device
        
        # 1. Calculate Sizes
        lengths = [s['len'] for s in spans]
        total_len = sum(lengths)
        
        # 2. Allocate Ephemeral Buffers (Functionpilled: No Side Effects)
        rope_pos = torch.zeros(total_len, self.num_rope_dims, device=device)
        metric_coeffs = torch.zeros(total_len, self.max_spatial_dims, device=device)
        span_ids = torch.zeros(total_len, dtype=torch.int32, device=device)
        is_causal = torch.zeros(total_len, dtype=torch.bool, device=device)
        
        # 3. Fill Causal Dim (Dim 0) - Branchless
        # "Every single embedding is uniformly placed along the 0th (context) R^n RoPE dim"
        rope_pos[:, 0] = torch.arange(total_len, device=device, dtype=torch.float32)
        
        # 4. Fill Spans
        ptr = 0
        for i, span in enumerate(spans):
            length = span['len']
            p_end = ptr + length
            
            # Span Identity
            span_ids[ptr:p_end] = i
            
            # Modality Specifics
            if span.get('causal', True):
                # Text: Causal Masking
                is_causal[ptr:p_end] = True
            else:
                # Latent: Bidirectional Masking
                is_causal[ptr:p_end] = False
                
                # Spatial Geometry (Dims 1..K)
                if 'shape' in span:
                    # Generate coords from shape
                    coords = euclid_rpow_n(span['shape'], device=device)
                    D = min(coords.shape[1], self.max_spatial_dims)
                    
                    # Overwrite default zeros
                    rope_pos[ptr:p_end, 1:1+D] = coords[:, :D]
                    metric_coeffs[ptr:p_end, :D] = coords[:, :D]
                    
                elif 'coords' in span:
                    # Use provided coords
                    c = span['coords']
                    D = min(c.shape[1], self.max_spatial_dims)
                    rope_pos[ptr:p_end, 1:1+D] = c[:, :D]
                    metric_coeffs[ptr:p_end, :D] = c[:, :D]
            
            ptr = p_end
            
        # 5. Reference Scale
        scale = total_len / self.base_ref_len
        
        return GeometryState(rope_pos, metric_coeffs, span_ids, is_causal, scale, self.inv_freq)

    def get_mask(self, geo_state, radius=2.5):
        """
        Local/Tangle Mask.
        """
        N = geo_state.total_len
        span_ids = geo_state.span_ids
        is_causal = geo_state.is_causal
        coeffs = geo_state.metric_coeffs
        radius_sq = radius ** 2
        
        def mask_mod(b, h, q_idx, kv_idx):
            # A. Block Causality (Global)
            q_span = span_ids[q_idx]
            k_span = span_ids[kv_idx]
            is_history = q_span > k_span
            
            # B. Intra-Span Logic
            is_same_span = q_span == k_span
            
            # C. Modality Causality
            is_valid_time = (~is_causal[q_idx]) | (q_idx >= kv_idx)
            
            # D. Geometric Sparsity (Local Metric)
            q_c = coeffs[q_idx]
            k_c = coeffs[kv_idx]
            dist_sq = euclidean_sq(q_c, k_c)
            is_valid_space = dist_sq < radius_sq
            
            return is_history | (is_same_span & is_valid_time & is_valid_space)

        return create_block_mask(mask_mod, B=1, H=1, Q_LEN=N, KV_LEN=N)

    def get_global_mask(self, geo_state):
        """
        Global/Block-Causal Mask.
        No spatial radius constraints, just Span & Modality Causality.
        """
        N = geo_state.total_len
        span_ids = geo_state.span_ids
        is_causal = geo_state.is_causal
        
        def mask_mod(b, h, q_idx, kv_idx):
            # Block Causality
            q_span = span_ids[q_idx]
            k_span = span_ids[kv_idx]
            is_history = q_span > k_span
            
            # Intra-Span
            is_same_span = q_span == k_span
            
            # Modality Causality
            is_valid_time = (~is_causal[q_idx]) | (q_idx >= kv_idx)
            
            # No geometric check
            return is_history | (is_same_span & is_valid_time)
            
        return create_block_mask(mask_mod, B=1, H=1, Q_LEN=N, KV_LEN=N)

# --- 4. RnRoPE Module ---

class RnRoPE(nn.Module):
    def __init__(self, head_dim, num_rope_dims):
        super().__init__()
        self.head_dim = head_dim
        self.num_rope_dims = num_rope_dims
        self.orthogonal = HouseholderOrthogonal(head_dim, num_reflections=head_dim//2)
        
    def forward(self, q, k, geo_state):
        # 1. Householder Transform (Basis mixing)
        q = self.orthogonal(q, inverse=True)
        k = self.orthogonal(k, inverse=True)
        
        # 2. RoPE Feature Generation
        inv_freq_scaled = geo_state.inv_freq / geo_state.scale
        rope_pos = geo_state.rope_pos # [N, NumDims]
        
        freqs_list = []
        # Generate features for each dimension (Causal + Spatial_1...K)
        # Note: num_rope_dims matches the allocator config
        for d in range(self.num_rope_dims):
            vals = rope_pos[:, d]
            f = torch.outer(vals, inv_freq_scaled)
            freqs_list.append(f)
            
        # Concatenate features
        full_freqs = torch.cat(freqs_list, dim=-1)
        
        # Truncate/Pad to match head_dim
        if full_freqs.shape[-1] > self.head_dim:
            full_freqs = full_freqs[:, :self.head_dim]
        elif full_freqs.shape[-1] < self.head_dim:
            pad = torch.zeros(full_freqs.shape[0], self.head_dim - full_freqs.shape[-1], device=q.device)
            full_freqs = torch.cat([full_freqs, pad], dim=-1)
            
        # 3. Apply Rotation
        emb = torch.cat((full_freqs, full_freqs), dim=-1)
        cos, sin = emb.cos()[None], emb.sin()[None]
        
        def apply_rot(x, c, s):
            x1, x2 = x.chunk(2, dim=-1)
            return (x * c) + (torch.cat((-x2, x1), dim=-1) * s)
            
        q_rot = apply_rot(q, cos, sin)
        k_rot = apply_rot(k, cos, sin)
        
        # 4. Inverse Householder Transform
        return self.orthogonal(q_rot, inverse=False), self.orthogonal(k_rot, inverse=False)

# --- FFN Primitives ---

class SwiGLU(nn.Module):
    """
    Standard SwiGLU FFN.
    Parameters are roughly 3 * dim * hidden_dim.
    """
    def __init__(self, dim, hidden_dim, bias=False):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        
    def forward(self, x):
        # x: [B, S, D]
        # Fused gate + value projection
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        # Swish Gate
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

class SigmoidMoE(nn.Module):
    """
    Sigmoid-Gated Mixture of SwiGLU Experts.
    
    Design Philosophy:
    - No Softmax: Experts are selected independently.
    - No Load Balancing Loss: We use Jitter (Noise) to prevent early collapse.
    - Top-K: We still enforce Top-K for computational budget control, 
      but the weights are absolute sigmoid probabilities, not relative softmax.
    """
    def __init__(self, dim, hidden_dim, num_experts=8, num_active=2, jitter_noise=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.num_active = num_active
        self.jitter_noise = jitter_noise
        
        # Router
        self.router = nn.Linear(dim, num_experts)
        # Init router to 0 to ensure all experts start with 0.5 probability
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        
        # Experts (ModuleList for easy debugging/hacking)
        # Note: hidden_dim here defines the capacity of ONE expert.
        # Total parameters = num_experts * SwiGLU(dim, hidden_dim)
        self.experts = nn.ModuleList([
            SwiGLU(dim, hidden_dim) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        # x: [B, S, D]
        router_logits = self.router(x)
        
        # 1. Jitter (Training Only)
        # Prevents "Expert Dave" from winning everything early on.
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(router_logits) * self.jitter_noise
            router_logits = router_logits + noise
            
        # 2. Sigmoid Gating (Independent Probabilities)
        scores = torch.sigmoid(router_logits)
        
        # 3. Top-K Selection
        # Even though probabilities are independent, we only run the top K
        # to stay within compute budget.
        top_k_scores, top_k_indices = torch.topk(scores, self.num_active, dim=-1)
        
        # 4. Normalization? 
        # In Softmax MoE, weights sum to 1. 
        # In Sigmoid MoE, we often re-normalize to preserve magnitude, 
        # OR we just use the raw sigmoid scores to let the router learn "intensity".
        # Let's use L1 normalization over the selected K to keep variance stable.
        router_weights = top_k_scores / (top_k_scores.sum(dim=-1, keepdim=True) + 1e-6)
        
        # 5. Dispatch
        out = torch.zeros_like(x)
        for i in range(self.num_active):
            idx = top_k_indices[:, :, i]
            weight = router_weights[:, :, i:i+1]
            
            for e in range(self.num_experts):
                mask = (idx == e)
                if mask.any():
                    expert_out = self.experts[e](x)
                    out = out + (expert_out * weight * mask.unsqueeze(-1))
                    
        # Optional: Z-Loss to keep logits sane (no gradients > 10.0)
        # This is cheap and strictly helps stability without changing behavior.
        # z_loss = torch.mean(torch.log(torch.exp(router_logits).sum(dim=-1)) ** 2)
        # For sigmoid, just penalize raw magnitude
        aux_loss = 1e-2 * (router_logits ** 2).mean()
        
        return out, aux_loss

# --- Blocks ---

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

    def forward(self, fourier_features):
        return torch.exp(self.net(fourier_features))

class GatedAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, spatial_allocator, is_global=False, ffn_type='swiglu'):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.is_global = is_global
        self.spatial = spatial_allocator # Reference to shared allocator
        
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        #self.rn_rope = RnRoPE(self.head_dim, spatial_buffer)

        # NOTE: GatedAttentionBlock needs to be updated to initialize RnRoPE
        # using self.spatial.num_rope_dims. Assuming GatedAttentionBlock structure:
        self.rn_rope = RnRoPE(self.head_dim, spatial_allocator.num_rope_dims)

        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, dim)
        
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)
        
        self.ffn_type = ffn_type
        # Standard FFN size is usually 4 * dim, but SwiGLU has 3 matrices.
        # To match parameter counts of standard Transformers, hidden is often 4*dim * 2/3.
        # We'll just stick to 4*dim for raw power here.
        ffn_hidden = dim * 4
        
        if ffn_type == 'swiglu':
            self.ffn = SwiGLU(dim, ffn_hidden)
        elif ffn_type == 'moe':
            # Expert hidden dim. 
            # If we want total params to be higher, we keep it large.
            # Usually experts are smaller or same size. Let's make them same size.
            self.ffn = SigmoidMoE(dim, ffn_hidden, num_experts=8, num_active=3)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

        self.mlp_gate = nn.Linear(dim, dim) # Simple gate for the residual

    def forward(self, x, block_mask):
        B, S, D = x.shape
        resid = x
        x_norm = self.norm1(x)
        
        q, k, v = self.qkv(x_norm).chunk(3, dim=-1)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        #~~q, k = self.rn_rope(q, k, grid_size)~~
        # Apply RoPE using GeometryState
        q, k = self.spatial.get_rope(q, k, geo_state, scale=scale)
        
        attn = flex_attention(q, k, v, block_mask=block_mask)
        attn = attn.transpose(1, 2).contiguous().view(B, S, D)
        attn = self.out_proj(attn)
        
        # Gated Residual 1 (Attention)
        gate_attn = torch.sigmoid(self.gate_proj(attn))
        x = resid + (attn * gate_attn)
        
        # Gated Residual 2 (FFN)
        resid = x
        x_norm = self.norm2(x)
        
        aux_loss = 0.0
        if self.ffn_type == 'moe':
            ffn_out, aux_loss = self.ffn(x_norm)
        else:
            ffn_out = self.ffn(x_norm)
            
        # Post-FFN Gating (Standard in some DiTs, keeping your style)
        # Your previous code had `mlp_gate` splitting into gate/val
        # But SwiGLU is already gated internally. 
        # We'll add a simple learnable scalar gate on the residual block.
        # Or preserve your MLP gate structure? 
        # Your old code: g, val = mlp_gate(norm(x))... x + mlp_out(val * gelu(g))
        # That was a GLU. We replaced it with SwiGLU/MoE. 
        # So we just add directly? Let's add a gate for stability.
        #gate_ffn = torch.sigmoid(self.mlp_gate(x_norm))
        #x = resid + (ffn_out * gate_ffn)
        # what on earth was that...? reverting to a basic ah ah resnet.

        # totally ordinary post-ffn residual.
        x = resid + ffn_out

        
        return x, aux_loss

class MLPResBlock(nn.Module):
    """
    A simple Residual Block for the Interface MLPs.
    Pre-Norm structure: x = x + MLP(Norm(x))
    """
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
        
        # 1. Lift from Pixels to Hidden Space
        self.input_proj = nn.Linear(patch_flat_dim + 8, hidden_dim)
        
        # 2. Deep Interface (Residual Processing)
        # If mlp_depth > 1, we add (depth-1) residual blocks
        self.res_blocks = nn.Sequential(*[
            MLPResBlock(hidden_dim) for _ in range(max(0, mlp_depth - 1))
        ])
        
        # 3. Project to Embedding Dimension
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim), # Final norm before entering transformer
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x, spins):
        x_pad = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        patches = x_pad.unfold(2, self.context_size, self.stride).unfold(3, self.context_size, self.stride)
        B, C, GH, GW, K, _ = patches.shape
        num_tokens = GH * GW
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, num_tokens, -1)
        spins_expanded = spins.unsqueeze(1).expand(-1, num_tokens, -1)
        
        x = self.input_proj(torch.cat([patches, spins_expanded], dim=-1))
        x = self.res_blocks(x)
        x = self.output_proj(x)
        return x

class NonLinearOutputHead(nn.Module):
    def __init__(self, embed_dim, output_dim=12, mlp_depth=1): 
        super().__init__()
        
        # 1. Process in Embedding Space
        # Note: We assume input is already normalized by the Transformer's final LN
        self.res_blocks = nn.Sequential(*[
            MLPResBlock(embed_dim) for _ in range(max(0, mlp_depth - 1))
        ])
        
        # 2. Project to Output Pixels
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim)
        )
        
    def forward(self, x):
        x = self.res_blocks(x)
        return self.output_proj(x)

class HybridGemmaDiT(nn.Module):
    def __init__(self, mode='factorized', embed_dim=256, depth=8, ffn_type='swiglu', interface_depth=1):
        super().__init__()
        self.mode = mode
        
        self.patch_in = ContextualPatchEmbedder(
            input_dim=3, 
            hidden_dim=embed_dim, 
            embed_dim=embed_dim, 
            context_size=4,
            mlp_depth=interface_depth
        )
        
        self.fourier_dim = 8
        self.spin_encoder = FourierFeatures(num_bands=4)
        # Allocator: 2 Spatial Dims (X, Y) -> 3 Total RoPE Dims (Time, X, Y)
        # Head Dim splits roughly 64 // 3 = 21 per dim
        self.spatial = MultiSpanAllocator(head_dim=embed_dim//4, max_spatial_dims=2)
        self.mask_cache = {} 
        self.depth = depth
        
        self.layers = nn.ModuleList([
            GatedAttentionBlock(
                embed_dim, 4, self.spatial, 
                is_global=((i+1)%4==0),
                ffn_type=ffn_type
            )
            for i in range(depth)
        ])
        self.norm_final = nn.RMSNorm(embed_dim, elementwise_affine=False)

        #this is a hand-coded rgb color channel * samples-per-patch parsing.
        # feel free to refactor this at some time when it's not a confusing distraction
        self.patch_dim = 12
        self.output_head = NonLinearOutputHead(embed_dim, self.patch_dim, mlp_depth=interface_depth)
        
        if mode == 'factorized':
            self.scale_decoder = FourierScaleDecoder(self.fourier_dim, embed_dim, self.patch_dim)
            self.lambda_head = nn.Linear(embed_dim, 1)
        else:
            self.scale_decoder = None
            self.lambda_head = None

    def get_masks(self, grid_size):
        # Back-compat method
        spans = [{'len': grid_size*grid_size, 'shape': (grid_size, grid_size), 'causal': False}]
        geo_state = self.spatial.compute_geometry(spans)
        
        eff_res = math.sqrt(geo_state.total_len)
        base_res = 8.0; base_radius = 2.5; log_scale = 1.5
        
        if eff_res <= base_res:
            radius = base_radius
        else:
            radius = base_radius + log_scale * math.log2(eff_res / base_res)
            
        mask_local = self.spatial.get_mask(geo_state, radius=radius)
        mask_global = self.spatial.get_global_mask(geo_state)
        
        return mask_local, mask_global, geo_state


    def forward(self, z_t, logsnr, spans):
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
        B, C, H, W = z_t.shape
        grid_h, grid_w = H // 2, W // 2
        
        # 1. Compute Geometry State (Once per forward)
        geo_state = self.spatial.compute_geometry(spans)
        
        # 2. Compute Masks (Once per forward)
        # Logarithmic Radius Logic based on total sequence length
        # Proxy 'resolution' is sqrt(total_len)
        eff_res = math.sqrt(geo_state.total_len)
        base_res = 8.0; base_radius = 2.5; log_scale = 1.5
        
        if eff_res <= base_res:
            radius = base_radius
        else:
            radius = base_radius + log_scale * math.log2(eff_res / base_res)
            
        mask_local = self.spatial.get_mask(geo_state, radius=radius)
        mask_global = self.spatial.get_global_mask(geo_state)
        
        # 3. Embedding
        spins = self.spin_encoder(logsnr)
        x = self.patch_in(z_t, spins) 
        # x is [B, N, D]
        
        total_aux_loss = 0.0
        for layer in self.layers:
            mask = mask_global if layer.is_global else mask_local
            # Pass geo_state explicitly to layer/rope
            x, al = layer(x, mask, geo_state)
            total_aux_loss += al
            
        x = self.norm_final(x)
        z_pred = self.output_head(x)
        
        l_pred = None
        if self.mode == 'factorized':
            scale = self.scale_decoder(spins).unsqueeze(1)
            z_pred = z_pred * scale
            l_pred = self.lambda_head(x.mean(dim=1)).squeeze(-1)
            
        # Unflatten to original shape for compatibility with image-based training loops
        return z_pred.view(B, grid_h, grid_w, 3, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W), l_pred, total_aux_loss