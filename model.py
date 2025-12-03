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

class DynamicSpatialBuffer(nn.Module):
    def __init__(self, max_grid_size=16, head_dim=64, device='cuda'):
        super().__init__()
        self.max_grid_size = max_grid_size
        y = torch.arange(max_grid_size, device=device)
        x = torch.arange(max_grid_size, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        self.coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1).float()
        d = torch.cdist(self.coords, self.coords, p=2)
        self.register_buffer('dist_matrix', d)
        self.dim_half = head_dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim_half, 2, device=device).float() / self.dim_half))
        self.register_buffer('inv_freq', inv_freq)

    def get_rope(self, B, grid_size, base_grid_size=8):
        """
        RoPE with semantic distance invariance for 2D spatial grids.
        
        Principle:
            Two patches that span the same *fraction* of the image
            should have similar relative position encodings, regardless
            of the total image resolution.
        
        Example:
            - At 8×8 grid: Patches 2 units apart span 2/8 = 25% of image
            - At 32×32 grid: Patches 8 units apart span 8/32 = 25% of image
            → Both pairs should have the same RoPE rotation angle
        
        Implementation:
            rotation_angle = θ · distance
            
            For semantic invariance:
            θ_effective = θ_base · (grid_size / base_grid_size)
            
            This makes rotation_angle invariant when distance scales
            proportionally with grid_size.
        
        Why This Matters:
            - Checkerboards: A tile at 8×8 spans ~2 patches; at 32×32 spans ~8 patches.
            With scaling, both are recognized as "same tile" by attention.
            - Torii: Curvature features at 8×8 span ~4 patches; at 32×32 span ~16 patches.
            With scaling, the model learns shape geometry consistently.
        """
        scale = grid_size / base_grid_size
        scaled_base = self.base_freq * scale
        
        inv_freq = 1.0 / (scaled_base ** (
            torch.arange(0, self.dim_half, 2, device=self.inv_freq.device).float() 
            / self.dim_half
        ))
        
        # Get active coordinates
        if grid_size == self.max_grid_size:
            active_coords = self.coords
        else:
            y = torch.arange(grid_size, device=self.inv_freq.device)
            grid_y, grid_x = torch.meshgrid(y, y, indexing='ij')
            active_coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1).float()
        
        y_pos, x_pos = active_coords[:, 0], active_coords[:, 1]
        freqs_y = torch.einsum('i, j -> i j', y_pos, inv_freq)
        freqs_x = torch.einsum('i, j -> i j', x_pos, inv_freq)
        emb = torch.cat((torch.cat([freqs_y, freqs_x], dim=-1), 
                        torch.cat([freqs_y, freqs_x], dim=-1)), dim=-1)
        return emb.cos()[None], emb.sin()[None]
    def get_rope_spherical(self, angular_distance, bandwidth_L, base_bandwidth_L=16):
        """
        RoPE for SO(3)-equivariant spherical convolutions.
        
        Principle:
            Angular distance on a sphere is resolution-independent.
            However, the *discretization* (spherical harmonic bandwidth L)
            determines the Nyquist frequency.
        
        Example:
            - At L=16: Can resolve features with ~16 lobes
            - At L=64: Can resolve features with ~64 lobes
            → A 4-lobe pattern at L=16 is analogous to a 16-lobe pattern at L=64
        
        Implementation:
            For a feature with k lobes:
            θ_effective = θ_base · (bandwidth_L / base_bandwidth_L)
            
            This makes the rotation period match the *perceptual frequency*
            of features (lobes/harmonics) rather than raw angular distance.
        
        Note:
            Unlike 2D grids, spherical distance is already normalized [0, π].
            The scaling applies to the *frequency content* we can represent,
            not the coordinate system itself.
        """
        scale = bandwidth_L / base_bandwidth_L
        scaled_base = self.base_freq * scale
        
        inv_freq = 1.0 / (scaled_base ** (
            torch.arange(0, self.dim_half, 2).float() / self.dim_half
        ))
        
        # angular_distance ∈ [0, π] (geodesic on sphere)
        freqs = torch.einsum('i, j -> i j', angular_distance, inv_freq)
        return freqs.cos(), freqs.sin()
    def get_rope_graph(self, geodesic_dist, diameter, base_diameter=10):
        """
        RoPE for graph-structured data using geodesic distance.
        
        Principle:
            Nodes that are X% of the graph's diameter apart should
            have similar relative encodings, regardless of graph size.
        
        Example:
            - Small molecule (diameter=5): Nodes 2 hops apart are "distant" (40%)
            - Protein (diameter=50): Nodes 20 hops apart are "distant" (40%)
            → Both pairs should have similar attention affinity
        
        Implementation:
            Normalize geodesic distance by diameter:
            d_semantic = geodesic_dist / diameter
            
            Then scale RoPE:
            θ_effective = θ_base · (diameter / base_diameter)
            
            This makes attention depend on *relative position in the graph's
            structure*, not absolute hop count.
        
        Alternative (For Irregular Graphs):
            Use diffusion distance instead of geodesic:
            d_diffusion = -log(P_t(i → j))
            
            where P_t is the t-step random walk transition matrix.
            This captures "functional distance" (how information flows)
            rather than topological distance (shortest path).
        """
        scale = diameter / base_diameter
        scaled_base = self.base_freq * scale
        
        inv_freq = 1.0 / (scaled_base ** (
            torch.arange(0, self.dim_half, 2).float() / self.dim_half
        ))
        
        # geodesic_dist: [num_nodes, num_nodes] matrix of shortest path lengths
        freqs = torch.einsum('ij, k -> ijk', geodesic_dist.float(), inv_freq)
        return freqs.cos(), freqs.sin()
    def get_rope_char_lm(self, position, seq_length, base_seq_length=512):
        """
        RoPE for character-level language models (no BPE tokenization).
        
        Principle:
            Character-level models lack the "compression" of BPE, which
            naturally adapts to content density. A 10-character word is
            semantically similar whether embedded in a 100-char or 10,000-char
            document.
        
        Problem:
            Standard RoPE uses absolute position → "character 50" has the same
            encoding in a tweet (100 chars) and a novel (100k chars), even though
            its *semantic role* is very different.
        
        Solution:
            Scale RoPE by sequence length:
            θ_effective = θ_base · (seq_length / base_seq_length)
            
            This makes position encoding reflect *relative progress through
            the document*, not raw character index.
        
        Caveat:
            This assumes documents of different lengths have comparable
            *density* of information per character. For highly heterogeneous
            corpora (tweets + books), consider adaptive scaling based on
            local compression ratio (entropy per character).
        
        Why BPE Models Don't Need This:
            BPE tokens are already "semantically normalized" — common patterns
            (words, subwords) become single tokens regardless of context.
            A 100-token BPE sequence and a 10,000-token sequence are directly
            comparable (both represent ~100 and ~10,000 semantic units).
            Character-level models lack this normalization.
        """
        scale = seq_length / base_seq_length
        scaled_base = self.base_freq * scale
        
        inv_freq = 1.0 / (scaled_base ** (
            torch.arange(0, self.dim_half, 2).float() / self.dim_half
        ))
        
        # position: [seq_length] (0, 1, 2, ..., seq_length-1)
        freqs = torch.einsum('i, j -> i j', position.float(), inv_freq)
        return freqs.cos(), freqs.sin()
    """
    def get_mask(self, grid_size, radius=2.5):
        if grid_size == self.max_grid_size: d = self.dist_matrix
        else:
            y = torch.arange(grid_size, device=self.dist_matrix.device)
            grid_y, grid_x = torch.meshgrid(y, y, indexing='ij')
            active_coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1).float()
            d = torch.cdist(active_coords, active_coords, p=2)
        valid_mask = d < radius
        num_tokens = grid_size * grid_size
        def mask_mod(b, h, q_idx, kv_idx):
            q_safe, k_safe = q_idx.clamp(0, num_tokens - 1), kv_idx.clamp(0, num_tokens - 1)
            return valid_mask[q_safe, k_safe] & (q_idx < num_tokens) & (kv_idx < num_tokens)
        return create_block_mask(mask_mod, B=1, H=1, Q_LEN=num_tokens, KV_LEN=num_tokens)
    """
    def get_mask(self, grid_size, base_radius=2.5, base_grid=8.0):
        """
        Compute attention mask with logarithmically-scaled radius.
        
        Scaling Law:
            radius(G) = base_radius + log2(G / G_base) * scale_factor
        
        Intuition:
            At 8×8:   radius = 2.5 (covers ~20% of spatial extent)
            At 16×16: radius = 4.0 (covers ~12.5% of spatial extent)
            At 32×32: radius = 5.5 (covers ~8.6% of spatial extent)
            
            In absolute terms, the radius grows to capture
            finer details. In relative terms, it shrinks to
            maintain local inductive bias.
        
        Why Logarithmic (not linear, not constant):
            - Linear scaling: O(N²) attention cost, loses local bias
            - Constant radius: Fails to capture multi-tile patterns at high-res
            - Logarithmic: Balances detail capture with computational efficiency
        
        Why This Preserves Perceptual Consistency:
            The number of patches WITHIN a semantic structure
            (e.g., one checkerboard tile) grows linearly with resolution.
            But the number of ADJACENT structures we need to attend to
            grows logarithmically (due to hierarchical spatial decomposition).
        
        Connection to CNNs:
            This mimics how CNN receptive fields grow:
            - Early layers: Small, constant receptive field
            - Deep layers: Exponentially growing receptive field
            
            We implement this spatially (across resolution) rather than
            temporally (across layers).
        """
        # Logarithmic scaling coefficient
        # Tuned so that radius grows ~1.5 patches per doubling of resolution
        log_scale_factor = 1.5
        
        if grid_size <= base_grid:
            effective_radius = base_radius
        else:
            log_ratio = torch.log2(torch.tensor(grid_size / base_grid, dtype=torch.float32))
            effective_radius = base_radius + log_scale_factor * log_ratio.item()
        
        # Compute distance matrix
        if grid_size == self.max_grid_size:
            d = self.dist_matrix
        else:
            y = torch.arange(grid_size, device=self.dist_matrix.device)
            grid_y, grid_x = torch.meshgrid(y, y, indexing='ij')
            active_coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1).float()
            d = torch.cdist(active_coords, active_coords, p=2)
        
        valid_mask = d < effective_radius
        num_tokens = grid_size * grid_size
        
        def mask_mod(b, h, q_idx, kv_idx):
            """
            Block mask predicate for flex_attention.
            
            Ensures queries only attend to keys within the spatial neighborhood
            defined by effective_radius. This creates a spatially-localized
            attention pattern that scales gracefully across resolutions.
            """
            q_safe = q_idx.clamp(0, num_tokens - 1)
            k_safe = kv_idx.clamp(0, num_tokens - 1)
            return valid_mask[q_safe, k_safe] & (q_idx < num_tokens) & (kv_idx < num_tokens)
        
        return create_block_mask(
            mask_mod, 
            B=1, H=1, 
            Q_LEN=num_tokens, 
            KV_LEN=num_tokens
        )

class RnRoPE(nn.Module):
    def __init__(self, head_dim, spatial_buffer):
        super().__init__()
        self.buffer = spatial_buffer
        self.orthogonal = HouseholderOrthogonal(head_dim, num_reflections=head_dim//2)
        
    def forward(self, q, k, grid_size):
        q = self.orthogonal(q, inverse=True)
        k = self.orthogonal(k, inverse=True)
        cos, sin = self.buffer.get_rope(q.shape[0], grid_size)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        q1, q2 = q.chunk(2, dim=-1); k1, k2 = k.chunk(2, dim=-1)
        q_rot = (q * cos) + (torch.cat((-q2, q1), dim=-1) * sin)
        k_rot = (k * cos) + (torch.cat((-k2, k1), dim=-1) * sin)
        return self.orthogonal(q_rot, inverse=False), self.orthogonal(k_rot, inverse=False)

def get_global_mask(seq_len):
    def mask_mod(b, h, q_idx, kv_idx): return (q_idx < seq_len) & (kv_idx < seq_len)
    return create_block_mask(mask_mod, B=1, H=1, Q_LEN=seq_len, KV_LEN=seq_len)

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


class ContextualPatchEmbedder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, embed_dim=256, context_size=4, mlp_depth=1):
        super().__init__()
        self.context_size = context_size
        self.stride = 2
        self.padding = 1 
        
        patch_flat_dim = (context_size ** 2) * input_dim
        
        layers = []
        # Input Projection
        layers.append(nn.Linear(patch_flat_dim + 8, hidden_dim))
        layers.append(nn.SiLU())
        
        # Deep Interface Layers (The "Griddy" Scaling)
        for _ in range(mlp_depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            
        # Final Projection
        layers.append(nn.Linear(hidden_dim, embed_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x, spins):
        x_pad = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        patches = x_pad.unfold(2, self.context_size, self.stride).unfold(3, self.context_size, self.stride)
        B, C, GH, GW, K, _ = patches.shape
        num_tokens = GH * GW
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, num_tokens, -1)
        spins_expanded = spins.unsqueeze(1).expand(-1, num_tokens, -1)
        out = self.net(torch.cat([patches, spins_expanded], dim=-1))
        return out

class NonLinearOutputHead(nn.Module):
    def __init__(self, embed_dim, output_dim=12, mlp_depth=1): 
        super().__init__()
        
        layers = []
        # Input Projection
        layers.append(nn.Linear(embed_dim, embed_dim))
        layers.append(nn.SiLU())
        
        # Deep Interface Layers
        for _ in range(mlp_depth - 1):
            layers.append(nn.Linear(embed_dim, embed_dim))
            layers.append(nn.SiLU())
            
        # Final Projection
        layers.append(nn.Linear(embed_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

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
    def __init__(self, dim, num_heads, spatial_buffer, is_global=False, ffn_type='swiglu'):
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

    def forward(self, x, block_mask, grid_size):
        B, S, D = x.shape
        resid = x
        x_norm = self.norm1(x)
        
        q, k, v = self.qkv(x_norm).chunk(3, dim=-1)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        q, k = self.rn_rope(q, k, grid_size)
        
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
        gate_ffn = torch.sigmoid(self.mlp_gate(x_norm))
        x = resid + (ffn_out * gate_ffn)
        
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
        self.spatial = DynamicSpatialBuffer(max_grid_size=16, head_dim=embed_dim//4)
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
        self.patch_dim = 12
        
        self.output_head = NonLinearOutputHead(embed_dim, self.patch_dim, mlp_depth=interface_depth)
        
        if mode == 'factorized':
            self.scale_decoder = FourierScaleDecoder(self.fourier_dim, embed_dim, self.patch_dim)
            self.lambda_head = nn.Linear(embed_dim, 1)
        else:
            self.scale_decoder = None
            self.lambda_head = None

    def get_masks(self, grid_size):
        if grid_size in self.mask_cache: return self.mask_cache[grid_size]
        base_grid = 8.0; base_radius = 2.5
        scaled_radius = base_radius * (grid_size / base_grid)
        mask_local = self.spatial.get_mask(grid_size, radius=scaled_radius)
        mask_global = get_global_mask(grid_size * grid_size)
        self.mask_cache[grid_size] = (mask_local, mask_global)
        return mask_local, mask_global

    def forward(self, z_t, logsnr):
        B, C, H, W = z_t.shape
        grid_h, grid_w = H // 2, W // 2
        grid_size = grid_h 
        spins = self.spin_encoder(logsnr)
        x = self.patch_in(z_t, spins)
        mask_local, mask_global = self.get_masks(grid_size)
        total_aux_loss = 0.0
        for layer in self.layers:
            mask = mask_global if layer.is_global else mask_local
            x, al = layer(x, mask, grid_size)
            total_aux_loss += al
        x = self.norm_final(x)
        z_pred = self.output_head(x)
        l_pred = None
        if self.mode == 'factorized':
            scale = self.scale_decoder(spins).unsqueeze(1)
            z_pred = z_pred * scale
            l_pred = self.lambda_head(x.mean(dim=1)).squeeze(-1)
        return z_pred.view(B, grid_h, grid_w, 3, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W), l_pred, total_aux_loss