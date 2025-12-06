# ld_tformer.py
import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention
from nvllm_flex_attention import update_kv_cache

# === Initialization Helpers ===

def init_linear(m: nn.Linear, std=0.02):
    if hasattr(m, 'weight'):
        torch.nn.init.xavier_uniform_(m.weight)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.zeros_(m.bias)

def init_layer_norm(m):
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
    """
    Parametrized Orthogonal Matrix via product of Householder reflections.
    Used to project N-dimensional spatial coordinates into the rotation subspace.
    Reference: https://arxiv.org/abs/2504.06308
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
        Q = torch.eye(self.dim, device=self.vs.device)
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
    Generalized RoPE for R^n.
    Projects Topological Embeddings (Highway + Spatial) -> Rotation Frequencies.
    """
    def __init__(self, head_dim: int, topo_dim: int, num_reflections: int = 4):
        super().__init__()
        self.head_dim = head_dim
        # We project the [Topo_Dim] coordinate vector into [Head_Dim/2] logical pairs.
        # We use the Householder Orthogonal matrix to ensure this projection 
        # preserves the geometric structure of the manifold.
        self.ortho_proj = HouseholderOrthogonal(topo_dim, num_reflections)
        
        # We need to map from Topo Space to Frequency Space.
        # If Topo_Dim != Head_Dim/2, we need a Linear adapter.
        # Or, strictly following the paper, we might just learn a mixing.
        # Here we use a Linear layer initialized to project relevant dims.
        self.freq_linear = nn.Linear(topo_dim, head_dim // 2, bias=False)
        self.param_init()

    def param_init(self):
        self.ortho_proj.param_init()
        # Initialize frequency projection
        init_linear(self.freq_linear)

    def forward(self, x: torch.Tensor, topo_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, L, D]
            topo_embeds: [B, L, Topo_Dim]
        """
        # 1. Project Coordinates to Frequencies
        # topo_embeds is [B, L, T]. We want [B, L, D/2].
        # We assume topo_embeds are 'log-space' coordinates (linear distance).
        # We want to convert them to angular velocities.
        freqs = self.freq_linear(topo_embeds.float())
        
        # 2. Construct Complex Rotations
        # [B, L, D/2] -> [B, 1, L, D/2] for broadcasting over Heads
        freqs = freqs.unsqueeze(1)
        
        cos = freqs.cos()
        sin = freqs.sin()
        
        # 3. Apply Rotation (Standard pairs)
        # x: [..., D] -> x1 (even), x2 (odd)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        
        # Interleave back
        y = torch.stack([y1, y2], dim=-1).flatten(-2)
        return y.to(x.dtype)

# --- FFN & Blocks ---

class SwiGLU(nn.Module):
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
    def __init__(self, dim, hidden_dim, num_experts=8, num_active=2, jitter_noise=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.num_active = num_active
        self.jitter_noise = jitter_noise
        self.router = nn.Linear(dim, num_experts)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        self.experts = nn.ModuleList([SwiGLU(dim, hidden_dim) for _ in range(num_experts)])
        self.param_init()
        
    def param_init(self):
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        for expert in self.experts:
            expert.param_init()

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

# latent embedding units
class MLPResBlock(nn.Module):
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

class FourierFeatures(nn.Module):
    """
    Projects scalar fields into high-dimensional Fourier features.
    """
    def __init__(self, fourier_dim=16, scale=1.0):
        super().__init__()
        self.fourier_dim = fourier_dim
        self.scale = scale
        # Fixed frequencies: 2^0, 2^1, ... 
        # (Or random Gaussian, but powers of 2 are standard for position-like scalars)
        self.register_buffer("freqs", 2.0 ** torch.arange(0, fourier_dim // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., 1]
        x = x * self.scale
        args = x * self.freqs * math.pi
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class FourierScaleDecoder(nn.Module):
    """
    Decodes predicted Fourier-space features back into a scalar (LogSNR/Lambda).
    Used by the Unembedder to predict the noise level.
    """
    def __init__(self, fourier_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.param_init()

    def param_init(self):
        init_linear(self.net[0])
        # Initialize output layer near zero for stability
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def forward(self, f):
        # f: [..., Fourier_Dim]
        # We assume the network predicts Log(Lambda) or similar, 
        # but here we just return the raw scalar output.
        return torch.exp(self.net(f))

# latent unembedding units

class ContextualPatchEmbedder(nn.Module):
    """
    Tokenizes a single continuous latent raster + spatial logsnr map.
    Removes batch assumptions. Operates on [C, H, W].
    """
    def __init__(
        self, 
        input_channels: int = 3, 
        fourier_dim: int = 16, 
        embed_dim: int = 256, 
        patch_size: int = 2, 
        mlp_depth: int = 1
    ):
        super().__init__()
        self.patch_size = patch_size
        self.fourier_dim = fourier_dim
        # Patch Flat Dim: (C * P * P) + Fourier_Dim
        self.patch_flat_dim = (patch_size ** 2) * input_channels
        self.input_dim = self.patch_flat_dim + fourier_dim
        
        # 1. LogSNR Encoder
        self.fourier_enc = FourierFeatures(fourier_dim=fourier_dim)
        
        # 2. Input Projection        
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        self.res_blocks = nn.Sequential(*[
            MLPResBlock(embed_dim) for _ in range(max(0, mlp_depth - 1))
        ])
    self.param_init()
        
    def param_init(self):
        init_linear(self.input_proj)
        for block in self.res_blocks:
            block.param_init()

    def forward(
        self, 
        x: torch.Tensor, 
        logsnr_map: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Args:
            x: [C, H, W]
            logsnr_map: [1, H, W]
            
        Returns:
            z: [Num_Tokens, Embed_Dim]
            shape: (Grid_H, Grid_W)
        """
        C, H, W = x.shape
        P = self.patch_size
        
        if H % P != 0 or W % P != 0:
            raise ValueError(f"Input shape {H}x{W} not divisible by patch size {P}")
        
        # 1. Raster Patching
        # Unfold spatial dims: [C, H, W] -> [C, GH, GW, P, P]
        # Dim 1 is H, Dim 2 is W
        patches = x.unfold(1, P, P).unfold(2, P, P)
        
        GH, GW = patches.shape[1], patches.shape[2]
        
        # Permute to [GH, GW, C, P, P] -> Flatten to [GH*GW, C*P*P]
        patches = patches.permute(1, 2, 0, 3, 4).reshape(GH * GW, -1)
        
        # 2. LogSNR Pooling & Encoding
        # [1, H, W] -> AvgPool -> [1, GH, GW]
        # We need 4D input for avg_pool2d usually, or 3D works? 
        # F.avg_pool2d supports [C, H, W] or [B, C, H, W].
        logsnr_pooled = F.avg_pool2d(logsnr_map, kernel_size=P, stride=P)
        
        # Flatten: [1, GH, GW] -> [GH*GW, 1]
        logsnr_flat = logsnr_pooled.view(1, GH * GW).permute(1, 0)
        
        # Fourier: [GH*GW, F]
        logsnr_emb = self.fourier_enc(logsnr_flat)
        
        # 3. Concatenate
        raw_input = torch.cat([patches, logsnr_emb], dim=-1)
        
        # 4. Embed
        h = self.input_proj(raw_input)
        z = self.res_blocks(h)
        
        return z, (GH, GW)

class ContextualPatchUnembedder(nn.Module):
    """
    Reconstructs a single raster + logsnr map from tokens.
    """
    def __init__(
        self, 
        output_channels: int = 3, 
        fourier_dim: int = 16,
        embed_dim: int = 256, 
        patch_size: int = 2, 
        mlp_depth: int = 1
    ):
        super().__init__()
        self.patch_size = patch_size
        self.output_channels = output_channels
        self.fourier_dim = fourier_dim
        
        self.raster_flat_dim = output_channels * (patch_size ** 2)
        total_out_dim = self.raster_flat_dim + fourier_dim
        
        self.res_blocks = nn.Sequential(*[
            MLPResBlock(embed_dim) for _ in range(max(0, mlp_depth - 1))
        ])
        
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim), 
            nn.Linear(embed_dim, total_out_dim)
        )
        #i don't care if 256 is too big, having magic numbers is bad
        self.logsnr_decoder = FourierScaleDecoder(fourier_dim, hidden_dim=embed_dim, output_dim=1)
        self.param_init()

    def param_init(self):
        for block in self.res_blocks:
            block.param_init()
        init_layer_norm(self.output_proj[0])
        init_linear(self.output_proj[1])
        self.logsnr_decoder.param_init()

    def forward(
        self, 
        z: torch.Tensor, 
        shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Args:
            z: [Num_Tokens, Embed_Dim]
            shape: (Grid_H, Grid_W)
            
        Returns:
            [Output_Channels + 1, H, W]
        """
        L, D = z.shape
        P = self.patch_size
        GH, GW = shape
        
        if L != GH * GW:
            raise ValueError(f"Token count {L} does not match shape {GH}x{GW}")

        # 1. Decode Features
        h = self.res_blocks(z)
        flat = self.output_proj(h) # [L, Raster_Dim + Fourier_Dim]
        
        raster_part = flat[:, :self.raster_flat_dim]
        fourier_part = flat[:, self.raster_flat_dim:]
        
        # 2. Reconstruct Raster
        # [L, C*P*P] -> [GH, GW, C, P, P]
        patches = raster_part.reshape(GH, GW, self.output_channels, P, P)
        
        # Permute to [C, GH, P, GW, P] -> [C, H, W]
        patches = patches.permute(2, 0, 3, 1, 4)
        rasters = patches.reshape(self.output_channels, GH * P, GW * P)
        
        # 3. Reconstruct LogSNR
        # [L, F] -> [L, 1]
        logsnr_pred = self.logsnr_decoder(fourier_part)
        
        # Reshape to grid [1, GH, GW]
        logsnr_grid = logsnr_pred.view(GH, GW).unsqueeze(0)
        
        # Upsample [1, H, W]
        # Need 4D for interpolate: [1, 1, GH, GW]
        logsnr_pixel = F.interpolate(
            logsnr_grid.unsqueeze(0), 
            scale_factor=P, 
            mode='nearest'
        ).squeeze(0)
        
        # 4. Concat
        return torch.cat([rasters, logsnr_pixel], dim=0)

# ===== OUTSIDE MODEL: Span Processor =====

@dataclass
class Span:
    type: str  # 'text' | 'latent'
    start_idx: int
    end_idx: int
    shape: Tuple[int, ...]  # ~~(H, W) for images, () for text~~
    # wait that's not right at all. shape needs to be dim1, dim2, dim3, dim4, ... dim_final for images.
    # and shape needs to be (L) for text.
    causal: bool

class SpanEmbedder:
    def __init__(self, text_embedder, patch_embedder):
        self.text_emb = text_embedder
        self.patch_emb = patch_embedder
        
    def embed(
        self, 
        spans_metadata: List[Dict],
        text_tokens: Optional[List[torch.Tensor]] = None,
        images: Optional[List[torch.Tensor]] = None,
        logsnr_maps: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[Span], List[int]]:
        
        all_embeds = []
        span_objects = []
        cursor = 0
        text_idx = 0
        img_idx = 0
        
        from ld_tformer_embedding_functional import generate_content_hash_stream
        hash_spans = []

        for i, meta in enumerate(spans_metadata):
            span_type = meta['type']
            span_len = meta['len']
            
            if span_type == 'text':
                tokens = text_tokens[text_idx]
                emb = self.text_emb(tokens)
                text_idx += 1
                
                hash_spans.append({
                    'type': 'text',
                    'shape': (span_len,),
                    'data': tokens.cpu().tolist()
                })
                
            elif span_type == 'latent':
                img = images[img_idx]
                logsnr = logsnr_maps[img_idx]
                emb, grid_shape = self.patch_emb(img, logsnr)
                img_idx += 1
                
                meta['shape'] = grid_shape
                
                hash_spans.append({
                    'type': 'latent',
                    'shape': grid_shape,
                    'id': meta.get('id', f'img_{i}')
                })
            
            all_embeds.append(emb)
            span_objects.append(Span(
                type=span_type,
                start_idx=cursor,
                end_idx=cursor + span_len,
                shape=meta.get('shape', ()),
                causal=meta.get('causal', True)
            ))
            cursor += span_len
            
        content_hashes = generate_content_hash_stream(hash_spans)
        return torch.cat(all_embeds, dim=0), span_objects, content_hashes

class SpanUnembedder:
    """
    Converts flat [L_total, D] -> heterogeneous outputs.
    """
    def __init__(self, text_head, patch_unembedder):
        self.text_head = text_head
        self.patch_unembed = patch_unembedder
        
    def decode(
        self,
        z: torch.Tensor,  # [L_total, D]
        spans: List[Span]
    ) -> Dict[str, Any]:
        """
        Returns list ofs dict with:
            'text_logits': Tensor - [len(span), Vocab] per text span
            'image_vpreds': Tensor - [C, H, W] per image span
            'image_logsnrs': Tensor - [1, H, W] per image span
        """
        outputs = []
        for span in spans:
            spandict = {}
            z_span = z[span.start_idx:span.end_idx]
            
            #if span.type == 'text':
            # commented out because this branch is arbitrary; 
            # we might want to look at the logits for image patches.
            # it's up to the downstream model consumer to appreciate 
            # that there isn't a loss metric joining a 
            # [L, Vocab] <-> [C, H, W] tensor.
            # specifically, because 
            # L == H * W, and C != Vocab.
            logits = self.text_head(z_span)  # [L, Vocab]
            spandict['text_logits']=logits
                
            #elif span.type == 'latent':
            # Need to know grid shape to unflatten
            grid_shape = span.shape  # Should be (GH, GW) from metadata
            # is this a superfluous special case?
            # we should make sure we consider text spans to be shapeful.
            # and their shape is [len(span)]!
            
            # [L, D] + shape -> [C+1, H, W]
            reconstruction = self.patch_unembed(z_span, grid_shape)
            
            raster = reconstruction[:-1]  # [C, H, W]
            logsnr = reconstruction[-1:]   # [1, H, W]
            
            spandict['image_vpreds']=raster
            spandict['image_logsnrs']=logsnr
            outputs.append(spandict)
        # we now have vpred and logsnr field predictions for every single embedding even if this doesn't make sense!
        # hehe :)
        return outputs

def build_composed_mask(
    spans: List[Span],
    topo_active: torch.Tensor,  # [L_active, Topo_Dim] - for Q positions
    topo_heap: torch.Tensor,    # [Capacity, Topo_Dim] - for KV positions  
    page_table: PageTable,
    batch_idx: torch.Tensor,
    window_size: float = 10.0
) -> BlockMask:
    """
    Composes: Block-Causal AND Sliding-Window AND Paged-Lookup
    """
    from torch.nn.attention.flex_attention import create_block_mask
    from ld_tformer_embedding_functional import get_block_causal_mod, get_sliding_window_mod
    
    # 1. Build doc_ids from spans
    doc_ids = []
    for i, span in enumerate(spans):
        doc_ids.extend([i] * (span.end_idx - span.start_idx))
    doc_ids = torch.tensor(doc_ids, device=topo_embeds.device)
    
    # 2. Compose mask_mod
    def composed_mod(b, h, q_idx, kv_idx):
        # Both indices now index into spaces with GLOBAL coordinates
        
        # q_idx: index into active tokens (but they have global coords)
        # kv_idx: index into heap (which has global coords)
        
        # Spatial distance uses global coordinates
        q_highway = topo_active[q_idx, 0]  # e.g., 336
        k_highway = topo_heap[kv_idx, 0]   # e.g., 150
        
        # Highway enforces causality: can't attend to future
        causal = q_highway >= k_highway
        
        # Spatial dims for sliding window
        q_spatial = topo_active[q_idx, 1:]
        k_spatial = topo_heap[kv_idx, 1:]
        
        # For text tokens, spatial = [0, 0, ...] so distance is just highway
        # For image tokens, spatial = [x, y, ...] so distance includes geometry
        spatial_dist = torch.norm(q_spatial - k_spatial)
        
        return causal & (spatial_dist < window_size)
    
    # 3. Create logical mask
    logical_mask = create_block_mask(
        composed_mod,
        B=None, H=None,  # Will be inferred
        Q_LEN=len(doc_ids),
        KV_LEN=len(doc_ids)
    )
    
    # 4. Convert to physical mask
    physical_mask = page_table.convert_logical_block_mask(
        logical_mask,
        batch_idx
    )
    
    return physical_mask

class LDTformerAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, topo_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rope = RnRoPE(self.head_dim, topo_dim)

    # In LDTformerAttention.forward():
    def forward(
        self,
        x: torch.Tensor,           # [B, L, D] - ACTIVE tokens only
        topo_active: torch.Tensor, # [B, L_active, Topo_Dim] - GLOBAL COORDS
        k_cache: torch.Tensor,     # [1, H, Capacity, D] - FULL heap
        v_cache: torch.Tensor,     # [1, H, Capacity, D] - FULL heap
        slot_mapping: torch.Tensor,
        block_mask: object         # Already composed using HEAP topology
    ):
        """
        Stateless Attention:
        1. Projects Inputs
        2. Applies RoPE using Topology
        3. Commits New Data to Paged Heap
        4. Attends over Paged Heap using Physical Mask
        """
        B, L, D = x.shape
        
        # 1. Compute Q, K, V for NEW/ACTIVE tokens
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        # [3, B, L, H, D_head] -> [3, B, H, L, D_head]
        q, k, v = qkv.permute(0, 1, 3, 2, 4).unbind(0)
        
        # Apply RoPE using GLOBAL coordinates
        # topo_active[i] contains the ABSOLUTE highway position + spatial coords
        # So token at position 336 in the sequence gets highway=336, not highway=0
        q = self.rope(q, topo_active)  # Uses global positions
        k = self.rope(k, topo_active)  # Uses global positions
        
        # 3. Cache Write (Side Effect)
        # Transform from [B, H, L, D] -> [B*L, H, 1, D] semantics for scatter writer
        # We treat the batch as a flat stream of writes.
        k_write = k.transpose(1, 2).reshape(B * L, self.num_heads, 1, self.head_dim)
        v_write = v.transpose(1, 2).reshape(B * L, self.num_heads, 1, self.head_dim)
        
        update_kv_cache(k_write, v_write, k_cache, v_cache, slot_mapping)
        
        # Attention uses HEAP topology (via the mask)
        # The mask_mod already captures distances in the full heap
        out = flex_attention(q, k_cache, v_cache, block_mask=block_mask)
        
        # 5. Projection
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)

class LDTformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, topo_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.attn = LDTformerAttention(dim, num_heads, topo_dim)
        
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)
        
        # Using SigmoidMoE for FFN
        hidden_dim = int(dim * mlp_ratio)
        self.moe = SigmoidMoE(dim, hidden_dim, num_experts=8, num_active=3, ) # Defaults: 8 experts, 3 active
        self.gate_proj = nn.Linear(dim)

    def forward(self, x, topo, k_cache, v_cache, slots, mask):
        # Attention Sub-block
        h = self.norm1(x)
        h = self.attn(h, topo, k_cache, v_cache, slots, mask)
        gh = torch.sigmoid(self.gate_proj(h))
        x = x + (h*gh)
        
        # MoE Sub-block
        h_moe, aux_loss = self.moe(self.norm2(x))
        x = x + h_moe
        
        return x, aux_loss

# ===== INSIDE MODEL: Metadata-Agnostic =====

class coolerLDTformer(nn.Module):
    def __init__(self, dim=256, depth=8, num_heads=8, topo_dim=4, vocab_size=65536):
        super().__init__()
        
        # Embedding heads (used by SpanEmbedder)
        self.text_embed = nn.Embedding(vocab_size, dim)
        self.patch_embedder = ContextualPatchEmbedder(
            input_channels=3,
            embed_dim=dim,
            patch_size=2
        )
        
        # Transformer trunk
        self.layers = nn.ModuleList([
            LDTformerBlock(dim, num_heads, topo_dim) for _ in range(depth)
        ])
        
        # Output heads (used by SpanUnembedder)
        self.text_head = nn.Linear(dim, vocab_size)
        self.patch_unembedder = ContextualPatchUnembedder(
            output_channels=3,
            embed_dim=dim,
            patch_size=2
        )
        
        self.final_norm = nn.LayerNorm(dim)
        # Initialize everything
        self.param_init()
    

    def param_init(self):
        # Top level params
        torch.nn.init.normal_(self.text_embed.weight, mean=0.0, std=0.02)
        init_linear(self.text_head)
        init_layer_norm(self.final_norm)
        
        # Recursively init custom modules
        self.patch_embedder.param_init()
        self.patch_unembedder.param_init()
        for layer in self.layers:
            layer.param_init()

    def forward(
        self,
        z: torch.Tensor,           # [B, L_total, D] - FLAT
        topo_embeds: torch.Tensor, # [B, L_total, Topo_Dim] - FLAT
        k_caches: list,
        v_caches: list,
        slot_mapping: torch.Tensor,
        block_mask: object
    ) -> Tuple[torch.Tensor, float]:
        """
        Pure transformer pass. No span logic.
        
        Returns:
            z_out: [B, L_total, D] - transformed features
            aux_loss: scalar
        """
        x = z
        total_aux = 0.0
        
        for i, layer in enumerate(self.layers):
            x, aux = layer(x, topo_embeds, k_caches[i], v_caches[i], 
                          slot_mapping, block_mask)
            total_aux += aux
        
        x = self.final_norm(x)
        return x, total_aux

    # === LIFECYCLE METHODS ===
    
    def dump(self) -> Dict[str, torch.Tensor]:
        """Return reference to parameters (no move)."""
        return {k: v.clone() for k, v in self.state_dict().items()}

    def flush(self):
        """Zero out all parameters and gradients to simulate fresh state."""
        for p in self.parameters():
            p.data.zero_()
            if p.grad is not None:
                p.grad.zero_()
        
    def param_load(self, state_dict):
        """Load a specific parameter set."""
        self.load_state_dict(state_dict)

"""
# === Training ===

# 1. Prepare heterogeneous batch
spans_meta = [
    {'type': 'text', 'len': 128},
    {'type': 'latent', 'len': 144, 'shape': (12, 12), 'causal': False},
    {'type': 'latent', 'len': 256, 'shape': (16, 16), 'causal': False}
]

embedder = SpanEmbedder(model.text_embed, model.patch_embedder)
unembedder = SpanUnembedder(model.text_head, model.patch_unembedder)

# 2. Embed (outside model)
z_flat, span_objects = embedder.embed(
    spans_meta,
    text_tokens=[tokens_128],
    images=[img_144, img_256],
    logsnr_maps=[logsnr_144, logsnr_256]
)  # [528, D]

# 3. Run model (inside, metadata-agnostic)
z_out, aux_loss = model(
    z_flat.unsqueeze(0),  # [1, 528, D]
    topo_embeds.unsqueeze(0),
    k_caches, v_caches, slot_mapping, block_mask
)

# 4. Decode (outside model)
outputs = unembedder.decode(z_out.squeeze(0), span_objects)

# 5. Compute losses (user's choice)
text_loss = F.cross_entropy(
    outputs['text_logits'][0], 
    text_targets
)

img_loss = sum(
    F.mse_loss(pred, target) 
    for pred, target in zip(outputs['image_vpreds'], image_targets)
)

total_loss = text_loss + img_loss + aux_loss
"""