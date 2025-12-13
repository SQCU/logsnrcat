# ld_tformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, BlockMask
from typing import Tuple, List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import math

from nvllm_flex_attention import update_kv_cache
from memory_manager import KVTManager, PageTable

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
    def __init__(self, head_dim: int, topo_dim: int, rope_base: float = 500.0):
        super().__init__()
        self.head_dim = head_dim
        self.topo_dim = topo_dim
        self.freq_dim = head_dim // 2
        
        # Householder rotation for latent space projection
        self.orthogonal = HouseholderOrthogonal(head_dim, num_reflections=head_dim//2)
        
        # Calculate how many frequency bands each topology dimension gets.
        # e.g., Head=64 -> Freq=32. Topo=3 -> 10 bands per dim.
        # This fixes the utilization issue; previous logic used only half capacity.
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
        Args:
            q, k: [B, H, L, D]
            topo_embeds: [B, L, Topo_Dim]
            scale: Scaling factor for context length generalization (inv_freq / scale).
        """
        B, H, L, D = q.shape
        
        # 1. Rotate into frequency-friendly space
        # Collapse B, H, L for efficient matmul
        q = self.orthogonal(q.reshape(B*H*L, D), inverse=True).reshape(B, H, L, D)
        k = self.orthogonal(k.reshape(B*H*L, D), inverse=True).reshape(B, H, L, D)
        
        # 2. Vectorized Frequency Computation
        # Slice inputs to supported dimensions (handles implicit truncation if input has extra dims)
        t_embeds = topo_embeds[..., :self.topo_dim] # [B, L, Topo_Dim]
        
        # Scale frequencies (Context Generalization)
        inv_freq_scaled = self.inv_freq / scale # [Subspace_Dim]
        
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
        cos = full_freqs.cos().unsqueeze(1).repeat(1, 1, 1, 2)[..., :D]
        sin = full_freqs.sin().unsqueeze(1).repeat(1, 1, 1, 2)[..., :D]
        
        # 5. Apply RoPE (Standard Rotate Half)
        def rotate_half(x):
            x1, x2 = x[..., :D//2], x[..., D//2:]
            return torch.cat([-x2, x1], dim=-1)
        
        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        
        # 6. Rotate back to original basis
        q_out = self.orthogonal(q_rot.reshape(B*H*L, D), inverse=False).reshape(B, H, L, D)
        k_out = self.orthogonal(k_rot.reshape(B*H*L, D), inverse=False).reshape(B, H, L, D)
        
        return q_out, k_out

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
        # 1. Routing
        B, L, D = x.shape
        router_logits = self.router(x)
        if self.training and self.jitter_noise > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.jitter_noise
            
        scores = torch.sigmoid(router_logits)
        top_k_scores, top_k_indices = torch.topk(scores, self.num_active, dim=-1)
        
        # Normalize weights
        router_weights = top_k_scores / (top_k_scores.sum(dim=-1, keepdim=True) + 1e-6)
        
        # 2. Flatten for efficient indexing
        # Shape: [N, D] where N = B*L
        x_flat = x.view(-1, D)
        out_flat = torch.zeros_like(x_flat)
        
        # Shape: [N, K]
        indices_flat = top_k_indices.view(-1, self.num_active)
        weights_flat = router_weights.view(-1, self.num_active)
        
        # 3. Process Experts (Static Loop for compilation)
        for e in range(self.num_experts):
            # A. Identify tokens for this expert
            # mask: [N, K] bool
            match_mask = (indices_flat == e)
            
            # token_mask: [N] bool (True if token picked expert 'e' in ANY slot)
            token_mask = match_mask.any(dim=-1)
            
            # B. Get Indices (Dynamic Shape, but graph-safe)
            active_indices = torch.nonzero(token_mask).flatten()
            
            # C. Gather Weights & Input
            # Aggregate weight for expert 'e' per token (usually just one slot, but sums if dupes)
            # [N] -> [Num_Selected]
            active_weights = (weights_flat * match_mask.float()).sum(dim=-1)[active_indices]
            
            # Gather inputs: [Num_Selected, D]
            # (If active_indices is empty, this is [0, D], which works fine in Linear layers)
            active_x = x_flat[active_indices]
            
            # D. Compute Expert
            expert_out = self.experts[e](active_x)
            
            # E. Scale & Scatter Add
            weighted_out = expert_out * active_weights.unsqueeze(-1)
            out_flat.index_add_(0, active_indices, weighted_out)
        
        aux_loss = 1e-2 * (router_logits ** 2).mean()
        return out_flat.view(B, L, D), aux_loss

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
        return self.net(f)

# latent unembedding units

class ContextualPatchEmbedder(nn.Module):
    def __init__(self, input_channels=3, fourier_dim=16, embed_dim=256, context_size=4, stride=2, mlp_depth=1):
        super().__init__()
        self.context_size = context_size
        self.stride = stride
        self.padding = (context_size - stride) // 2
        self.fourier_enc = FourierFeatures(fourier_dim=fourier_dim)
        self.input_dim = (context_size ** 2) * input_channels + fourier_dim
        self.input_proj = nn.Linear(self.input_dim, embed_dim)
        self.res_blocks = nn.Sequential(*[MLPResBlock(embed_dim) for _ in range(max(0, mlp_depth - 1))])
        self.param_init()
    def param_init(self):
        for block in self.res_blocks: block.param_init()
        init_linear(self.input_proj)
    def forward(self, x, logsnr_map):
        # x: [C, H, W], logsnr_map: [1, H, W]
        x_pad = F.pad(x, (self.padding,)*4, mode='reflect')
        logsnr_pad = F.pad(logsnr_map, (self.padding,)*4, mode='reflect')
        patches_img = x_pad.unfold(1, self.context_size, self.stride).unfold(2, self.context_size, self.stride)
        GH, GW = patches_img.shape[1], patches_img.shape[2]
        patches_img = patches_img.permute(1, 2, 0, 3, 4).reshape(GH * GW, -1)
        patches_logsnr = logsnr_pad.unfold(1, self.context_size, self.stride).unfold(2, self.context_size, self.stride)
        patches_logsnr = patches_logsnr.permute(1, 2, 0, 3, 4).reshape(GH * GW, -1).mean(dim=-1, keepdim=True)
        logsnr_features = self.fourier_enc(patches_logsnr)
        raw_input = torch.cat([patches_img, logsnr_features], dim=-1)
        h = self.input_proj(raw_input)
        z = self.res_blocks(h)
        return z, (GH, GW)


class ContextualPatchUnembedder(nn.Module):
    def __init__(self, output_channels=3, fourier_dim=16, embed_dim=256, patch_size=2, mlp_depth=1):
        super().__init__()
        self.patch_size = patch_size
        self.output_channels = output_channels
        self.raster_flat_dim = output_channels * (patch_size ** 2)
        self.res_blocks = nn.Sequential(*[MLPResBlock(embed_dim) for _ in range(max(0, mlp_depth - 1))])
        self.output_proj = nn.Sequential(nn.LayerNorm(embed_dim, elementwise_affine=False), nn.Linear(embed_dim, self.raster_flat_dim + fourier_dim))
        self.logsnr_decoder = FourierScaleDecoder(fourier_dim, hidden_dim=embed_dim, output_dim=1)
        self.param_init()
    def param_init(self):
        for block in self.res_blocks: block.param_init()
        init_layer_norm(self.output_proj[0]); init_linear(self.output_proj[1])
        self.logsnr_decoder.param_init()
    def forward(self, z, shape):
        L, D = z.shape
        P = self.patch_size
        if len(shape) == 2: GH, GW = shape
        elif len(shape) == 1: GH, GW = 1, shape[0]
        else: GH, GW = 1, L
        if L != GH * GW: GH, GW = 1, L # Fallback

        flat = self.output_proj(self.res_blocks(z))
        raster_part = flat[:, :self.raster_flat_dim]
        fourier_part = flat[:, self.raster_flat_dim:]
        
        patches = raster_part.reshape(GH, GW, self.output_channels, P, P)
        patches = patches.permute(2, 0, 3, 1, 4)
        rasters = patches.reshape(self.output_channels, GH * P, GW * P)
        
        logsnr_pred = self.logsnr_decoder(fourier_part)
        logsnr_grid = logsnr_pred.view(GH, GW).unsqueeze(0)
        logsnr_pixel = F.interpolate(logsnr_grid.unsqueeze(0), scale_factor=P, mode='nearest').squeeze(0)
        return torch.cat([rasters, logsnr_pixel], dim=0)


# ===== OUTSIDE MODEL: Span Processor =====
@dataclass
class Span:
    type: str  # 'text' | 'latent'
    start_idx: int
    end_idx: int
    shape: Tuple[int, ...] 
    causal: bool    
    doc_id: int 

@dataclass
class ContextBlock:
    """
    Canonical atomic unit of the dataset.
    Holds raw data and its topological metadata.
    """
    content: Union[torch.Tensor, str] # [3, H, W] or String
    type: str = 'latent'
    causal: bool = True
    # Metadata
    shape_meta: Tuple[int, ...] = field(default_factory=tuple)
    logsnr: Optional[torch.Tensor] = None # [1, H, W]
    group_id: int = 0
    id: str = ""

    def __post_init__(self):
        if not self.shape_meta:
             if isinstance(self.content, torch.Tensor) and self.type == 'latent':
                 h, w = self.content.shape[-2:]
                 self.shape_meta = (h // 2, w // 2)
             elif isinstance(self.content, str) and self.type == 'text':
                 self.shape_meta = (len(self.content),)
             elif isinstance(self.content, torch.Tensor) and self.type == 'text':
                 self.shape_meta = (self.content.shape[0],)

class SpanEmbedder:
    def __init__(self, text_embedder, patch_embedder):
        self.text_emb = text_embedder
        self.patch_emb = patch_embedder
        
    def embed(self, context_blocks: List[ContextBlock]) -> Tuple[torch.Tensor, List[Span], List[int]]:
        all_embeds = []
        span_objects = []
        cursor = 0
        
        from ld_tformer_embedding_functional import generate_content_hash_stream
        hash_spans = []

        for block in context_blocks:
            if block.type == 'text':
                # Assuming content is already tokenized tensor or handle tokenization externally?
                # The benchmark scripts pass tokenized tensors. 
                # Let's assume input is Tensor[Long].
                tokens = block.content
                if isinstance(tokens, str): raise ValueError("SpanEmbedder expects tokenized text tensors, not strings.")
                
                emb = self.text_emb(tokens)
                span_len = tokens.shape[0]
                hash_spans.append({'type': 'text', 'shape': (span_len,), 'data': tokens.cpu().tolist()})
                
            elif block.type == 'latent':
                img = block.content
                logsnr = block.logsnr
                # Direct 3D Tensor processing
                emb, grid_shape = self.patch_emb(img, logsnr)
                span_len = emb.shape[0]
                hash_spans.append({'type': 'latent', 'shape': grid_shape, 'id': block.id})
            
            all_embeds.append(emb)
            span_objects.append(Span(
                type=block.type,
                start_idx=cursor,
                end_idx=cursor + span_len,
                shape=block.shape_meta,
                causal=block.causal,
                doc_id=block.group_id
            ))
            cursor += span_len
            
        content_hashes = generate_content_hash_stream(hash_spans)
        return torch.cat(all_embeds, dim=0), span_objects, content_hashes

class SpanUnembedder:
    def __init__(self, text_head, patch_unembedder):
        self.text_head = text_head
        self.patch_unembed = patch_unembedder
        
    def decode(self, z: torch.Tensor, spans: List[Span]) -> List[Dict[str, Any]]:
        outputs = []
        for span in spans:
            spandict = {}
            z_span = z[span.start_idx:span.end_idx]
            
            # Text Head (Always computable)
            spandict['text_logits'] = self.text_head(z_span)
            
            # Latent Head (Always computable, handles 1D/2D)
            reconstruction = self.patch_unembed(z_span, span.shape)
            spandict['image_vpreds'] = reconstruction[:-1]
            spandict['image_logsnrs'] = reconstruction[-1:]
            
            outputs.append(spandict)
        return outputs

def build_dual_masks(
    spans: List[Span],
    topo_active: torch.Tensor,
    topo_heap: torch.Tensor,
    page_table: Optional[PageTable] = None,
    flat_page_table: Optional[torch.Tensor] = None,
    inverse_page_table: Optional[torch.Tensor] = None,
    window_size: float = 10.0
) -> Tuple[BlockMask, BlockMask]:
    """
    Returns (local_mask, global_mask).
    Global mask ignores spatial constraints but respects Document/Causal boundaries.
    """
    from torch.nn.attention.flex_attention import create_block_mask
    
    device = topo_active.device
    L_active = topo_active.shape[0]
    L_heap = topo_heap.shape[0]
    block_size = page_table.block_size
    
    # 1. Build doc_ids for ACTIVE tokens
    # USE THE EXPLICIT doc_id FROM THE SPAN
    doc_ids_active = []
    for span in spans:
        doc_ids_active.extend([span.doc_id] * (span.end_idx - span.start_idx))
    doc_ids_active_t = torch.tensor(doc_ids_active, dtype=torch.long, device=topo_active.device)
    
    # 2. Build doc_ids for HEAP (EFFICIENT VERSION)
    # Initialize heap as -1
    L_heap = topo_heap.shape[0]
    doc_ids_heap_t = torch.full((L_heap,), -1, dtype=torch.long, device=topo_active.device)
    
    block_size = page_table.block_size
    cursor = 0
    for span in spans:
        span_len = span.end_idx - span.start_idx
        
        # Get logical block range for this span
        start_block = cursor // block_size
        end_block = (cursor + span_len - 1) // block_size + 1
        
        # Map logical blocks -> physical slots
        for log_block_idx in range(start_block, end_block):
            if log_block_idx >= len(flat_page_table):
                break
                
            phys_block = flat_page_table[log_block_idx].item()
            
            # Calculate which tokens in this span map to this physical block
            block_start_in_span = max(0, log_block_idx * block_size - cursor)
            block_end_in_span = min(span_len, (log_block_idx + 1) * block_size - cursor)
            
            # Physical slot range
            offset_start = (cursor + block_start_in_span) % block_size
            offset_end = offset_start + (block_end_in_span - block_start_in_span)
            
            phys_start = phys_block * block_size + offset_start
            phys_end = phys_block * block_size + offset_end
            
            # Mark these slots as belonging to this document
            doc_ids_heap_t[phys_start:phys_end] = span.doc_id
            if flat_page_table is not None:
             # Identity mapping shortcut for ZC
                doc_ids_heap_t[cursor:cursor+span_len] = span.doc_id
        
        cursor += span_len
    
    # 3. Decompose Topology
    topo_active_cols = topo_active.unbind(dim=-1)
    highway_active = topo_active_cols[0]
    spatial_active = topo_active_cols[1:]
    
    topo_heap_cols = topo_heap.unbind(dim=-1)
    highway_heap = topo_heap_cols[0]
    spatial_heap = topo_heap_cols[1:]
    
    win_sq = torch.tensor(window_size * window_size, device=device, dtype=topo_active.dtype)

    # 1. The Core Connectivity Logic (Shared)
    def base_connectivity(q_idx, kv_idx):
        q_doc = doc_ids_active_t[q_idx]
        k_doc = doc_ids_heap_t[kv_idx]
        same_doc = (q_doc == k_doc) & (k_doc >= 0)
        
        q_time = highway_active[q_idx]
        k_time = highway_heap[kv_idx]
        causal = q_time >= k_time
        
        return same_doc & causal

    # 2. Local Mod (Spatial Window)
    def mask_mod_local(b, h, q_idx, kv_idx):
        base = base_connectivity(q_idx, kv_idx)
        
        dist_sq = 0.0
        for q_col, k_col in zip(spatial_active, spatial_heap):
            d = q_col[q_idx] - k_col[kv_idx]
            dist_sq = dist_sq + (d * d)
            
        spatial_ok = dist_sq < win_sq
        return base & spatial_ok

    # 3. Global Mod (Infinite Window)
    def mask_mod_global(b, h, q_idx, kv_idx):
        return base_connectivity(q_idx, kv_idx)

    # 4. Compile
    local_mask = create_block_mask(
        mask_mod_local, B=None, H=None, Q_LEN=L_active, KV_LEN=L_heap
    )
    global_mask = create_block_mask(
        mask_mod_global, B=None, H=None, Q_LEN=L_active, KV_LEN=L_heap
    )
    
    return local_mask, global_mask

def build_composed_mask(
    spans: List[Span],
    topo_active: torch.Tensor,      # [L_active, Topo_Dim] - Active tokens only
    topo_heap: torch.Tensor,        # [Capacity, Topo_Dim] - Full heap
    page_table: PageTable,
    flat_page_table: torch.Tensor,  # [Num_Logical_Blocks] -> Physical_Block_ID
    inverse_page_table: torch.Tensor,  # [Capacity_Blocks] -> Logical_Block_ID
    window_size: float = 10.0
) -> BlockMask:
    """
    Composes masks and converts to physical B=1 space.
    EFFICIENT: Uses tensor operations instead of Python loops.
    """
    from torch.nn.attention.flex_attention import create_block_mask
    
    device = topo_active.device
    L_active = topo_active.shape[0]
    L_heap = topo_heap.shape[0]
    block_size = page_table.block_size
    
    # 1. Build doc_ids for ACTIVE tokens
    doc_ids_active = []
    for i, span in enumerate(spans):
        doc_ids_active.extend([i] * (span.end_idx - span.start_idx))
    doc_ids_active_t = torch.tensor(doc_ids_active, dtype=torch.long, device=device)
    
    # 2. Build doc_ids for HEAP (EFFICIENT VERSION)
    # Initialize heap as unallocated
    doc_ids_heap_t = torch.full((L_heap,), -1, dtype=torch.long, device=device)
    
    # For each span, compute which physical slots it occupies
    cursor = 0
    for rid, span in enumerate(spans):
        span_len = span.end_idx - span.start_idx
        
        # Get logical block range for this span
        start_block = cursor // block_size
        end_block = (cursor + span_len - 1) // block_size + 1
        
        # Map logical blocks -> physical slots
        for log_block_idx in range(start_block, end_block):
            if log_block_idx >= len(flat_page_table):
                break
                
            phys_block = flat_page_table[log_block_idx].item()
            
            # Calculate which tokens in this span map to this physical block
            block_start_in_span = max(0, log_block_idx * block_size - cursor)
            block_end_in_span = min(span_len, (log_block_idx + 1) * block_size - cursor)
            
            # Physical slot range
            offset_start = (cursor + block_start_in_span) % block_size
            offset_end = offset_start + (block_end_in_span - block_start_in_span)
            
            phys_start = phys_block * block_size + offset_start
            phys_end = phys_block * block_size + offset_end
            
            # Mark these slots as belonging to this document
            doc_ids_heap_t[phys_start:phys_end] = rid
        
        cursor += span_len
    
    # 3. Decompose Topology
    topo_active_cols = topo_active.unbind(dim=-1)
    highway_active = topo_active_cols[0]
    spatial_active = topo_active_cols[1:]
    
    topo_heap_cols = topo_heap.unbind(dim=-1)
    highway_heap = topo_heap_cols[0]
    spatial_heap = topo_heap_cols[1:]
    
    win_sq = torch.tensor(window_size * window_size, device=device, dtype=topo_active.dtype)
    
    # 4. Build Physical Mask Mod (PURE FUNCTIONAL)
    # 4. Build Physical Mask Mod (PURE FUNCTIONAL)
    def physical_mask_mod(b, h, q_idx, kv_idx):
        """
        Pure functional mask - all operations are tensor ops.
        """
        # Document matching
        q_doc = doc_ids_active_t[q_idx]
        k_doc = doc_ids_heap_t[kv_idx]
        same_doc = (q_doc == k_doc) & (k_doc >= 0)
        
        # Causal constraint
        q_time = highway_active[q_idx]
        k_time = highway_heap[kv_idx]
        causal = q_time >= k_time
        
        # Spatial window
        # FIX: Use python float 0.0 instead of torch.tensor(0.0).
        # Creating 0-d tensors inside mask_mod confuses Inductor/Triton compilation.
        dist_sq = 0.0
        for q_col, k_col in zip(spatial_active, spatial_heap):
            d = q_col[q_idx] - k_col[kv_idx]
            dist_sq = dist_sq + (d * d)
        
        spatial_ok = dist_sq < win_sq
        
        return same_doc & causal & spatial_ok
    
    # 5. Create Physical Mask
    physical_mask = create_block_mask(
        physical_mask_mod,
        B=None, H=None,
        Q_LEN=L_active,
        KV_LEN=L_heap
    )
    
    return physical_mask

class LDTformerAttentionKVC(nn.Module):
    def __init__(self, dim: int, num_heads: int, topo_dim: int, is_global=False, rope_base: float = 500.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rope = RnRoPE(self.head_dim, topo_dim, rope_base=rope_base)
        # Initialize immediately
        self.param_init()

    def param_init(self):
        init_linear(self.qkv)
        init_linear(self.proj)
        self.rope.param_init()

    # In LDTformerAttention.forward():
    def forward(
        self,
        x: torch.Tensor,           # [B, L, D] - ACTIVE tokens only
        topo_active: torch.Tensor, # [B, L_active, Topo_Dim] - GLOBAL COORDS
        k_cache: torch.Tensor,     # [1, H, Capacity, D] - FULL heap
        v_cache: torch.Tensor,     # [1, H, Capacity, D] - FULL heap
        slot_mapping: torch.Tensor,
        block_mask: object,         # Already composed using HEAP topology
        scale: float = 1.0
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
        # [B, L, 3, H, D_head] -> [3, B, H, L, D_head]
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        
        # Apply RoPE using GLOBAL coordinates
        # topo_active[i] contains the ABSOLUTE highway position + spatial coords
        # So token at position 336 in the sequence gets highway=336, not highway=0
        q, k = self.rope(q, k, topo_active, scale=scale)  # Uses global positions
        
        # 3. Cache Write (Side Effect)
        # Transform from [B, H, L, D] -> [B*L, H, 1, D] semantics for scatter writer
        # We treat the batch as a flat stream of writes.
        k_write = k.transpose(1, 2).reshape(B * L, self.num_heads, 1, self.head_dim).clone()
        v_write = v.transpose(1, 2).reshape(B * L, self.num_heads, 1, self.head_dim).clone()
        
        update_kv_cache(k_write, v_write, k_cache, v_cache, slot_mapping)
        
        # Attention uses HEAP topology (via the mask)
        # The mask_mod already captures distances in the full heap
        out = flex_attention(q, k_cache, v_cache, block_mask=block_mask)
        
        # 5. Projection
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)

class LDTformerAttentionZC(nn.Module):
    def __init__(self, dim: int, num_heads: int, topo_dim: int, rope_base: float = 500.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rope = RnRoPE(self.head_dim, topo_dim, rope_base=rope_base)
        # Initialize immediately
        self.param_init()

    def param_init(self):
        init_linear(self.qkv)
        init_linear(self.proj)
        self.rope.param_init()

    # In LDTformerAttention.forward():
    def forward(
        self,
        x: torch.Tensor,           # [B, L, D] - ACTIVE tokens only
        topo_active: torch.Tensor, # [B, L_active, Topo_Dim] - GLOBAL COORDS
        slot_mapping: torch.Tensor,
        block_mask: object,         # Already composed using HEAP topology
        scale: float = 1.0
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
        # [B, L, 3, H, D_head] -> [3, B, H, L, D_head]
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        
        # Apply RoPE using GLOBAL coordinates
        # topo_active[i] contains the ABSOLUTE highway position + spatial coords
        # So token at position 336 in the sequence gets highway=336, not highway=0
        q, k = self.rope(q, k, topo_active, scale=scale)  # Uses global positions
        
        # Attention uses HEAP topology (via the mask)
        # The mask_mod already captures distances in the full heap
        out = flex_attention(q, k, v, block_mask=block_mask)
        
        # 5. Projection
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)

class LDTformerBlockKVC(nn.Module):
    def __init__(self, dim: int, num_heads: int, topo_dim: int, mlp_ratio: float = 4.0, is_global=False, num_experts=8, num_active=3, rope_base: float = 500.0):
        super().__init__()
        self.is_global = is_global
        self.rope_base = rope_base*(100**is_global)
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.attn = LDTformerAttentionKVC(dim, num_heads, topo_dim, rope_base=self.rope_base)        
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)
        
        # Using SigmoidMoE for FFN
        hidden_dim = int(dim * mlp_ratio)
        self.moe = SigmoidMoE(dim, hidden_dim, num_experts=num_experts, num_active=num_active, ) # Defaults: 8 experts, 3 active
        self.gate_proj = nn.Linear(dim, dim)
        # Initialize immediately
        self.param_init()

    def param_init(self):
        init_layer_norm(self.norm1)
        self.attn.param_init()
        init_layer_norm(self.norm2)
        self.moe.param_init()
        init_linear(self.gate_proj)

    def forward(self, x, topo, k_cache, v_cache, slots, mask, scale: float = 1.0):
        # Attention Sub-block
        h = self.norm1(x)
        h = self.attn(h, topo, k_cache, v_cache, slots, mask, scale=scale)
        gh = torch.sigmoid(self.gate_proj(h))
        x = x + (h*gh)
        
        # MoE Sub-block
        h_moe, aux_loss = self.moe(self.norm2(x))
        x = x + h_moe
        
        return x, aux_loss

class LDTformerBlockZC(nn.Module):
    def __init__(self, dim: int, num_heads: int, topo_dim: int, mlp_ratio: float = 4.0, is_global=False, num_experts=8, num_active=3, rope_base: float = 500.0):
        super().__init__()
        self.is_global = is_global
        self.rope_base = rope_base*(100**is_global)
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.attn = LDTformerAttentionZC(dim, num_heads, topo_dim, rope_base=self.rope_base)
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)
        
        # Using SigmoidMoE for FFN
        hidden_dim = int(dim * mlp_ratio)
        self.moe = SigmoidMoE(dim, hidden_dim, num_experts=num_experts, num_active=num_active, ) # Defaults: 8 experts, 3 active
        self.gate_proj = nn.Linear(dim, dim)
        # Initialize immediately
        self.param_init()

    def param_init(self):
        init_layer_norm(self.norm1)
        self.attn.param_init()
        init_layer_norm(self.norm2)
        self.moe.param_init()
        init_linear(self.gate_proj)

    def forward(self, x, topo, slots, mask, scale: float = 1.0):
        # Attention Sub-block
        h = self.norm1(x)
        h = self.attn(h, topo, slots, mask, scale=scale)
        gh = torch.sigmoid(self.gate_proj(h))
        x = x + (h*gh)
        
        # MoE Sub-block
        h_moe, aux_loss = self.moe(self.norm2(x))
        x = x + h_moe
        
        return x, aux_loss

# ===== INSIDE MODEL: Metadata-Agnostic =====

class coolerLDTformerKVC(nn.Module):
    def __init__(self, dim=256, depth=8, num_heads=8, topo_dim=4, mlp_depth=1, vocab_size=65536, global_layer_interval=4, num_experts=8, num_active=3, rope_base: int = 500):
        super().__init__()
        
        # Embedding heads (used by SpanEmbedder)
        self.global_layer_interval = global_layer_interval
        self.text_embed = nn.Embedding(vocab_size, dim)
        self.patch_embedder = ContextualPatchEmbedder(
            input_channels=3,
            embed_dim=dim,
            context_size= 4,  # ← Add this back!
            stride= 2,         # ← Add this back!
            mlp_depth=mlp_depth
        )
        
        # Transformer trunk
        self.layers = nn.ModuleList([
            LDTformerBlockKVC(dim, num_heads, topo_dim, is_global=((i+1)%global_layer_interval==0), num_experts=num_experts, num_active=num_active, rope_base=rope_base) for i in range(depth) 
        ])
        
        # Output heads (used by SpanUnembedder)
        self.text_head = nn.Linear(dim, vocab_size)
        self.patch_unembedder = ContextualPatchUnembedder(
            output_channels=3,
            embed_dim=dim,
            patch_size=2,
            mlp_depth=mlp_depth
        )
        
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
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
        block_masks: Tuple[object, object],  # Receives (Local, Global)
        scale: float = 1.0
    ) -> Tuple[torch.Tensor, float]:
        """
        Pure transformer pass. No span logic.
        
        Returns:
            z_out: [B, L_total, D] - transformed features
            aux_loss: scalar
        """
        mask_local, mask_global = block_masks
        x = z
        total_aux = 0.0
        
        for i, layer in enumerate(self.layers):
            block_mask = mask_global if layer.is_global else mask_local
            x, aux = layer(x, topo_embeds, k_caches[i], v_caches[i], 
                          slot_mapping, block_mask, scale=scale)
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


class coolerLDTformerZC(nn.Module):
    def __init__(self, dim=256, depth=8, num_heads=8, topo_dim=4, mlp_depth=1, vocab_size=65536, global_layer_interval=4, num_experts=8, num_active=3, rope_base:int = 500):
        super().__init__()
        
        # Embedding heads (used by SpanEmbedder)
        self.global_layer_interval = global_layer_interval
        self.text_embed = nn.Embedding(vocab_size, dim)
        self.patch_embedder = ContextualPatchEmbedder(
            input_channels=3,
            embed_dim=dim,
            context_size= 4,  # ← Add this back!
            stride= 2,         # ← Add this back!
            mlp_depth=mlp_depth
        )
        
        # Transformer trunk
        self.layers = nn.ModuleList([
            LDTformerBlockZC(dim, num_heads, topo_dim, is_global=((i+1)%global_layer_interval==0), num_experts=num_experts, num_active=num_active) for i in range(depth) 
        ])
        
        # Output heads (used by SpanUnembedder)
        self.text_head = nn.Linear(dim, vocab_size)
        self.patch_unembedder = ContextualPatchUnembedder(
            output_channels=3,
            embed_dim=dim,
            patch_size=2,
            mlp_depth=mlp_depth
        )
        
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
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
        slot_mapping: torch.Tensor,
        block_masks: Tuple[object, object],
        scale: float = 1.0 # Receives (Local, Global)
    ) -> Tuple[torch.Tensor, float]:
        """
        Pure transformer pass. No span logic.
        
        Returns:
            z_out: [B, L_total, D] - transformed features
            aux_loss: scalar
        """
        mask_local, mask_global = block_masks
        x = z
        total_aux = 0.0
        
        for i, layer in enumerate(self.layers):
            block_mask = mask_global if layer.is_global else mask_local
            x, aux = layer(x, topo_embeds,
                          slot_mapping, block_mask, scale=scale)
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