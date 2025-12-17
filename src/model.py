# src/model.py - Transformer architecture
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, BlockMask
from typing import Tuple, List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import math

try:
    from nvllm_flex_attention import update_kv_cache
except ImportError:
    update_kv_cache = None  # Not needed for ZC mode

### MODEL PROVIDES PAGETABLE, CORE DATA STRUCTURES ALLOWING INFERENCE

class PageTable:
    """
    Manages the mapping between Logical Blocks (Sequence) and Physical Blocks (Heap).
    Implements the 'convert_logical_block_mask' primitive for Paged FlexAttention.
    """
    def __init__(self, 
                 num_blocks: int, 
                 block_size: int, 
                 max_batch_size: int, 
                 max_logical_blocks: int,
                 device='cuda'):
        
        self.block_size = block_size
        self.device = device
        
        # [logical_batch_idx, logical_block_idx] -> physical_page_idx
        self.page_table = torch.full(
            (max_batch_size, max_logical_blocks), -1, 
            dtype=torch.int32, device=device
        )
        
        # [logical_batch_idx, physical_page_idx] -> logical_page_idx
        # Used by the mask_mod to reverse-lookup logical positions
        self.physical_to_logical = torch.full(
            (max_batch_size, num_blocks), -1,
            dtype=torch.int32, device=device
        )

    def convert_logical_block_mask(
        self,
        logical_mask: BlockMask,
        batch_idx: torch.Tensor 
    ) -> BlockMask:
        """
        Teleports a BlockMask from Logical Space to Physical Space.
        
        Args:
            logical_mask: Mask computed on logical sequences (e.g. 0..L).
            batch_idx: [B] Tensor mapping Kernel Batch Index -> Logical Request ID.
                       (Used to look up the specific Page Table row).
        
        Returns:
            A new BlockMask instance valid for the Paged KV Cache.
        """
        
        # 1. Identify Active Page Tables
        # Select the rows corresponding to the active requests in this kernel batch
        # shape: [B, Max_Logical_Blocks]
        active_page_table = self.page_table[batch_idx.long()]
        
        # 2. Extract Logical Indices (Sparse)
        # These are indices into the Logical Block sequence (0, 1, 2...)
        # kv_indices shape: [B, H, Q_Blocks, K_Blocks_Sparse]
        # (Note: FlexAttention usually broadcasts B if masks are identical, 
        #  but for PagedAttention we assume uniqueness per batch item or handle broadcast).
        
        # We assume logical_mask batch dim matches active_page_table batch dim (B).
        # If logical_mask is shared (B=1) but we have multiple requests, expand it.
        B_kernel = batch_idx.size(0)
        
        kv_indices = logical_mask.kv_indices
        full_kv_indices = logical_mask.full_kv_indices
        kv_num_blocks = logical_mask.kv_num_blocks
        full_kv_num_blocks = logical_mask.full_kv_num_blocks

        if kv_indices.size(0) == 1 and B_kernel > 1:
            kv_indices = kv_indices.expand(B_kernel, -1, -1, -1)
            full_kv_indices = full_kv_indices.expand(B_kernel, -1, -1, -1)
            kv_num_blocks = kv_num_blocks.expand(B_kernel, -1, -1)
            full_kv_num_blocks = full_kv_num_blocks.expand(B_kernel, -1, -1)

        # 3. Map to Physical Indices
        # We need to gather the physical block IDs using the logical block IDs.
        # active_page_table: [B, Max_Log]
        # indices: [B, H, Q, K_Sparse]
        
        # Reshape page_table for broadcasting against H, Q dimensions
        # [B, 1, 1, Max_Log]
        pt_view = active_page_table.view(B_kernel, 1, 1, -1)
        
        # Gather Physical Indices for Partial Blocks
        phys_kv_indices = torch.gather(
            pt_view.expand(-1, kv_indices.size(1), kv_indices.size(2), -1),
            3, 
            kv_indices.long()
        )
        
        # Gather Physical Indices for Full Blocks
        phys_full_kv_indices = torch.gather(
            pt_view.expand(-1, full_kv_indices.size(1), full_kv_indices.size(2), -1),
            3, 
            full_kv_indices.long()
        )

        # 4. Wrap the Mask Mod
        # The kernel calls mask_mod(b, h, q, k_phys).
        # We must translate k_phys -> k_log to check the original geometry condition.
        
        original_mod = logical_mask.mask_mod
        
        def physical_mask_mod(b, h, q_idx, k_phys_idx):
            # 1. Get Logical Request ID
            # b is the kernel batch index (0..B-1)
            logical_req_id = batch_idx[b]
            
            # 2. Get Logical Block ID
            phys_block = k_phys_idx // self.block_size
            offset = k_phys_idx % self.block_size
            
            # self.physical_to_logical: [Max_Reqs, Max_Phys]
            log_block = self.physical_to_logical[logical_req_id, phys_block]
            
            # 3. Reconstruct Logical K Index
            log_k_idx = log_block * self.block_size + offset
            
            # 4. Delegate to Original Logic
            return original_mod(b, h, q_idx, log_k_idx)

        # 5. Construct New BlockMask
        # We clone the object (shallow copy) and overwrite the tensor attributes.
        physical_mask = copy.copy(logical_mask)
        
        physical_mask.kv_indices = phys_kv_indices.int()
        physical_mask.full_kv_indices = phys_full_kv_indices.int()
        physical_mask.kv_num_blocks = kv_num_blocks.int()
        physical_mask.full_kv_num_blocks = full_kv_num_blocks.int()
        physical_mask.mask_mod = physical_mask_mod
        
        # We retain BLOCK_SIZE and other metadata from logical_mask
        
        return physical_mask
        
    def convert_flattened_block_mask(
        self,
        logical_mask: BlockMask,
        flat_page_table: torch.Tensor,     # [Total_Logical_Blocks] -> Phys_Block
        inverse_page_table: torch.Tensor   # [Capacity_Blocks] -> Log_Block
    ) -> BlockMask:
        """
        Teleports a BlockMask from Logical Space to Physical Space for B=1 (Flattened) execution.
        """
        # 1. Map Logical Indices to Physical Indices (Sparse)
        # logical_mask.kv_indices has shape [1, H, Q_blocks, K_sparse_blocks]
        phys_kv_indices = flat_page_table[logical_mask.kv_indices.long()]
        
        # 2. Map Full Blocks (Dense)
        phys_full_kv_indices = flat_page_table[logical_mask.full_kv_indices.long()]

        # 3. Wrap Mask Mod
        original_mod = logical_mask.mask_mod
        
        def physical_mask_mod(b, h, q_idx, k_phys_idx):
            # Map Physical Heap Index -> Logical Sequence Index
            phys_block = k_phys_idx // self.block_size
            offset = k_phys_idx % self.block_size
            
            # Lookup
            log_block = inverse_page_table[phys_block]
            
            # Reconstruct Logical Index
            log_k_idx = log_block * self.block_size + offset
            
            return original_mod(b, h, q_idx, log_k_idx)

        # 4. Construct New BlockMask
        physical_mask = copy.copy(logical_mask)
        physical_mask.kv_indices = phys_kv_indices.int()
        physical_mask.full_kv_indices = phys_full_kv_indices.int()
        physical_mask.mask_mod = physical_mask_mod
        
        return physical_mask

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
        # FIX 1: Prevent float promotion by casting result back to input dtype
        denom = top_k_scores.sum(dim=-1, keepdim=True) + 1e-6
        router_weights = top_k_scores / denom
        router_weights = router_weights.to(x.dtype)
        
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
            # FIX 2: Use .to(x.dtype) instead of .float()
            mask_cast = match_mask.to(x.dtype)
            
            # Aggregate weight for expert 'e' per token
            # [N] -> [Num_Selected]
            active_weights = (weights_flat * mask_cast).sum(dim=-1)[active_indices]
            
            # Gather inputs: [Num_Selected, D]
            active_x = x_flat[active_indices].to(x.dtype)
            
            # D. Compute Expert
            expert_out = self.experts[e](active_x)
            
            # E. Scale & Scatter Add
            # ensure active_weights is [Num_Selected, 1] for broadcasting
            weighted_out = expert_out * active_weights.unsqueeze(-1).to(x.dtype)
            
            # weighted_out is now strictly x.dtype (e.g. bf16)
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

    def _pad_and_patch(self, x):
        # 1. Standard Reflection Pad (Context Window)
        x_pad = F.pad(x, (self.padding,)*4, mode='reflect')
        
        # 2. Dynamic Safety Pad (Stride Alignment)
        # Unfold drops the last patch if dimensions don't fit perfectly.
        # We calculate the remaining length after context, and if it's not div by stride, we pad.
        _, h, w = x_pad.shape
        
        # Effective area available for striding after the first context window
        h_rem = (h - self.context_size) % self.stride
        w_rem = (w - self.context_size) % self.stride
        
        pad_bottom = (self.stride - h_rem) % self.stride
        pad_right = (self.stride - w_rem) % self.stride
        
        if pad_bottom > 0 or pad_right > 0:
            x_pad = F.pad(x_pad, (0, pad_right, 0, pad_bottom), mode='replicate')
            
        patches = x_pad.unfold(1, self.context_size, self.stride).unfold(2, self.context_size, self.stride)
        return patches

    def forward(self, x, logsnr_map):
        # x: [C, H, W], logsnr_map: [1, H, W]
        patches_img = self._pad_and_patch(x)
        
        GH, GW = patches_img.shape[1], patches_img.shape[2]
        patches_img = patches_img.permute(1, 2, 0, 3, 4).reshape(GH * GW, -1)
        
        # Apply EXACT SAME padding/striding logic to logsnr
        patches_logsnr = self._pad_and_patch(logsnr_map)
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
class ContextBlock:
    """
    Canonical atomic unit of the dataset.
    Holds raw data and its topological metadata.
 
    shape_meta: For latents, this is (H, W) - the spatial dimensions of content.
                For text, this is (seq_len,) - the token count.
                Must broadcast correctly with content for logsnr map operations.
    """
    content: Union[torch.Tensor, str] # [C, H, W] for latent or [L] for text tokens
    type: str = 'latent'
    causal: bool = True
    # Metadata
    shape_meta: Tuple[int, ...] = field(default_factory=tuple)
    logsnr: Optional[torch.Tensor] = None # [1, H, W] for latents
    group_id: int = 0
    id: str = ""
    source: str = "unknown"
 
    def __post_init__(self):
        # Derive shape_meta from content if not explicitly set
        # This is a fallback - prefer explicit setting in iterators
        if not self.shape_meta and isinstance(self.content, torch.Tensor):
            if self.type == 'latent':
                # shape_meta = (H, W) for broadcasting with (C, H, W) content
                #h, w = self.content.shape[-2:]
                #self.shape_meta = (h, w)
                print(f"type inference of pre-pooling image shape from an unknown tensor of unknown source is not possible.")
                raise TypeError("we are crashing this run... with no survivors.")
            elif self.type == 'text':
                # shape_meta = (seq_len,) for text tokens
                self.shape_meta = (self.content.shape[0],)

@dataclass
class Span:
    type: str  # 'text' | 'latent'
    start_idx: int
    end_idx: int
    shape: Tuple[int, ...] 
    causal: bool    
    doc_id: int 
    # NEW: Store original unpadded dimensions
    original_shape: Optional[Tuple[int, ...]] = None


class SpanEmbedder:
    def __init__(self, text_embedder, patch_embedder):
        self.text_emb = text_embedder
        self.patch_emb = patch_embedder
        
    def embed(self, context_blocks: List[ContextBlock]) -> Tuple[torch.Tensor, List[Span], List[int]]:
        all_embeds = []
        span_objects = []
        cursor = 0
        
        hash_spans = []

        for block in context_blocks:
            original_shape = None
            
            if block.type == 'text':
                tokens = block.content
                if isinstance(tokens, str): raise ValueError("SpanEmbedder expects tokenized text tensors, not strings.")
                
                emb = self.text_emb(tokens)
                span_len = tokens.shape[0]
                actual_shape = (span_len,)
                hash_spans.append({'type': 'text', 'shape': actual_shape, 'data': tokens.cpu().tolist()})
                
            elif block.type == 'latent':
                img = block.content
                logsnr = block.logsnr
                
                # CAPTURE ORIGINAL SHAPE
                original_shape = img.shape[-2:]
                
                # Direct 3D Tensor processing
                emb, grid_shape = self.patch_emb(img, logsnr)
                span_len = emb.shape[0]
                actual_shape = grid_shape
                hash_spans.append({'type': 'latent', 'shape': grid_shape, 'id': block.id})
            
            all_embeds.append(emb)
            span_objects.append(Span(
                type=block.type,
                start_idx=cursor,
                end_idx=cursor + span_len,
                shape=actual_shape,
                causal=block.causal,
                doc_id=block.group_id,
                original_shape=original_shape # PERSIST IT
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
            
            # Latent Head
            if span.type == 'latent':
                # Reconstruct full padded grid
                reconstruction = self.patch_unembed(z_span, span.shape)
                
                # CROP LOGIC: Slice back to original resolution
                if span.original_shape is not None:
                    orig_h, orig_w = span.original_shape
                    # reconstruction is [C+1, H_pad, W_pad]
                    reconstruction = reconstruction[:, :orig_h, :orig_w]
                
                spandict['image_vpreds'] = reconstruction[:-1]
                spandict['image_logsnrs'] = reconstruction[-1:]
            
            outputs.append(spandict)
        return outputs

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
    def __init__(self, dim: int, num_heads: int, topo_dim: int, mlp_ratio: float = 4.0, is_global=False, num_experts=8, num_active=3, jitter_noise: float = 0.1, rope_base: float = 500.0):
        super().__init__()
        self.is_global = is_global
        self.rope_base = rope_base*(100**is_global)
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.attn = LDTformerAttentionKVC(dim, num_heads, topo_dim, rope_base=self.rope_base)        
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)
        
        # Using SigmoidMoE for FFN
        hidden_dim = int(dim * mlp_ratio)
        self.moe = SigmoidMoE(dim, hidden_dim, num_experts=num_experts, num_active=num_active, jitter_noise=jitter_noise)
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
    def __init__(self, dim: int, num_heads: int, topo_dim: int, mlp_ratio: float = 4.0, is_global=False, num_experts=8, num_active=3, jitter_noise: float = 0.1, rope_base: float = 500.0):
        super().__init__()
        self.is_global = is_global
        self.rope_base = rope_base*(100**is_global)
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.attn = LDTformerAttentionZC(dim, num_heads, topo_dim, rope_base=self.rope_base)
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)
        
        # Using SigmoidMoE for FFN
        hidden_dim = int(dim * mlp_ratio)
        self.moe = SigmoidMoE(dim, hidden_dim, num_experts=num_experts, num_active=num_active, jitter_noise=jitter_noise)
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
        self.window_size = window_size
        self.text_embed = nn.Embedding(vocab_size, dim)
        self.patch_embedder = ContextualPatchEmbedder(
            input_channels=3,
            embed_dim=dim,
            context_size=context_size,
            stride=stride,
            fourier_dim=fourier_dim,
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
            patch_size=stride,
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
    def __init__(self, dim=256, depth=8, num_heads=8, topo_dim=4, mlp_depth=1, vocab_size=65536, global_layer_interval=4, num_experts=8, num_active=3, rope_base:int = 500, mlp_ratio: float = 4.0, jitter_noise: float = 0.1, context_size: int = 4, stride: int = 2, fourier_dim: int = 16, window_size: float = 10.0):
        super().__init__()
        
        # Embedding heads (used by SpanEmbedder)
        self.global_layer_interval = global_layer_interval
        self.window_size = window_size
        self.text_embed = nn.Embedding(vocab_size, dim)
        self.patch_embedder = ContextualPatchEmbedder(
            input_channels=3,
            embed_dim=dim,
            context_size=context_size,
            stride=stride,
            fourier_dim=fourier_dim,
            mlp_depth=mlp_depth
        )
        
        # Transformer trunk
        self.layers = nn.ModuleList([
            LDTformerBlockZC(dim, num_heads, topo_dim, mlp_ratio=mlp_ratio, is_global=((i+1)%global_layer_interval==0), num_experts=num_experts, num_active=num_active, jitter_noise=jitter_noise, rope_base=rope_base) for i in range(depth) 
        ])
        
        # Output heads (used by SpanUnembedder)
        self.text_head = nn.Linear(dim, vocab_size)
        self.patch_unembedder = ContextualPatchUnembedder(
            output_channels=3,
            embed_dim=dim,
            patch_size=stride,
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

# coupled concerns for embeddings stuff

import torch
import xxhash
import numpy as np
import math
from typing import List, Dict, Tuple, Any, Callable, Optional

# =========================================================
# 1. CONTENT IDENTITY (Hashing Policy)
# =========================================================

def generate_content_hash_stream(spans: List[Any]) -> List[int]:
    """
    Transforms a list of Spans (or dicts) into a linear stream of Atomic Content IDs.
    These IDs are used by the BlockManager to detect identical content 
    (Prefix Caching).
    """
    stream = []
    
    for span in spans:
        # Support both Dataclass and Dict interfaces
        if isinstance(span, dict):
            span_type = span.get('type', 'latent')
            shape = span['shape']
            span_id = span.get('id', 0)
            data = span.get('data', None)
        else:
            span_type = getattr(span, 'type', 'latent')
            shape = getattr(span, 'shape', ())
            span_id = getattr(span, 'id', 0) # Assumes Span has 'id' if needed, else 0
            data = getattr(span, 'data', None)

        num_tokens = math.prod(shape)
        
        if span_type == 'text':
            # Text Identity = The Token ID itself
            if data is None:
                # If we don't have data (e.g. inference placeholder), use 0 or handle error
                # For hashing purposes, we need content.
                raise ValueError("Text spans must provide 'data' (token IDs) for hashing.")
            
            # Ensure data is flat list
            if hasattr(data, 'tolist'): data = data.tolist()
            
            if len(data) != num_tokens:
                 # Adjust shape or warn? Strict for now.
                 # Text shapes are often (L,), so prod is L.
                 pass 
                 
            stream.extend([int(t) for t in data])
            
        elif span_type == 'latent':
            # Latent Identity = Hash(Unique_Span_ID, Relative_Index)
            # Seed from the Span ID
            if isinstance(span_id, str):
                seed = xxhash.xxh64(span_id).intdigest()
            else:
                seed = int(span_id)
            
            # Generate deterministic stream: Hash(Seed + Index)
            # Using xxhash for speed and distribution quality
            base_hasher = xxhash.xxh64(seed=seed)
            for i in range(num_tokens):
                base_hasher.reset()
                # Determine "Relative Index" uniqueness
                base_hasher.update(i.to_bytes(8, 'little'))
                stream.append(base_hasher.intdigest())
                
    return stream

# =========================================================
# 2. GEOMETRY (Topology Policy)
# =========================================================

def render_topology_embeddings(
    spans: List[Span],
    max_dims: int,
    device: torch.device,
    highway_offset: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Renders Global Topology.
    Fix: Text spans get (0,0) spatial coords. Image spans get Grid coords.
    Both share the global Highway timeline.
    """
    highway_idx = []
    manifold_coords = []
    doc_ids = []
    
    current_highway = highway_offset
    spatial_dim_capacity = max_dims - 1
    
    for i, span in enumerate(spans):
        # Flattened length
        if span.type == 'text':
            num_tokens = span.shape[0]
        else:
            num_tokens = math.prod(span.shape) # e.g. H*W
            
        # 1. Highway (Shared Global Time)
        h_range = torch.arange(current_highway, current_highway + num_tokens, device=device)
        highway_idx.append(h_range)
        current_highway += num_tokens
        
        # 2. Manifold (Spatial)
        if span.type == 'text':
            # Text exists at the "singularity" (0,0) of the spatial manifold
            coords = torch.zeros((num_tokens, spatial_dim_capacity), device=device)
        else:
            # Latents exist on a grid
            dims = [torch.arange(d, device=device) for d in span.shape]
            mesh = torch.meshgrid(*dims, indexing='ij')
            coords = torch.stack([m.flatten() for m in mesh], dim=-1)
            
            # Pad spatial dims if needed (e.g., 2D grid in 3D manifold)
            curr_dim = coords.shape[-1]
            if curr_dim < spatial_dim_capacity:
                padding = torch.zeros((num_tokens, spatial_dim_capacity - curr_dim), device=device)
                coords = torch.cat([coords, padding], dim=-1)
                
        manifold_coords.append(coords)
        doc_ids.append(torch.full((num_tokens,), span.doc_id, device=device, dtype=torch.int32))

    # Stack
    flat_highway = torch.cat(highway_idx).unsqueeze(-1).float()
    flat_manifold = torch.cat(manifold_coords).float()
    topo_embeds = torch.cat([flat_highway, flat_manifold], dim=-1)
    flat_doc_ids = torch.cat(doc_ids)
    
    return topo_embeds, flat_doc_ids

# =========================================================
# 3. CONNECTIVITY (Masking Policy)
# =========================================================

def get_block_causal_mod(doc_ids: torch.Tensor) -> Callable:
    """
    Returns a pure Python closure defining the Block-Causal connectivity rules.
    """
    def block_causal_mod(b, h, q_idx, kv_idx):
        q_doc = doc_ids[q_idx]
        k_doc = doc_ids[kv_idx]
        return (q_doc == k_doc) | (q_doc > k_doc)
        
    return block_causal_mod

def get_sliding_window_mod(
    topo_embeds: torch.Tensor, 
    window_size: float,
    doc_ids: Optional[torch.Tensor] = None
) -> Callable:
    """
    Returns a closure that enforces a Spatial Sliding Window in R^n.
    Optionally combines with Block-Causal logic if doc_ids are provided.
    
    Args:
        topo_embeds: [Total_L, 1 + N_Dims] (Col 0 is Highway, Cols 1..N are Space)
        window_size: Maximum Euclidean distance for connection.
        doc_ids: (Optional) If provided, enforces block-causal rules AND spatial window.
    """
    
    def swa_mod(b, h, q_idx, kv_idx):
        # 1. Spatial Rule (R^n)
        # We slice off the Highway dimension (Col 0) to get purely spatial coords
        q_pos = topo_embeds[q_idx, 1:]
        k_pos = topo_embeds[kv_idx, 1:]
        
        # Calculate Rn Distance
        dist = compute_rn_distance(q_pos, k_pos)
        spatial_mask = dist < window_size
        
        # 2. Block-Causal Rule (Optional Composition)
        if doc_ids is not None:
            q_doc = doc_ids[q_idx]
            k_doc = doc_ids[kv_idx]
            causal_mask = (q_doc == k_doc) | (q_doc > k_doc)
            return spatial_mask & causal_mask
            
        return spatial_mask

    return swa_mod

def build_dual_masks(
    spans: List[Span],
    topo_active: torch.Tensor,
    topo_heap: torch.Tensor,
    page_table: Optional[PageTable] = None,
    flat_page_table: Optional[torch.Tensor] = None,
    inverse_page_table: Optional[torch.Tensor] = None,
    window_size: float = 10.0,
    return_mask_closures: bool = False
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
    doc_ids_active = []         # <--- The Batch Flattener Is At It Again
    span_ids_active = []        # <--- Returning Once Again
    causal_modes_active = []    # <--- Returning Once Again: Track causality per token
    for i, span in enumerate(spans):
        span_len = span.end_idx - span.start_idx
        doc_ids_active.extend([span.doc_id] * span_len)
        span_ids_active.extend([i] * span_len) # Monotonic Span ID
        causal_modes_active.extend([span.causal] * span_len)
        
    doc_ids_active_t = torch.tensor(doc_ids_active, dtype=torch.long, device=device)
    span_ids_active_t = torch.tensor(span_ids_active, dtype=torch.long, device=device)
    causal_modes_active_t = torch.tensor(causal_modes_active, dtype=torch.bool, device=device)

    # 2. Build doc_ids for HEAP (EFFICIENT VERSION)
    # Initialize heap as -1
    L_heap = topo_heap.shape[0]
    doc_ids_heap_t = torch.full((L_heap,), -1, dtype=torch.long, device=device)
    span_ids_heap_t = torch.full((L_heap,), -1, dtype=torch.long, device=device) # <--- NEW
    block_size = page_table.block_size
    cursor = 0
    for i, span in enumerate(spans):
        span_len = span.end_idx - span.start_idx
        
        # Trivial Case (Training/ZC)
        if flat_page_table is None:
             doc_ids_heap_t[cursor : cursor+span_len] = span.doc_id
             span_ids_heap_t[cursor : cursor+span_len] = i
        else:
            # Inference Case (Paged) - Iterate Logical Blocks
            start_block = cursor // block_size
            end_block = (cursor + span_len - 1) // block_size + 1
            
            for log_block_idx in range(start_block, end_block):
                if log_block_idx >= len(flat_page_table): break
                
                phys_block = flat_page_table[log_block_idx].item()
                
                # Intersection of Span and Block
                block_start_global = log_block_idx * block_size
                block_end_global = (log_block_idx + 1) * block_size
                
                start_in_span = max(0, block_start_global - cursor)
                end_in_span = min(span_len, block_end_global - cursor)
                
                # Global offsets
                global_start = cursor + start_in_span
                global_end = cursor + end_in_span
                
                # Physical offsets
                offset_start = global_start % block_size
                offset_end = offset_start + (end_in_span - start_in_span)
                
                phys_start = phys_block * block_size + offset_start
                phys_end = phys_block * block_size + offset_end
                
                doc_ids_heap_t[phys_start : phys_end] = span.doc_id
                span_ids_heap_t[phys_start : phys_end] = i
                
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
    # --- THE CONNECTIVITY LOGIC FIX ---
    def base_connectivity(q_idx, kv_idx):
        # 1. Document Separation
        q_doc = doc_ids_active_t[q_idx]
        k_doc = doc_ids_heap_t[kv_idx]
        same_doc = (q_doc == k_doc)
        
        # 2. Span Identification
        q_span = span_ids_active_t[q_idx]
        k_span = span_ids_heap_t[kv_idx]
    
        # 3. Block Causal Logic (Global Hierarchy)
        block_condition = (q_span > k_span)
        same_span = (q_span == k_span)
        
        # 4. Intra-Span Logic (Local Visibility)
        # Only evaluated if q_span == k_span.
        is_ar = causal_modes_active_t[q_idx]
        # If AR: Enforce Time. If BiDir: Allow All.
        internal_condition = (~is_ar) | (highway_active[q_idx] >= highway_heap[kv_idx])
        
        # 5. Composition
        # Visible if: (Same Doc) AND ( (Strictly Past Span) OR (Same Span AND Internal Condition) )
        valid_connection = block_condition | (same_span & internal_condition)
        
        return same_doc & valid_connection

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

    debug_dict={'mask_mod_local':mask_mod_local, 'mask_mod_global':mask_mod_global}
    if return_mask_closures:
        return local_mask, global_mask, debug_dict
    else: 
        return local_mask, global_mask

def materialize_mask_for_analysis(spans: List[Span], topo_active: torch.Tensor) -> torch.Tensor:
    # Convenience wrapper for 1-to-1 analysis
    # Reuse build_dual_masks logic logic internally
    _, _, debug = build_dual_masks_debug(spans, topo_active, topo_active)
    mod = debug['mask_mod_global']
    L = topo_active.shape[0]
    dev = topo_active.device
    q = torch.arange(L, device=dev).unsqueeze(1).expand(L, L)
    k = torch.arange(L, device=dev).unsqueeze(0).expand(L, L)
    return mod(q, k)