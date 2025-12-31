# Sparse AE Integration: Complete Analysis

## 1. What model.py Provides

### Core Data Structures

**ContextBlock** (lines 591-622): Atomic unit holding content + metadata
- `content`: Tensor [C,H,W] for latent or [L] for text
- `type`: 'latent' | 'text'
- `causal`: Per-block causality flag
- `logsnr`: [1,H,W] spatial noise field
- `group_id`: Document isolation ID
- `shape_meta`: Original dimensions before padding

**Span** (lines 625-633): Tracks embedded token ranges
- `type`, `start_idx`, `end_idx`, `shape`
- `causal`: Copied from ContextBlock
- `doc_id`: For cross-document isolation
- `original_shape`: Pre-padding dims for cropping on decode

**SpanEmbedder** (lines 636-686): Wraps text_emb + patch_emb
- `embed()`: Takes List[ContextBlock] -> (z_flat, spans, hashes)
- Calls `patch_emb(img, logsnr)` -> (embeddings, grid_shape)
- Creates Span objects with proper doc_id from block.group_id

**SpanUnembedder** (lines 688-717): Wraps text_head + patch_unembed
- `decode()`: Takes z, spans -> List[Dict] with predictions
- Crops to original_shape if stored

### Topology System

**render_topology_embeddings** (lines 1208-1270):
- Input: List[Span], max_dims, device
- Output: (topo_embeds [L, topo_dim], doc_ids [L])
- Column 0: "highway" - monotonic sequential position
- Columns 1+: Spatial coords (grid for images, zeros for text)
- Text tokens exist at spatial origin (0,0,...)
- Images get actual grid coordinates

### Mask Construction

**build_dual_masks** (lines 1323-1473):
- Input: spans, topo_active, topo_heap, page_table, window_size
- Output: (local_mask, global_mask) as BlockMask objects

**Key implementation details:**
1. Builds per-token tensors from span metadata:
   - `doc_ids_active_t`: Document ID per token
   - `span_ids_active_t`: Span index per token
   - `causal_modes_active_t`: Causality flag per token

2. Decomposes topology:
   - `highway_active/heap`: Column 0 (temporal position)
   - `spatial_active/heap`: Columns 1+ (spatial coords)

3. Defines `base_connectivity(q_idx, kv_idx)`:
   - Document isolation: `same_doc = (q_doc == k_doc)`
   - Block causal: `q_span > k_span` allows connection
   - Intra-span: Uses causal flag + highway ordering
   - Returns: `same_doc & valid_connection`

4. Defines mask_mod functions:
   - `mask_mod_local`: base_connectivity AND spatial distance < window
   - `mask_mod_global`: base_connectivity only (infinite window)

5. Creates BlockMasks via `create_block_mask(mask_mod, B=None, H=None, Q_LEN, KV_LEN)`

**Critical insight**: Mask closures capture TENSORS and index into them. The q_idx/kv_idx are iteration variables from flex_attention. This is what makes torch.compile work - tensors are fixed, only indices vary.

### Attention Layers

**LDTformerAttentionZC** (lines 823-874):
- Standard QKV projection
- RnRoPE with topology coordinates
- `flex_attention(q, k, v, block_mask=block_mask)`

**LDTformerBlockZC** (lines 912-946):
- Pre-norm + attention + gated residual
- Pre-norm + SigmoidMoE FFN + residual
- `gate_proj` for sigmoid gating on attention output

**coolerLDTformerZC** (lines 1045-1135):
- Contains `text_embed`, `patch_embedder`, `layers`, `text_head`, `patch_unembedder`
- Forward: iterates layers with alternating local/global masks
- Returns (z_out, aux_loss)

### Embedder/Unembedder Interfaces

**ContextualPatchEmbedder** (lines 497-549):
- `.stride` attribute (required)
- `forward(x: [C,H,W], logsnr: [1,H,W]) -> (z: [L,D], (GH,GW))`
- Uses FourierFeatures for logsnr encoding
- Reflection padding + unfold for context windows

**ContextualPatchUnembedder** (lines 552-585):
- `forward(z: [L,D], shape: (GH,GW)) -> [C+1, H, W]`
- Output includes RGB + logsnr channel
- Uses FourierScaleDecoder for logsnr prediction

---

## 2. What Sparse AE Needs

The sparse AE processes images for compression/reconstruction. When used for diffusion training:
- May process 32,000 images in a batch
- Each image is a separate "document" for attention isolation
- Needs efficient sliding window attention (not O(N^2) across all patches)
- Must integrate with SpanEmbedder/SpanUnembedder interfaces

### Required Attention Features

1. **Document Isolation**: Each image = separate doc_id
2. **Sliding Window**: Local spatial attention within each image
3. **Optional BigBird**: Global register tokens for long-range dependencies
4. **Batch Flattening**: All images flattened into single sequence with proper masking
5. **torch.compile Compatibility**: Static flex_attention kernels

### Interface Requirements

**SparseAEPatchEmbedder** must provide:
- `.stride` attribute matching patch_size
- `forward(x: [C,H,W], logsnr: [1,H,W]) -> (z: [L,D], (GH,GW))`

**SparseAEPatchUnembedder** must provide:
- `forward(z: [L,D], shape: (GH,GW)) -> [C+1, H, W]`

---

## 3. Minimal Additions to model.py

### 3.1 Simplified Image-Only Mask Constructor

Add a function that creates masks for image-only batches:

```python
def build_image_batch_masks(
    n_images: int,
    patches_per_image: int,
    grid_shape: Tuple[int, int],
    window_size: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> Tuple[BlockMask, BlockMask]:
    """
    Build attention masks for a batch of images (no text).

    Each image is a separate document. All images flattened into
    a single sequence for efficient batch processing.

    Args:
        n_images: Number of images in batch
        patches_per_image: L = GH * GW per image
        grid_shape: (GH, GW) patch grid dimensions
        window_size: Spatial window for local attention
        device, dtype: Target device and dtype

    Returns:
        (local_mask, global_mask) - BlockMasks for flex_attention
    """
    total_len = n_images * patches_per_image
    GH, GW = grid_shape

    # Build per-token metadata tensors
    doc_ids = torch.arange(n_images, device=device).repeat_interleave(patches_per_image)

    # Spatial coords: tile the same grid for each image
    gy = torch.arange(GH, device=device, dtype=dtype)
    gx = torch.arange(GW, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
    spatial = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)  # [L, 2]
    spatial_all = spatial.repeat(n_images, 1)  # [total_len, 2]

    win_sq = torch.tensor(window_size * window_size, device=device, dtype=dtype)

    def base_connectivity(q_idx, kv_idx):
        # Same document only (images don't see each other)
        return doc_ids[q_idx] == doc_ids[kv_idx]

    def mask_mod_local(b, h, q_idx, kv_idx):
        base = base_connectivity(q_idx, kv_idx)
        dy = spatial_all[q_idx, 0] - spatial_all[kv_idx, 0]
        dx = spatial_all[q_idx, 1] - spatial_all[kv_idx, 1]
        dist_sq = dy * dy + dx * dx
        return base & (dist_sq < win_sq)

    def mask_mod_global(b, h, q_idx, kv_idx):
        return base_connectivity(q_idx, kv_idx)

    from torch.nn.attention.flex_attention import create_block_mask
    local_mask = create_block_mask(mask_mod_local, B=None, H=None, Q_LEN=total_len, KV_LEN=total_len)
    global_mask = create_block_mask(mask_mod_global, B=None, H=None, Q_LEN=total_len, KV_LEN=total_len)

    return local_mask, global_mask
```

### 3.2 BigBird Mask Extension (Optional)

For register tokens, extend the mask builder:

```python
def build_image_batch_masks_bigbird(
    n_images: int,
    patches_per_image: int,
    grid_shape: Tuple[int, int],
    window_size: float,
    n_registers: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> Tuple[BlockMask, BlockMask]:
    """
    Like build_image_batch_masks, but with BigBird-style register tokens.

    Register tokens are prepended to each image's patch sequence.
    They attend globally within their document.
    """
    # ... similar structure, with register token handling
```

### 3.3 GQA Support

Current `LDTformerAttentionZC` uses standard multi-head attention. For GQA:

```python
class GQAAttention(nn.Module):
    """Grouped Query Attention - fewer KV heads than query heads."""
    def __init__(self, dim: int, n_query_heads: int, n_kv_heads: int, topo_dim: int, rope_base: float = 500.0):
        super().__init__()
        self.n_query_heads = n_query_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_query_heads

        self.q_proj = nn.Linear(dim, n_query_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rope = RnRoPE(self.head_dim, topo_dim, rope_base=rope_base)

    def forward(self, x, topo, block_mask, scale=1.0):
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.n_query_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k, topo, scale=scale)

        # Expand KV heads to match query heads
        heads_per_kv = self.n_query_heads // self.n_kv_heads
        k = k.repeat_interleave(heads_per_kv, dim=1)
        v = v.repeat_interleave(heads_per_kv, dim=1)

        out = flex_attention(q, k, v, block_mask=block_mask)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)
```

---

## 4. sparse_ae_model.py Structure

The new file should IMPORT from model.py, not reimplement:

```python
# sparse_ae_model.py
import torch
import torch.nn as nn
from src.model import (
    # Data structures
    FourierFeatures,
    FourierScaleDecoder,
    MLPResBlock,
    # Mask construction
    build_image_batch_masks,
    # Attention (if we add GQA to model.py)
    GQAAttention,  # or use existing LDTformerAttentionZC
    # Initialization
    init_linear,
    init_layer_norm
)

class SparseAETransformerBlock(nn.Module):
    """Transformer block using model.py attention with sparse AE specifics."""
    def __init__(self, dim, n_heads, topo_dim, ...):
        # Use GQAAttention or LDTformerAttentionZC from model.py
        self.attn = GQAAttention(dim, n_heads, n_kv_heads, topo_dim)
        # FFN can use SwiGLU from model.py
        self.ffn = SwiGLU(dim, dim * 4)
        ...

class SparsePerDimFSQAutoencoder(nn.Module):
    """Hierarchical sparse FSQ autoencoder."""
    def __init__(self, ...):
        # Keep FSQ, sparsity, patchify logic
        # Replace custom TransformerEncoder with blocks using model.py attention
        ...

    def forward(self, images, logsnr_map):
        B, C, H, W = images.shape
        patches = self.patchify(images)  # [B, L, patch_dim]

        # Flatten batch for efficient attention
        patches_flat = patches.view(1, B * L, -1)

        # Build masks using model.py function
        local_mask, global_mask = build_image_batch_masks(
            n_images=B,
            patches_per_image=L,
            grid_shape=(H // patch_size, W // patch_size),
            window_size=self.window_size,
            device=patches.device
        )

        # Process through encoder (uses model.py attention)
        # ...

class SparseAEPatchEmbedder(nn.Module):
    """Interface wrapper matching ContextualPatchEmbedder."""
    def __init__(self, ae, embed_dim):
        self.ae = ae
        self.stride = ae.patch_size  # Required interface attribute
        self.code_proj = nn.Linear(ae.code_dim * ae.n_levels, embed_dim)

    def forward(self, x, logsnr_map):
        # x: [C, H, W], logsnr_map: [1, H, W]
        codes, _ = self.ae.encode(x.unsqueeze(0), logsnr_map.unsqueeze(0))
        z = self.code_proj(torch.cat(codes, dim=-1)).squeeze(0)
        GH, GW = x.shape[-2] // self.stride, x.shape[-1] // self.stride
        return z, (GH, GW)

class SparseAEPatchUnembedder(nn.Module):
    """Interface wrapper matching ContextualPatchUnembedder."""
    def forward(self, z, shape):
        # z: [L, D], shape: (GH, GW)
        # Decode and return [C+1, H, W]
        ...
```

---

## 5. Refactoring Candidates (Documentation Only)

### 5.1 SpanEmbedder/SpanUnembedder Naming
Current names obscure the multimodal nature. Consider:
- `MultimodalEmbedder` with explicit `embed_text()`, `embed_image()` methods
- Clearer distinction between "embedding" and "span tracking"

### 5.2 Mask Construction Complexity
`build_dual_masks` handles both training (ZC) and inference (paged) cases. Consider:
- Separate functions for each case
- Cleaner abstraction for the paging logic

### 5.3 Config Access
Legacy `.get()` patterns hide errors. Already flagged for cleanup.

### 5.4 Training Loop Structure
`train_autoembed` and `train_denoise` share significant setup. Consider:
- Single training function with phase flags
- Extracted shared setup logic

---

## 6. Implementation Order

1. **Add `build_image_batch_masks` to model.py** - minimal, self-contained
2. **Add `GQAAttention` to model.py if needed** - or verify existing attention suffices
3. **Create sparse_ae_model.py** - imports from model.py, minimal new code
4. **Update main.py imports** - point to sparse_ae_model.py
5. **Test with sparse_ae_fractal.toml** - verify compilation and training

---

## 7. What NOT to Do

- Do NOT create parallel attention implementations
- Do NOT use nn.TransformerEncoder as a "simpler" alternative
- Do NOT cache BlockMasks by sequence length with Python int captures
- Do NOT assume dense tensor stacking works (variable shapes require flattening)
- Do NOT bypass the span system for "simplicity"
