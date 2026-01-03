# Distance Metrics for FlexAttention Masks: Construction, Caching, and Consumption

## Overview

This document describes how custom distance metrics are constructed, cached, and consumed by flex_attention mask closures in the LDTformer architecture. The system enables config-driven switching between pixel-space and latent-space diffusion with zero code changes—only the geometry computation differs.

The key insight: **flex_attention mask_mod functions are closures that capture pre-computed coordinate tensors**. Distance computation happens inside the closure using tensor indexing, which is fully compatible with torch.compile and CUDA graph capture.

---

## The Mask Construction Pipeline

### 1. Pre-Compute Coordinates (Outside torch.compile)

```python
# For latent diffusion with multi-level codes:
topo_embeds, level_ids, patch_ids = render_latent_topology_embeddings(
    n_patches=H * W,
    n_levels=6,
    grid_shape=(H, W),
    device=device,
    level_scale=cfg['topology']['level_scale']
)
# topo_embeds: [n_patches * n_levels, 4] -> [highway, spatial_x, spatial_y, level]
```

### 2. Build BlockMask with Captured Coordinates

```python
def build_latent_diffusion_mask(...):
    # Extract coordinate columns as 1D tensors
    spatial_x = mesh_x.flatten().repeat(n_levels)  # [total_tokens]
    spatial_y = mesh_y.flatten().repeat(n_levels)  # [total_tokens]
    level_coords = torch.arange(n_levels).repeat_interleave(n_patches)

    # Closure captures these tensors
    def mask_mod_latent(b, h, q_idx, kv_idx):
        dx = spatial_x[q_idx] - spatial_x[kv_idx]
        dy = spatial_y[q_idx] - spatial_y[kv_idx]
        spatial_dist_sq = dx * dx + dy * dy

        dl = level_coords[q_idx] - level_coords[kv_idx]
        level_dist_sq = (level_lambda * dl) ** 2

        # Vertical tube: same spatial position = free cross-level attention
        if vertical_free:
            same_position = (spatial_dist_sq == 0.0)
            effective_dist_sq = torch.where(same_position, 0.0, spatial_dist_sq + level_dist_sq)
        else:
            effective_dist_sq = spatial_dist_sq + level_dist_sq

        return effective_dist_sq < window_sq

    return create_block_mask(mask_mod_latent, ...)
```

### 3. Cache the BlockMask

```python
_latent_mask_cache: Dict[Tuple, Optional[BlockMask]] = {}

def get_cached_latent_mask(n_patches, n_levels, grid_shape, window_size, ...):
    key = (n_patches, n_levels, grid_shape, window_size, level_lambda, vertical_free, mode, str(device))
    if key not in _latent_mask_cache:
        _latent_mask_cache[key] = build_latent_diffusion_mask(...)
    return _latent_mask_cache[key]
```

### 4. Consume in flex_attention (Inside torch.compile)

```python
# In forward pass (compiled)
out = flex_attention(q, k, v, block_mask=cached_mask)
```

---

## Distance Metrics

### Euclidean (Default)

Standard L2 distance in the manifold coordinates:

```
dist² = Σᵢ (qᵢ - kᵢ)²
```

For latent diffusion with levels:
```
dist² = (qₓ - kₓ)² + (qᵧ - kᵧ)² + (λ · (q_level - k_level))²
```

### Vertical Tube (Recommended for Latent Diffusion)

Same-position cross-level attention is always allowed:

```python
if spatial_dist_sq == 0:
    effective_dist = 0  # Always attend
else:
    effective_dist² = spatial_dist² + (λ · level_dist)²
```

This creates "tubes" through the level stack where each spatial position can attend freely to all its level variants, while cross-position attention still respects the window constraint.

### Product Geodesic (Advanced)

For graph-structured data, distances can be pre-computed as shortest-path geodesics:

```python
# Pre-compute all-pairs shortest paths
geodesic_dist = floyd_warshall(adjacency)  # [N, N]

# Embed into R^n via MDS
coords = metric_mds(geodesic_dist, n_dims=topo_dim)  # [N, topo_dim]

# Use coords as topology embeddings
# RoPE and SWA will approximate geodesic distances
```

---

## Config-Driven Mode Selection

### Pixel-Space Diffusion (Default)

```toml
[training.sparse_ae.topology]
diffusion_space = "pixel"
include_level_dim = false
```

- Tokens = patch embeddings from ContextualPatchEmbedder
- Topology = [highway, spatial_x, spatial_y] (3D)
- SWA = standard spatial windowing
- topo_dim = 3 in model config

### Latent-Space Diffusion

```toml
[training.sparse_ae.topology]
diffusion_space = "latent"
include_level_dim = true  # Auto-enabled
level_lambda = 0.5
level_scale = 1.0
vertical_attention_free = true
```

- Tokens = AE code embeddings (flattened [n_patches * n_levels, code_dim])
- Topology = [highway, spatial_x, spatial_y, level] (4D)
- SWA = level-aware windowing with vertical tubes
- topo_dim = 4 in model config (must update!)

---

## Implementation Checklist for Latent Diffusion Mode

When `diffusion_space = "latent"`:

1. **Model topo_dim**: Must be 4 (not 3) to accommodate level dimension
   ```toml
   [model]
   topo_dim = 4  # Changed from 3
   ```

2. **Input embedding**: Flatten AE codes to [B, N_patches * N_levels, code_dim]
   ```python
   codes_flat = codes_list.view(B, -1, code_dim)  # [B, HW*L, D]
   ```

3. **Topology rendering**: Use `render_latent_topology_embeddings`
   ```python
   topo, level_ids, patch_ids = render_latent_topology_embeddings(
       n_patches=H*W, n_levels=6, grid_shape=(H, W), device=device
   )
   ```

4. **Mask construction**: Use `get_cached_latent_mask`
   ```python
   mask = get_cached_latent_mask(
       n_patches=H*W, n_levels=6, grid_shape=(H, W),
       window_size=cfg['window_size'],
       level_lambda=cfg['topology']['level_lambda'],
       vertical_free=cfg['topology']['vertical_attention_free'],
       mode='local', device=device
   )
   ```

5. **Output reshaping**: Unflatten v-field to [B, N_patches, N_levels, code_dim]
   ```python
   v_field = model_output.view(B, n_patches, n_levels, code_dim)
   ```

---

## Cache Management

### Why Caching Matters

BlockMask construction via `create_block_mask` is expensive:
- Runs the mask_mod function over all (Q, KV) pairs to build block structure
- Must happen outside torch.compile (mask_mod closures capture Python state)
- For 256×256 images with 6 levels: 1024 patches × 6 = 6144 tokens → 37M pairs

### Cache Keys

The cache key must include all parameters that affect mask structure:

```python
key = (n_patches, n_levels, grid_shape, window_size, level_lambda, vertical_free, mode, device)
```

### Cache Invalidation

Clear caches when:
- Moving to different device
- Changing geometry parameters mid-training
- Resolution bucketing changes active resolution

```python
from src.context_manager import clear_latent_mask_cache
clear_latent_mask_cache()
```

---

## Relationship to nanovllm Pattern

This architecture mirrors the batch-packing pattern from nanovllm:

| Concept | nanovllm | LDTformer |
|---------|----------|-----------|
| Heterogeneous input | Multiple sequences in batch | Mixed text/image/levels |
| Flattening | `[B, L]` → `[B*L]` | `[B, patches, levels]` → `[B, patches*levels]` |
| Logical structure | Batch boundaries | Document IDs, level IDs |
| Physical layout | Contiguous tensor | Contiguous tensor |
| Isolation mechanism | Block attention mask | Block attention mask |
| Position encoding | Per-sequence indices | Topology coordinates |

The key abstraction: **physical contiguity ≠ logical adjacency**. Masks translate between them.

---

## Performance Considerations

### Mask Construction: O(Q × KV) but Cached

- First call: expensive (builds block structure)
- Subsequent calls: O(1) lookup
- Memory: O(num_blocks) for BlockMask structure

### Distance Computation: O(1) per (q, k) Pair

- Closure captures coordinate tensors
- Indexing is O(1)
- Arithmetic is fused by torch.compile

### SWA Sparsity Benefit

For window_size=2 on 32×32 grid with 6 levels:
- Full attention: 6144² = 37.7M pairs
- SWA (3×3 spatial + vertical): ~6144 × (9 + 6) ≈ 92K pairs
- **400× reduction** in attention computation

---

## Files

| File | Purpose |
|------|---------|
| `src/config.py` | `TopologyGeometryConfig` schema |
| `src/context_manager.py` | `render_latent_topology_embeddings`, `build_latent_diffusion_mask`, cache functions |
| `src/model.py` | Re-exports geometry functions |
| `configs/sparse_ae_swiglu.toml` | `[training.sparse_ae.topology]` section |
| `RNROPE.md` | Companion doc on R^n position embeddings |
