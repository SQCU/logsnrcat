# RnRoPE: N-Dimensional Rotary Position Embeddings with Pre-Compiled Geometric Distance Fields

## Overview

This codebase implements a generalized position encoding system that treats **all contexts as sequences of embeddings** with positions defined by **pre-computed topology coordinates** rather than naive sequential indices. The system was designed from the ground up to support:

1. **Arbitrary distance relationships** via topology embedding into local R^n manifolds
2. **Graph geodesic distances** that pack high-dimensional or non-Euclidean spaces into low-dimensional local topology
3. **Multi-modal multi-turn multi-image contexts** without spurious inductive biases

The core insight: position is not an integer index, but a point in an N-dimensional coordinate space where distance relationships are explicitly computed and materialized before being fed to RoPE and attention masks.

---

## The Highway + Manifold Decomposition

### Topology Structure: `[highway, spatial_1, spatial_2, ...]`

Every token receives coordinates in an N-dimensional topology space (`topo_dim`, typically 3):

```
Column 0:   highway      - Monotonic context position across ALL tokens
Columns 1+: manifold     - Spatial coordinates (2D grid for images, origin for text)
```

**The Highway Dimension** (`ctx-dim`) represents sequential ordering within the context window. It increases monotonically across all tokens regardless of modality:

```
[text_tok_0, text_tok_1, ..., image_patch_0, image_patch_1, ..., text_tok_N, ...]
     0           1                 K            K+1                  K+HW
```

**The Manifold Dimensions** (`space_1`, `space_2`) encode spatial relationships:
- **Images**: Grid coordinates `(i, j)` from the patch grid
- **Text**: The spatial "singularity" at `(0, 0)` - all text tokens share the same spatial position

### Why This Matters: Eliminating Multi-Image Ordering Bias

Consider two images in a multimodal context. Naive flattening creates a spurious bias:

```
# WRONG: Naive sequential embedding
image1_patches = [p00, p01, p02, ..., p77]  # positions 0-63
image2_patches = [q00, q01, q02, ..., q77]  # positions 64-127

# This implies: image1's right edge (position 7) is "before" image2's left edge (position 64)
# RoPE will encode: patch p07 is closer to p06 than to q00
# But semantically: p07 and q00 might both be "left edges" of their respective images!
```

**The correct approach (implemented here)**:

```
# Highway: Sequential context position
# Manifold: Spatial position WITHIN the image

image1_patch[3,4] -> topo = [highway=K,   spatial=(3, 4)]
image2_patch[3,4] -> topo = [highway=K+64, spatial=(3, 4)]

# Same spatial position, different context positions
# RoPE encodes: these patches are spatially coincident but temporally separated
```

This explicitly disposes of inductive biases like "the first image's right side is to the left of the second image's right side" - a real problem in many multimodal architectures that linearize images before position encoding.

---

## RnRoPE: The Geometric Embedding

### Architecture (`src/rope.py`)

```python
class RnRoPE(nn.Module):
    """
    N-dimensional Rotary Position Embedding.

    Extends standard RoPE to handle multi-dimensional topology coordinates
    (e.g., temporal highway + 2D spatial grid for images).
    """
    def __init__(self, head_dim: int, topo_dim: int, rope_base: float = 500.0):
        # Householder rotation for latent space projection
        self.orthogonal = HouseholderOrthogonal(head_dim, num_reflections=head_dim//2)

        # Frequency bands split across topology dimensions
        self.features_per_subspace = (head_dim // 2) // topo_dim
```

**Key Operations**:

1. **Householder Orthogonal Projection**: Learned orthogonal transformation that projects Q/K into a frequency-friendly basis before applying rotations
2. **Per-Dimension Frequencies**: Each topology dimension gets `head_dim // (2 * topo_dim)` frequency bands
3. **Outer Product Phase Computation**: `freqs[b,l,d,f] = topo_coords[b,l,d] * inv_freq[f]`
4. **Standard RoPE Application**: Rotate-half with computed phases, then project back

### The Householder Parameterization

Instead of directly parameterizing rotation angles, we use a product of Householder reflections to create a learnable orthogonal matrix:

```python
class HouseholderOrthogonal(nn.Module):
    """Parametrized Orthogonal Matrix via product of Householder reflections."""

    def get_matrix(self):
        Q = torch.eye(self.dim)
        for v in self.vs:
            # H = I - 2vv^T / ||v||^2 (Householder reflection)
            Q = Q - (2 / ||v||^2) * v @ v.T @ Q
        return Q
```

This ensures the transformation is always exactly orthogonal (no gradient-induced drift), which is critical for RoPE's distance-preserving properties.

---

## Pre-Computed Distance Fields and Graph Geodesics

### The Core Insight

The topology coordinates are **not** required to be literal grid positions. They can be **any embedding** that encodes distance relationships, including:

1. **Euclidean grid coordinates**: Standard 2D image patches
2. **Graph geodesic distances**: Shortest-path distances on a graph, embedded into R^n
3. **Manifold embeddings**: High-dimensional spaces compressed via MDS, UMAP, or learned embeddings
4. **Hierarchical coordinates**: Multi-scale positions (e.g., residual levels + spatial)

### Graph Geodesic Example

Consider a graph with complex connectivity. To use RnRoPE:

1. Compute all-pairs shortest paths on the graph
2. Embed the distance matrix into R^n using metric MDS or similar
3. Use the embedded coordinates as topology coordinates

```python
# Conceptual: embedding graph geodesics
distances = floyd_warshall(adjacency_matrix)  # [N, N] shortest paths
coords = metric_mds(distances, n_dims=topo_dim-1)  # [N, topo_dim-1]
topo_embeds = torch.cat([highway, coords], dim=-1)  # [N, topo_dim]
```

The RoPE frequencies will then encode approximate geodesic distances, with distortion controlled by the embedding quality.

### Distortion as a Feature

When embedding high-dimensional spaces into low-dimensional R^n, distortion is inevitable. However:

1. **Local distances are preserved better than global** - exactly what attention needs
2. **The Householder projection learns to compensate** - the orthogonal basis adapts to the embedding's structure
3. **Attention masks provide hard constraints** - SWA masks enforce exact connectivity where needed

---

## Sliding Window Attention with Topology Distance

### Mask Construction (`src/context_manager.py`)

```python
def mask_mod_local(b, h, q_idx, kv_idx):
    base = base_connectivity(q_idx, kv_idx)  # Document isolation + causality

    # Spatial distance in manifold coordinates
    dist_sq = 0.0
    for q_col, k_col in zip(spatial_active, spatial_heap):
        d = q_col[q_idx] - k_col[kv_idx]
        dist_sq = dist_sq + (d * d)

    spatial_ok = dist_sq < window_size_squared
    return base & spatial_ok
```

**Key Properties**:

1. **Distance is computed in topology space**, not sequence index space
2. **Different modalities can have different effective neighborhoods**:
   - Images: 3x3 spatial window based on grid distance
   - Text: All text tokens at `(0,0)` are "spatially adjacent" to each other
3. **The highway dimension is NOT used for spatial windowing** - it only affects causal ordering

### Multi-Image Spatial Overlap

Two images at the same resolution have **overlapping spatial coordinates**:

```
image1[2,3] -> spatial = (2, 3), highway = K
image2[2,3] -> spatial = (2, 3), highway = K + HW

# Spatial distance = 0 (same manifold position)
# Highway distance = HW (different context positions)
```

This means: patches at the same spatial position in different images are "close" in the spatial SWA sense, allowing cross-image spatial attention when masks permit.

---

## Connection to FlexAttention Block Masking (nanovllm Pattern)

### The Shared Paradigm

Both this codebase and nanovllm's flex_attention implementation follow the same fundamental pattern:

1. **Flatten heterogeneous structures into single tensors**
   - Here: Mixed text/image tokens → flat `[L_total, D]` sequence
   - nanovllm: Multiple sequences in batch → single `[B*L, D]` tensor

2. **Use block attention masks to maintain logical isolation**
   - Here: `doc_id` tracking prevents cross-document attention
   - nanovllm: Batch boundaries encoded in mask prevent cross-sequence attention

3. **Separate physical layout from logical structure**
   - Here: Paged KV cache with `PageTable` mapping logical→physical blocks
   - nanovllm: Similar block-slot mapping for flattened batches

4. **Pre-compute masks outside the compiled forward pass**
   - Here: `build_dual_masks()` creates `BlockMask` before `torch.compile` traces
   - nanovllm: Block attention document masks created before graph capture

### The Key Abstraction

Both systems implement the same insight: **treat position as metadata, not memory layout**.

```
Physical Position: Where the tensor element lives in memory
Logical Position:  What the element's "position" means for attention

# Physical contiguity ≠ Logical adjacency
# Mask functions translate between them
```

This allows efficient dense tensor operations (good for GPU) while maintaining complex logical structures (good for models).

---

## Extending to Residual Quantization Levels

### The Three-Space Problem

When extending to multi-level residual quantization (as in the SwiGLU FSQ AE), we have:

```
Dimension 0: highway (context position across all tokens)
Dimension 1: spatial_x (2D grid x-coordinate)
Dimension 2: spatial_y (2D grid y-coordinate)
Dimension 3: level (residual quantization level) [NEW]
```

### Distance Metric Considerations

The level dimension has different semantics than spatial dimensions:

1. **Spatial**: Euclidean distance, symmetric, continuous
2. **Level**: Represents scale hierarchy, semantically asymmetric (coarse→fine)

A principled distance metric might be:

```python
dist = sqrt(spatial_dist_sq) + lambda * abs(level_diff)
# or
dist = sqrt(spatial_dist_sq + (lambda * level_diff)^2)
```

Where `lambda` controls the "cost" of crossing levels relative to spatial movement.

### Graph Geodesic Interpretation

The multi-level structure naturally forms a graph:

```
Nodes: (patch_position, level) tuples
Edges within level: 2D spatial neighborhood (SWA)
Edges across levels: Residual connections (same position, adjacent levels)
```

Geodesic distance = shortest path on this product graph.

This can be pre-computed and embedded into R^n topology coordinates, then fed to RnRoPE just like any other distance field.

---

## Implementation Files

| File | Purpose |
|------|---------|
| `src/rope.py` | `RnRoPE`, `HouseholderOrthogonal` - core position encoding |
| `src/context_manager.py` | `render_topology_embeddings`, `build_dual_masks` - geometry + masks |
| `src/model.py` | `LDTformerAttentionZC` - attention with RnRoPE integration |
| `src/paging.py` | `PageTable` - physical↔logical block mapping |

---

## Summary

This codebase solves several problems that are typically ignored or handled incorrectly:

1. **Multi-image contexts without ordering bias**: Images overlap in spatial coordinates, differ only in highway position
2. **Arbitrary distance relationships**: Topology coordinates can encode any distance field, not just grid positions
3. **Graph geodesic support**: Pre-compute shortest paths, embed into R^n, use as topology
4. **Unified handling of modalities**: Text at spatial origin, images on grid, same RoPE and SWA machinery
5. **Physical/logical separation**: Paged KV cache and block masks decouple memory layout from attention patterns

The system was designed for harder problems than typical DNN repositories address - the R^n RoPE and topology embedding infrastructure is deliberately general enough to handle manifold embeddings, graph distances, and hierarchical multi-scale positions with minimal modification.
