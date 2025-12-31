# Refactoring Agenda: model.py Decomposition and Sparse AE Integration

## Executive Summary

The codebase has three interrelated problems:

1. **model.py is too large** (1483 lines) - Contains 8+ distinct concerns that should be separate modules
2. **Sparse AE integration is broken** - The `kmaze_ae/model_sparse_dim.py` file has mask caching bugs that cause recompilation
3. **Attention architecture mismatch** - Original model.py avoided attention in embedders, but sparse AE needs attention internally

## Critical Finding: The Mask Caching Bug

In `kmaze_ae/model_sparse_dim.py:93-104`:

```python
@classmethod
def _get_sliding_block_mask(cls, seq_len: int, window_size: int):
    cache_key = ('sliding', seq_len, window_size, 0)  # <-- Python ints!
    if cache_key not in cls._block_mask_cache:
        def mask_mod_sliding(b, h, q_idx, kv_idx):
            return torch.abs(q_idx - kv_idx) <= window_size  # <-- Captures Python int!
```

This violates the flex_attention contract. The mask_mod captures `window_size` as a Python int, which gets baked into the compiled graph. When seq_len or window_size changes, a NEW graph is compiled. The correct pattern (from model.py) is:

```python
# Correct: Capture TENSORS, not Python values
win_sq = torch.tensor(window_size * window_size, device=device, dtype=dtype)

def mask_mod_local(b, h, q_idx, kv_idx):
    # Index into pre-built tensors
    dist_sq = ...
    return base & (dist_sq < win_sq)  # Tensor comparison
```

## File Structure Proposal

### Current: Everything in model.py

```
src/model.py (1483 lines)
├── PageTable                    (lines 13-183)
├── Initialization helpers       (lines 185-208)
├── HouseholderOrthogonal        (lines 224-256)
├── RnRoPE                       (lines 257-336)
├── SwiGLU, SigmoidMoE           (lines 339-433)
├── MLPResBlock                  (lines 435-447)
├── FourierFeatures/Decoder      (lines 449-493)
├── ContextualPatchEmbedder      (lines 497-549)
├── ContextualPatchUnembedder    (lines 552-585)
├── ContextBlock, Span           (lines 590-633)
├── SpanEmbedder/Unembedder      (lines 636-717)
├── update_kv_cache              (lines 720-757)
├── LDTformerAttention*          (lines 759-946)
├── coolerLDTformer*             (lines 950-1136)
├── generate_content_hash_stream (lines 1145-1201)
├── render_topology_embeddings   (lines 1208-1270)
└── build_dual_masks             (lines 1323-1484)
```

### Proposed: Modular Structure

```
src/
├── model.py                 (~400 lines) - Just LDTformer models
│   ├── LDTformerAttentionZC
│   ├── LDTformerBlockZC
│   ├── coolerLDTformerZC
│   └── coolerLDTformerKVC (if kept)
│
├── context_utils.py         (~300 lines) - Context management
│   ├── ContextBlock
│   ├── Span
│   ├── SpanEmbedder
│   ├── SpanUnembedder
│   ├── generate_content_hash_stream
│   ├── render_topology_embeddings
│   └── build_dual_masks
│
├── rope.py                  (~150 lines) - Position encoding
│   ├── HouseholderOrthogonal
│   └── RnRoPE
│
├── embedders.py             (~200 lines) - Patch/text embedding
│   ├── FourierFeatures
│   ├── FourierScaleDecoder
│   ├── ContextualPatchEmbedder
│   └── ContextualPatchUnembedder
│
├── blocks.py                (~150 lines) - Building blocks
│   ├── SwiGLU
│   ├── SigmoidMoE
│   ├── MLPResBlock
│   └── init_* helpers
│
├── paging.py                (~200 lines) - KV cache paging
│   ├── PageTable
│   └── update_kv_cache
│
├── sparse_ae.py             (~500 lines) - Sparse autoencoder
│   ├── (Moved from kmaze_ae/model_sparse_dim.py)
│   ├── SparsePerDimFSQAutoencoder
│   ├── SparseAEPatchEmbedder
│   └── SparseAEPatchUnembedder
│
└── utils.py                 (~300 lines) - Rename to reflect actual content
    ├── Block/BlockManager (already here)
    ├── KVTManager (already here)
    └── run_model_forward, etc.
```

## Key Insight: SpanEmbed/SpanUnembed as API Shield

From the user's note:
> "correcting the apis for using the patching and unpatching methods should be basically salvageable by correcting the apis at the spanembed and spanunembed methods"

The SpanEmbedder takes `patch_emb` as a constructor argument:

```python
class SpanEmbedder:
    def __init__(self, text_embedder, patch_embedder):
        self.text_emb = text_embedder
        self.patch_emb = patch_embedder  # <-- Swappable!
```

This means:
- SparseAEPatchEmbedder can replace ContextualPatchEmbedder
- As long as it provides `.stride` and `forward(x, logsnr) -> (z, grid_shape)`
- No broad rewrites needed upstream

## Attention Architecture Issue

### Original Design (model.py)
- ContextualPatchEmbedder: **No attention** - just MLP residual blocks
- ContextualPatchUnembedder: **No attention** - just MLP residual blocks
- Attention happens in LDTformerBlockZC during the main forward pass
- This was intentional: embedders are fast, attention uses the complex topology/masking system

### Sparse AE Design (kmaze_ae)
- SparseLevelEncoder: **Uses TransformerEncoder with attention**
- SparseLevelDecoder: **Uses TransformerDecoder with attention**
- Attention happens INSIDE the AE, separate from main model

### The Problem
The sparse AE's internal attention creates a parallel attention system that:
1. Doesn't use the topology/span infrastructure
2. Has its own mask caching (buggy)
3. Uses different patterns (GQA with bigbird) vs main model (MoE with local/global)

### Solutions

**Option A: Simplify Sparse AE Attention**
- Use `nn.TransformerEncoder` with no custom masking
- 256 patches for 256x256 image = O(65K) attention ops, acceptable
- Remove all flex_attention complexity from AE
- AE trains independently, interfaces via embedder wrappers

**Option B: Integrate Sparse AE with Main Attention**
- Create `build_image_only_masks()` in context_utils.py
- Sparse AE calls this each forward pass
- Enables future joint text+image attention in AE
- More complex, higher payoff if needed

**Recommendation: Option A first**, then Option B if joint attention is needed.

## Reference Code Comparison

### Original kmaze (parent dir): `/mnt/f/dox/repos/ai/model_sparse_dim.py`
- Clean, simple
- Uses `F.scaled_dot_product_attention(q, k, v)` - no flex_attention
- No mask caching
- 287 lines total

### Mangled version: `kmaze_ae/model_sparse_dim.py`
- 966 lines (3.4x larger)
- Added flex_attention with buggy mask caching
- Added bigbird/gemma patterns
- Added register tokens
- Added logsnr handling (useful)
- Added interface wrappers (useful)

### What to Keep from Mangled Version
1. `SparseAEPatchEmbedder` / `SparseAEPatchUnembedder` interface wrappers
2. LogSNR handling in encoder/decoder
3. `patchify_logsnr` / `unpatchify_logsnr` utilities
4. Per-level logsnr estimators concept

### What to Discard
1. `GQATransformerLayer` with mask caching
2. `_get_sliding_block_mask` / `_get_bigbird_block_mask`
3. `TransformerEncoder`/`TransformerDecoder` custom classes
4. All flex_attention usage in AE (for now)

## Execution Order

### Phase 1: Stabilize (Don't Break Working Code)
1. Create `src/context_utils.py` - move context management code
2. Create `src/embedders.py` - move embedding code
3. Create `src/rope.py` - move RoPE code
4. Update imports in `train.py`, `utils.py`, etc.
5. Test that existing training still works

### Phase 2: Fix Sparse AE
1. Simplify `kmaze_ae/model_sparse_dim.py`
2. Replace flex_attention with standard attention
3. Keep logsnr handling and interface wrappers
4. Test standalone AE training

### Phase 3: Integrate
1. Move fixed sparse AE to `src/sparse_ae.py`
2. Wire up SparseAEPatchEmbedder to SpanEmbedder
3. Update config to select embedder type
4. Test end-to-end training

### Phase 4: Cleanup
1. Remove `kmaze_ae/` directory
2. Update all documentation
3. Clean up unused code paths

## Files to Modify

| File | Action | Priority |
|------|--------|----------|
| src/model.py | Extract ~1000 lines | High |
| src/context_utils.py | Create | High |
| src/embedders.py | Create | High |
| src/rope.py | Create | Medium |
| src/blocks.py | Create | Medium |
| src/paging.py | Create | Low |
| kmaze_ae/model_sparse_dim.py | Fix/Simplify | High |
| src/utils.py | Rename + clean | Medium |
| src/train.py | Update imports | After extraction |

## Success Criteria

1. `python -m main configs/sparse_ae_fractal.toml` runs without error
2. No torch.compile recompilation on batch size changes
3. Reconstruction loss decreases during AE training
4. Each src/*.py file < 500 lines
5. No circular imports
6. All existing tests pass (if any)

## Immediate Next Steps

1. [ ] Fix the mask caching bug in kmaze_ae as a standalone fix
2. [ ] Create context_utils.py with ContextBlock, Span, SpanEmbedder, SpanUnembedder
3. [ ] Create embedders.py with ContextualPatchEmbedder, ContextualPatchUnembedder, FourierFeatures
4. [ ] Test that imports work correctly
5. [ ] Simplify sparse AE attention to use nn.TransformerEncoder
