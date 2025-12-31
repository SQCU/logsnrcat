# Sparse AE Integration: Analysis and Path Forward

## What model.py Does That Works

### Attention Infrastructure
The working attention in `model.py` succeeds because of these structural decisions:

1. **Topology-Driven Masks**: Masks are built from explicit topology tensors (`topo_active`, `topo_heap`) containing spatial coordinates and highway (temporal) positions. The `mask_mod` closures index into these pre-built tensors rather than computing relationships from raw indices.

2. **Span System**: Variable-length sequences are managed through `Span` objects with explicit `start_idx`, `end_idx`, `doc_id`, and `causal` flags. Batch flattening happens through span concatenation, not dense tensor stacking.

3. **Tensor Captures in mask_mod**: The mask_mod functions capture GPU tensors (`doc_ids_active_t`, `span_ids_active_t`, etc.) and index into them. They don't capture Python integers that get baked into the graph.

4. **Dual Mask Pattern**: `build_dual_masks` creates both local (spatially-windowed) and global masks from the same topology, allowing layer-wise switching between patterns.

### Key Observation
The flex_attention API works with dynamic compilation because the *mask structure* is fixed (same mask_mod logic) while the *tensor contents* vary. The BlockMask is created fresh per forward pass from current topology, not cached by sequence length.

## What kmaze_ae/model_sparse_dim.py Does Wrong

1. **Parallel Attention System**: Created `GQATransformerLayer` with its own mask caching keyed by `(seq_len, window_size)`. This assumes masks are reusable across calls with same shapes, which conflicts with dynamic compilation.

2. **Python Int Captures**: Mask_mod functions capture `window_size` as Python int:
   ```python
   def mask_mod_sliding(b, h, q_idx, kv_idx):
       return torch.abs(q_idx - kv_idx) <= window_size  # window_size is Python int
   ```
   This gets baked into compiled graphs, causing recompilation on any change.

3. **No Span Integration**: Uses standard `[B, N, D]` layout with no awareness of the span system. Can't participate in the main model's attention patterns.

4. **Separate BlockMask Cache**: Maintains `_block_mask_cache` at class level, which doesn't interact with model.py's mask construction.

## Integration Strategy

### Option A: Sparse AE as Leaf Module (Simpler)
The sparse AE's internal transformers don't need full span machinery if they only process single images. The fix:

1. Remove the GQATransformerLayer entirely
2. Use standard `nn.TransformerEncoder` with no custom attention (or simple sliding window via `attn_mask`)
3. Let torch handle the attention - it's a small enough sequence (256 patches for 256x256 image with 16px patches)
4. The interface wrappers handle translation to/from the span system

**Rationale**: The sparse AE's job is image compression, not document/sequence modeling. It doesn't need doc_id awareness, causal masking, or cross-document isolation.

### Option B: Full Integration (Complex)
Make sparse AE use model.py's attention:

1. Create a simplified `build_image_mask` in model.py that constructs local/global masks for single-image patch sequences
2. Sparse AE's transformers call this function each forward pass
3. Remove all custom mask caching from kmaze_ae

**Rationale**: If we want the sparse AE to eventually participate in joint attention with text tokens, it needs the same infrastructure.

### Recommended: Option A First
Option A is simpler and sufficient for AE training. The sparse AE processes one image at a time, producing codes that get projected into the span system via `SparseAEPatchEmbedder`. The embedder/unembedder interfaces are the integration point, not internal attention.

## Interface Requirements

`SparseAEPatchEmbedder` must match `ContextualPatchEmbedder`:
- `.stride` attribute (int)
- `forward(x: [C,H,W], logsnr: [1,H,W]) -> (z: [L,D], grid_shape: (GH,GW))`

`SparseAEPatchUnembedder` must match `ContextualPatchUnembedder`:
- `forward(z: [L,D], shape: Tuple) -> [C+1, H, W]` (RGB + logsnr channel)

These interfaces are the contract. Internal implementation can change freely.

## Refactoring Candidates (Document Only)

### 1. SpanEmbedder/SpanUnembedder Indirection
Current flow:
```
image -> SparseAEPatchEmbedder -> codes
codes -> SpanEmbedder.patch_emb (which IS SparseAEPatchEmbedder) -> embeddings
```
This double-wrapping is confusing. SpanEmbedder takes `patch_emb` as constructor arg, then its `embed_latent` method calls `patch_emb.forward()`. The indirection exists to unify text and latent embedding, but the naming obscures data flow.

**Candidate refactor**: Rename to make roles clearer: `MultimodalEmbedder` containing `text_embedder` and `image_embedder`, with explicit methods `embed_text()` and `embed_image()`.

### 2. train_autoembed vs train_denoise Separation
Both functions share significant setup (optimizer, scheduler, iterator, logging). The separation exists because AE warmup was added later.

**Candidate refactor**: Single `train()` function with phase flags, or a `TrainingPhase` abstraction that encapsulates phase-specific logic.

### 3. Config Access Patterns
Mix of `config['key']` (correct) and `config.get('key', default)` (legacy). The `.get()` pattern hides missing config errors until runtime.

**Candidate refactor**: Already in plan - remove all `.get()` calls, let KeyError surface immediately.

### 4. ContextBlock as Universal Container
`ContextBlock` holds both text tokens and image tensors via `.content` with `.type` discriminator. This works but requires checking `.type` everywhere.

**Candidate refactor**: Separate `TextBlock` and `ImageBlock` types with shared base, or a proper union type.

## Implementation Steps

1. **Simplify Sparse AE Attention** (model_sparse_dim.py)
   - Remove `GQATransformerLayer` and its mask caching
   - Replace `TransformerEncoder`/`TransformerDecoder` with `nn.TransformerEncoder` using standard attention
   - For sliding window, use `nn.MultiheadAttention` with `attn_mask` parameter
   - Or just use full attention - 256 patches is small enough

2. **Verify Interface Compliance**
   - Test `SparseAEPatchEmbedder.forward()` returns correct shapes
   - Test `SparseAEPatchUnembedder.forward()` returns `[C+1, H, W]`
   - Ensure `.stride` attribute exists and matches patch_size

3. **Fix train_autoembed**
   - Remove any remaining torch.stack assumptions
   - Ensure sparse_ae is called correctly through the wrappers or directly

4. **Test Compilation**
   - Run with `compile=True, compile_dynamic=True`
   - Verify no InductorError
   - Check that different resolutions work without excessive recompilation

## Success Criteria

- `python -m main configs/sparse_ae_fractal.toml` runs without error
- AE training shows decreasing reconstruction loss
- Sparsity metrics are logged correctly
- Diffusion training after AE warmup works
- Sampling produces coherent outputs
