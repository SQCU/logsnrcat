# Performance Fixes

Date: 2025-12-31

---

## Sprite Atlas Throughput Fix

### Problem

Run vii regressed 50% (9416 → 4870 batch/hr) after adding sprite_atlas dataset.

Root cause in `src/sprite_atlas.py` `generate_batch_list`:
- `Image.open()` per sprite (disk I/O in loop)
- `np.array()` intermediate (CPU)
- `torch.from_numpy()` → CPU tensor
- `.to(device)` per sprite (**blocking H2D transfer per sprite**)

### Fix

1. **GPU spritesheet cache** (`_sheet_cache`):
   - Spritesheets loaded to GPU once, cached (up to 64 sheets)
   - FIFO eviction when at capacity
   - Single H2D transfer per unique sheet, not per sprite

2. **GPU-side cropping**:
   - Group sprites by sheet path
   - Extract sprites via GPU tensor slicing (no H2D)
   - `sheet[:, y0:y0+96, x0:x0+96]` is a view, not a copy

3. **GPU-native background generation**:
   - `_generate_background()` uses `torch.rand()` on device
   - Removed `self._rng.random()` → CPU tensor → H2D pattern

### Expected Improvement

- Before: batch_size × H2D transfers per batch
- After: ~1-5 H2D transfers per batch (cached sheets)
- Should recover throughput close to run vi levels

---

# Graph Capture Variable-Size Mask Fix

Date: 2025-12-31

## Problem

CUDA graph capture in `src/graph_runner.py` assumed fixed mask shapes based on the first warmup batch. With variable resolution bucketing (e.g., mixed wide/short and tall/narrow images), subsequent batches with different sequence lengths would fail with shape mismatch errors during `_copy_inputs_to_buffers`.

## Changes Made

### 1. `src/graph_runner.py`

#### `_create_static_buffers` (lines 100-140)
- **Before**: Cloned mask tensors from first warmup - shapes determined by that batch's sequence length
- **After**: Allocates mask buffers at `max_blocks` in Q-block dimension (first dim), preserving KV-blocks-per-Q dimension from attention pattern

```python
# Get KV dimension from incoming mask (determined by attention pattern, not seq len)
local_kv_dim = mask_local.kv_indices.shape[1] if mask_local.kv_indices.dim() > 1 else 1
# ... allocate at [max_blocks, kv_dim] instead of cloning
```

#### `_copy_inputs_to_buffers` (lines 158-200)
- **Before**: Direct `.copy_()` assuming shapes match
- **After**:
  1. Gets actual block count from incoming mask
  2. Zeros out static buffers (kv_num_blocks=0 disables attention for OOB)
  3. Copies into valid slice `[:actual_blocks]`

```python
actual_blocks_local = mask_local.kv_num_blocks.shape[0]
self._buffers.mask_local_kv_num_blocks.zero_()
self._buffers.mask_local_kv_indices[:actual_blocks_local].copy_(mask_local.kv_indices)
```

#### `_create_static_masks` (lines 142-168)
- Added update of BlockMask shape metadata (`num_rows`, `num_cols`) to reflect `max_blocks`

### 2. `configs/sparse_ae_fractal.toml`

Added graph capture config section:

```toml
[training.graph_capture]
enabled = true
warmup_steps = 5
capture_after_warmup = true
use_dedicated_stream = true
```

### 3. `src/config.py`

No changes needed - already correctly structured:
- `GraphCaptureConfig` class defined with defaults (`enabled=False`)
- Nested under `TrainingConfig.graph_capture`
- `model_dump()` handles nested Pydantic models correctly

## Architecture Note

The graph capture region is correctly scoped:

```
OUTSIDE GRAPH (variable shapes):
├── span_embedder.embed()     <- sparse AE encode (patchify)
├── render_topology_embeddings()
├── build_dual_masks()

GRAPH CAPTURABLE (fixed shapes via static buffers):
└── model() transformer forward

OUTSIDE GRAPH (variable shapes):
└── span_unembedder.decode()  <- sparse AE decode (unpatchify)
```

## Expected Behavior

After warmup_steps iterations with real data:
1. Static buffers allocated at max_blocks capacity
2. Graph captured on dedicated CUDA stream
3. Subsequent iterations use graph replay (single CUDA call for transformer)
4. Variable-sized batches work by copying into valid slices of static buffers

## Throughput Context

From `analyze_throughput.py` on sparse_ae_fractal logs:
- Runs i-iv (dense attention, graph capture disabled): ~1500-1900 batch/hr
- Run v (gemma-bigbird, zero graph breaks): 5270 batch/hr
- Run vi (same + optimized): 9416 batch/hr

The 5x speedup came from architectural changes (gemma-bigbird attention + eliminating graph breaks), not from CUDA graph capture which was disabled. Enabling graph capture should provide additional speedup by eliminating per-iteration kernel launch overhead.
