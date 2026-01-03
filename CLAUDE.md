# Claude Session Notes - logsnrcat Project

## Project Overview

Field diffusion training with heterogeneous optimization, sparse autoencoders, and multi-resolution bucketing.

---

## File Map: Where Features Live

### Core Training Pipeline
| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `main.py` | Entry point, config loading, component assembly | `build_model()`, `build_components()` |
| `src/train.py` | Training loops (AE warmup + denoiser) | `train_autoembed()`, `train_denoise()`, `OnlineVarianceTracker` |
| `src/config.py` | Pydantic schema - ALL defaults live here | `ExperimentConfig`, `OptimizerConfig`, `MuonConfig`, etc. |
| `configs/*.toml` | User-facing config files | - |

### Model Architecture
| File | Purpose | Key Classes |
|------|---------|-------------|
| `src/model.py` | Main transformer models | `coolerLDTformerZC`, `LDTformerBlockZC`, `LDTformerAttentionZC` |
| `src/blocks.py` | Sparse AE transformer blocks | `TransformerEncoder`, `TransformerDecoder`, `EncoderAttention`, `SwiGLU`, `SigmoidMoE` |
| `src/embedders.py` | Patch embed/unembed, Fourier features | `ContextualPatchEmbedder`, `ContextualPatchUnembedder`, `FourierFeatures` |
| `src/context_manager.py` | Span embedding for mixed modality | `SpanEmbedder`, `SpanUnembedder`, `ContextBlock` |
| `src/rope.py` | Rotary position embeddings | `RnRoPE`, `HouseholderOrthogonal` |

### Optimization
| File | Purpose | Key Classes |
|------|---------|-------------|
| `src/optim_utils.py` | Heterogeneous optimizer (Muon+AdamW) | `Muon`, `OptimizerGroup`, `FP8Linear`, `build_optimizer_group()`, `classify_parameters()` |

**Parameter Classification for Heterogeneous Optimizer:**
- `transformer`: Muon (orthogonalized momentum, bf16 state safe)
- `embedding`: AdamW (stable for high-magnitude params)
- `norm`: AdamW (no weight decay)
- `fsq_adjacent`: AdamW (sigmoid STE attenuates gradients, sparse masks zero most)
  - Patterns: `code_proj`, `code_unproj`, `fsq`, `sparsity`, `dim_logits`, `attn_gate`, `logsnr`

### Data Pipeline
| File | Purpose | Key Classes |
|------|---------|-------------|
| `src/data_iterator.py` | Unified data iteration | `CompositeIterator`, `FunctionalIterator` |
| `src/data.py` | Dataset implementations | `VideoFolderIterator`, `TorusIterator`, `CheckerboardIterator` |
| `src/fractal.py` | Fractal generation | `FractalIterator`, `CUDAGraphFractalGenerator` |
| `src/sprite_atlas.py` | Sprite atlas dataset | `SpriteAtlasIterator`, `SpriteAtlasDataset` |
| `src/bucket_manager.py` | Resolution bucketing | `BucketManager`, `ResolutionBucket` |

### Inference & Sampling
| File | Purpose | Key Classes |
|------|---------|-------------|
| `src/sample.py` | Sampling, diffusion steps | `euler_forward_step()`, `euler_reverse_step()`, `MultiTurnContext` |
| `src/graph_runner.py` | CUDA graph capture for inference | `GraphRunner`, `GraphBuffers` |
| `src/paging.py` | KV cache paging | `PageTable` |

### Utilities
| File | Purpose | Key Classes |
|------|---------|-------------|
| `src/plotting.py` | Visualization, logging | `ExperimentLogger`, `plot_multimetric_analysis()` |
| `src/utils.py` | Blocks, context management | `Block`, `BlockManager`, `KVTManager` |
| `src/patches/` | Upstream bug workarounds | `triton_windows_int64.py` |

---

## Problem-Solving Guide: Optimization Issues

### NaN / Training Instability

1. **Check optimizer numerical stability** → `src/optim_utils.py`
   - Muon `_orthogonalize()`: Must run in fp32 (bf16 matmuls accumulate error)
   - AdamW state: Always fp32 (v accumulator numerically sensitive)

2. **Check precision mismatches** → `src/train.py`, `src/optim_utils.py`
   - State dtype vs compute dtype vs parameter dtype
   - Muon momentum buffer can be bf16 (no division), AdamW cannot

3. **Check scheduler configuration** → `src/optim_utils.py:_build_scheduler()`
   - Each optimizer needs its own `max_lr` via `max_lr_override`
   - Wrong LR = silent degradation or explosion

4. **Check FSQ-adjacent parameter classification** → `src/optim_utils.py:classify_parameters()`
   - Params feeding into sigmoid FSQ have attenuated gradients (sigmoid' max = 0.5)
   - Sparse masking zeros gradients for (code_dim - k) dimensions
   - `dim_logits` uses MoE router STE (sigmoid for gradient, topk for forward)
   - These params should use AdamW, not Muon (attenuated gradients)

### Wrong Learning Rate / Optimizer Not Working

1. **Verify config schema** → `src/config.py`
   - `OptimizerConfig` must have all fields for heterogeneous mode
   - Validator `validate_heterogeneous_requires_subconfigs` catches missing muon/adamw

2. **Check build path** → `src/optim_utils.py:build_optimizer_group()`
   - `_build_heterogeneous()` vs `_build_single_adamw()` dispatch
   - Parameter classification in `classify_parameters()`

3. **Print optimizer summary** → `print_optimizer_summary(optimizer_group, model)`
   - Shows actual LR per group, param counts, percentages

### Mask / Attention Issues

1. **flex_attention masks** → `src/blocks.py`, `src/model.py`
   - Masks must be created DURING forward pass (inside compile trace)
   - GraphRunner pre-creates masks → conflicts with torch.compile dynamic shapes

2. **GQA configuration** → `src/config.py:GQAConfig`, `src/model.py`
   - n_kv_heads must divide evenly into num_heads

### Memory / OOM Issues

1. **Resolution bucketing** → `src/bucket_manager.py`
   - Check bucket batch sizes scale inversely with resolution²

2. **Graph capture** → `src/graph_runner.py`
   - Pre-allocated buffers sized for max_ctx
   - Warmup steps before capture

3. **FP8 weights** → `src/optim_utils.py:FP8Linear`
   - Reduces weight memory, but requires Hopper/Ada for native support

### Config Not Applied

1. **Schema validation** → `src/config.py`
   - Missing Pydantic field = silent fallback to default
   - Add field to schema FIRST, then use in code

2. **Config sanitization** → `src/config.py:sanitize_config()`
   - Converts Pydantic models to dicts
   - Must be called before accessing nested fields

---

# Architecture Principles

## Configuration Architecture: The Single Source of Truth

### CRITICAL: No Defensive `.get()` with Defaults in Method Internals

**WRONG:**
```python
def _build_heterogeneous(model, opt_cfg, total_steps):
    muon_cfg = opt_cfg.get('muon', {})  # EVIL - hides missing config
    lr = muon_cfg.get('lr', 0.02)       # EVIL - silent fallback
```

**RIGHT:**
```python
def _build_heterogeneous(model, opt_cfg, total_steps):
    muon_cfg = opt_cfg['muon']  # CRASH if missing - config schema is wrong
    lr = muon_cfg['lr']          # CRASH if missing - schema incomplete
```

### Why This Matters

1. **Defensive programming inside validated code is EVIL** - Once past Pydantic schema validation, all required fields MUST exist. If `.get()` with defaults appears in downstream code, it means:
   - The schema (config.py) is incomplete
   - Silent failures will occur when configs are wrong
   - Debugging becomes impossible ("why is it using 0.02 when I set 0.01?")

2. **Crash early, crash loudly** - A KeyError with a clear traceback is infinitely better than silent fallback to a default buried in method internals.

3. **Defaults belong in exactly two places:**
   - Pydantic models in `config.py` (schema defaults)
   - TOML config files (user-facing overrides)
   - **NEVER** in function implementations

### The Config Flow

```
TOML file (user input)
    ↓
load_config() → Pydantic validation (ExperimentConfig)
    ↓
sanitize_config() → Dict[str, Any]
    ↓
Functions receive complete, validated dicts
    ↓
Direct access with [] - crash on missing = schema bug
```

### When Implementing Features with Conditional Inputs

1. **Update config.py first** - Add Pydantic models for new config sections
2. **Update TOML configs** - Add the new fields with appropriate values
3. **Use direct dict access in code** - `config['training']['optimizer']['muon']['lr']`
4. **Never add fallback defaults in methods** - If it crashes, the schema is wrong

### Data Iterators Pattern

For variable/streaming data (not static config), use iterator patterns:
- `CompositeIterator` consumes configs to produce data streams
- Iterators encapsulate the variability
- Consumer code receives complete, typed objects

### Common Anti-Patterns to Avoid

```python
# BAD: Hidden default
value = config.get('key', 'default')

# BAD: Defensive None check after validation
if config.get('section') is not None:
    ...

# BAD: or-fallback pattern
section = config.get('section') or {}

# GOOD: Direct access - crash reveals schema bugs
value = config['key']
section = config['section']
```

### Session Persistence Notes

These principles must survive context compaction. When working on this codebase:
- All feature additions requiring conditional behavior → update config.py + TOML
- No `.get()` with defaults in src/*.py methods (except actual optional fields properly marked in schema)
- Schema validation is the ONLY guard - trust it completely downstream

---

## Future Feature: Gradient Anomaly Detection & Visualization

**Status:** Spec only - implement when needed for debugging, NOT as preemptive defense

### Motivation
When training destabilizes, we need to diagnose WHERE and WHY - not mask it with clipping.
Gradient clipping conceals defects. Anomaly detection reveals them.

### Requirements

1. **Gradient Statistics Tracker**
   - Per-layer/per-param-group gradient norms (L2, Linf)
   - Running EMA of gradient magnitudes
   - Detect when current grad >> EMA (anomaly threshold configurable)

2. **Anomaly Detection (NOT clipping)**
   - Flag steps where grad norm exceeds k * EMA
   - Log which parameters triggered the anomaly
   - Optionally: pause training and dump state for analysis

3. **Visualization**
   - Per-layer gradient norm over time
   - Heatmap of gradient flow through model
   - Histogram of gradient magnitudes per param group

4. **Integration**
   - Optional diagnostic mode (not on by default)
   - Zero overhead when disabled
   - Logs to experiment directory for post-hoc analysis

### Anti-patterns to avoid
- NO gradient clipping as "fix"
- NO silent clamping or normalization
- Detection/visualization only - never modify gradients

### When to implement
When we encounter instability and need to trace it to root cause.
NOT preemptively. The fix for instability is fixing the instability, not hiding it.
