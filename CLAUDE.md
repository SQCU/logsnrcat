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

---

## ContextBlock Access Patterns

### CRITICAL: ContextBlocks Are Heterogeneous By Design

`ContextBlock` is the canonical unit for batched inference over tensors of **different literal shapes**. This is the core abstraction enabling multi-resolution training.

**NEVER do this:**
```python
# WRONG - ContextBlocks have different shapes, this will crash
blocks = iterator.generate_batch_list(batch_size=16)
images = torch.stack([b.content for b in blocks])  # RuntimeError: stack expects equal sizes
```

**Correct patterns:**

1. **For training/inference (heterogeneous batches):**
   ```python
   # Pass blocks directly to SpanEmbedder - it handles grouping internally
   blocks = iterator.generate_batch_list(batch_size=16)
   embedded, spans, hashes = span_embedder.embed(blocks)
   ```

2. **For eval requiring homogeneous batches:**
   ```python
   # Option A: Use generate_from_split for a single data source
   blocks = iterator.generate_from_split('fractal', count=16, resolution=64)

   # Option B: Filter by shape after generation
   blocks = iterator.generate_batch_list(batch_size=64)
   matching = [b.content for b in blocks if b.content.shape[-1] == target_res]
   images = torch.stack(matching[:n_samples])
   ```

3. **For iteration/processing:**
   ```python
   # Process individually or group by shape yourself
   for block in blocks:
       process_single(block.content, block.shape_meta)
   ```

### Why This Exists

Multi-resolution bucketing means a single batch contains images at 64px, 128px, 256px, etc. The `SpanEmbedder` groups blocks by spatial shape `(GH, GW)` for efficient batched attention, then reassembles the heterogeneous output. Callers should never need to stack blocks directly - if you're doing that, you're bypassing the abstraction incorrectly.

---

## Autocast Context Requirements

### CRITICAL: All Model Calls Must Use Training's Autocast Context

Training runs under `torch.amp.autocast(device_type='cuda', dtype=dtype)` with bf16/fp16. Model weights are stored in that dtype. Any code calling the model (eval, sampling, sensitivity sweeps) **must** use the same autocast context.

**Symptom of violation:**
```
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
```

**WRONG:**
```python
# Missing autocast - model weights are bf16 but inputs are float32
with torch.no_grad():
    output = model(images)  # CRASH: dtype mismatch
```

**RIGHT:**
```python
# Match training's autocast context
use_amp = (dtype == torch.bfloat16) or (dtype == torch.float16)
with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
    output = model(images)  # Works: autocast handles dtype conversion
```

### The Pattern

All code paths that call models must:
1. Have access to the training `dtype` (from config)
2. Wrap model calls in `torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_amp)`
3. This applies to: eval loops, sampling, sensitivity sweeps, visualization code, etc.

### Why This Happens

- Training uses autocast → model weights stored in bf16
- Eval code often uses only `torch.no_grad()` → inputs stay float32
- Linear layers fail on `bf16_weights @ float32_inputs`
- Autocast automatically casts inputs to match the context dtype

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

---

## Interactive Model Diagnostics (Eval Server Probes)

These are the probes Claude Code finds most useful when debugging model failures interactively.

### Grey Sludge / Mode Collapse

```python
# Per-layer activation variance (healthy ≈ 1, dead → 0 or ∞)
def activation_variance_sweep(model, x):
    variances = {}
    hooks = []
    def capture(name):
        def hook(m, inp, out):
            variances[name] = out.var().item()
        return hook
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(capture(name)))
    model(x)
    [h.remove() for h in hooks]
    return variances

# Batch diversity (mode collapse → mean ≈ 1)
def batch_cosine_similarity(hidden_states):
    flat = F.normalize(hidden_states.flatten(1), dim=-1)
    sim = flat @ flat.T
    off_diag = sim[~torch.eye(sim.shape[0], dtype=bool, device=sim.device)]
    return {'mean': off_diag.mean().item(), 'std': off_diag.std().item()}
```

### Sampling Correctness

```python
# Verify v-prediction at different noise levels
def v_prediction_sanity(model, x0, logsnr_values):
    results = []
    for logsnr in logsnr_values:
        alpha = (logsnr.sigmoid()).sqrt()
        sigma = ((-logsnr).sigmoid()).sqrt()
        eps = torch.randn_like(x0)
        x_t = alpha * x0 + sigma * eps
        v_true = alpha * eps - sigma * x0
        v_pred = model.predict_v(x_t, logsnr)
        x0_pred = alpha * x_t - sigma * v_pred
        results.append({
            'logsnr': logsnr.item(),
            'v_mse': F.mse_loss(v_pred, v_true).item(),
            'x0_pred_mean': x0_pred.mean().item(),
            'x0_pred_std': x0_pred.std().item(),
        })
    return results
```

### Sparse AE / FSQ Health

```python
# Codebook utilization per level
def codebook_usage(ae, images):
    codes_list = ae.encode(images)
    return [{
        'level': i,
        'unique_codes': codes.flatten(0,1).unique(dim=0).shape[0],
        'sparsity': (codes.abs() < 1e-6).float().mean().item(),
    } for i, codes in enumerate(codes_list)]

# Per-level reconstruction importance
def per_level_importance(ae, images):
    codes = ae.encode(images)
    full_recon = ae.decode(codes)
    base_mse = F.mse_loss(full_recon, images).item()
    importance = []
    for level in range(len(codes)):
        ablated = [c if i != level else torch.zeros_like(c) for i, c in enumerate(codes)]
        ablated_mse = F.mse_loss(ae.decode(ablated), images).item()
        importance.append(ablated_mse - base_mse)
    return importance
```

### Hidden State Geometry

```python
# Effective dimensionality via SVD
def effective_dim(hidden_states, threshold=0.99):
    centered = hidden_states.flatten(0, -2) - hidden_states.flatten(0, -2).mean(0)
    S = torch.linalg.svdvals(centered.float())
    cumvar = (S**2).cumsum(0) / (S**2).sum()
    return (cumvar < threshold).sum().item() + 1

# Gradient norm per layer (for detecting dead/exploding gradients)
def gradient_norms(model, loss):
    loss.backward(retain_graph=True)
    norms = {name: p.grad.norm().item() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    return norms
```

### One-Shot Health Check

```python
def health_check(model, ae, batch):
    """First thing to run on any failing model."""
    recon = ae.decode(ae.encode(batch))
    return {
        'output_range': (batch.min().item(), batch.max().item()),
        'output_variance': batch.var().item(),
        'roundtrip_mse': F.mse_loss(recon, batch).item(),
        'activation_vars': activation_variance_sweep(model, batch),
        'codebook_usage': codebook_usage(ae, batch),
        'batch_diversity': batch_cosine_similarity(recon),
    }
```

---

## Eval Server Architecture (`src/eval_server.py`)

### Purpose

Ephemeral server for interactive model probing. **Network-yeet weights directly from training - NO FILESYSTEM PERSISTENCE.** Exposes `eval()` endpoint for arbitrary Python diagnostics.

### Architecture

```
Training Loop                          Eval Server
┌─────────────┐                       ┌─────────────┐
│ model.dump()│ ─── POST /yeet ──────▶│ param_load()│
└─────────────┘    (raw bytes)        └──────┬──────┘
                                             │
                                      POST /eval
                                             │
                                      ┌──────▼──────┐
                                      │ Claude Code │
                                      └─────────────┘
```

### Design Constraints

1. **No filesystem** - weights are network-yeet'd via `model.dump()` → bytes → HTTP → `param_load()`
2. **Supports gradients** - not just no_grad inference (can run backward for gradient analysis)
3. **Ephemeral** - for testing "hot weights" we might discard after probing
4. **Matches training context** - same `build_model()` path, same autocast
5. **Easier than files** - `yeet_to_server(model)` is one line

### Usage

```bash
# Start server (builds model architecture, waits for weights)
python -m src.eval_server -f configs/exp.toml --port 8421
```

```python
# In training loop - yeet weights to server
from src.eval_server import yeet_to_server
if step % eval_interval == 0:
    yeet_to_server(model, 'http://localhost:8421')

# From Claude Code - probe interactively
from src.eval_server import probe_server
result = probe_server("health_check(model, ae, batch)")
print(result)

# Or via curl
curl -X POST http://localhost:8421/eval -d '{"code": "health_check(model, ae, batch)"}'
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/yeet` | POST | Receive raw state_dict bytes from training |
| `/eval` | POST | Execute Python code in model namespace |
| `/flush` | POST | Zero out model weights (uses `model.flush()`) |
| `/health` | GET | Health check with weights_loaded status |
| `/status` | GET | Detailed status (dtype, device, ae_present, deps_loaded) |
| `/load_deps` | GET | Load plotting/visualization dependencies into namespace |

### Namespace Available to eval()

| Name | Type | Description |
|------|------|-------------|
| `model` | nn.Module | Loaded model (coolerLDTformerZC) |
| `ae` | nn.Module | Sparse AE if present |
| `config` | dict | Sanitized config |
| `batch` | Tensor | Pre-generated [4, C, 64, 64] test batch |
| `ctx` | EvalContext | Server context for get_batch(), autocast() |
| `get_batch(res, bs)` | function | Generate batch at specific resolution |
| `autocast()` | context | Matching training's autocast |
| `load_deps()` | function | Load plotting deps (returns confirmation dict) |
| `torch`, `nn`, `F` | modules | PyTorch imports |
| `health_check`, etc. | functions | All diagnostic functions from above |

### Loading Plotting Dependencies

To avoid repeating import boilerplate in every probe, hit `/load_deps` once to add visualization tools to the namespace:

```bash
# One-time setup
curl http://localhost:8421/load_deps
```

Or from Claude Code:
```python
from src.eval_server import probe_server
probe_server("load_deps()")  # Adds plt, np, make_grid, etc.

# Now use them
probe_server("plt.imshow(batch[0].permute(1,2,0).cpu()); plt.savefig('/tmp/debug.png')")
```

**Dependencies loaded by `/load_deps`:**
| Name | Module | Purpose |
|------|--------|---------|
| `plt` | matplotlib.pyplot | Plotting |
| `np` | numpy | Array ops |
| `make_grid` | torchvision.utils | Image grids |
| `save_image` | torchvision.utils | Save tensors as images |
| `plot_multimetric_analysis` | src.plotting | Project plotting |
| `render_checkerboard`, etc. | src.data_functional | Test data generation |

### Training Integration

Training automatically yeets weights to eval server at the end of training when enabled in config.

**Config (`[logging.eval_server]` in TOML):**
```toml
[logging.eval_server]
enabled = true           # Yeet weights to eval server at end of training
url = "http://localhost:8421"
health_check = true      # Query health after yeet to verify transfer
```

**Schema (`src/config.py`):**
```python
class EvalServerConfig(BaseModel):
    enabled: bool = False
    url: str = "http://localhost:8421"
    health_check: bool = True
```

**What happens at end of `main.py`:**
1. If `cfg['logging']['eval_server']['enabled']` is True:
2. Yeets model weights via `yeet_to_server(model, url)`
3. If `health_check` is True, queries `/health` and prints confirmation

**Client helpers (`src/eval_server.py`):**
```python
from src.eval_server import yeet_to_server, query_health

# Yeet weights (returns True on success)
yeet_to_server(model, 'http://localhost:8421')

# Query health (returns status dict)
health = query_health('http://localhost:8421')
print(health['weights_loaded'])  # True if yeet worked
```

### Why Network-Yeet Instead of Filesystem

- **Models should NEVER PERSIST EVER** - hot weights are transient artifacts
- **Easier than files** - `yeet_to_server(model)` vs `torch.save(...)` + path management
- **Leverages existing API** - `model.dump()` / `model.flush()` / `model.param_load()` already exist
- **Serialization boundary is clear** - bytes over network, not file handles
- **Training code stays clean** - no checkpoint management logic polluting main.py
