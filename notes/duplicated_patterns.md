# Duplicated Patterns Requiring Refactor

Patterns that have been copy-pasted across multiple locations, creating maintenance burden and bug surface area. Each section describes the duplication and suggests consolidation.

---

## 1. Deferred Tensor Stats Logging

**Locations:**
- `train_denoise()` lines ~725-743, ~804-818
- `train_latent_diffusion()` lines ~1194-1203, ~1236-1263

**Pattern:**
```python
# Collection phase (inside training loop)
deferred_stats.append({
    'step': i,
    'source': getattr(block, 'source', 'unknown'),
    'type': 'latent',
    'loss_tensor': loss.detach(),
    'logsnr_tensor': logsnr.mean().detach(),
    'resolution': res,
})

# Conversion phase (at logging intervals)
for stat in deferred_stats:
    converted = {'step': stat['step'], 'source': stat['source'], 'type': stat['type']}
    for key in list(stat.keys()):
        if key.endswith('_tensor'):
            base_key = key[:-7]
            val = stat[key]
            converted[base_key] = val.item() if isinstance(val, torch.Tensor) else val
        elif key not in ('step', 'source', 'type'):
            converted[key] = stat[key]
    history.append(converted)
deferred_stats.clear()
```

**Why it exists:** Avoid per-step CPU-GPU sync overhead; batch `.item()` calls at logging intervals.

**Refactor suggestion:** Extract to `DeferredStatsCollector` class with `.append()`, `.flush_to_history()` methods.

---

## 2. Patch Unembedder Output Format

**Locations:**
- `kmaze_ae/model_sparse_dim.py` `SparseAEPatchUnembedder.forward()`
- `kmaze_ae/model_swiglu.py` `SwiGLUPatchUnembedder.forward()`
- `src/embedders.py` `ContextualPatchUnembedder.forward()`

**Pattern:**
All must return `[C+1, H, W]` or `[B, C+1, H, W]` where last channel is logsnr prediction.
Consumer code in `context_manager.py` does:
```python
spandict['image_vpreds'] = reconstruction[:-1]  # RGB
spandict['image_logsnrs'] = reconstruction[-1:]  # logsnr
```

**Failure mode:** If any unembedder returns `[C, H, W]` without logsnr channel, the slice gives wrong channel count, causing shape mismatches downstream in sampling.

**Refactor suggestion:**
- Define `UnembedderOutput` protocol/base class enforcing shape contract
- Or: change consumer to explicitly request channels rather than assuming `[:-1]` slice

---

## 3. Attention Mask Mode Mapping

**Locations:**
- `context_manager.py` `SpanEmbedder._get_cached_mask()` line ~131
- `context_manager.py` `SpanUnembedder._get_cached_mask()` line ~344

**Pattern:**
```python
mode = self.attn_config.get('mode', 'full')
if mode == 'sliding':
    mode = 'local'  # build_encoder_mask expects 'local' not 'sliding'
```

**Why it exists:** Config uses `mode='sliding'` but `build_encoder_mask()` only accepts `'full'`, `'local'`, `'bigbird'`.

**Refactor suggestion:**
- Either rename config field to match expected values
- Or: add mapping inside `build_encoder_mask()` itself
- Or: define canonical mode enum used everywhere

---

## 4. BCE/MSE Loss Schedule Config

**Locations:**
- `config.py` `LossScheduleConfig` (for AE)
- `config.py` `DiffusionLossScheduleConfig` (for v-field)
- `losses.py` `scheduled_mse_bce_loss()`
- `losses.py` `scheduled_mse_bce_velocity_loss()`

**Pattern:**
```python
class SomeLossScheduleConfig:
    enabled: bool
    mse_start: float = 1.0
    bce_start: float = 0.0
    mse_end: float
    bce_end: float
    schedule: Literal["linear", "cosine", "step"]
    pct_switch: float = 0.8  # for step schedule
```

Both loss functions compute lerp weights identically:
```python
t = step / max(total_steps - 1, 1)
if schedule == 'linear':
    mse_weight = mse_start + t * (mse_end - mse_start)
    bce_weight = bce_start + t * (bce_end - bce_start)
# ... cosine, step variants
```

**Refactor suggestion:** Single `LossScheduleConfig` + `compute_schedule_weights(cfg, step, total_steps)` utility function.

---

## 5. Logsnr Decoder Initialization

**Locations:**
- `embedders.py` `ContextualPatchUnembedder.__init__()` (FourierScaleDecoder)
- `kmaze_ae/model_sparse_dim.py` `SparseAEPatchUnembedder.__init__()`
- `kmaze_ae/model_swiglu.py` `SwiGLUPatchUnembedder.__init__()`

**Pattern:**
```python
self.logsnr_decoder = nn.Linear(dim, 1)
with torch.no_grad():
    self.logsnr_decoder.weight.zero_()
    self.logsnr_decoder.bias.zero_()
```

**Why it exists:** Logsnr prediction should start near zero for training stability.

**Refactor suggestion:** `make_zero_init_linear(in_dim, out_dim)` factory function.

---

## 6. Per-Sample Stats Collection for SNR Binning

**Locations:**
- `train_denoise()` collects per-block
- `train_latent_diffusion()` collects per-sample in batch

**Required columns for `plot_multimetric_analysis()`:**
- `type == 'latent'` (for filtering)
- `logsnr` (for SNR binning)
- `loss` (for y-axis)
- `resolution` (for resolution plots)
- `source` (for per-source curves)

**Failure mode:** Missing/constant `logsnr` values = single-bin plots with one point.

**Refactor suggestion:** Document required DataFrame schema; add validation in plotting code that warns if logsnr has zero variance.

---

## Priority

1. **Deferred stats pattern** - Most duplicated, highest maintenance cost
2. **Unembedder output format** - Caused actual runtime bug, needs contract enforcement
3. **Loss schedule** - Two nearly-identical configs and functions
4. **Mode mapping** - Small but annoying
5. **Zero-init** - Minor, one-liner

---

## Notes

These patterns emerged from iterative feature addition without upfront design. The duplication was expedient but is now technical debt. Consolidation should happen before adding more AE variants or training modes.
