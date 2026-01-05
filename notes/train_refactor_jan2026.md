# train.py Refactoring: Stats Collection and Batch Preparation

**Date:** January 2026
**Files:** `src/train.py`, `src/embedders.py`, `src/optim_closure_bullshit.py`

## Executive Summary

Reduced `train.py` from 1322 to 979 lines (26% reduction) by:
1. Extracting optimizer boilerplate to `TrainingContext` pattern
2. Unifying stats collection with `DeferredStatsCollector` class
3. Decoupling batch embedding preparation from model application

---

## Motivation

### The Original Problems

**1. Optimizer Ceremony Duplication**
Both `train_denoise` and `train_latent_diffusion` had identical blocks:
- GradScaler setup for fp16
- FP8 optimizer construction
- Autocast context management
- `zero_grad() → backward() → scaler.step() → scheduler.step()` dance

This was ~50 lines of boilerplate repeated twice, with subtle divergences creeping in.

**2. Stats Collection Sprawl**
Each training function built stats dicts inline:
```python
deferred_stats.append({
    'step': i,
    'source': getattr(block, 'source', 'unknown'),
    'type': 'latent',
    'loss_tensor': loss.detach(),
    'logsnr_tensor': target_l.mean().detach(),
    ...
})
```

Then at logging intervals:
```python
for stat in deferred_stats:
    converted = {}
    for key in stat:
        if key.endswith('_tensor'):
            converted[key[:-7]] = stat[key].item()
    ...
```

Same pattern, different key sets, copy-pasted everywhere.

**3. Latent Diffusion Forward Pass Conflation**
The forward pass mixed three concerns:
- Embedding creation (encode → noise → project)
- Model application (LDTformer forward)
- Loss computation

This made it impossible to reuse the embedding preparation for different purposes (e.g., AE-only gradients vs denoising gradients).

---

## Solutions Implemented

### 1. TrainingContext (optim_closure_bullshit.py)

A dataclass wrapping optimizer ceremony into closures:

```python
@dataclass
class TrainingContext:
    zero_grad: Callable[[], None]
    backward: Callable[[torch.Tensor], None]
    step: Callable[[], None]
    autocast: Callable[[], Any]  # Returns context manager

    # For introspection
    optimizer_group: OptimizerGroup
    fp8_optimizer: Optional[Any]
    scaler: Optional[torch.amp.GradScaler]
    dtype: torch.dtype
    device: torch.device
```

Usage in training loops:
```python
ctx = build_training_context(model, config, total_steps=steps)

for step in range(steps):
    ctx.zero_grad()
    with ctx.autocast():
        loss = compute_loss(...)
    ctx.backward(loss)
    ctx.step()
```

The closures internally handle:
- FP8 vs standard weights
- bf16 vs fp16 vs fp32 precision
- GradScaler for fp16
- Heterogeneous vs single optimizer
- Scheduler stepping

### 2. DeferredStatsCollector (train.py:62-96)

Class that collects stats with deferred tensor→float conversion:

```python
class DeferredStatsCollector:
    def __init__(self):
        self._entries = []

    def add(self, step: int, source: str, type: str,
            resolution: int = None, **tensor_stats):
        """Add a stat entry. Keys ending in '_tensor' converted on flush."""
        entry = {'step': step, 'source': source, 'type': type}
        if resolution is not None:
            entry['resolution'] = resolution
        entry.update(tensor_stats)
        self._entries.append(entry)

    def flush(self, extra_stats: dict = None) -> list:
        """Convert tensors to floats and return entries. Clears buffer."""
        converted_list = []
        for entry in self._entries:
            converted = {}
            for key, val in entry.items():
                if key.endswith('_tensor'):
                    base_key = key[:-7]
                    converted[base_key] = val.item() if isinstance(val, torch.Tensor) else val
                else:
                    converted[key] = val
            if extra_stats:
                converted.update(extra_stats)
            converted_list.append(converted)
        self._entries.clear()
        return converted_list
```

Key design decisions:
- `*_tensor` suffix convention signals deferred conversion
- `resolution` is optional (text samples don't have it)
- `flush()` accepts `extra_stats` for step-level metrics (variance tracker state, etc.)
- Clears buffer on flush to prevent memory growth

### 3. prepare_latent_batch() (embedders.py)

Decoupled function that prepares embeddings WITHOUT running the model:

```python
def prepare_latent_batch(
    group: list,
    grid_shape: Tuple[int, int],
    sparse_ae,
    patch_embedder,
    topo_config: dict,
    device: torch.device,
) -> dict:
    """
    Prepare batched embeddings for latent diffusion training.

    Returns dict with:
        imgs: [B, C, H, W] - stacked images
        codes: List[Tensor] - per-level codes from encoder
        pre_quant_flat: [B, N*L, code_dim] - flattened pre-quant codes
        noisy_codes: [B, N*L, code_dim] - after noise injection
        target_v: [B, N*L, code_dim] - velocity targets
        alpha, sigma: [B, 1, 1] - noise schedule coefficients
        h_input: [B, N*L, model_dim] - projected input for LDTformer
        topo_embeds: [N*L, model_dim] - topology embeddings
        latent_mask: BlockMask - SWA mask for attention
        decoder_masks: List[Tensor] - per-level decoder masks
        grid_shape, n_patches, n_levels, batch_size
    """
```

The calling code then explicitly applies the model:
```python
batch = prepare_latent_batch(group, grid_shape, sparse_ae, patch_emb, topo_config, device)

# Model application is explicit and separate
h = model.forward_latent_diffusion(
    batch['h_input'],
    topo_embeds=batch['topo_embeds'].unsqueeze(0).expand(B, -1, -1),
    block_mask=batch['latent_mask'],
)
v_pred = patch_unemb.latent_code_unproj(h)
```

**Why this decoupling matters:**
- The same `batch` dict can feed AE reconstruction loss (codes → decode → pixel loss)
- AND latent diffusion loss (h_input → model → v_pred → code-space loss)
- No duplicate encoding, no duplicate mask building

---

## Line Count Changes

| File | Before | After | Delta |
|------|--------|-------|-------|
| train.py | 1322 | 979 | -343 (26%) |
| embedders.py | ~400 | 549 | +149 |
| optim_closure_bullshit.py | 0 | 242 | +242 |
| **Net** | ~1722 | 1770 | +48 |

The net increase is misleading - we added significant new capability:
- `TrainingContext` eliminates future optimizer boilerplate
- `DeferredStatsCollector` is reusable for any training function
- `prepare_latent_batch` can serve multiple training modes

---

## Removed Code

1. **`convert_deferred_stats()`** - Replaced by `DeferredStatsCollector.flush()`
2. **`AEModule` / `LatentDiffusionModule`** - Wrapper classes that added indirection without value
3. **Inline variance tracker construction** - Replaced by `build_variance_tracker()` factory
4. **`calculate_global_max_resolution()`** - Dead code

---

## Usage Patterns

### Adding a New Training Function

```python
def train_new_thing(components, config, iterator):
    config = sanitize_config(config)
    model = components[0]
    device = config['device']

    # 1. Build context (handles all optimizer complexity)
    ctx = build_training_context(model, config, total_steps=steps)

    # 2. Initialize stats collector
    stats_collector = DeferredStatsCollector()
    history = []

    for i in range(steps):
        ctx.zero_grad()
        with ctx.autocast():
            # ... compute loss ...

            # Collect stats (tensors stay on GPU)
            stats_collector.add(
                step=i, source='my_source', type='my_type',
                loss_tensor=loss.detach(),
                custom_metric_tensor=metric.detach(),
            )

        ctx.backward(loss)
        ctx.step()

        # Flush at intervals (CPU sync happens here)
        if i % log_interval == 0:
            history.extend(stats_collector.flush())

    return pd.DataFrame(history)
```

### Reusing Batch Preparation

```python
# For joint AE + denoising training:
batch = prepare_latent_batch(group, grid_shape, sparse_ae, patch_emb, topo_config, device)

# AE reconstruction path (codes → decode → pixel loss)
recon = sparse_ae.decode(batch['codes'], grid_shape, batch['decoder_masks'])
ae_loss = F.mse_loss(recon, batch['imgs'])

# Latent diffusion path (h_input → model → v_pred → code loss)
h = model.forward_latent_diffusion(batch['h_input'], ...)
v_pred = patch_unemb.latent_code_unproj(h)
denoise_loss = F.mse_loss(v_pred, batch['target_v'])

# Both losses from same batch, no duplicate encoding
total_loss = ae_loss + denoise_loss
```

---

## Future Work

1. **Extend `prepare_latent_batch` for pixel diffusion** - Similar function for pixel-space denoising
2. **Generalize `DeferredStatsCollector`** - Could accept schema definition for type safety
3. **TrainingContext for multi-model** - Current design assumes single model; joint training may need extension

---

## Related Files

- `notes/latent_vs_pixel_diffusion.md` - Architectural distinction between diffusion spaces
- `notes/refactor_agenda.md` - Broader model.py decomposition plan
- `CLAUDE.md` - Configuration architecture principles (no defensive `.get()`)
