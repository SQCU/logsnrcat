# REINFORCE Histogram Fidelity Experiment

**Date:** 2026-01-04
**Goal:** Fine-tune autoencoder via policy gradient to improve color histogram fidelity on Vaporeon sprites

## Overview

Used REINFORCE-style policy optimization with LoRA adapters to improve histogram divergence between input images and AE reconstructions. The hypothesis: AEs can "cheat" by reconstructing backgrounds well while losing foreground sprite color information. Histogram divergence serves as a verification signal for color fidelity.

## Eval Server API Pattern

The experiment used a remote eval server running on Windows (accessible from WSL at `172.26.160.1:8421`) with the trained model loaded. This pattern enables:

- **Live model probing** without restarting training
- **Iterative experimentation** with immediate feedback
- **State persistence** across API calls via `ctx` object

### Key Endpoints

```python
# Health check
GET /health -> {"status": "ok", "weights_loaded": true, "params": 332930337}

# Execute arbitrary Python code with model access
POST /eval {"code": "..."} -> {"success": true, "result": ..., "error": null}
```

### Execution Pattern

```python
def eval_code(code: str, host: str, port: int, timeout: int = 180) -> dict:
    url = f"http://{host}:{port}/eval"
    resp = requests.post(url, json={"code": code}, timeout=timeout)
    return resp.json()

# Multi-line code via f-strings, results stored in ctx._*
setup_code = f'''
import torch
# ... setup code ...
ctx._my_result = computed_value
"Setup complete"  # Last expression becomes result
'''
result = eval_code(setup_code, host, port)

# Fetch stored results separately (exec returns "executed", not values)
data = eval_code("ctx._my_result", host, port)['result']
```

### Gotchas

1. **exec vs eval**: Multi-line code uses `exec()` which returns `None`/"executed". Store results in `ctx._*` and fetch separately.
2. **JSON serialization**: NumPy arrays must be `.tolist()` before returning through API
3. **State persistence**: `ctx` object persists between calls - useful for iterators, optimizers, history tracking
4. **Model modification persists**: LoRA wrappers stay attached to model between runs. Design for reinitialization.

## LoRA Adapter Pattern

### Target Selection

Targeted projection layers in encoder/decoder transformers:
```python
target_patterns = [
    "encoders.0.amplitude_proj",      # Input projection (768 -> 256)
    "encoders.0.wavelet_proj",        # Input projection
    "encoders.0.transformer.layers.0.attn.out_proj",  # Attention output
    "encoders.0.transformer.layers.1.attn.out_proj",
    "decoders.0.wav_embed",           # Code embedding (64 -> 256)
    "decoders.0.amp_embed",
    "decoders.0.transformer.layers.0.attn.out_proj",
    "decoders.0.transformer.layers.1.attn.out_proj",
]
```

**Why these layers:**
- Projection layers have most impact on information flow
- Attention output projects combined context back to residual stream
- Embedding layers control how codes are interpreted

### LoRA Implementation

```python
class LoRALinear(nn.Module):
    """Wraps a frozen linear layer with trainable LoRA adapter."""
    def __init__(self, base_layer, rank=8):
        super().__init__()
        self.base = base_layer
        self.rank = rank

        # LoRA decomposition: W' = W + B @ A
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Freeze base layer
        for p in self.base.parameters():
            p.requires_grad = False

        self.enabled = True  # Toggle for reference comparison

    def forward(self, x):
        base_out = self.base(x)
        if self.enabled:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
            return base_out + lora_out
        return base_out
```

**Key design choices:**
- `enabled` toggle allows reference model comparison (disable LoRA for baseline)
- B initialized to zeros = adapter starts as identity
- A initialized with small random values for symmetry breaking
- Scale=1.0 (full contribution) - constraint via loss, not scaling

### Reinitialization Pattern

```python
# Handle already-wrapped layers from previous runs
for name, module in ae.named_modules():
    if name in target_patterns:
        if hasattr(module, 'lora_A'):  # Already wrapped
            module.lora_A.data.normal_(0, 0.01)
            module.lora_B.data.zero_()
            module.enabled = True
            ctx._lora_layers[name] = module
        elif isinstance(module, nn.Linear):  # Fresh - wrap it
            # ... wrap with LoRALinear ...
```

## Loss Functions

### Histogram Divergence (Reward Signal)

Non-differentiable JS divergence computed on CPU for reward:
```python
def compute_js_divergence(imgs, recons, n_bins=32):
    js_total = 0.0
    for c in range(3):  # RGB channels
        hist_in, _ = np.histogram(imgs[:, c].flatten(), bins=n_bins, range=(0, 1))
        hist_re, _ = np.histogram(recons[:, c].flatten(), bins=n_bins, range=(0, 1))
        # Normalize to probability
        hist_in = hist_in / (hist_in.sum() + 1e-10)
        hist_re = hist_re / (hist_re.sum() + 1e-10)
        # JS divergence
        m = 0.5 * (hist_in + hist_re)
        kl_pm = np.sum(hist_in * np.log(hist_in / m + 1e-10))
        kl_qm = np.sum(hist_re * np.log(hist_re / m + 1e-10))
        js_total += 0.5 * kl_pm + 0.5 * kl_qm
    return js_total / 3.0
```

### Soft Histogram Loss (Differentiable)

Gaussian kernel approximation for gradient flow:
```python
def soft_histogram_loss(img, ref, n_bins=32, sigma=0.05):
    bins = torch.linspace(0, 1, n_bins, device=img.device)
    total_loss = 0.0
    for c in range(3):
        img_flat = img[:, c].reshape(-1)
        ref_flat = ref[:, c].reshape(-1)
        # Soft bin assignment via Gaussian
        img_dists = (img_flat.unsqueeze(1) - bins.unsqueeze(0)) / sigma
        img_hist = torch.exp(-0.5 * img_dists ** 2).sum(0)
        img_hist = img_hist / (img_hist.sum() + 1e-8)
        # ... same for ref_hist ...
        # JS divergence (differentiable)
        m = 0.5 * (img_hist + ref_hist)
        kl_pm = (ref_hist * torch.log(ref_hist / m + 1e-8)).sum()
        kl_qm = (img_hist * torch.log(img_hist / m + 1e-8)).sum()
        total_loss += 0.5 * kl_pm + 0.5 * kl_qm
    return total_loss / 3
```

### Combined Loss (PPO-style)

```python
# Trust region constraints
kl_codes = F.mse_loss(codes_lora_flat, codes_ref_flat.detach())  # Code drift
mse_ref = F.mse_loss(recon_lora, recon_ref.detach())  # Output drift
recon_mse = F.mse_loss(recon_lora, images)  # Reconstruction quality

# REINFORCE advantage weighting
reward_weight = 1.0 + max(0, advantage * 2.0)

# Final loss
loss = (reward_weight * hist_loss
        + kl_weight * (kl_codes + mse_ref)
        + mse_weight * recon_mse)
```

**Finding:** MSE constraint (`mse_weight`) more effective than trust region (`kl_weight`) for this task. The reference model isn't necessarily good at histogram fidelity, so staying close to it isn't helpful.

## Super-REINFORCE (Double Autoencoding)

Policy rollout simulation - encode/decode twice to penalize compounding errors:

```python
# First pass
codes_lora = ae.encode(images, ...)
recon_lora = ae.decode(codes_lora, ...)

# Second pass (rollout)
if double_ae:
    codes_double = ae.encode(recon_lora, ...)
    recon_double = ae.decode(codes_double, ...)

    # Penalize round-trip error
    double_mse = F.mse_loss(recon_double, images)
    double_hist = soft_histogram_loss(recon_double, images)
    loss += double_weight * (double_mse + double_hist)
```

**Rationale:** If the AE compounds errors, the double-pass reconstruction will be much worse. Penalizing this encourages more stable representations.

## Visualization Patterns

### Color Histogram Ribbon

Compact visual representation of color distribution:
```python
def make_color_ribbon(img, height=64, width=16, n_bins=32):
    """Colors occupy vertical space proportional to frequency."""
    ribbon = np.zeros((height, width, 3))
    pixels = img.reshape(-1, 3)

    # Bin and count colors
    binned = (np.clip(pixels, 0, 1) * (n_bins - 1)).astype(int)
    color_counts = {}
    for p in binned:
        key = tuple(p)
        color_counts[key] = color_counts.get(key, 0) + 1

    # Sort by frequency, fill ribbon proportionally
    sorted_colors = sorted(color_counts.items(), key=lambda x: -x[1])
    total = sum(c for _, c in sorted_colors)
    y = 0
    for (r, g, b), count in sorted_colors:
        h = max(1, int(height * count / total))
        color = np.array([r, g, b]) / (n_bins - 1)
        ribbon[y:y+h, :, :] = color
        y += h
        if y >= height:
            break
    return ribbon
```

**Usage:** Place thin ribbon next to each image (original, baseline, policy) to show distributional differences at a glance.

### Comparison Grid Layout

```python
# 4 samples x 6 columns: [image, ribbon] x 3 conditions
fig, axes = plt.subplots(4, 6, figsize=(12, 12),
                          gridspec_kw={'width_ratios': [4, 1, 4, 1, 4, 1]})
```

### Run Numbering

```python
parser.add_argument("--run-id", type=str, default=None)
if args.run_id is None:
    args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

output_path = output_dir / f"reinforce_ppo_{args.run_id}{suffix}.png"
```

Enables systematic comparison across hyperparameter sweeps.

## Results Summary

| Run | Config | Start JS | End JS | Improvement | End MSE |
|-----|--------|----------|--------|-------------|---------|
| run001 | KL=0.01, MSE=1.0 | 0.31 | 0.20 | +0.11 | 0.09 |
| run002 | KL=0.0, MSE=2.0 | 0.38 | 0.21 | +0.17 | 0.09 |
| run003 | Super-REINFORCE | 0.24 | **0.17** | +0.07 | 0.09 |
| run004 | KL=0.0, MSE=1.0 | 0.29 | 0.19 | +0.10 | 0.10 |

**Key findings:**
1. **MSE constraint > trust region** for this task
2. **Super-REINFORCE achieves best absolute JS** (0.17)
3. **All configs improve** histogram fidelity while maintaining reconstruction quality
4. **LoRA on projection layers** is effective intervention target

## Vaporeon Sampling

Used `SpriteAtlasIterator` with logit adjustments for Vaporeon-biased sampling:
```python
sampling_config = {
    "split": "all",
    "mode": "uniform_sprites",
    "adjustment_mode": "additive",
    "adjustments": {"134": 1.0, "*.134": 1.0}  # +1 logit bias
}
```

**Note:** +10 bias too strong (mostly 134.134 fusions). +1 gives good mix of 134.X and X.134 fusions with more color diversity.

## Files

- `scripts/reinforce_ppo_style.py` - Main training script
- `scripts/probe_histogram_divergence.py` - Histogram analysis tool
- `scripts/reinforce_histogram.py` - Earlier iteration (code-space adapters)
- `experiments_swiglu_ae/main_run_091/reinforce_ppo_*.png` - Output plots

## Future Directions

1. **Gradient accumulation** for larger effective batch size (reduce variance)
2. **Fixed validation set** for more stable evaluation
3. **Value estimator LoRA** for actor-critic style updates
4. **Foreground masking** to focus histogram on sprite pixels, not background
5. **Per-image reward** instead of batch-level for finer credit assignment
