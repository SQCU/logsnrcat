# FSQ-SwiGLU Hyperparameter Tuning Strategy

## Our Implementation vs FSQ Paper

### Paper's FSQ Formula (arxiv 2309.15505)
```
z_i → round(⌊L/2⌋ * tanh(z_i))
```
- Maps continuous z to L discrete levels: {-⌊L/2⌋, ..., 0, ..., ⌊L/2⌋}
- Codebook size = product of levels across d dimensions
- Example: d=4 dims with L=[8,6,5,5] → 8×6×5×5 = 1200 codes
- Paper targets codebook sizes matching VQ-VAE: ~2^10 to 2^14

### Our BinaryFSQ Implementation
```python
soft = sigmoid(x)
hard = (soft > 0.5).float()  # {0, 1}
codes = hard * 2 - 1         # {-1, +1}
return hard - soft.detach() + soft  # STE
```

**Key difference**: We use L=2 (binary) per dimension, paper uses L=3-8.

### Effective Codebook Analysis

| Config | Formula | Effective Codes |
|--------|---------|-----------------|
| Paper: d=6, L=[8,8,8,5,5,5] | 8³×5³ | 64,000 |
| Paper: d=4, L=[8,5,5,5] | 8×5³ | 1,000 |
| **Ours: code_dim=128, k=4, binary** | C(128,4)×2⁴ | ~10.7M theoretically |

Our sparse binary approach creates a *combinatorially larger* implicit codebook than paper's dense multi-level approach. This is both a strength (expressivity) and weakness (harder to learn, no structure).

---

## Implementation Correctness Check

### What we do correctly:
1. **STE gradient flow**: `hard - soft.detach() + soft` matches paper
2. **Per-dimension quantization**: Each dim quantized independently
3. **No auxiliary losses**: No commitment loss, codebook collapse handling

### Potential issues:
1. **Binary is extreme**: L=2 is minimum granularity. Paper typically uses L=5-8.
2. **Sparsity interaction**: Paper doesn't combine FSQ with top-k sparsity
3. **tanh vs sigmoid**: Paper uses tanh (symmetric around 0), we use sigmoid (asymmetric)

### Recommended fix for multi-level FSQ:
```python
class MultilevelFSQ(nn.Module):
    """FSQ with configurable levels per dimension (paper's approach)."""
    def __init__(self, levels: List[int]):
        super().__init__()
        # levels = [8, 5, 5, 5] means 4 dims with those level counts
        self.levels = levels
        self.d = len(levels)
        # Half-levels for scaling
        self.register_buffer("half_L", torch.tensor([L // 2 for L in levels]))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [..., d] last dim must match len(levels)
        # Bound with tanh, scale by half_L, round
        bounded = torch.tanh(z) * self.half_L
        quantized = torch.round(bounded)
        # STE
        return quantized - bounded.detach() + bounded
```

---

## Hyperparameter Search Strategy

### Phase 1: Validate Binary FSQ Baseline (Current)
Run current config with metrics tracking:
```toml
[training.sparse_ae]
ae_type = "swiglu"
code_dim = 128
k_per_patch = 4      # ~97% sparsity
n_levels = 6         # residual hierarchy levels
```

**Metrics to track:**
- Reconstruction MSE per level
- Codebook utilization (unique codes seen)
- Gradient magnitudes through FSQ
- Sparsity actual vs target

### Phase 2: Level Count Sweep
The paper finds diminishing returns beyond ~2^12 codes. Test:

| Experiment | n_levels | code_dim | k_per_patch | Theoretical codes |
|------------|----------|----------|-------------|-------------------|
| sparse-4   | 4        | 128      | 4           | C(128,4)×2⁴×4 levels |
| sparse-6   | 6        | 128      | 4           | baseline |
| sparse-8   | 8        | 128      | 4           | more levels |
| dense-6    | 6        | 64       | 64 (all)    | 2⁶⁴×6 (no sparsity) |

### Phase 3: Sparsity vs Density Tradeoff
Key question: Is sparse binary better than dense multi-level?

| Experiment | code_dim | k | L (levels) | Style |
|------------|----------|---|------------|-------|
| sparse-binary | 128 | 4 | 2 | Current |
| sparse-3bit | 128 | 4 | 8 | 3-bit + sparse |
| dense-binary | 32 | 32 | 2 | No sparsity, binary |
| dense-multiL | 8 | 8 | [8,6,5,5,5,5,5,5] | Paper style |

### Phase 4: Architecture Interaction
Test how FSQ interacts with:
1. **Residual scale**: 1.0, 2.0, 4.0 (affects level independence)
2. **Decoder complexity**: n_layers 2, 4, 6
3. **Hidden dim**: 128, 256, 512

---

## Config Templates for Experiments

### Experiment A: Paper-style dense FSQ
```toml
[training.sparse_ae]
ae_type = "swiglu"
n_levels = 6
patch_size = 16
hidden_dim = 256
code_dim = 8          # Only 8 dims!
k_per_patch = 8       # Keep all (no sparsity)
fsq_levels = [8, 6, 5, 5, 5, 5, 5, 5]  # NEW: multi-level config
residual_scale = 2.0
n_layers = 4
```
Codebook: 8×6×5⁶ = 750,000 codes

### Experiment B: Hybrid sparse multi-level
```toml
[training.sparse_ae]
ae_type = "swiglu"
n_levels = 6
code_dim = 32
k_per_patch = 8       # 25% density
fsq_levels = 8        # 3-bit per dim (if uniform)
residual_scale = 2.0
```
Codebook: C(32,8)×8⁸ = massive

### Experiment C: Current baseline (reference)
```toml
[training.sparse_ae]
ae_type = "swiglu"
n_levels = 6
code_dim = 128
k_per_patch = 4       # ~3% density
# fsq_levels = 2 (implicit binary)
residual_scale = 2.0
```

---

## Metrics Dashboard

For each experiment, log:
```python
metrics = {
    'recon_mse': final_reconstruction_error,
    'codebook_usage': unique_codes / theoretical_max,
    'level_contrib': [mse_reduction_per_level],
    'sparsity_actual': (codes == 0).mean(),
    'grad_norm_fsq': gradient_magnitude_through_quantizer,
    'bits_per_pixel': code_dim * k * log2(L) / (patch_size^2 * 3),
}
```

**Target**: Match or beat VQ-VAE reconstruction at similar bits/pixel.

---

## Decision Tree

```
START
  │
  ├─ Q: Is codebook utilization < 10%?
  │    YES → Reduce code_dim or increase k
  │    NO  → Continue
  │
  ├─ Q: Is early level contributing < 5% reconstruction?
  │    YES → Reduce n_levels or increase residual_scale
  │    NO  → Continue
  │
  ├─ Q: Are gradients through FSQ vanishing?
  │    YES → Try multi-level FSQ (smoother STE)
  │    NO  → Binary is fine
  │
  └─ Q: Is reconstruction good but latent diffusion poor?
       YES → Latent space too discontinuous, try dense FSQ
       NO  → Current config is working
```

---

## References

- FSQ Paper: https://arxiv.org/abs/2309.15505
- Key insight: FSQ achieves VQ-VAE quality without codebook collapse
- Our extension: Sparse FSQ trades codebook structure for combinatorial expressivity
