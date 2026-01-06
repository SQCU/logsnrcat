# Composite Multi-Reward REINFORCE Results

## Run: composite_001
- **Steps**: 210
- **Learning Rate**: 5e-4
- **Batch Size**: 4 (server default)
- **LoRA Rank**: 8
- **Trust Region Weight**: 2.0
- **Ablation Rate**: 100%

### Architecture
Multi-group REINFORCE with four reward signals:
1. **JS Divergence** - histogram color distribution matching
2. **Joint MSE** - full reconstruction quality
3. **Wavelet-only MSE** - reconstruction with amplitude subspace ablated
4. **Amplitude-only MSE** - reconstruction with wavelet subspace ablated

Trust region constraint: penalize if joint MSE degrades from baseline.

### Results

| Metric | Start | End | Baseline | Δ vs Start | Δ vs Baseline |
|--------|-------|-----|----------|------------|---------------|
| JS Divergence | 0.437 | 0.308 | 0.434 | **+0.129** | **-0.126 (29%)** |
| MSE Joint | 0.0571 | 0.0535 | 0.0569 | +0.0036 | -0.0034 (6%) |
| MSE Wavelet-only | 0.0676 | 0.0518 | 0.0676 | **+0.0158** | **-0.0158 (23%)** |
| MSE Amplitude-only | 0.0704 | 0.0508 | 0.0695 | **+0.0196** | **-0.0187 (27%)** |

### Key Observations

1. **All four reward groups improved simultaneously** - the multi-objective optimization successfully balanced competing objectives.

2. **JS divergence improved 29%** - histogram color matching significantly better. Policy reconstructions have color distributions much closer to originals.

3. **Subspace ablation recovery strong**:
   - Wavelet-only: 23% MSE reduction
   - Amplitude-only: 27% MSE reduction
   - These improvements suggest the adapter learned to encode information more robustly across both subspaces.

4. **Trust region constraint worked** - Joint MSE improved 6% rather than degrading. The constraint prevented the adapter from sacrificing reconstruction quality for histogram matching.

5. **Advantage dynamic range** - Multi-group rewards provide richer gradient signals than single-task REINFORCE. Each group can independently signal improvement direction.

### Comparison to Single-Task Runs

| Run | Task | Steps | LR | Improvement |
|-----|------|-------|-----|-------------|
| subspace_001 | Subspace MSE only | 120 | 5e-5 | Joint +0.0025, Wav +0.0029, Amp +0.0013 |
| subspace_002 | Subspace MSE only | 120 | 5e-4 | Joint +0.0130, Wav +0.0139, Amp +0.0134 |
| run_250_v2 | JS divergence only | 250 | 1e-4 | JS +0.085 (0.390→0.305) |
| **composite_001** | **All combined** | 210 | 5e-4 | **JS +0.129, Joint +0.0036, Wav +0.0158, Amp +0.0196** |

The composite run achieved better JS improvement than the JS-only run (+0.129 vs +0.085) while also improving all subspace metrics.

### Files Generated
- `composite_composite_001_curves.png` - Training curves (3x3 grid)
- `composite_composite_001_step{000,030,...,180}.png` - Comparison grids at checkpoints

---

## Proper LoRA Implementation

The original composite_001 used a "code space adapter" - a low-rank transformation applied to latent codes between encode/decode. This was NOT actual LoRA (Low-Rank Adaptation), which injects trainable `W + BA` into specific weight matrices.

Proper LoRA was implemented with:
- `LoRALinear` class wrapping `nn.Linear` modules
- `inject_lora()` to patch target modules by name pattern
- `lora_disabled()` context manager for baseline comparison

---

## Run: lora_decoder_001 (Decoder-only LoRA)

- **Steps**: 200
- **Learning Rate**: 5e-4
- **LoRA Rank**: 8
- **Trust Region Weight**: 2.0
- **LoRA Targets**: Decoder only (60 modules, 375K params)
  - `decoders.{0,1}.transformer.layers.*` (attention + MLP)
  - `decoders.{0,1}.wav_embed`, `amp_embed`

### Results

| Metric | Start | End | Change |
|--------|-------|-----|--------|
| JS Divergence | 0.479 | 0.246 | **-0.234 (49%)** |
| MSE Joint | 0.067 | 0.078 | +0.011 (17% worse) |
| MSE Wavelet | 0.079 | 0.060 | -0.020 (25%) |
| MSE Amplitude | 0.067 | 0.063 | -0.004 (6%) |

### Observations

- JS divergence improved dramatically (49%) - much better than code-space adapter
- **Joint MSE degraded** - trust region weight=2.0 insufficient
- Decoder-only LoRA is too expressive - can sacrifice reconstruction for histogram matching
- Subspace MSEs still improved

---

## Run: lora_full_001 (Encoder + Decoder LoRA)

- **Steps**: 200
- **Learning Rate**: 5e-4
- **LoRA Rank**: 8
- **Trust Region Weight**: 5.0 (increased from 2.0)
- **LoRA Targets**: Encoder + Decoder (124 modules, 782K params)
  - `encoders.{0,1}.transformer.layers.*`
  - `encoders.{0,1}.wav_code_proj`, `amp_code_proj`
  - `encoders.{0,1}.amplitude_proj`, `wavelet_proj`
  - `decoders.{0,1}.transformer.layers.*`
  - `decoders.{0,1}.wav_embed`, `amp_embed`

### Results

| Metric | Start | End | Change |
|--------|-------|-----|--------|
| JS Divergence | 0.310 | 0.283 | -0.026 (8%) |
| MSE Joint | 0.099 | 0.057 | **-0.042 (42%)** |
| MSE Wavelet | 0.063 | 0.046 | -0.018 (28%) |
| MSE Amplitude | 0.061 | 0.048 | -0.013 (21%) |

### Observations

- **All metrics improved** - no trade-off between objectives
- MSE improved 42% (vs 17% degradation in decoder-only)
- Higher trust weight (5.0) successfully constrained optimization
- Encoder LoRA allows learning better representations, not just better decoding
- JS improvement smaller (8% vs 49%) but reconstruction quality preserved

---

## Comparison: Code-Space Adapter vs Proper LoRA

| Approach | JS Δ | MSE Joint Δ | Params | Notes |
|----------|------|-------------|--------|-------|
| Code-space adapter | +0.129 (29%) | +0.004 (6%) | ~4K | Limited expressivity |
| Decoder-only LoRA | +0.234 (49%) | -0.011 (worse) | 375K | Too much freedom |
| **Encoder+Decoder LoRA** | +0.026 (8%) | **+0.042 (42%)** | 782K | Balanced improvement |

**Key insight**: Encoder LoRA is critical. Decoder-only LoRA can overfit to histogram matching at the cost of reconstruction. With encoder adaptation, the model learns better latent representations that satisfy both objectives.
