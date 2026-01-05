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

### Next Steps
- composite_002_scaled: 1050 steps, lr=1e-3, batch_size=8 (currently running)
  - 5x more steps
  - 2x learning rate
  - 2x batch size
