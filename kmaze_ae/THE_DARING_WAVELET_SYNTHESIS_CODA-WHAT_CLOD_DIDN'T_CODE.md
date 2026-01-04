# Wavelet-Amplitude Subspace Routing for FSQ Autoencoder

## Context

We have a SwiGLU FSQ autoencoder in `model_swiglu.py` that uses sparse binary codes with top-k dimension selection. The current implementation includes a wavelet-gated decoder (`WaveletGatedDecoder`) that uses a sigmoid gate to blend between wavelet and amplitude reconstruction pathways.

This design has a fundamental redundancy: the FSQ sparsity mechanism already implements discrete routing through dimension selection, but we're adding a continuous sigmoid blend on top. The sigmoid gate is a continuous relaxation of something the discrete system can express more elegantly.

## The Core Insight

If we partition the code dimensions into subspaces (wavelet subspace, amplitude subspace), then the sparsity mask's distribution across subspaces **already encodes pathway selection**. With k=6 active dimensions split between two subspaces:

- 6 wavelet, 0 amplitude → pure frequency-domain reconstruction
- 0 wavelet, 6 amplitude → pure spatial reconstruction  
- 3 wavelet, 3 amplitude → balanced mixed representation

This is a **discretized rotation angle** through representation space, chosen per-patch by the sparsity router. No sigmoid blend needed.

## Files to Modify

- `model_swiglu.py`: Primary implementation file

## Required Changes

### 1. Remove Redundant Sigmoid Gating

The `WaveletGatedDecoder` class (around line 614) should be refactored to remove the sigmoid gate blend. Instead of:
```python
gate = torch.sigmoid(self.gate_router(h))
patches = gate * wavelet_recon + (1 - gate) * amplitude_recon
```

The decoder should sum contributions from each subspace, where the sparsity pattern determines the effective contribution magnitude.

### 2. Implement Subspace-Aware Code Structure

The decoder needs to know which code dimensions belong to which subspace. Add configuration for subspace partitioning:
```python
def __init__(self, code_dim, hidden_dim, patch_dim, n_wavelet_dims=None, ...):
    # Default: split evenly
    self.n_wavelet_dims = n_wavelet_dims or code_dim // 2
    self.n_amplitude_dims = code_dim - self.n_wavelet_dims
```

### 3. Separate Embedding Projections Per Subspace

Replace the single `input_proj` with per-subspace projections:
```python
self.wav_embed = nn.Linear(self.n_wavelet_dims, hidden_dim)
self.amp_embed = nn.Linear(self.n_amplitude_dims, hidden_dim)
```

The forward pass splits codes and embeds each subspace independently before combining for the transformer.

### 4. Decoder Output as Sum, Not Blend

Both output heads (wavelet coefficients → IDWT → pixels, and direct amplitude → pixels) produce outputs that are summed:
```python
wav_pixels = batch_idwt2d(self.wav_head(h))
amp_pixels = self.amp_head(h)
return wav_pixels + amp_pixels
```

When a subspace has all-zero codes, its contribution naturally vanishes.

### 5. Encoder Subspace Routing Statistics

The encoder's sparsity module should track how many active dimensions fall in each subspace. This replaces the sigmoid gate value as the diagnostic for pathway preference:
```python
# After top-k selection
wav_active = hard_mask[..., :self.n_wavelet_dims].sum(dim=-1)
amp_active = hard_mask[..., self.n_wavelet_dims:].sum(dim=-1)
```

Return these statistics for logging/visualization.

### 6. Fix Existing Issues in Wavelet Utilities

While refactoring, fix these issues in `batch_dwt2d` and `batch_idwt2d`:

- **Hardcoded `C = 3`**: Infer channel count from input dimensions
- **Module instantiation in functions**: Move `HaarDWT2d` and `HaarIDWT2d` to be module attributes of the decoder/encoder classes

### 7. Update WaveletGatedEncoder Similarly

The `WaveletGatedEncoder` (line 499) also has a sigmoid gate blending wavelet features with raw patch features at the input. Apply the same subspace principle: let the encoder learn to route information through code subspaces rather than blending input representations.

However, the encoder case is different—it's about input features, not output reconstruction. Consider whether encoder-side wavelet features are still valuable as a fixed preprocessing step (DWT on patches as additional input channels) without gating.

## New Class Structure
```
SubspaceRoutedDecoder:
    - n_wavelet_dims, n_amplitude_dims (partitioning)
    - wav_embed, amp_embed (separate input projections)
    - transformer (shared processing)
    - neighbor_head (SwiGLU spatial gathering)
    - wav_head → IDWT → wavelet contribution
    - amp_head → amplitude contribution
    - forward: split codes → embed each → combine → transform → dual heads → sum
```

## Logging and Diagnostics

Replace gate value logging with subspace routing statistics:

- `wav_active_mean`: Mean active dims in wavelet subspace per patch
- `amp_active_mean`: Mean active dims in amplitude subspace per patch
- `routing_entropy`: Entropy of the active-dim distribution across subspaces (detects collapse)

## Regularization

Add optional entropy regularization to prevent subspace collapse (all codes routing to one subspace):
```python
# Encourage diverse subspace usage
p_wav = wav_active / k  # fraction in wavelet subspace
p_amp = amp_active / k  # fraction in amplitude subspace
routing_entropy = -(p_wav * log(p_wav + eps) + p_amp * log(p_amp + eps))
# Maximize entropy → encourage balanced routing when appropriate
```

This should be gentle—we want the network to choose freely, but not collapse entirely.

## Testing

After implementation:

1. Verify that all-zeros in one subspace produces near-zero contribution from that pathway
2. Check that routing statistics vary across patch content (edges vs smooth regions)
3. Compare reconstruction quality to the sigmoid-gated version
4. Monitor for subspace collapse during training

## Design Principle

The guiding principle: **don't add continuous gates on top of discrete routing—make the discrete routing expressive enough to encode the choice**. The sparsity mechanism is already a learned router; we're just making it subspace-aware rather than treating all code dimensions as interchangeable.