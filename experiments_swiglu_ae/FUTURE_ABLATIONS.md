# Future Ablation Ideas for SwiGLU FSQ Autoencoder

## Cross-Residual Hidden State Attention

### Current Architecture (Independent Levels)

Each residual level in `SwiGLUFSQAutoencoder` has:
- Independent encoder transformer
- Independent decoder transformer
- Independent register tokens (if using bigbird/gemma_bigbird)
- No cross-level attention - levels only see their own hidden states

Residual connection between levels is purely additive:
```
level_0_recon = decoder_0(encoder_0(patches))
level_1_recon = decoder_1(encoder_1((patches - level_0_recon.detach()) * scale))
...
```

### Proposed: Cross-Residual Attention

Allow higher residual levels to attend to hidden states from lower levels. This enables:
- Refinement levels to see what coarse levels already captured
- Gradual information flow from coarse → fine
- Better coordination of what each level encodes

### Residual-Causal Masking Pattern

The key insight is that cross-residual attention should be **residual-causal**:
- Level N can attend to levels 0, 1, ..., N-1 (lower = coarser)
- Level N cannot attend to levels N+1, N+2, ... (future refinements)
- Within each level, use gemma_bigbird (local + registers)

This prevents information leakage from fine→coarse while allowing coarse→fine context.

### Implementation Sketch

1. **Topology Embedding**: Extend RnRoPE coordinates to 4D: `[highway, spatial_x, spatial_y, level]`
   - See `render_latent_topology_embeddings()` in `src/context_manager.py`
   - Level coordinate enables cross-level position encoding

2. **Residual-Causal Mask**: New mask mode `'residual_blockcausal'`
   - Within-level: gemma_bigbird pattern (local + registers)
   - Cross-level: causal (level N sees level 0..N-1 only)
   - Registers per level act as level-local memory

3. **Hidden State Stacking**: Instead of independent transformers per level:
   ```
   # Current: separate encode/decode per level
   codes_0 = encoder_0(patches)
   codes_1 = encoder_1(residual_1)

   # Proposed: stacked hidden states with cross-attention
   h_0 = encoder.layer_0(patches)  # level 0 hidden states
   h_1 = encoder.layer_1(residual_1, cross_attend=h_0)  # level 1 sees level 0
   ```

### Graph Distance Alternative

An alternative to strict level ordering is **graph/manifold distance**:
- Treat (spatial_x, spatial_y, level) as 3D coordinates
- Attention weight decays with distance in this space
- Allows soft locality that respects both spatial AND level proximity

See `render_latent_topology_embeddings()` for how spatial+level coordinates could be rendered.

### Why This Is a Future Ablation

Per discussion with co-contributors: cross-attending to hidden states (not just codes) "gets really funky" and can destabilize training. The current independent-level design is stable and working.

Recommended ablation path:
1. First verify gemma_bigbird works well independently per level (current change)
2. Then try `residual_blockcausal` mode as a controlled ablation
3. Compare reconstruction quality, training stability, codebook utilization

### Config Key

If implemented, would add new mode to `attn_config`:
```python
attn_config = {
    'mode': 'residual_blockcausal',  # Cross-residual with causal masking
    'window_size': 2,
    'n_global_tokens': 4,
    'bigbird_layout': [2, 2],
    # ... existing fields
}
```

This would change `TransformerEncoder`/`TransformerDecoder` to accept hidden states from previous levels as cross-attention context.

---

## Other Potential Ablations

### Window Size Sweep
Current: `window_size: 2` (very local)
Ablate: `window_size: 3, 4, 5` with same gemma_bigbird layout
Hypothesis: Larger window reduces tiling, but more computation

### Layout Variations
Current: `bigbird_layout: [2, 2]` (2 local, 2 bigbird per cycle)
Ablate: `[1, 1]`, `[3, 1]`, `[1, 3]`
Hypothesis: More bigbird layers = better global coherence, slower

### Register Count
Current: `n_global_tokens: 4`
Ablate: `2, 8, 16`
Hypothesis: More registers = more global memory capacity

### Subspace-Specific Attention
Current: Same attention pattern for wavelet and amplitude pathways
Ablate: Different patterns per subspace (e.g., wavelet needs more local, amplitude more global)
