# Patch Review: clean_impl_swiglu_reference

Comparative analysis of the external patch against the reference implementation (`main.py`, `src/*.py`, `configs/sparse_ae_fractal.toml`).

---

## Data Flow Analysis: External Patch

### 1. 2D Rotary Position Embeddings

The external patch implements 2D RoPE in `get_2d_rope_freqs` (lines 40-72) using a straightforward half-split strategy: the head dimension is bisected, with the first half encoding x-coordinates and the second half encoding y-coordinates. Frequencies are computed via the standard `1 / (base ** (arange / half_dim))` formula, then outer-producted with grid positions to produce phase angles. The interleaving pattern `torch.stack([freqs, freqs], dim=-1).flatten(-2)` prepares for the standard rotate-half application. This design is clean and transparent but fundamentally committed to 2D spatial grids—there is no pathway for higher-dimensional or mixed-modality (temporal + spatial) topologies.

### 2. Post-SDPA Sigmoid Gating

The `GQATransformerLayerRoPE` (lines 100-154) introduces an unusual post-attention gate: after `scaled_dot_product_attention` computes the attention output, a per-head sigmoid gate is applied via `einsum('bhnd,hd->bhn', q, attn_gate) + bias`. This query-dependent gating modulates the attention contribution before the output projection. The mechanism is parameter-efficient (only `n_heads * head_dim + n_heads` parameters) and provides a learned "volume control" per head. The gate is applied elementwise to the attention output, creating a soft selection of which heads contribute to which positions.

### 3. SwiGLU FFN in Transformer Layers

Following attention, each `GQATransformerLayerRoPE` applies a SwiGLU feedforward: `w3(silu(w1(x)) * w2(x))` (line 153). The expansion ratio is 4x, creating a 3-projection structure where the gate path (`w1 + SiLU`) modulates the value path (`w2`) before compression (`w3`). This matches modern LLM conventions (LLaMA, Mistral) and provides better gradient flow than plain ReLU-gated variants.

### 4. 3x3 Neighbor Gathering with Reflect Padding

The `SwiGLUNeighborDecoder` (lines 210-267) implements spatial-aware decoding by gathering 3x3 neighborhoods around each patch. The implementation uses `F.pad(mode='reflect')` to handle boundaries, then iterates through grid positions to collect 9-neighbor tensors via slicing. This O(N) explicit loop is simple but not vectorized—each position appends to a list before `torch.stack`. The SwiGLU pattern is then applied to the concatenated 9*hidden_dim neighborhood: gate and value projections are computed from the full neighborhood, not just the center, giving the decoder access to local context during reconstruction.

### 5. MultiScaleRouter with 2-bit FSQ

The `MultiScaleRouter` (lines 270-318) implements dimension-level sparsity via learned logits. For each level, a topk selection determines which code dimensions are active, and the values at those dimensions undergo 2-bit quantization (4 levels mapped to {-1.5, -0.5, 0.5, 1.5}, then normalized to [-1, 1]). This is a *fixed* routing scheme—the same dimensions are selected for all images within a level. The router's `dim_logits` parameter learns which dimensions are important per level, while `level_values` learns their quantized magnitudes. Sparsity comes from masking: inactive dimensions are zeroed.

### 6. Binary FSQ with STE

The encoder's quantization (lines 423-426) uses binary FSQ: `sigmoid(logits) > 0.5` produces hard bits, then straight-through estimation (`hard - soft.detach() + soft`) preserves gradients. The hard values are mapped to [-1, 1] via `hard * 2 - 1`. This is simpler than multi-level FSQ and maximizes compression at the cost of reconstruction fidelity—each code dimension is 1 bit.

### 7. Hierarchical Residual Refinement

The multi-level structure (lines 401-454) follows a coarse-to-fine pattern: level 0 encodes the full image, subsequent levels encode `(image - cumulative_recon) * residual_scale`. The residual scaling amplifies differences for better gradient signal. Reconstruction accumulates additively, with the final output being the sum of all level contributions (each divided back by residual_scale except level 0).

### 8. K-Annealing Schedule

Training uses per-level exponential decay of sparsity (lines 484-491): `k = k_start * ((k_end / k_start) ** t)` over `anneal_steps`. This curriculum gradually increases compression, starting with more active dimensions (easier task) and progressively constraining to the target sparsity. The config shows 6 levels all annealing from k=64 to k=8 over 5000 steps.

---

## Comparative Analysis: Reference Implementation

### Contrast 1: RnRoPE vs 2D RoPE

The reference implementation (`src/rope.py:48-141`) uses RnRoPE—N-dimensional RoPE via Householder reflections. Rather than hardcoding a 2D split, it learns an orthogonal projection matrix through products of Householder reflections (`HouseholderOrthogonal`), then applies frequency bands proportionally across an arbitrary `topo_dim` (e.g., 3 for temporal highway + 2D spatial). This is dramatically more flexible: the same architecture handles text sequences, images, video, and mixed-modality contexts. The external patch's 2D RoPE is locked to image grids, while RnRoPE's learnable projection adapts to any topology.

### Contrast 2: Sigmoid Gate Placement

The reference implementation (`src/model.py:267-278`, `src/blocks.py:299-302`) applies a sigmoid gate to the attention output before the residual connection: `x = x + h * sigmoid(gate_proj(h))`. Crucially, this gates the *projected* attention output, not the raw SDPA result. The external patch instead computes the gate from queries and applies it post-SDPA but pre-output-projection (`attn_out = attn_out * gate` at line 146, then `out_proj(attn_out)` at line 149). The reference pattern modulates the residual contribution, while the external pattern modulates the attention values before they're mixed across heads.

### Contrast 3: MoE vs Dense SwiGLU

The reference uses `SigmoidMoE` (`src/blocks.py:62-143`) in transformer blocks—a mixture of SwiGLU experts with sigmoid routing. This provides capacity scaling without proportional compute: only `num_active` of `num_experts` are evaluated per token. The external patch uses dense SwiGLU throughout (no sparsity in the FFN). For the same parameter count, MoE offers higher effective capacity at the cost of load balancing complexity.

### Contrast 4: Per-Dimension Sparsity vs Router-Based Sparsity

The reference's `PerDimSparsity` (`kmaze_ae/model_sparse_dim.py:58-101`) learns per-patch, per-dimension gates: `gate_proj(latent)` produces dimension-wise logits, topk selects active dims *per patch*, and STE maintains gradients. This is content-dependent: different patches can activate different dimensions. The external patch's `MultiScaleRouter` uses *level-global* dimension selection—all patches at a level share the same active dimensions. The reference trades uniformity for adaptivity; the external trades adaptivity for simpler codebook structure.

### Contrast 5: 3-bit vs Binary FSQ

The reference uses `ThreeBitFSQ` (`kmaze_ae/model_sparse_dim.py:42-55`): 8 quantization levels via `round(sigmoid(x) * 7)`. This provides 3 bits per active dimension vs the external patch's 1 bit. Combined with the reference's per-dimension sparsity (e.g., k=4 of 128 dims active), the effective bit budget is ~12 bits per patch. The external patch with 6 levels × (8 active dims) × 1 bit = ~48 bits but less structured. The reference's approach is sparser but each active dimension carries more information.

### Consonance 1: Hierarchical Residual Structure

Both implementations share the fundamental hierarchical pattern: level 0 encodes the full signal, level n>0 encodes `(target - cumulative) * scale`, and reconstruction sums `decoded_n / scale` contributions. Both use `residual_scale` (2.0 in both configs) to amplify residuals for training. This coarse-to-fine decomposition is a core architectural choice that both respect.

### Consonance 2: Fourier Feature Conditioning

Both condition encoding on auxiliary signals via Fourier features. The reference's `SparseLevelEncoder` (`kmaze_ae/model_sparse_dim.py:104-147`) concatenates `logsnr_fourier(logsnr_patches)` to patch features before encoding—enabling SNR-aware compression. The external patch lacks logsnr conditioning, but both implementations use similar Fourier embedding patterns for continuous values.

### Consonance 3: Transformer-Based Encode/Decode

Both route patches through transformer layers before quantization and after dequantization. The external patch uses `TransformerBlockRoPE` with GQA; the reference uses `TransformerEncoder`/`TransformerDecoder` with configurable attention modes (full, sliding, bigbird, gemma_bigbird) and register tokens. The core insight—that transformer attention enables global context for patch-wise codes—is shared.

### Contrast 6: Attention Mode Flexibility

The reference's `TransformerEncoder` (`src/blocks.py:324-425`) supports multiple attention patterns: full O(n²), sliding window, BigBird (local + global registers), Gemma (alternating local/global), and hybrid gemma_bigbird. This enables efficiency-quality tradeoffs for different sequence lengths. The external patch hardcodes full attention via `scaled_dot_product_attention`—simple but O(n²) always.

### Contrast 7: LogSNR Prediction Head

The reference's decoder predicts per-patch logsnr (`logsnr_head` in `SparseLevelDecoder`), enabling the autoencoder to estimate noise levels at decode time. This supports latent diffusion workflows where the decoder must reconstruct both signal and noise metadata. The external patch's decoder outputs only pixels—no auxiliary predictions.

### Contrast 8: Neighbor Gathering Implementation

The external patch's explicit loop over grid positions for neighbor gathering is readable but inefficient. The reference doesn't use explicit neighbor gathering—instead, the transformer's attention implicitly aggregates spatial context. When local structure matters, the reference relies on attention window patterns (sliding, bigbird) rather than hardcoded 3x3 neighborhoods. This is a flexibility vs locality tradeoff: the external patch guarantees local context access, the reference learns what context to attend to.

---

## Summary

| Aspect | External Patch | Reference Implementation |
|--------|----------------|-------------------------|
| Position encoding | 2D RoPE (hardcoded) | RnRoPE (learnable, N-dimensional) |
| Gate placement | Post-SDPA, pre-output-proj | Post-output-proj, pre-residual |
| FFN | Dense SwiGLU | SigmoidMoE (sparse) |
| Sparsity | Level-global dimension selection | Per-patch per-dimension gates |
| Quantization | Binary FSQ (1 bit) | 3-bit FSQ (8 levels) |
| Auxiliary prediction | None | LogSNR per patch |
| Attention patterns | Full only | Full/sliding/bigbird/gemma |
| Neighbor context | Explicit 3x3 gather | Implicit via attention |

The external patch is a cleaner, more self-contained implementation suitable for quick experimentation on fixed-size image grids. The reference implementation is more flexible and production-oriented, with support for variable topologies, efficient attention patterns, noise-aware encoding, and mixture-of-experts scaling.
