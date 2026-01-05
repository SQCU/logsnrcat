# Latent vs Pixel Diffusion: Architecture and Implementation Guide

## 1. The LogSNR Field

LogSNR (log signal-to-noise ratio) is the **continuous conditioning signal** that tells the denoiser how much noise is present in its input. Unlike discrete timestep conditioning (t=0...T), logsnr is a continuous scalar (or spatially-varying field) in the range approximately [-20, +20].

The relationship to the noise schedule is:
```
alpha = sqrt(sigmoid(logsnr))    # Signal attenuation
sigma = sqrt(sigmoid(-logsnr))   # Noise magnitude
```

At high logsnr (e.g., +10), the signal dominates (alpha ≈ 1, sigma ≈ 0) - nearly clean.
At low logsnr (e.g., -10), noise dominates (alpha ≈ 0, sigma ≈ 1) - nearly pure noise.

The logsnr field can be:
- **Scalar per sample**: Same noise level across entire image/sequence
- **Spatially varying**: Different noise levels at different spatial locations (enables progressive denoising, inpainting, etc.)

The denoiser receives logsnr as input conditioning AND predicts a logsnr field as output. The predicted logsnr enables self-consistency checks and iterative refinement.

## 2. Noise Augmentation Strategy

The forward diffusion process adds noise to create training pairs:
```
x_t = alpha * x_0 + sigma * noise
```

Where `x_0` is the clean signal, `noise` is Gaussian, and `x_t` is the noised version.

**CRITICAL DISTINCTION:**

- **Pixel Diffusion**: `x_0` is pixels [B, C, H, W]. Noise is added to pixels.
- **Latent Diffusion**: `x_0` is codes [B, N, code_dim] or [B, N, L, code_dim]. Noise is added to codes. **Pixels are NEVER noised.**

In latent diffusion:
1. Clean pixels → AE encoder → clean codes
2. Noise added to codes: `codes_t = alpha * codes_0 + sigma * noise`
3. Noisy codes → denoiser → predicted v-field (in code space)
4. Clean codes recovered: `codes_0_pred = alpha * codes_t - sigma * v_pred`

The AE encoder/decoder are **fixed reference frames** - they transform between pixel space and code space but are not part of the diffusion process itself.

## 3. Output Format for LogSNRCat LDTformer Denoiser

The LDTformer denoiser outputs TWO fields:
1. **V-field** (velocity): Used to recover the clean signal via `x_0 = alpha * x_t - sigma * v`
2. **LogSNR field**: Predicted noise level at each position

The output is in the **SAME SPACE as the input**:
- Pixel diffusion: v-field is [B, C, H, W] - pixel-space velocity
- Latent diffusion: v-field is [B, N*L, code_dim] - code-space velocity

**The v-field is NOT an image.** Its values can range outside [0,1]. It represents the "direction" to move from noisy to clean in the signal space.

## 4. Pixel vs Latent Diffusion Data Flow

### Pixel Diffusion (latent_diffusion = false)

```
Training:
  clean_pixels ──────────────────────────────────────┐
       │                                              │
       ▼                                              │
  euler_forward_step(pixels, logsnr, noise)          │
       │                                              │
       ├──► noisy_pixels ──► patch_embed ──► tokens  │
       │                          │                   │
       │                          ▼                   │
       │                     transformer              │
       │                          │                   │
       │                          ▼                   │
       │                    patch_unembed             │
       │                          │                   │
       │                          ▼                   │
       │                    v_pred (pixels)           │
       │                          │                   │
       └──► v_target (pixels) ────┼──► MSE loss
                                  │
                            (pixel space)
```

The patch_embedder/unembedder are simple MLPs that convert between pixel patches and token embeddings. No AE involved.

### Latent Diffusion (latent_diffusion = true)

```
Training:
  clean_pixels
       │
       ▼
  AE.encode() ──► clean_codes [B, N, L, code_dim]
       │                │
       │                ▼
       │         euler_forward_step(codes, logsnr, noise)
       │                │
       │                ├──► noisy_codes
       │                │         │
       │                │         ▼
       │                │    code_proj ──► tokens [B, N*L, model_dim]
       │                │                      │
       │                │                      ▼
       │                │                 transformer
       │                │                      │
       │                │                      ▼
       │                │                 code_unproj
       │                │                      │
       │                │                      ▼
       │                │               v_pred (codes) [B, N*L, code_dim]
       │                │                      │
       │                └──► v_target (codes) ─┼──► MSE loss (CODE SPACE)
       │                                       │
       │                                       ▼
       │                              codes_0_pred = alpha * noisy_codes - sigma * v_pred
       │                                       │
       │                                       ▼
       │                                 AE.decode()
       │                                       │
       └──────────────────────────────────────►├──► reconstruction loss (OPTIONAL, PIXEL SPACE)
                                               │
                                          recon_pixels
```

**Key points:**
1. Noise is added to CODES, not pixels
2. V-field prediction happens in CODE SPACE
3. MSE loss for v-field is in CODE SPACE
4. AE decode is ONLY for optional reconstruction loss - it's not in the main diffusion path
5. The AE encoder/decoder weights should NOT receive gradients from the v-field loss

## 5. The Bug Pattern to Avoid

The catastrophic bug we encountered:

```python
# WRONG: Mixing pixel and latent concepts
noisy_pixels = euler_forward_step(pixels, logsnr, noise)  # Pixel-space noising
codes = ae.encode(noisy_pixels)                           # Encode noisy pixels
tokens = code_proj(codes)
output_tokens = transformer(tokens)
output_codes = code_unproj(output_tokens)
output_pixels = ae.decode(output_codes)                   # Decode to pixels
loss = MSE(output_pixels, pixel_velocity_target)          # WRONG SPACE!
```

This fails because:
1. The AE was trained on CLEAN images, not noisy images
2. The AE decoder outputs values in [0,1] (image range), not velocity range
3. Gradients from pixel-velocity loss corrupt the AE decoder

**CORRECT for latent diffusion:**
```python
# RIGHT: Everything in code space
clean_codes = ae.encode(clean_pixels)                     # Encode clean pixels
noisy_codes = euler_forward_step(clean_codes, logsnr, noise)  # Code-space noising
tokens = code_proj(noisy_codes)
output_tokens = transformer(tokens)
v_pred_codes = code_unproj(output_tokens)
loss = MSE(v_pred_codes, code_velocity_target)            # CODE SPACE loss!
# Optional: decode for reconstruction
clean_codes_pred = alpha * noisy_codes - sigma * v_pred_codes
recon = ae.decode(clean_codes_pred)
recon_loss = MSE(recon, clean_pixels)                     # Separate loss term
```

## 6. Implementation Checklist

When implementing or modifying diffusion training:

- [ ] **Identify the diffusion space**: Is noise added to pixels or codes?
- [ ] **Match input/output spaces**: V-field must be in same space as noised input
- [ ] **Loss must be in diffusion space**: If diffusing codes, MSE loss is on code-space v-field
- [ ] **AE is a reference frame**: In latent diffusion, AE transforms between spaces but is not in the diffusion path
- [ ] **AE gradients**: In latent diffusion, AE should not receive gradients from v-field loss (only from reconstruction loss if used)
- [ ] **Don't mix embedders**: Pixel diffusion uses ContextualPatchEmbedder. Latent diffusion uses code projections, NOT SwiGLUPatchEmbedder (which encodes/decodes through AE)

## 7. Training Function Responsibilities

| Function | Diffusion Space | Noise Added To | V-Field Space | AE Role |
|----------|-----------------|----------------|---------------|---------|
| `train_denoise` | Pixel | Pixels | Pixels | None (or WRONG if sparse_ae enabled) |
| `train_latent_diffusion` | Latent | Codes | Codes | Encode input, decode for recon loss |
| `train_autoembed` | N/A | None | N/A | Direct encode/decode training |

**WARNING**: `train_denoise` should NOT be used with `sparse_ae` enabled. The data flow is architecturally incompatible. Use `train_latent_diffusion` instead.

## 8. Symptoms of Architectural Confusion

If you see these symptoms, check for pixel/latent space mixing:

1. **Grey sludge output**: AE decoder weights corrupted by wrong-space gradients
2. **Loss doesn't decrease**: Comparing quantities in different spaces
3. **AE quality degrades after denoiser training**: Wrong gradients flowing to AE
4. **V-field values outside expected range**: Decoder outputting images, not velocities
5. **Reconstruction perfect but generation fails**: Different code paths for train vs inference

## 9. File Map for Diffusion Implementation

| File | Pixel Diffusion | Latent Diffusion |
|------|-----------------|------------------|
| `src/train.py:train_denoise` | Correct | DO NOT USE |
| `src/train.py:train_latent_diffusion` | N/A | Correct |
| `src/embedders.py:ContextualPatchEmbedder` | Used | Not used |
| `kmaze_ae/model_swiglu.py:SwiGLUPatchEmbedder` | Not used | Not used (for denoising) |
| `src/context_manager.py:SpanEmbedder` | Routes to embedder | Routes to embedder |

The `SwiGLUPatchEmbedder` exists for compatibility but should NOT be in the denoising path for latent diffusion. Latent diffusion uses direct code projections (`latent_code_proj`, `latent_code_unproj`) without going through the full AE encode/decode cycle.
