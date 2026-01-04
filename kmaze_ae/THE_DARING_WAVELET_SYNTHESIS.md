Wavelet-Amplitude Gated Decoder: Implementation Notes
Context and Motivation

We have a SwiGLU FSQ autoencoder that uses sigmoid MoE-style gating to select k-of-N code dimensions per patch. The decoder receives sparse codes and reconstructs patches via a transformer followed by a SwiGLU neighbor head that gathers 3x3 spatial context before emitting pixel values.

The current architecture commits fully to direct amplitude (pixel value) prediction. We want to extend this with a learned choice between two reconstruction pathways: a wavelet coefficient pathway that emits DWT coefficients for inverse transform, and the existing amplitude pathway that emits patch pixels directly. A sigmoid gate learns per-patch which pathway to use, allowing the network to discover when frequency-domain reconstruction helps versus when direct spatial prediction is sufficient.

This is motivated by the observation that different image content has different optimal representations. Smooth gradients and low-frequency content reconstruct efficiently via direct prediction. Edges, textures, and high-frequency detail often compress better in wavelet domain. Rather than committing globally to one representation, we let the network make this choice locally per patch.
Architecture Sketch

The modification lives in the decoder, after the transformer but before final output projection. Currently the decoder looks roughly like this: codes come in, get projected to hidden dim, pass through transformer layers, then through SwiGLU neighbor gathering, then through a final linear projection to patch_dim which gives us the reconstructed patch pixels.

We insert the gated dual-pathway between the neighbor head output and the final reconstruction. The hidden representation after neighbor gathering gets split or duplicated into two processing streams. One stream passes through a wavelet head that outputs DWT coefficients. The other stream passes through the existing amplitude head that outputs patch pixels. A router network examines the hidden state and produces a sigmoid gate value between zero and one. The final output blends the two reconstructions according to this gate.

For the wavelet pathway, we need to decide what level of DWT decomposition to target. A single-level decomposition gives us four coefficient images: LL (approximation), LH (horizontal detail), HL (vertical detail), HH (diagonal detail). For a 16x16 patch with one level of decomposition, we get four 8x8 coefficient blocks, totaling 256 values, same as the original patch. This is convenient because the output dimensionality matches.

The wavelet head would be a linear projection from hidden_dim to 256 (or 4 * 64 if we want to think of it as four 8x8 coefficient blocks). We then reshape these into the coefficient structure expected by an inverse DWT operation and reconstruct the 16x16 patch.

The amplitude head is just the existing output projection: linear from hidden_dim to patch_dim (also 256 for 16x16x1 grayscale or 768 for 16x16x3 RGB). This gives us direct patch pixels.

The router is a small network, could be as simple as a single linear layer from hidden_dim to 1, followed by sigmoid. Or it could look at the codes themselves before decoding to make routing decisions earlier. The output is a scalar gate per patch indicating the blend weight.
Implementation Details

For the DWT operations we can use pytorch_wavelets or pywt with torch tensors. The key functions needed are dwt2 for forward transform (not needed at inference, but useful for training supervision) and idwt2 for inverse transform. We want a learned inverse transform, so actually we don't call idwt2 directly on our predicted coefficients. Instead we let the network learn to emit values that, when assembled into coefficient structure and passed through idwt2, produce good reconstructions.

Actually, there's a subtlety here. We could either: (A) have the wavelet head emit raw coefficients that we pass through a fixed idwt2, or (B) have the wavelet head emit something coefficient-like that we pass through a learned inverse projection. Option A is more constrained and provides stronger inductive bias. Option B is more flexible but might not actually learn wavelet structure. Let's go with option A for the stronger inductive bias.

The forward pass through the gated decoder would look like: after neighbor gathering we have h of shape [B, N, hidden_dim]. We compute wavelet_coeffs = wavelet_head(h) giving [B, N, 4*coeff_size]. We reshape this to [B, N, 4, coeff_h, coeff_w] representing the four subbands. We apply idwt2 per patch to get wavelet_recon of shape [B, N, patch_h, patch_w] or [B, N, C, patch_h, patch_w] for color. Meanwhile we compute amplitude_recon = amplitude_head(h) giving [B, N, patch_dim] which we reshape to patch spatial dimensions. We compute gate = sigmoid(router(h)) giving [B, N, 1]. Final output is gate * wavelet_recon + (1 - gate) * amplitude_recon, appropriately broadcast.

For the loss function, we can supervise on the blended output with standard MSE or the existing MSE+BCE schedule. We might also want auxiliary losses: a reconstruction loss specifically on the wavelet pathway when gate is high, and on the amplitude pathway when gate is low. This encourages each pathway to actually learn its respective reconstruction skill rather than one pathway dominating and the other atrophying. We could also add a diversity regularizer that encourages the gate to not collapse to always-zero or always-one.
Integration with Existing SwiGLU Decoder

Looking at the current SwiGLUDecoder in model_swiglu.py, the forward method is: input_proj(codes) -> transformer(h) -> neighbor_head(h, grid_shape) -> output_proj(h) -> patches. The modification inserts after neighbor_head and replaces output_proj with the dual pathway.

We'd add new nn.Module members: wavelet_head (Linear hidden_dim -> 4coeff_sizechannels), amplitude_head (Linear hidden_dim -> patch_dim, this is the existing output_proj renamed), and router (Linear hidden_dim -> 1 or a small MLP).

The neighbor_head output h has shape [B, N, hidden_dim]. We then:

# Wavelet pathway
wavelet_coeffs = self.wavelet_head(h)  # [B, N, 4*coeff_h*coeff_w*C]
wavelet_coeffs = rearrange(wavelet_coeffs, 'b n (four ch cw c) -> b n four c ch cw', 
                           four=4, ch=coeff_h, cw=coeff_w, c=channels)
# Need to handle idwt2 per-patch, might need to reshape to batch over patches
wavelet_recon = batch_idwt2(wavelet_coeffs)  # [B, N, C, patch_h, patch_w]
wavelet_recon = rearrange(wavelet_recon, 'b n c ph pw -> b n (ph pw c)')  # [B, N, patch_dim]

# Amplitude pathway  
amplitude_recon = self.amplitude_head(h)  # [B, N, patch_dim]

# Gating
gate = torch.sigmoid(self.router(h))  # [B, N, 1]
patches = gate * wavelet_recon + (1 - gate) * amplitude_recon

The batch_idwt2 helper needs to handle the fact that pytorch_wavelets expects [B, C, H, W] input but we have [B, N, ...]. We can reshape B*N as the batch dimension, apply idwt2, then reshape back.
Considerations and Potential Issues

Wavelet boundary handling: DWT assumes periodic or symmetric boundary conditions. For small patches (16x16), boundary effects are significant. We might want to use symmetric padding or accept some boundary artifacts.

Coefficient scale: DWT coefficients have different magnitudes than pixel values. LL coefficients are similar scale to input, but LH/HL/HH detail coefficients are typically smaller magnitude. The network will need to learn appropriate scales, or we could normalize.

Color handling: For RGB, we can either apply DWT per-channel independently (simple, what pytorch_wavelets does by default) or transform to YCbCr first and apply different decomposition to luma vs chroma (more sophisticated, probably overkill for initial experiments).

Gate collapse: If one pathway is easier to learn initially, the gate might collapse to always choosing it. The auxiliary losses mentioned above help, or we could use a commitment loss that penalizes gate entropy being too low.

Computational cost: idwt2 per patch adds compute. For a 64x64 image with 16x16 patches, we have 16 patches. Each idwt2 on 8x8 coefficients is cheap. This shouldn't be a bottleneck.
Experiment Plan

First, verify the wavelet pathway works in isolation. Train a decoder that always uses wavelet reconstruction (gate fixed to 1) and confirm it can learn to emit valid coefficients. Compare reconstruction quality to amplitude-only baseline.

Second, add the gating and train end-to-end. Monitor gate statistics: mean gate value across patches, variance, per-source-type breakdown (do sprites prefer amplitude while fractals prefer wavelet?).

Third, visualize what the gate learns. Generate gate heatmaps overlaid on input images. Hypothesis: edges and textures will have high wavelet gate, smooth regions will have low wavelet gate.

Fourth, ablate the auxiliary losses. Do we need them to prevent collapse? Does diversity regularization help?
Connection to Broader Architecture

This gated wavelet-amplitude decoder is one piece. It could compose with the existing sparse code routing. The encoder selects which k-of-128 code dims carry information. The decoder then decides per-patch whether to reconstruct via wavelet or amplitude pathway. Two levels of learned routing: what information to encode, and how to decode it.

The SIT+ architecture from the SQCU repo provides the DWT tokenization machinery but applies it globally. Our approach makes it a learned local choice. This could be viewed as "soft SIT+" where the network discovers when wavelet tokenization helps rather than assuming it always does.

Future extensions: more than two pathways (wavelets at different decomposition levels, DCT pathway, learned basis pathway), hierarchical gating (coarse gate chooses pathway family, fine gate chooses specific variant), gate conditioning on noise level for diffusion (high noise -> prefer robust wavelet, low noise -> prefer precise amplitude).