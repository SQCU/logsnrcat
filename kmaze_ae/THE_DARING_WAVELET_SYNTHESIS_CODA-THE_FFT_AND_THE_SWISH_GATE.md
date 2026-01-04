This is a beautiful question that gets at something deep about representation learning.
The Rotation Angle Interpretation

The Fractional Fourier Transform (FrFT) generalizes the standard Fourier transform via a rotation angle α in the time-frequency plane. At α=0 you have the identity (pure time/space domain). At α=π/2 you have the standard Fourier transform (pure frequency domain). Intermediate angles give you hybrid representations that are "partially transformed."

The Type A wavelet-amplitude gate is doing something conceptually similar but discretized: gate=0 means "pure spatial domain" (amplitude pathway), gate=1 means "frequency-decomposed domain" (wavelet pathway). The sigmoid gate is learning where on this continuum the optimal representation lies, but it's forced into a soft binary choice rather than a continuous rotation.

If you squint, the gate value is a learned rotation angle, just quantized to {0, 1} with soft interpolation via sigmoid.
Chirplets and Adaptive Basis Selection

Chirplets extend wavelets by adding a chirp rate parameter - they can represent signals with time-varying frequency (linear frequency modulation). A chirplet basis adapts not just where in time-frequency space to localize, but also how the frequency changes across that locality.

The connection to Type A: with a fixed idwt2, we've committed to a specific wavelet basis (say, Haar or Daubechies). The network learns coefficients but not the basis itself. This is like using a fixed FrFT angle and learning which frequencies are present.

A chirplet-like extension would be: learn the basis, not just the coefficients. This is closer to Type B ("learned inverse projection") but with structure. Instead of a fully arbitrary linear transform, you'd parameterize a family of bases (say, wavelets with learnable filter coefficients, or FrFT with learnable angle, or chirplets with learnable chirp rate) and learn which point in that family to use.
The Hierarchy of Adaptivity

There's a spectrum of how much structure you impose:
Approach	Fixed	Learned	Inductive Bias
Type A (fixed idwt2)	Basis (wavelet family)	Coefficients only	Strong: must decompose into wavelets
FrFT (fixed angle)	Rotation angle	Coefficients	Strong: must use that specific hybrid
FrFT (learned angle)	Nothing	Angle + coefficients	Medium: continuous rotation, but still FrFT structure
Chirplet (learned params)	Chirplet structure	Scale, location, chirp rate, amplitude	Medium: richer than FrFT but constrained family
Type B (learned inverse)	Nothing	Arbitrary linear map	Weak: might not learn anything wavelet-like

The sigmoid gate in our architecture is learning to select between two fixed points in this space (wavelet and identity). A more general version would learn a continuous position in the time-frequency rotation space.
What a "Soft FrFT Gate" Would Look Like

Instead of:

output = gate * wavelet_recon + (1 - gate) * amplitude_recon

You could imagine:

alpha = self.angle_router(h) * (pi/2)  # learned rotation angle ∈ [0, π/2]
output = fractional_fourier_2d(amplitude_coeffs, alpha)

Where alpha=0 gives you the spatial representation and alpha=π/2 gives you frequency representation. The network learns per-patch what rotation angle produces the best reconstruction.

The catch: FrFT is a global transform, not naturally local. For patches this might be okay (16x16 is small), but you lose the multi-scale property of wavelets. Chirplets or windowed FrFT recover locality.
The Deeper Unity

What wavelets, FrFT, and chirplets share: they're all ways of tiling the time-frequency plane with different trade-offs:

    Fourier: Perfect frequency resolution, no time resolution (vertical strips)
    Wavelets: Adaptive tiling (wide low-frequency, narrow high-frequency)
    STFT: Fixed-size rectangles everywhere
    Chirplets: Rotated/sheared rectangles (can follow frequency modulation)
    FrFT: Rotates the entire coordinate system

The sigmoid gate is asking: "should this patch be represented with wavelet tiling or trivial (identity) tiling?" A learned FrFT angle would ask: "what rotation of the tiling is optimal?" A learned chirplet would ask: "what rotation and shear of the tiling is optimal?"
Practical Implications

For the immediate implementation (Type A with fixed idwt2), we're betting that the binary choice wavelet-vs-amplitude captures most of the benefit. This is probably true for natural images where some regions are smooth (amplitude wins) and some have edges/texture (wavelet wins).

But if you wanted to go further: parameterize a learnable 2D FrFT or chirplet transform, and have the router predict transform parameters rather than a binary gate. This would let the network discover intermediate representations that aren't quite wavelets and aren't quite pixels - hybrid representations tuned to the specific image content.

The computational cost increases (FrFT is more expensive than DWT), but the representation power increases too. Whether that trade-off is worth it depends on whether the binary wavelet/amplitude choice is "good enough" for your data distribution.

TL;DR: The sigmoid gate is a discretized, learned FrFT angle. Type A fixes the basis (strong bias, efficient). Chirplets/learned-FrFT would make the basis itself learnable (weaker bias, more expressive). The deep connection is that all of these are ways of learning where in time-frequency space to represent signal content, with different granularities of choice.