#!/usr/bin/env python3
"""Modular noise/pattern generators. Each class is standalone and stackable."""

import math
import random
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class NoiseGenerator(ABC):
    """Base class for all noise/pattern generators."""

    name: str = "base"

    @abstractmethod
    def generate(self, size: int, device: torch.device) -> torch.Tensor:
        """Generate a single (3, H, W) image tensor in [0, 1]."""
        pass

    def generate_batch(self, size: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate batch of images. Override for faster batched impl."""
        return torch.stack([self.generate(size, device) for _ in range(batch_size)])


# ============ UTILITIES ============

_coord_cache = {}  # Cache coordinate grids by (size, device)

def get_coords(size, device):
    """Get (x, y) coordinate grids (cached)."""
    key = (size, device)
    if key not in _coord_cache:
        y, x = torch.meshgrid(torch.arange(size, device=device),
                              torch.arange(size, device=device), indexing='ij')
        _coord_cache[key] = (x.float(), y.float())
    return _coord_cache[key]


def gaussian_kernel_1d(sigma, device):
    """Create a 1D Gaussian kernel."""
    kernel_size = int(6 * sigma + 1) | 1
    kernel_size = max(3, kernel_size)
    x = torch.arange(kernel_size, device=device).float() - kernel_size // 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    return gauss / gauss.sum()


def separable_blur(img, sigma, device):
    """Fast separable Gaussian blur on (C, H, W) tensor."""
    if sigma < 0.5:
        return img
    c, h, w = img.shape

    # For very large sigmas, use FFT blur (faster when kernel > ~50)
    if sigma > 25:
        return fft_blur(img, sigma, device)

    kernel = gaussian_kernel_1d(sigma, device)
    ksize = len(kernel)
    pad = ksize // 2
    k_h = kernel.view(1, 1, 1, ksize).expand(c, 1, 1, ksize)
    k_v = kernel.view(1, 1, ksize, 1).expand(c, 1, ksize, 1)
    img = img.unsqueeze(0)
    img = F.conv2d(img, k_h, padding=(0, pad), groups=c)
    img = F.conv2d(img, k_v, padding=(pad, 0), groups=c)
    return img.squeeze(0)


def fft_blur(img, sigma, device):
    """FFT-based Gaussian blur (faster for large sigma)."""
    c, h, w = img.shape

    # Create Gaussian kernel in frequency domain
    freq_y = torch.fft.fftfreq(h, device=device).view(-1, 1)
    freq_x = torch.fft.fftfreq(w, device=device).view(1, -1)
    freq_sq = freq_y**2 + freq_x**2

    # Gaussian in freq domain: exp(-2 * pi^2 * sigma^2 * freq^2)
    gauss_freq = torch.exp(-2 * (math.pi * sigma)**2 * freq_sq)

    # FFT, multiply, IFFT
    fft = torch.fft.fft2(img)
    filtered = fft * gauss_freq.unsqueeze(0)
    return torch.fft.ifft2(filtered).real


# ============ VORONOI ============

class VoronoiNoise(NoiseGenerator):
    """Random Voronoi cell patterns."""

    name = "voronoi"

    def __init__(self, min_cells=5, max_cells=50, colored=True):
        self.min_cells = min_cells
        self.max_cells = max_cells
        self.colored = colored

    @torch.no_grad()
    def generate(self, size, device):
        n_cells = random.randint(self.min_cells, self.max_cells)

        y, x = torch.meshgrid(torch.arange(size, device=device),
                              torch.arange(size, device=device), indexing='ij')
        coords = torch.stack([x, y], dim=-1).float().view(-1, 2)

        centers = torch.rand(n_cells, 2, device=device) * size
        colors = torch.rand(n_cells, 3, device=device) if self.colored else torch.rand(n_cells, 1, device=device).expand(-1, 3)

        dists = torch.cdist(coords, centers)
        closest = dists.argmin(dim=1)
        img = colors[closest].view(size, size, 3).permute(2, 0, 1)
        return img


# ============ PINK NOISE ============

class PinkNoise(NoiseGenerator):
    """1/f^alpha noise with variable sharpness."""

    name = "pink_noise"
    _freq_cache = {}  # Cache freq_mag grids by (size, device)

    def __init__(self, min_sharpness=0.5, max_sharpness=2.5):
        self.min_sharpness = min_sharpness
        self.max_sharpness = max_sharpness

    @classmethod
    def _get_freq_mag(cls, size, device):
        """Get cached frequency magnitude grid."""
        key = (size, device)
        if key not in cls._freq_cache:
            freq_y = torch.fft.fftfreq(size, device=device).view(-1, 1)
            freq_x = torch.fft.fftfreq(size, device=device).view(1, -1)
            freq_mag = torch.sqrt(freq_y**2 + freq_x**2)
            freq_mag[0, 0] = 1  # Avoid div by zero
            cls._freq_cache[key] = freq_mag
        return cls._freq_cache[key]

    @torch.no_grad()
    def generate(self, size, device):
        sharpness = random.uniform(self.min_sharpness, self.max_sharpness)

        noise = torch.randn(3, size, size, device=device)
        fft = torch.fft.fft2(noise)

        freq_mag = self._get_freq_mag(size, device)
        falloff = 1.0 / (freq_mag ** sharpness)
        falloff[0, 0] = 0

        filtered = fft * falloff.unsqueeze(0)
        result = torch.fft.ifft2(filtered).real

        # Normalize per-channel (faster than global)
        result = result.view(3, -1)
        mins = result.min(dim=1, keepdim=True)[0]
        maxs = result.max(dim=1, keepdim=True)[0]
        result = (result - mins) / (maxs - mins + 1e-8)
        return result.view(3, size, size)

    @torch.no_grad()
    def generate_batch(self, size, batch_size, device):
        """Fully batched FFT implementation."""
        sharpness = torch.empty(batch_size, device=device).uniform_(self.min_sharpness, self.max_sharpness)

        noise = torch.randn(batch_size, 3, size, size, device=device)
        fft = torch.fft.fft2(noise)

        freq_y = torch.fft.fftfreq(size, device=device).view(1, 1, -1, 1)
        freq_x = torch.fft.fftfreq(size, device=device).view(1, 1, 1, -1)
        freq_mag = torch.sqrt(freq_y**2 + freq_x**2)
        freq_mag = freq_mag.expand(batch_size, 1, size, size).clone()
        freq_mag[:, :, 0, 0] = 1

        sharpness = sharpness.view(-1, 1, 1, 1)
        falloff = 1.0 / (freq_mag ** sharpness)
        falloff[:, :, 0, 0] = 0

        filtered = fft * falloff
        result = torch.fft.ifft2(filtered).real

        mins = result.view(batch_size, 3, -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        maxs = result.view(batch_size, 3, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        result = (result - mins) / (maxs - mins + 1e-8)
        return result


# ============ GEOMETRIC SHAPES ============

class GeometricShapes(NoiseGenerator):
    """Random overlapping geometric shapes."""

    name = "shapes"

    def __init__(self, min_shapes=3, max_shapes=15, shape_types=None):
        self.min_shapes = min_shapes
        self.max_shapes = max_shapes
        self.shape_types = shape_types or ['circle', 'square', 'triangle']

    @torch.no_grad()
    def generate(self, size, device):
        x, y = get_coords(size, device)
        img = torch.zeros(3, size, size, device=device)

        n_shapes = random.randint(self.min_shapes, self.max_shapes)

        for _ in range(n_shapes):
            shape_type = random.choice(self.shape_types)
            color = torch.rand(3, device=device)
            cx = random.uniform(0, size)
            cy = random.uniform(0, size)
            s = random.uniform(size * 0.05, size * 0.3)
            angle = random.uniform(0, math.pi)
            alpha = random.uniform(0.5, 1.0)

            dx, dy = x - cx, y - cy

            if shape_type == 'circle':
                mask = (dx**2 + dy**2) < s**2
            elif shape_type == 'square':
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                rx = dx * cos_a + dy * sin_a
                ry = -dx * sin_a + dy * cos_a
                mask = (rx.abs() < s) & (ry.abs() < s)
            elif shape_type == 'triangle':
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                rx = dx * cos_a + dy * sin_a
                ry = -dx * sin_a + dy * cos_a
                mask = (ry > -s * 0.5) & (ry < s) & (rx.abs() < (s - ry) * 0.6)
            else:
                continue

            mask_f = mask.float()
            for c in range(3):
                img[c] = img[c] * (1 - mask_f * alpha) + color[c] * mask_f * alpha

        return img

    @torch.no_grad()
    def generate_batch(self, size, batch_size, device):
        """Batched shapes - process all shapes across all images in parallel."""
        x, y = get_coords(size, device)
        x = x.unsqueeze(0)  # (1, H, W)
        y = y.unsqueeze(0)

        img = torch.zeros(batch_size, 3, size, size, device=device)
        max_shapes = self.max_shapes

        # Pre-generate all shape parameters for all images
        n_shapes = torch.randint(self.min_shapes, max_shapes + 1, (batch_size,), device=device)
        total_shapes = batch_size * max_shapes

        # Parameters for all shapes: (B, max_shapes)
        shape_types = torch.randint(0, 3, (batch_size, max_shapes), device=device)  # 0=circle, 1=square, 2=triangle
        colors = torch.rand(batch_size, max_shapes, 3, device=device)
        cx = torch.rand(batch_size, max_shapes, device=device) * size
        cy = torch.rand(batch_size, max_shapes, device=device) * size
        s = torch.rand(batch_size, max_shapes, device=device) * (size * 0.25) + (size * 0.05)
        angles = torch.rand(batch_size, max_shapes, device=device) * math.pi
        alphas = torch.rand(batch_size, max_shapes, device=device) * 0.5 + 0.5

        # Valid shape mask
        shape_idx = torch.arange(max_shapes, device=device).unsqueeze(0)
        valid = shape_idx < n_shapes.unsqueeze(1)  # (B, max_shapes)

        # Process each shape slot across all images
        for si in range(max_shapes):
            # Get params for this shape slot
            st = shape_types[:, si]  # (B,)
            col = colors[:, si]  # (B, 3)
            cxi, cyi = cx[:, si], cy[:, si]  # (B,)
            si_size = s[:, si]
            ang = angles[:, si]
            alph = alphas[:, si]
            v = valid[:, si]  # (B,)

            # Compute dx, dy for all images: (B, H, W)
            dx = x - cxi.view(-1, 1, 1)
            dy = y - cyi.view(-1, 1, 1)

            # Compute rotated coords
            cos_a = torch.cos(ang).view(-1, 1, 1)
            sin_a = torch.sin(ang).view(-1, 1, 1)
            rx = dx * cos_a + dy * sin_a
            ry = -dx * sin_a + dy * cos_a

            si_size = si_size.view(-1, 1, 1)

            # Circle mask
            circle_mask = (dx**2 + dy**2) < si_size**2
            # Square mask
            square_mask = (rx.abs() < si_size) & (ry.abs() < si_size)
            # Triangle mask
            tri_mask = (ry > -si_size * 0.5) & (ry < si_size) & (rx.abs() < (si_size - ry) * 0.6)

            # Select mask based on shape type
            st_exp = st.view(-1, 1, 1)
            mask = torch.where(st_exp == 0, circle_mask,
                   torch.where(st_exp == 1, square_mask, tri_mask))

            # Apply validity mask
            mask = mask & v.view(-1, 1, 1)
            mask_f = mask.float().unsqueeze(1)  # (B, 1, H, W)
            alph = alph.view(-1, 1, 1, 1)
            col = col.view(-1, 3, 1, 1)

            img = img * (1 - mask_f * alph) + col * mask_f * alph

        return img


# ============ PATTERNS ============

class PatternNoise(NoiseGenerator):
    """Geometric patterns with transforms and effects."""

    name = "patterns"

    def __init__(self, pattern_types=None, enable_overlay=True, enable_blur=True, enable_vignette=True):
        self.pattern_types = pattern_types or ['checker', 'stripes', 'grid', 'dots']
        self.enable_overlay = enable_overlay
        self.enable_blur = enable_blur
        self.enable_vignette = enable_vignette

    @torch.no_grad()
    def generate(self, size, device):
        x_base, y_base = get_coords(size, device)
        x_base, y_base = x_base - size / 2, y_base - size / 2

        radial_dist = torch.sqrt(x_base**2 + y_base**2) / (size * 0.5 * math.sqrt(2))

        ptype = random.choice(self.pattern_types)

        # Random transforms
        scale = math.exp(random.uniform(math.log(8), math.log(64)))
        angle = random.uniform(0, math.pi * 2)
        shift_x = random.uniform(-size * 0.5, size * 0.5)
        shift_y = random.uniform(-size * 0.5, size * 0.5)
        aspect = random.uniform(0.5, 2.0)

        color1 = torch.rand(3, device=device)
        color2 = torch.rand(3, device=device)

        x = x_base - shift_x
        y = y_base - shift_y

        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rx = (x * cos_a + y * sin_a) * aspect
        ry = -x * sin_a + y * cos_a

        if ptype == 'checker':
            mask = ((rx // scale).long() + (ry // scale).long()) % 2 == 0
        elif ptype == 'stripes':
            mask = (rx // scale).long() % 2 == 0
        elif ptype == 'grid':
            line_w = scale * random.uniform(0.1, 0.4)
            mask = (rx % scale < line_w) | (ry % scale < line_w)
        else:  # dots
            dot_r = scale * random.uniform(0.2, 0.45)
            cx = (rx // scale + 0.5) * scale
            cy = (ry // scale + 0.5) * scale
            mask = torch.sqrt((rx - cx)**2 + (ry - cy)**2) < dot_r

        img = torch.zeros(3, size, size, device=device)
        mask_f = mask.float()
        for c in range(3):
            img[c] = color1[c] * mask_f + color2[c] * (1 - mask_f)

        # Optional overlay
        if self.enable_overlay and random.random() < 0.5:
            scale2 = scale * random.uniform(0.3, 3)
            angle2 = random.uniform(0, math.pi * 2)
            shift_x2 = random.uniform(-size * 0.3, size * 0.3)
            shift_y2 = random.uniform(-size * 0.3, size * 0.3)
            opacity = random.uniform(0.2, 0.7)
            overlay_color = torch.rand(3, device=device)

            x2 = x_base - shift_x2
            y2 = y_base - shift_y2
            cos_a2, sin_a2 = math.cos(angle2), math.sin(angle2)
            rx2 = x2 * cos_a2 + y2 * sin_a2
            ry2 = -x2 * sin_a2 + y2 * cos_a2
            overlay = ((rx2 // scale2).long() + (ry2 // scale2).long()) % 2 == 0
            overlay_f = overlay.float()

            for c in range(3):
                img[c] = img[c] * (1 - overlay_f * opacity) + overlay_color[c] * overlay_f * opacity

        # Radial blur
        if self.enable_blur and random.random() < 0.6:
            blur_sigma = random.uniform(2, 20)
            blur_mode = random.choice(['edge', 'center', 'uniform'])

            if blur_mode == 'uniform':
                img = separable_blur(img, blur_sigma, device)
            else:
                blurred = separable_blur(img, blur_sigma, device)
                if blur_mode == 'edge':
                    blend = radial_dist.clamp(0, 1) ** random.uniform(0.5, 2)
                else:
                    blend = (1 - radial_dist).clamp(0, 1) ** random.uniform(0.5, 2)
                img = img * (1 - blend) + blurred * blend

        # Vignette
        if self.enable_vignette and random.random() < 0.4:
            strength = random.uniform(0.2, 0.8)
            if random.random() < 0.5:
                vignette = 1 - radial_dist.clamp(0, 1) ** random.uniform(0.5, 2) * strength
            else:
                vignette = 1 - (1 - radial_dist).clamp(0, 1) ** random.uniform(0.5, 2) * strength
            img = img * vignette

        return img

    @torch.no_grad()
    def generate_batch(self, size, batch_size, device):
        """Batched patterns (simplified - core pattern only)."""
        x_base, y_base = get_coords(size, device)
        x_base, y_base = x_base - size / 2, y_base - size / 2
        x_base = x_base.unsqueeze(0)  # (1, H, W)
        y_base = y_base.unsqueeze(0)

        # Pattern types: 0=checker, 1=stripes, 2=grid, 3=dots
        ptypes = torch.randint(0, 4, (batch_size,), device=device)

        # Parameters
        log_scale_min, log_scale_max = math.log(8), math.log(64)
        scales = torch.empty(batch_size, device=device).uniform_(log_scale_min, log_scale_max).exp()
        angles = torch.rand(batch_size, device=device) * math.pi * 2
        shift_x = torch.rand(batch_size, device=device) * size - size * 0.5
        shift_y = torch.rand(batch_size, device=device) * size - size * 0.5
        aspects = torch.rand(batch_size, device=device) * 1.5 + 0.5
        color1 = torch.rand(batch_size, 3, device=device)
        color2 = torch.rand(batch_size, 3, device=device)

        # Transform coords
        x = x_base - shift_x.view(-1, 1, 1)
        y = y_base - shift_y.view(-1, 1, 1)

        cos_a = torch.cos(angles).view(-1, 1, 1)
        sin_a = torch.sin(angles).view(-1, 1, 1)
        aspects = aspects.view(-1, 1, 1)
        rx = (x * cos_a + y * sin_a) * aspects
        ry = -x * sin_a + y * cos_a

        scales = scales.view(-1, 1, 1)

        # Compute all pattern types
        checker = ((rx // scales).long() + (ry // scales).long()) % 2 == 0
        stripes = (rx // scales).long() % 2 == 0
        line_w = scales * 0.25
        grid = (rx % scales < line_w) | (ry % scales < line_w)
        dot_r = scales * 0.35
        cx = (rx // scales + 0.5) * scales
        cy = (ry // scales + 0.5) * scales
        dots = torch.sqrt((rx - cx)**2 + (ry - cy)**2) < dot_r

        # Select based on type
        ptypes_exp = ptypes.view(-1, 1, 1)
        mask = torch.where(ptypes_exp == 0, checker,
               torch.where(ptypes_exp == 1, stripes,
               torch.where(ptypes_exp == 2, grid, dots)))

        mask_f = mask.float().unsqueeze(1)  # (B, 1, H, W)
        c1 = color1.view(batch_size, 3, 1, 1)
        c2 = color2.view(batch_size, 3, 1, 1)

        return c1 * mask_f + c2 * (1 - mask_f)


# ============ BLURRED NOISE (from visualize_blurred_noise.py) ============

class BlurredNoise(NoiseGenerator):
    """Spatially-correlated noise via blurred Gaussian noise with color shift."""

    name = "blurred_noise"

    def __init__(self, noise_min=0.01, noise_max=1.0, blur_min=0.5, blur_max=160.0,
                 color_shift_min=0.5, color_shift_max=2.0):
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.blur_min = blur_min
        self.blur_max = blur_max
        self.color_shift_min = color_shift_min
        self.color_shift_max = color_shift_max

    @torch.no_grad()
    def generate(self, size, device):
        # Sample noise level (log-uniform)
        noise_level = math.exp(random.uniform(math.log(self.noise_min), math.log(self.noise_max)))

        # Sample blur sigma (log-uniform)
        blur_sigma = math.exp(random.uniform(math.log(self.blur_min), math.log(self.blur_max)))

        # Generate per-pixel IID noise
        noise = torch.randn(3, size, size, device=device) * noise_level

        # Blur the noise
        blurred_noise = separable_blur(noise, blur_sigma, device)

        # Random per-channel color shift (log-uniform)
        log_min, log_max = math.log(self.color_shift_min), math.log(self.color_shift_max)
        scales = torch.tensor([math.exp(random.uniform(log_min, log_max)) for _ in range(3)], device=device)
        shifted_noise = blurred_noise * scales.view(3, 1, 1)

        # Center around 0.5 and clamp
        img = (0.5 + shifted_noise).clamp(0, 1)
        return img

    @torch.no_grad()
    def generate_batch(self, size, batch_size, device):
        """Batched blurred noise using FFT blur."""
        log_noise_min, log_noise_max = math.log(self.noise_min), math.log(self.noise_max)
        log_blur_min, log_blur_max = math.log(self.blur_min), math.log(self.blur_max)
        log_color_min, log_color_max = math.log(self.color_shift_min), math.log(self.color_shift_max)

        # Sample parameters
        noise_levels = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(log_noise_min, log_noise_max).exp()
        blur_sigmas = torch.empty(batch_size, device=device).uniform_(log_blur_min, log_blur_max).exp()
        color_scales = torch.empty(batch_size, 3, 1, 1, device=device).uniform_(log_color_min, log_color_max).exp()

        # Generate noise
        noise = torch.randn(batch_size, 3, size, size, device=device) * noise_levels

        # FFT blur (batched) - use average sigma for simplicity, or group by sigma
        # For full correctness, group by similar sigmas. For speed, use FFT with per-image sigma
        freq_y = torch.fft.fftfreq(size, device=device).view(1, 1, -1, 1)
        freq_x = torch.fft.fftfreq(size, device=device).view(1, 1, 1, -1)
        freq_sq = freq_y**2 + freq_x**2

        # Per-image Gaussian in freq domain
        sigma_sq = (blur_sigmas ** 2).view(-1, 1, 1, 1)
        gauss_freq = torch.exp(-2 * (math.pi ** 2) * sigma_sq * freq_sq)

        fft = torch.fft.fft2(noise)
        blurred = torch.fft.ifft2(fft * gauss_freq).real

        # Apply color shift
        result = blurred * color_scales

        return (0.5 + result).clamp(0, 1)


# ============ SOLID COLOR ============

class SolidColor(NoiseGenerator):
    """Random solid color."""

    name = "solid"

    @torch.no_grad()
    def generate(self, size, device):
        color = torch.rand(3, device=device)
        return color.view(3, 1, 1).expand(3, size, size).clone()

    @torch.no_grad()
    def generate_batch(self, size, batch_size, device):
        """Batched solid colors."""
        colors = torch.rand(batch_size, 3, 1, 1, device=device)
        return colors.expand(batch_size, 3, size, size).clone()


# ============ GRADIENT ============

class GradientNoise(NoiseGenerator):
    """Random linear or radial gradients."""

    name = "gradient"

    def __init__(self, gradient_types=None):
        self.gradient_types = gradient_types or ['linear', 'radial', 'angular']

    @torch.no_grad()
    def generate(self, size, device):
        x, y = get_coords(size, device)
        x, y = x / size - 0.5, y / size - 0.5

        gtype = random.choice(self.gradient_types)
        color1 = torch.rand(3, device=device)
        color2 = torch.rand(3, device=device)

        if gtype == 'linear':
            angle = random.uniform(0, math.pi * 2)
            t = (x * math.cos(angle) + y * math.sin(angle) + 0.5).clamp(0, 1)
        elif gtype == 'radial':
            cx, cy = random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3)
            t = torch.sqrt((x - cx)**2 + (y - cy)**2) * random.uniform(1, 3)
            t = t.clamp(0, 1)
        else:  # angular
            cx, cy = random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3)
            t = (torch.atan2(y - cy, x - cx) / math.pi + 1) / 2

        img = torch.zeros(3, size, size, device=device)
        for c in range(3):
            img[c] = color1[c] * (1 - t) + color2[c] * t

        return img

    @torch.no_grad()
    def generate_batch(self, size, batch_size, device):
        """Batched gradients."""
        x, y = get_coords(size, device)
        x, y = x / size - 0.5, y / size - 0.5
        x = x.unsqueeze(0)  # (1, H, W)
        y = y.unsqueeze(0)

        # Random gradient types (0=linear, 1=radial, 2=angular)
        gtypes = torch.randint(0, 3, (batch_size,), device=device)
        color1 = torch.rand(batch_size, 3, device=device)
        color2 = torch.rand(batch_size, 3, device=device)

        # Parameters
        angles = torch.rand(batch_size, device=device) * math.pi * 2
        cx = torch.rand(batch_size, device=device) * 0.6 - 0.3
        cy = torch.rand(batch_size, device=device) * 0.6 - 0.3
        radial_scale = torch.rand(batch_size, device=device) * 2 + 1

        # Compute t for all types
        cos_a = torch.cos(angles).view(-1, 1, 1)
        sin_a = torch.sin(angles).view(-1, 1, 1)
        t_linear = (x * cos_a + y * sin_a + 0.5).clamp(0, 1)

        cx_exp = cx.view(-1, 1, 1)
        cy_exp = cy.view(-1, 1, 1)
        t_radial = (torch.sqrt((x - cx_exp)**2 + (y - cy_exp)**2) * radial_scale.view(-1, 1, 1)).clamp(0, 1)
        t_angular = (torch.atan2(y - cy_exp, x - cx_exp) / math.pi + 1) / 2

        # Select based on type
        gtypes_exp = gtypes.view(-1, 1, 1)
        t = torch.where(gtypes_exp == 0, t_linear,
            torch.where(gtypes_exp == 1, t_radial, t_angular))

        # Interpolate colors
        c1 = color1.view(batch_size, 3, 1, 1)
        c2 = color2.view(batch_size, 3, 1, 1)
        t = t.unsqueeze(1)  # (B, 1, H, W)

        return c1 * (1 - t) + c2 * t


# ============ FBM (FRACTAL BROWNIAN MOTION) ============

class FBMNoise(NoiseGenerator):
    """Fractal Brownian Motion - layered octaves of noise."""

    name = "fbm"

    def __init__(self, min_octaves=3, max_octaves=8, lacunarity_range=(1.8, 2.2),
                 persistence_range=(0.4, 0.6)):
        self.min_octaves = min_octaves
        self.max_octaves = max_octaves
        self.lacunarity_range = lacunarity_range
        self.persistence_range = persistence_range

    @torch.no_grad()
    def generate(self, size, device):
        octaves = random.randint(self.min_octaves, self.max_octaves)
        lacunarity = random.uniform(*self.lacunarity_range)
        persistence = random.uniform(*self.persistence_range)
        base_freq = random.uniform(2, 8)

        x, y = get_coords(size, device)
        x, y = x / size, y / size  # normalize to [0, 1]

        # Random phase offsets per channel for color variation
        phase = torch.rand(3, 2, device=device) * 1000

        result = torch.zeros(3, size, size, device=device)
        amplitude = 1.0
        freq = base_freq
        max_val = 0.0

        for _ in range(octaves):
            for c in range(3):
                # Simple sin-based noise (fast approximation)
                nx = x * freq + phase[c, 0]
                ny = y * freq + phase[c, 1]
                # Combine multiple sin waves for noise-like appearance
                noise = torch.sin(nx * 6.28) * torch.cos(ny * 6.28)
                noise += torch.sin((nx + ny) * 4.5) * 0.5
                noise += torch.cos((nx - ny) * 3.7) * 0.5
                result[c] += noise * amplitude

            max_val += amplitude
            amplitude *= persistence
            freq *= lacunarity
            phase += 1.337  # shift phase each octave

        # Normalize to [0, 1]
        result = (result / max_val + 1) / 2
        return result.clamp(0, 1)

    @torch.no_grad()
    def generate_batch(self, size, batch_size, device):
        """Batched FBM noise."""
        x, y = get_coords(size, device)
        x, y = x / size, y / size
        x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        y = y.unsqueeze(0).unsqueeze(0)

        # Sample parameters for all images
        octaves = self.max_octaves  # Use max for batching, mask unused
        lacunarity = torch.empty(batch_size, device=device).uniform_(*self.lacunarity_range)
        persistence = torch.empty(batch_size, device=device).uniform_(*self.persistence_range)
        base_freq = torch.empty(batch_size, device=device).uniform_(2, 8)
        n_octaves = torch.randint(self.min_octaves, self.max_octaves + 1, (batch_size,), device=device)

        # Phase offsets: (B, 3, 2)
        phase = torch.rand(batch_size, 3, 2, device=device) * 1000

        result = torch.zeros(batch_size, 3, size, size, device=device)
        amplitude = torch.ones(batch_size, 1, 1, 1, device=device)
        freq = base_freq.view(-1, 1, 1, 1)
        max_val = torch.zeros(batch_size, 1, 1, 1, device=device)

        persistence = persistence.view(-1, 1, 1, 1)
        lacunarity = lacunarity.view(-1, 1, 1, 1)

        for oct_idx in range(octaves):
            # Mask for images that use this octave
            active = (oct_idx < n_octaves).view(-1, 1, 1, 1).float()

            for c in range(3):
                px = phase[:, c, 0].view(-1, 1, 1, 1)
                py = phase[:, c, 1].view(-1, 1, 1, 1)
                nx = x * freq + px
                ny = y * freq + py
                noise = torch.sin(nx * 6.28) * torch.cos(ny * 6.28)
                noise = noise + torch.sin((nx + ny) * 4.5) * 0.5
                noise = noise + torch.cos((nx - ny) * 3.7) * 0.5
                result[:, c:c+1] += noise * amplitude * active

            max_val += amplitude * active
            amplitude = amplitude * persistence
            freq = freq * lacunarity
            phase = phase + 1.337

        result = (result / max_val.clamp(min=1e-6) + 1) / 2
        return result.clamp(0, 1)


class FBMPerlin(NoiseGenerator):
    """FBM using proper gradient noise (Perlin-style)."""

    name = "fbm_perlin"
    _grad_cache = {}  # Cache for precomputed gradient lookup tables

    def __init__(self, min_octaves=3, max_octaves=8):
        self.min_octaves = min_octaves
        self.max_octaves = max_octaves

    @classmethod
    def _get_grad_table(cls, device):
        """Get cached gradient lookup table (256 precomputed unit vectors)."""
        if device not in cls._grad_cache:
            angles = torch.linspace(0, 2 * math.pi, 256, device=device)
            cls._grad_cache[device] = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        return cls._grad_cache[device]

    @torch.no_grad()
    def _perlin_octave_vectorized(self, x, y, freqs, device):
        """Vectorized octave for 3 channels at once. freqs is shape (3,)."""
        # x, y are (H, W), freqs is (3,)
        # Output is (3, H, W)
        n_channels = len(freqs)
        h, w = x.shape

        # Scale coordinates for each channel: (3, H, W)
        xs = x.unsqueeze(0) * freqs.view(-1, 1, 1)
        ys = y.unsqueeze(0) * freqs.view(-1, 1, 1)

        # Grid cell coords
        x0 = xs.floor().long()
        y0 = ys.floor().long()

        # Fractional part
        fx = xs - xs.floor()
        fy = ys - ys.floor()

        # Smoothstep
        u = fx * fx * (3 - 2 * fx)
        v = fy * fy * (3 - 2 * fy)

        # Hash and lookup gradients (vectorized)
        grad_table = self._get_grad_table(device)

        def grad_dot_vec(gx, gy, dx, dy):
            # Hash to 0-255 index
            h = ((gx * 374761393 + gy * 668265263) % 256).long()
            grads = grad_table[h]  # (3, H, W, 2)
            return grads[..., 0] * dx + grads[..., 1] * dy

        n00 = grad_dot_vec(x0, y0, fx, fy)
        n10 = grad_dot_vec(x0 + 1, y0, fx - 1, fy)
        n01 = grad_dot_vec(x0, y0 + 1, fx, fy - 1)
        n11 = grad_dot_vec(x0 + 1, y0 + 1, fx - 1, fy - 1)

        # Bilinear interpolation
        nx0 = n00 * (1 - u) + n10 * u
        nx1 = n01 * (1 - u) + n11 * u
        return nx0 * (1 - v) + nx1 * v

    @torch.no_grad()
    def generate(self, size, device):
        octaves = random.randint(self.min_octaves, self.max_octaves)
        lacunarity = random.uniform(1.8, 2.2)
        persistence = random.uniform(0.4, 0.6)
        base_freq = random.uniform(2, 8)

        x, y = get_coords(size, device)
        x, y = x / size, y / size

        # Process all 3 channels together
        result = torch.zeros(3, size, size, device=device)
        freqs = torch.tensor([base_freq, base_freq + 0.5, base_freq + 1.0], device=device)
        amp = 1.0
        total = 0.0

        for _ in range(octaves):
            result += self._perlin_octave_vectorized(x, y, freqs, device) * amp
            total += amp
            amp *= persistence
            freqs = freqs * lacunarity

        result = result / total

        # Normalize to [0, 1]
        result = (result + 1) / 2
        return result.clamp(0, 1)

# ============ CROSSHATCH ============

class CrosshatchNoise(NoiseGenerator):
    """High-frequency crosshatch line patterns."""

    name = "crosshatch"

    def __init__(self, min_layers=2, max_layers=5, min_spacing=4, max_spacing=50,
                 min_thickness=1, max_thickness=6):
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.min_spacing = min_spacing
        self.max_spacing = max_spacing
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness

    @torch.no_grad()
    def generate(self, size, device):
        x, y = get_coords(size, device)
        x, y = x - size / 2, y - size / 2

        n_layers = random.randint(self.min_layers, self.max_layers)
        bg_color = torch.rand(3, device=device)

        img = bg_color.view(3, 1, 1).expand(3, size, size).clone()

        for _ in range(n_layers):
            angle = random.uniform(0, math.pi)
            # Uniform sampling for spacing and thickness
            spacing = random.uniform(self.min_spacing, self.max_spacing)
            thickness = random.uniform(self.min_thickness, self.max_thickness)
            line_color = torch.rand(3, device=device)
            opacity = random.uniform(0.3, 0.9)

            # Rotate coordinates
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            rx = x * cos_a + y * sin_a

            # Create line pattern
            dist_to_line = torch.abs(rx % spacing - spacing / 2)
            mask = (dist_to_line < thickness / 2).float()

            # Blend
            for c in range(3):
                img[c] = img[c] * (1 - mask * opacity) + line_color[c] * mask * opacity

        return img.clamp(0, 1)

    @torch.no_grad()
    def generate_batch(self, size, batch_size, device):
        """Batched crosshatch."""
        x, y = get_coords(size, device)
        x, y = x - size / 2, y - size / 2
        x = x.unsqueeze(0)  # (1, H, W)
        y = y.unsqueeze(0)

        max_layers = self.max_layers
        n_layers = torch.randint(self.min_layers, max_layers + 1, (batch_size,), device=device)

        # Background colors
        bg_colors = torch.rand(batch_size, 3, 1, 1, device=device)
        img = bg_colors.expand(batch_size, 3, size, size).clone()

        # Pre-generate all layer params
        angles = torch.rand(batch_size, max_layers, device=device) * math.pi
        spacings = torch.rand(batch_size, max_layers, device=device) * (self.max_spacing - self.min_spacing) + self.min_spacing
        thicknesses = torch.rand(batch_size, max_layers, device=device) * (self.max_thickness - self.min_thickness) + self.min_thickness
        line_colors = torch.rand(batch_size, max_layers, 3, device=device)
        opacities = torch.rand(batch_size, max_layers, device=device) * 0.6 + 0.3

        layer_idx = torch.arange(max_layers, device=device).unsqueeze(0)
        valid = layer_idx < n_layers.unsqueeze(1)

        for li in range(max_layers):
            ang = angles[:, li]
            spa = spacings[:, li]
            thi = thicknesses[:, li]
            col = line_colors[:, li]
            opa = opacities[:, li]
            v = valid[:, li]

            cos_a = torch.cos(ang).view(-1, 1, 1)
            sin_a = torch.sin(ang).view(-1, 1, 1)
            rx = x * cos_a + y * sin_a

            spa = spa.view(-1, 1, 1)
            thi = thi.view(-1, 1, 1)
            dist_to_line = torch.abs(rx % spa - spa / 2)
            mask = (dist_to_line < thi / 2).float()

            # Apply validity
            mask = mask * v.view(-1, 1, 1).float()
            mask = mask.unsqueeze(1)  # (B, 1, H, W)
            opa = opa.view(-1, 1, 1, 1)
            col = col.view(-1, 3, 1, 1)

            img = img * (1 - mask * opa) + col * mask * opa

        return img.clamp(0, 1)


class MoireNoise(NoiseGenerator):
    """Moiré interference patterns from overlapping grids."""

    name = "moire"

    def __init__(self, min_freq=10, max_freq=40):
        self.min_freq = min_freq
        self.max_freq = max_freq

    @torch.no_grad()
    def generate(self, size, device):
        x, y = get_coords(size, device)
        x, y = x / size, y / size  # normalize

        result = torch.zeros(3, size, size, device=device)

        for c in range(3):
            # Two grids at slightly different angles/frequencies
            freq1 = random.uniform(self.min_freq, self.max_freq)
            freq2 = freq1 * random.uniform(0.95, 1.05)
            angle1 = random.uniform(0, math.pi)
            angle2 = angle1 + random.uniform(0.01, 0.1)

            cos1, sin1 = math.cos(angle1), math.sin(angle1)
            cos2, sin2 = math.cos(angle2), math.sin(angle2)

            grid1 = torch.sin((x * cos1 + y * sin1) * freq1 * 2 * math.pi)
            grid2 = torch.sin((x * cos2 + y * sin2) * freq2 * 2 * math.pi)

            # Interference
            result[c] = (grid1 * grid2 + 1) / 2

        return result.clamp(0, 1)

    @torch.no_grad()
    def generate_batch(self, size, batch_size, device):
        """Batched moire patterns."""
        x, y = get_coords(size, device)
        x, y = x / size, y / size
        x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        y = y.unsqueeze(0).unsqueeze(0)

        # Per image, per channel params: (B, 3)
        freq1 = torch.rand(batch_size, 3, device=device) * (self.max_freq - self.min_freq) + self.min_freq
        freq2 = freq1 * (torch.rand(batch_size, 3, device=device) * 0.1 + 0.95)
        angle1 = torch.rand(batch_size, 3, device=device) * math.pi
        angle2 = angle1 + torch.rand(batch_size, 3, device=device) * 0.09 + 0.01

        cos1 = torch.cos(angle1).view(batch_size, 3, 1, 1)
        sin1 = torch.sin(angle1).view(batch_size, 3, 1, 1)
        cos2 = torch.cos(angle2).view(batch_size, 3, 1, 1)
        sin2 = torch.sin(angle2).view(batch_size, 3, 1, 1)
        freq1 = freq1.view(batch_size, 3, 1, 1)
        freq2 = freq2.view(batch_size, 3, 1, 1)

        grid1 = torch.sin((x * cos1 + y * sin1) * freq1 * 2 * math.pi)
        grid2 = torch.sin((x * cos2 + y * sin2) * freq2 * 2 * math.pi)

        result = (grid1 * grid2 + 1) / 2
        return result.clamp(0, 1)


# ============ BULGE/PINCH DISTORTION ============

class BulgeDistortion(NoiseGenerator):
    """Apply random bulge/pinch distortions to local regions."""

    name = "bulge"

    def __init__(self, min_regions=1, max_regions=5, min_strength=-1.0, max_strength=1.0,
                 min_size=0.1, max_size=1.5, min_twist=-2.0, max_twist=2.0):
        self.min_regions = min_regions
        self.max_regions = max_regions
        self.min_strength = min_strength  # negative = pinch, positive = bulge
        self.max_strength = max_strength
        self.min_size = min_size  # as fraction of image size
        self.max_size = max_size
        self.min_twist = min_twist  # twist in radians at center
        self.max_twist = max_twist

    @torch.no_grad()
    def generate(self, size, device):
        """Generate a distortion displacement field (for applying to other images)."""
        # Returns a neutral gray image - use apply_to() for actual distortion
        return torch.ones(3, size, size, device=device) * 0.5

    @torch.no_grad()
    def apply_to(self, img, device=None):
        """Apply bulge/pinch distortion to an image."""
        if device is None:
            device = img.device

        c, h, w = img.shape
        size = h

        # Create base coordinate grid
        y, x = torch.meshgrid(torch.arange(h, device=device),
                              torch.arange(w, device=device), indexing='ij')
        x, y = x.float(), y.float()

        # Normalize to [-1, 1] for grid_sample
        x_norm = x / (w - 1) * 2 - 1
        y_norm = y / (h - 1) * 2 - 1

        # Displacement accumulators
        dx = torch.zeros_like(x)
        dy = torch.zeros_like(y)

        n_regions = random.randint(self.min_regions, self.max_regions)

        for _ in range(n_regions):
            # Random region properties
            shape = random.choice(['circle', 'square', 'ellipse', 'triangle'])
            strength = random.uniform(self.min_strength, self.max_strength)
            twist = random.uniform(self.min_twist, self.max_twist)
            region_size = random.uniform(self.min_size, self.max_size) * size

            # Random center (allow anywhere, even outside for large global effects)
            cx = random.uniform(-region_size * 0.3, w + region_size * 0.3)
            cy = random.uniform(-region_size * 0.3, h + region_size * 0.3)

            # Distance from center
            dist_x = x - cx
            dist_y = y - cy

            if shape == 'circle':
                dist = torch.sqrt(dist_x**2 + dist_y**2)
                radius = region_size / 2
                mask = (dist < radius).float()
                # Smooth falloff
                falloff = (1 - (dist / radius).clamp(0, 1)) ** 2
                falloff = falloff * mask

            elif shape == 'square':
                angle = random.uniform(0, math.pi / 2)
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                rx = dist_x * cos_a + dist_y * sin_a
                ry = -dist_x * sin_a + dist_y * cos_a
                half_size = region_size / 2
                mask = ((rx.abs() < half_size) & (ry.abs() < half_size)).float()
                # Distance from edge for falloff
                edge_dist = torch.min(half_size - rx.abs(), half_size - ry.abs()).clamp(0)
                falloff = (edge_dist / (half_size * 0.5)).clamp(0, 1) ** 2 * mask
                dist = torch.sqrt(dist_x**2 + dist_y**2)
                radius = region_size / 2

            elif shape == 'ellipse':
                angle = random.uniform(0, math.pi)
                aspect = random.uniform(0.3, 1.0)
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                rx = dist_x * cos_a + dist_y * sin_a
                ry = (-dist_x * sin_a + dist_y * cos_a) / aspect
                dist = torch.sqrt(rx**2 + ry**2)
                radius = region_size / 2
                mask = (dist < radius).float()
                falloff = (1 - (dist / radius).clamp(0, 1)) ** 2 * mask

            elif shape == 'triangle':
                angle = random.uniform(0, 2 * math.pi)
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                rx = dist_x * cos_a + dist_y * sin_a
                ry = -dist_x * sin_a + dist_y * cos_a
                # Triangle mask
                tri_h = region_size
                in_tri = (ry > -tri_h/3) & (ry < tri_h*2/3) & (rx.abs() < (tri_h*2/3 - ry) * 0.6)
                mask = in_tri.float()
                dist = torch.sqrt(dist_x**2 + dist_y**2)
                radius = region_size / 2
                falloff = mask * (1 - (dist / radius).clamp(0, 1)).clamp(0, 1)

            # Calculate displacement (bulge pushes away from center, pinch pulls toward)
            dist_safe = dist.clamp(min=1e-6)
            dir_x = dist_x / dist_safe
            dir_y = dist_y / dist_safe

            # Bulge displacement
            bulge_mag = falloff * strength * region_size * 0.3
            dx += dir_x * bulge_mag
            dy += dir_y * bulge_mag

            # Twist displacement (rotate around center)
            twist_angle = falloff * twist
            cos_t = torch.cos(twist_angle)
            sin_t = torch.sin(twist_angle)
            # Rotated position relative to center
            new_dist_x = dist_x * cos_t - dist_y * sin_t
            new_dist_y = dist_x * sin_t + dist_y * cos_t
            # Displacement is difference from original
            dx += (new_dist_x - dist_x) * falloff
            dy += (new_dist_y - dist_y) * falloff

        # Apply displacement to normalized coords
        x_warped = x_norm + dx / (w / 2)
        y_warped = y_norm + dy / (h / 2)

        # Stack into grid format for grid_sample
        grid = torch.stack([x_warped, y_warped], dim=-1).unsqueeze(0)

        # Sample image at warped coordinates
        img_batch = img.unsqueeze(0)
        warped = torch.nn.functional.grid_sample(
            img_batch, grid, mode='bilinear', padding_mode='border', align_corners=True
        )

        return warped.squeeze(0)


# ============ BINARY NOISE ============

class BinaryNoise(NoiseGenerator):
    """Pure black/white random pixels - correlated channels (posterized look)."""

    name = "binary"

    def __init__(self, threshold_min=0.3, threshold_max=0.7):
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max

    @torch.no_grad()
    def generate(self, size, device):
        # Random threshold per channel for color variety
        thresholds = torch.empty(3, 1, 1, device=device).uniform_(self.threshold_min, self.threshold_max)
        noise = torch.rand(3, size, size, device=device)
        return (noise > thresholds).float()


class IndependentChannelPatterns(NoiseGenerator):
    """Each RGB channel gets a different pattern - completely independent structure."""

    name = "independent_channels"

    # Only use fast generators (<4ms) for this composite pattern
    FAST_CANDIDATES = None  # Will be populated on first use

    def __init__(self, pattern_classes=None):
        # Classes that can generate patterns (will be converted to grayscale per channel)
        self.pattern_classes = pattern_classes

    @torch.no_grad()
    def generate(self, size, device):
        # Lazy import to avoid circular deps, use only fast generators
        if IndependentChannelPatterns.FAST_CANDIDATES is None:
            from noise_classes import (VoronoiNoise, GeometricShapes, PatternNoise,
                                       CrosshatchNoise, MoireNoise, GradientNoise, FBMNoise)
            # Exclude: PinkNoise (16ms), FBMPerlin (13ms), BlurredNoise (9ms)
            IndependentChannelPatterns.FAST_CANDIDATES = [
                VoronoiNoise, GeometricShapes, PatternNoise,
                CrosshatchNoise, MoireNoise, GradientNoise, FBMNoise
            ]

        result = torch.zeros(3, size, size, device=device)

        for c in range(3):
            # Pick random pattern class for this channel
            cls = random.choice(IndependentChannelPatterns.FAST_CANDIDATES)
            pattern = cls().generate(size, device)
            # Convert to grayscale and use for this channel
            gray = pattern.mean(dim=0)  # average RGB to get grayscale
            result[c] = gray

        return result


# ============ REGISTRY ============

ALL_GENERATORS = [
    VoronoiNoise,
    PinkNoise,
    GeometricShapes,
    PatternNoise,
    BlurredNoise,
    SolidColor,
    GradientNoise,
    FBMNoise,
    FBMPerlin,
    CrosshatchNoise,
    MoireNoise,
    BulgeDistortion,
    IndependentChannelPatterns,
]


def get_all_generators(**kwargs):
    """Instantiate all generators with default params."""
    return [cls(**kwargs) if kwargs else cls() for cls in ALL_GENERATORS]


def get_generator_by_name(name):
    """Get generator class by name."""
    for cls in ALL_GENERATORS:
        if cls.name == name:
            return cls
    raise ValueError(f"Unknown generator: {name}")


# ============ COMBINED GENERATOR ============

class CombinedGenerator(NoiseGenerator):
    """Randomly picks from multiple generators."""

    name = "combined"

    def __init__(self, generators=None):
        self.generators = generators or [cls() for cls in ALL_GENERATORS]

    def generate(self, size, device):
        gen = random.choice(self.generators)
        return gen.generate(size, device)

    def generate_batch(self, size, batch_size, device):
        return torch.stack([self.generate(size, device) for _ in range(batch_size)])


# ============ QUICK TEST ============

if __name__ == "__main__":
    import numpy as np
    from PIL import Image

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    size = 256
    cols = 9

    all_rows = []
    for GenClass in ALL_GENERATORS:
        gen = GenClass()
        print(f"Generating {gen.name}...")
        row_imgs = []
        for _ in range(cols):
            img = gen.generate(size, device)
            img_np = img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            row_imgs.append(img_np)
        row = np.concatenate(row_imgs, axis=1)
        all_rows.append(row)

    grid = np.concatenate(all_rows, axis=0)
    final = Image.fromarray((grid * 255).astype(np.uint8))
    final.save("noise_classes_test.png")
    print(f"Saved noise_classes_test.png ({len(ALL_GENERATORS)} rows x {cols} cols)")
