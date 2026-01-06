# src/procedural.py
"""
Procedural noise/pattern generator iterator.

Wraps the batched GPU generators from clean_multiscale_dsets as a data iterator
compatible with CompositeIterator. Produces varied synthetic images through
layered composition of patterns, noise, and distortions.

Each generated image is a composition of:
1. Base layer: solid, gradient, pattern, noise, shapes, etc.
2. 1-10 effect layers: blended using photoshop-style modes
3. Optional distortions: bulge/pinch/twist

All operations are batched on GPU for high throughput.
"""

import math
import random
from typing import Dict, Any, List, Optional

import torch
import torch.nn.functional as F

from .model import ContextBlock


# ============ COORDINATE UTILITIES ============

_coord_cache = {}


def get_coords(size: int, device: torch.device):
    """Get (x, y) coordinate grids (cached)."""
    key = (size, device)
    if key not in _coord_cache:
        y, x = torch.meshgrid(
            torch.arange(size, device=device),
            torch.arange(size, device=device),
            indexing='ij'
        )
        _coord_cache[key] = (x.float(), y.float())
    return _coord_cache[key]


# ============ BATCHED GENERATORS ============

@torch.no_grad()
def generate_pink_noise_batch(batch_size: int, size: int, device: torch.device,
                               min_sharp: float = 0.5, max_sharp: float = 2.5):
    """Batched pink noise via FFT."""
    sharpness = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(min_sharp, max_sharp)
    noise = torch.randn(batch_size, 3, size, size, device=device)
    fft = torch.fft.fft2(noise)

    freq_y = torch.fft.fftfreq(size, device=device).view(1, 1, -1, 1)
    freq_x = torch.fft.fftfreq(size, device=device).view(1, 1, 1, -1)
    freq_mag = torch.sqrt(freq_y**2 + freq_x**2)
    freq_mag[..., 0, 0] = 1

    falloff = 1.0 / (freq_mag ** sharpness)
    falloff[..., 0, 0] = 0

    filtered = fft * falloff
    result = torch.fft.ifft2(filtered).real

    result = result.view(batch_size, 3, -1)
    mins = result.min(dim=-1, keepdim=True)[0]
    maxs = result.max(dim=-1, keepdim=True)[0]
    result = (result - mins) / (maxs - mins + 1e-8)
    return result.view(batch_size, 3, size, size)


@torch.no_grad()
def generate_blurred_noise_batch(batch_size: int, size: int, device: torch.device):
    """Batched blurred noise using FFT blur."""
    log_noise_min, log_noise_max = math.log(0.01), math.log(1.0)
    log_blur_min, log_blur_max = math.log(0.5), math.log(160.0)
    log_color_min, log_color_max = math.log(0.5), math.log(2.0)

    noise_levels = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(log_noise_min, log_noise_max).exp()
    blur_sigmas = torch.empty(batch_size, device=device).uniform_(log_blur_min, log_blur_max).exp()
    color_scales = torch.empty(batch_size, 3, 1, 1, device=device).uniform_(log_color_min, log_color_max).exp()

    noise = torch.randn(batch_size, 3, size, size, device=device) * noise_levels

    freq_y = torch.fft.fftfreq(size, device=device).view(1, 1, -1, 1)
    freq_x = torch.fft.fftfreq(size, device=device).view(1, 1, 1, -1)
    freq_sq = freq_y**2 + freq_x**2

    sigma_sq = (blur_sigmas ** 2).view(-1, 1, 1, 1)
    gauss_freq = torch.exp(-2 * (math.pi ** 2) * sigma_sq * freq_sq)

    fft = torch.fft.fft2(noise)
    blurred = torch.fft.ifft2(fft * gauss_freq).real

    result = blurred * color_scales
    return (0.5 + result).clamp(0, 1)


@torch.no_grad()
def generate_solid_batch(batch_size: int, size: int, device: torch.device):
    """Batched solid colors."""
    colors = torch.rand(batch_size, 3, 1, 1, device=device)
    return colors.expand(batch_size, 3, size, size).clone()


@torch.no_grad()
def generate_gradient_batch(batch_size: int, size: int, device: torch.device):
    """Batched gradients (linear, radial, angular)."""
    y = torch.arange(size, device=device).view(1, -1, 1).float() / size - 0.5
    x = torch.arange(size, device=device).view(1, 1, -1).float() / size - 0.5

    gtypes = torch.randint(0, 3, (batch_size,), device=device)
    color1 = torch.rand(batch_size, 3, device=device)
    color2 = torch.rand(batch_size, 3, device=device)

    angles = torch.rand(batch_size, device=device) * math.pi * 2
    cx = torch.rand(batch_size, device=device) * 0.6 - 0.3
    cy = torch.rand(batch_size, device=device) * 0.6 - 0.3
    radial_scale = torch.rand(batch_size, device=device) * 2 + 1

    cos_a = torch.cos(angles).view(-1, 1, 1)
    sin_a = torch.sin(angles).view(-1, 1, 1)
    t_linear = (x * cos_a + y * sin_a + 0.5).clamp(0, 1)

    cx_exp = cx.view(-1, 1, 1)
    cy_exp = cy.view(-1, 1, 1)
    t_radial = (torch.sqrt((x - cx_exp)**2 + (y - cy_exp)**2) * radial_scale.view(-1, 1, 1)).clamp(0, 1)
    t_angular = (torch.atan2(y - cy_exp, x - cx_exp) / math.pi + 1) / 2

    gtypes_exp = gtypes.view(-1, 1, 1)
    t = torch.where(gtypes_exp == 0, t_linear,
                    torch.where(gtypes_exp == 1, t_radial, t_angular))

    c1 = color1.view(batch_size, 3, 1, 1)
    c2 = color2.view(batch_size, 3, 1, 1)
    t = t.unsqueeze(1)

    return c1 * (1 - t) + c2 * t


@torch.no_grad()
def generate_moire_batch(batch_size: int, size: int, device: torch.device,
                          min_freq: int = 10, max_freq: int = 40):
    """Batched moiré interference patterns."""
    x = torch.arange(size, device=device).view(1, 1, 1, -1).float() / size
    y = torch.arange(size, device=device).view(1, 1, -1, 1).float() / size

    freq1 = torch.rand(batch_size, 3, 1, 1, device=device) * (max_freq - min_freq) + min_freq
    freq2 = freq1 * (torch.rand(batch_size, 3, 1, 1, device=device) * 0.1 + 0.95)
    angle1 = torch.rand(batch_size, 3, 1, 1, device=device) * math.pi
    angle2 = angle1 + torch.rand(batch_size, 3, 1, 1, device=device) * 0.09 + 0.01

    cos1, sin1 = torch.cos(angle1), torch.sin(angle1)
    cos2, sin2 = torch.cos(angle2), torch.sin(angle2)

    grid1 = torch.sin((x * cos1 + y * sin1) * freq1 * 2 * math.pi)
    grid2 = torch.sin((x * cos2 + y * sin2) * freq2 * 2 * math.pi)

    return ((grid1 * grid2 + 1) / 2).clamp(0, 1)


@torch.no_grad()
def generate_crosshatch_batch(batch_size: int, size: int, device: torch.device):
    """Batched crosshatch line patterns."""
    x = torch.arange(size, device=device).float() - size / 2
    y = torch.arange(size, device=device).float() - size / 2
    x = x.view(1, 1, -1)
    y = y.view(1, -1, 1)

    max_layers = 5
    n_layers = torch.randint(2, max_layers + 1, (batch_size,), device=device)

    bg_colors = torch.rand(batch_size, 3, 1, 1, device=device)
    img = bg_colors.expand(batch_size, 3, size, size).clone()

    angles = torch.rand(batch_size, max_layers, device=device) * math.pi
    spacings = torch.rand(batch_size, max_layers, device=device) * 46 + 4
    thicknesses = torch.rand(batch_size, max_layers, device=device) * 5 + 1
    line_colors = torch.rand(batch_size, max_layers, 3, device=device)
    opacities = torch.rand(batch_size, max_layers, device=device) * 0.6 + 0.3

    layer_idx = torch.arange(max_layers, device=device).unsqueeze(0)
    valid = layer_idx < n_layers.unsqueeze(1)

    for li in range(max_layers):
        ang = angles[:, li].view(-1, 1, 1)
        spa = spacings[:, li].view(-1, 1, 1)
        thi = thicknesses[:, li].view(-1, 1, 1)
        col = line_colors[:, li].view(-1, 3, 1, 1)
        opa = opacities[:, li].view(-1, 1, 1, 1)
        v = valid[:, li].view(-1, 1, 1).float()

        rx = x * torch.cos(ang) + y * torch.sin(ang)
        dist_to_line = torch.abs(rx % spa - spa / 2)
        mask = ((dist_to_line < thi / 2).float() * v).unsqueeze(1)

        img = img * (1 - mask * opa) + col * mask * opa

    return img.clamp(0, 1)


@torch.no_grad()
def generate_fbm_batch(batch_size: int, size: int, device: torch.device):
    """Batched Fractal Brownian Motion noise."""
    x = torch.arange(size, device=device).float() / size
    y = torch.arange(size, device=device).float() / size
    x = x.view(1, 1, 1, -1)
    y = y.view(1, 1, -1, 1)

    octaves = 8
    n_octaves = torch.randint(3, octaves + 1, (batch_size,), device=device)
    lacunarity = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(1.8, 2.2)
    persistence = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(0.4, 0.6)
    base_freq = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(2, 8)
    phase = torch.rand(batch_size, 3, 2, device=device) * 1000

    result = torch.zeros(batch_size, 3, size, size, device=device)
    amplitude = torch.ones(batch_size, 1, 1, 1, device=device)
    freq = base_freq
    max_val = torch.zeros(batch_size, 1, 1, 1, device=device)

    for oct_idx in range(octaves):
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


@torch.no_grad()
def generate_patterns_batch(batch_size: int, size: int, device: torch.device):
    """Batched geometric patterns (checker, stripes, grid, dots)."""
    x = torch.arange(size, device=device).float() - size / 2
    y = torch.arange(size, device=device).float() - size / 2
    x = x.view(1, 1, -1)
    y = y.view(1, -1, 1)

    ptypes = torch.randint(0, 4, (batch_size,), device=device)
    scales = torch.empty(batch_size, 1, 1, device=device).uniform_(math.log(8), math.log(64)).exp()
    angles = torch.rand(batch_size, 1, 1, device=device) * math.pi * 2
    shift_x = torch.rand(batch_size, 1, 1, device=device) * size - size * 0.5
    shift_y = torch.rand(batch_size, 1, 1, device=device) * size - size * 0.5
    aspects = torch.rand(batch_size, 1, 1, device=device) * 1.5 + 0.5
    color1 = torch.rand(batch_size, 3, 1, 1, device=device)
    color2 = torch.rand(batch_size, 3, 1, 1, device=device)

    x = x - shift_x
    y = y - shift_y
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    rx = (x * cos_a + y * sin_a) * aspects
    ry = -x * sin_a + y * cos_a

    checker = ((rx // scales).long() + (ry // scales).long()) % 2 == 0
    stripes = (rx // scales).long() % 2 == 0
    line_w = scales * 0.25
    grid = (rx % scales < line_w) | (ry % scales < line_w)
    dot_r = scales * 0.35
    cx = (rx // scales + 0.5) * scales
    cy = (ry // scales + 0.5) * scales
    dots = torch.sqrt((rx - cx)**2 + (ry - cy)**2) < dot_r

    ptypes_exp = ptypes.view(-1, 1, 1)
    mask = torch.where(ptypes_exp == 0, checker,
                       torch.where(ptypes_exp == 1, stripes,
                                   torch.where(ptypes_exp == 2, grid, dots)))

    mask_f = mask.float().unsqueeze(1)
    return color1 * mask_f + color2 * (1 - mask_f)


@torch.no_grad()
def generate_shapes_batch(batch_size: int, size: int, device: torch.device):
    """Batched random geometric shapes."""
    x = torch.arange(size, device=device).float()
    y = torch.arange(size, device=device).float()
    x = x.view(1, 1, -1).expand(batch_size, size, size)
    y = y.view(1, -1, 1).expand(batch_size, size, size)

    img = torch.zeros(batch_size, 3, size, size, device=device)
    max_shapes = 15
    n_shapes = torch.randint(3, max_shapes + 1, (batch_size,), device=device)

    shape_types = torch.randint(0, 3, (batch_size, max_shapes), device=device)
    colors = torch.rand(batch_size, max_shapes, 3, device=device)
    cx = torch.rand(batch_size, max_shapes, device=device) * size
    cy = torch.rand(batch_size, max_shapes, device=device) * size
    s = torch.rand(batch_size, max_shapes, device=device) * (size * 0.25) + (size * 0.05)
    angles = torch.rand(batch_size, max_shapes, device=device) * math.pi
    alphas = torch.rand(batch_size, max_shapes, device=device) * 0.5 + 0.5

    shape_idx = torch.arange(max_shapes, device=device).unsqueeze(0)
    valid = shape_idx < n_shapes.unsqueeze(1)

    for si in range(max_shapes):
        st = shape_types[:, si].view(-1, 1, 1)
        col = colors[:, si].view(-1, 3, 1, 1)
        cxi = cx[:, si].view(-1, 1, 1)
        cyi = cy[:, si].view(-1, 1, 1)
        si_size = s[:, si].view(-1, 1, 1)
        ang = angles[:, si]
        alph = alphas[:, si].view(-1, 1, 1, 1)
        v = valid[:, si].view(-1, 1, 1)

        dx = x - cxi
        dy = y - cyi
        cos_a = torch.cos(ang).view(-1, 1, 1)
        sin_a = torch.sin(ang).view(-1, 1, 1)
        rx = dx * cos_a + dy * sin_a
        ry = -dx * sin_a + dy * cos_a

        circle = (dx**2 + dy**2) < si_size**2
        square = (rx.abs() < si_size) & (ry.abs() < si_size)
        tri = (ry > -si_size * 0.5) & (ry < si_size) & (rx.abs() < (si_size - ry) * 0.6)

        mask = torch.where(st == 0, circle, torch.where(st == 1, square, tri))
        mask = (mask & v).float().unsqueeze(1)

        img = img * (1 - mask * alph) + col * mask * alph

    return img


@torch.no_grad()
def generate_voronoi_batch(batch_size: int, size: int, device: torch.device,
                            min_cells: int = 5, max_cells: int = 50):
    """Batched Voronoi cell patterns."""
    # Sequential fallback - Voronoi is hard to batch efficiently
    result = torch.zeros(batch_size, 3, size, size, device=device)

    y, x = torch.meshgrid(torch.arange(size, device=device),
                          torch.arange(size, device=device), indexing='ij')
    coords = torch.stack([x, y], dim=-1).float().view(-1, 2)

    for b in range(batch_size):
        n_cells = random.randint(min_cells, max_cells)
        centers = torch.rand(n_cells, 2, device=device) * size
        colors = torch.rand(n_cells, 3, device=device)

        dists = torch.cdist(coords, centers)
        closest = dists.argmin(dim=1)
        result[b] = colors[closest].view(size, size, 3).permute(2, 0, 1)

    return result


# ============ BLEND MODES ============

BLEND_MODES = ['overlay', 'add', 'subtract', 'multiply', 'screen',
               'hard_light', 'soft_light', 'difference', 'exclusion']


def apply_blend_batched(base, overlay, mode: str, strength):
    """Batched blend: base/overlay are (B, 3, H, W), strength is (B, 1, 1, 1)."""
    if mode == 'overlay':
        mask = base < 0.5
        result = torch.where(mask, 2 * base * overlay,
                             1 - 2 * (1 - base) * (1 - overlay))
        return base * (1 - strength) + result * strength
    elif mode == 'add':
        return (base + (overlay - 0.5) * strength).clamp(0, 1)
    elif mode == 'subtract':
        return (base - (overlay - 0.5) * strength).clamp(0, 1)
    elif mode == 'multiply':
        return base * (1 - strength) + (base * overlay) * strength
    elif mode == 'screen':
        return base * (1 - strength) + (1 - (1 - base) * (1 - overlay)) * strength
    elif mode == 'hard_light':
        mask = overlay < 0.5
        result = torch.where(mask, 2 * base * overlay,
                             1 - 2 * (1 - base) * (1 - overlay))
        return base * (1 - strength) + result * strength
    elif mode == 'soft_light':
        result = torch.where(overlay < 0.5,
                             base - (1 - 2 * overlay) * base * (1 - base),
                             base + (2 * overlay - 1) * (torch.sqrt(base.clamp(min=1e-6)) - base))
        return base * (1 - strength) + result * strength
    elif mode == 'difference':
        return base * (1 - strength) + torch.abs(base - overlay) * strength
    elif mode == 'exclusion':
        result = base + overlay - 2 * base * overlay
        return base * (1 - strength) + result * strength
    return base


# ============ GENERATOR REGISTRY ============

BASE_GENERATORS = {
    'pink_noise': generate_pink_noise_batch,
    'blurred_noise': generate_blurred_noise_batch,
    'solid': generate_solid_batch,
    'gradient': generate_gradient_batch,
    'patterns': generate_patterns_batch,
    'shapes': generate_shapes_batch,
    'voronoi': generate_voronoi_batch,
}

EFFECT_GENERATORS = {
    'fbm': generate_fbm_batch,
    'pink_noise': generate_pink_noise_batch,
    'moire': generate_moire_batch,
    'crosshatch': generate_crosshatch_batch,
    'patterns': generate_patterns_batch,
    'shapes': generate_shapes_batch,
    'voronoi': generate_voronoi_batch,
}


# ============ HIGH-LEVEL COMPOSITION ============

@torch.no_grad()
def generate_base_batch(batch_size: int, size: int, device: torch.device,
                         generator_types: Optional[List[str]] = None):
    """Generate base images from specified or all generator types."""
    gen_names = generator_types or list(BASE_GENERATORS.keys())
    result = torch.zeros(batch_size, 3, size, size, device=device)
    assignments = [random.choice(gen_names) for _ in range(batch_size)]

    for name in set(assignments):
        indices = [i for i, a in enumerate(assignments) if a == name]
        n = len(indices)
        if name in BASE_GENERATORS:
            batch = BASE_GENERATORS[name](n, size, device)
            for j, idx in enumerate(indices):
                result[idx] = batch[j]

    return result


@torch.no_grad()
def generate_effect_batch(batch_size: int, size: int, device: torch.device,
                           effect_types: Optional[List[str]] = None):
    """Generate effect images for blending."""
    eff_names = effect_types or list(EFFECT_GENERATORS.keys())
    result = torch.zeros(batch_size, 3, size, size, device=device)
    assignments = [random.choice(eff_names) for _ in range(batch_size)]

    for name in set(assignments):
        indices = [i for i, a in enumerate(assignments) if a == name]
        n = len(indices)
        if name in EFFECT_GENERATORS:
            batch = EFFECT_GENERATORS[name](n, size, device)
            for j, idx in enumerate(indices):
                result[idx] = batch[j]

    return result


@torch.no_grad()
def generate_varied_batch(batch_size: int, size: int, device: torch.device,
                           min_layers: int = 1, max_layers: int = 10,
                           base_types: Optional[List[str]] = None,
                           effect_types: Optional[List[str]] = None,
                           blend_modes: Optional[List[str]] = None):
    """
    Generate a batch of varied synthetic images.

    Each image is a composition of:
    1. Base layer (solid, gradient, pattern, noise, etc.)
    2. 1-10 effect layers blended using photoshop-style modes

    Args:
        batch_size: Number of images to generate
        size: Image resolution (square)
        device: Torch device
        min_layers: Minimum number of effect layers
        max_layers: Maximum number of effect layers
        base_types: List of base generator names (None = all)
        effect_types: List of effect generator names (None = all)
        blend_modes: List of blend mode names (None = all)

    Returns:
        Tensor of shape (batch_size, 3, size, size) in [0, 1]
    """
    images = generate_base_batch(batch_size, size, device, base_types)
    modes = blend_modes or BLEND_MODES

    n_layers = torch.randint(min_layers, max_layers + 1, (batch_size,))
    max_n = n_layers.max().item()

    for layer in range(max_n):
        active = n_layers > layer
        n_active = active.sum().item()
        if n_active == 0:
            break

        active_indices = torch.where(active)[0]

        # Generate effects for active images
        effects = generate_effect_batch(len(active_indices), size, device, effect_types)
        layer_modes = [random.choice(modes) for _ in range(len(active_indices))]
        strengths = torch.empty(len(active_indices), 1, 1, 1, device=device).uniform_(0.2, 0.8)

        for i, idx in enumerate(active_indices):
            images[idx] = apply_blend_batched(
                images[idx:idx+1], effects[i:i+1], layer_modes[i], strengths[i:i+1]
            ).squeeze(0)

    return images.clamp(0, 1)


# ============ ITERATOR CLASS ============

class ProceduralIterator:
    """
    Data iterator for procedural noise/pattern generation.

    Compatible with CompositeIterator - produces ContextBlocks with
    varied synthetic images at requested resolutions.

    Config params:
        seed: Random seed for reproducibility
        min_layers: Minimum effect layers per image (default: 1)
        max_layers: Maximum effect layers per image (default: 10)
        base_types: List of base generator names (default: all)
        effect_types: List of effect generator names (default: all)
        blend_modes: List of blend modes to use (default: all)
        text_position: "none", "prefix", "suffix", "random" (default: none)
    """

    def __init__(self, device: torch.device, config: Dict[str, Any]):
        self.device = device
        self.config = config
        self.seed = config.get('seed', 42)
        self.min_layers = config.get('min_layers', 1)
        self.max_layers = config.get('max_layers', 10)
        self.base_types = config.get('base_types', None)  # None = all
        self.effect_types = config.get('effect_types', None)
        self.blend_modes = config.get('blend_modes', None)
        self.text_position = config.get('text_position', 'none')

        # Initialize RNG
        random.seed(self.seed)
        self._call_count = 0

    def generate_batch_list(self, batch_size: int, resolution: int = 64,
                             **kwargs) -> List[ContextBlock]:
        """
        Generate a batch of procedural images as ContextBlocks.

        Args:
            batch_size: Number of images to generate
            resolution: Output resolution (square images)
            **kwargs: Additional args (start_group_id, etc.)

        Returns:
            List of ContextBlocks containing generated images
        """
        start_group_id = kwargs.get('start_group_id', 0)

        # Generate images
        images = generate_varied_batch(
            batch_size=batch_size,
            size=resolution,
            device=self.device,
            min_layers=self.min_layers,
            max_layers=self.max_layers,
            base_types=self.base_types,
            effect_types=self.effect_types,
            blend_modes=self.blend_modes,
        )

        # Wrap in ContextBlocks
        blocks = []
        for i in range(batch_size):
            gid = start_group_id + i
            block = ContextBlock(
                content=images[i],
                type='latent',
                causal=False,
                shape_meta=(resolution, resolution),
                group_id=gid,
                id=f"proc_{self._call_count}_{gid}"
            )
            blocks.append(block)

        self._call_count += 1
        return blocks
