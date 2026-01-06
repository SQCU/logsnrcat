#!/usr/bin/env python3
"""Batched generation - actually use GPU parallelism."""

import random
import math
import torch
import numpy as np
from PIL import Image

from .noise_classes import (
    VoronoiNoise, PinkNoise, GeometricShapes, PatternNoise,
    BlurredNoise, SolidColor, GradientNoise, FBMNoise, FBMPerlin,
    CrosshatchNoise, MoireNoise, BulgeDistortion, IndependentChannelPatterns
)

# Config
IMAGE_SIZE = 256
MIN_LAYERS = 1
MAX_LAYERS = 10

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Base generators
BASE_CLASSES = [VoronoiNoise, PinkNoise, GeometricShapes, PatternNoise,
                BlurredNoise, SolidColor, GradientNoise]

# Effect generators (blendable)
EFFECT_CLASSES = [FBMNoise, FBMPerlin, CrosshatchNoise, MoireNoise,
                  PinkNoise, VoronoiNoise, PatternNoise, GeometricShapes]

BLEND_MODES = ['overlay', 'add', 'subtract', 'multiply', 'screen',
               'hard_light', 'soft_light', 'difference', 'exclusion']


def apply_blend_batched(base, overlay, mode, strength):
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


# ============ INLINE BATCHED GENERATORS (the fast ones) ============

@torch.no_grad()
def generate_pink_noise_batch(batch_size, size, device, min_sharp=0.5, max_sharp=2.5):
    """Batched pink noise - single FFT call for whole batch."""
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
def generate_blurred_noise_batch(batch_size, size, device):
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
def generate_solid_batch(batch_size, size, device):
    """Batched solid colors."""
    colors = torch.rand(batch_size, 3, 1, 1, device=device)
    return colors.expand(batch_size, 3, size, size).clone()


@torch.no_grad()
def generate_gradient_batch(batch_size, size, device):
    """Batched gradients."""
    # x, y: (1, H, W) -> will broadcast with (B, 1, 1)
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
    t_linear = (x * cos_a + y * sin_a + 0.5).clamp(0, 1)  # (B, H, W)

    cx_exp = cx.view(-1, 1, 1)
    cy_exp = cy.view(-1, 1, 1)
    t_radial = (torch.sqrt((x - cx_exp)**2 + (y - cy_exp)**2) * radial_scale.view(-1, 1, 1)).clamp(0, 1)
    t_angular = (torch.atan2(y - cy_exp, x - cx_exp) / math.pi + 1) / 2

    gtypes_exp = gtypes.view(-1, 1, 1)
    t = torch.where(gtypes_exp == 0, t_linear,
        torch.where(gtypes_exp == 1, t_radial, t_angular))  # (B, H, W)

    c1 = color1.view(batch_size, 3, 1, 1)
    c2 = color2.view(batch_size, 3, 1, 1)
    t = t.unsqueeze(1)  # (B, 1, H, W)

    return c1 * (1 - t) + c2 * t


@torch.no_grad()
def generate_moire_batch(batch_size, size, device, min_freq=10, max_freq=40):
    """Batched moire patterns."""
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
def generate_crosshatch_batch(batch_size, size, device):
    """Batched crosshatch."""
    # x, y: (1, H, W)
    x = torch.arange(size, device=device).float() - size / 2
    y = torch.arange(size, device=device).float() - size / 2
    x = x.view(1, 1, -1)  # (1, 1, W)
    y = y.view(1, -1, 1)  # (1, H, 1)

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

        rx = x * torch.cos(ang) + y * torch.sin(ang)  # (B, H, W)
        dist_to_line = torch.abs(rx % spa - spa / 2)
        mask = ((dist_to_line < thi / 2).float() * v).unsqueeze(1)  # (B, 1, H, W)

        img = img * (1 - mask * opa) + col * mask * opa

    return img.clamp(0, 1)


@torch.no_grad()
def generate_fbm_batch(batch_size, size, device):
    """Batched FBM noise."""
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
def generate_patterns_batch(batch_size, size, device):
    """Batched patterns."""
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
def generate_shapes_batch(batch_size, size, device):
    """Batched shapes."""
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


# Map classes to batch functions
BATCH_FUNCS = {
    PinkNoise: generate_pink_noise_batch,
    BlurredNoise: generate_blurred_noise_batch,
    SolidColor: generate_solid_batch,
    GradientNoise: generate_gradient_batch,
    MoireNoise: generate_moire_batch,
    CrosshatchNoise: generate_crosshatch_batch,
    FBMNoise: generate_fbm_batch,
    PatternNoise: generate_patterns_batch,
    GeometricShapes: generate_shapes_batch,
}


@torch.no_grad()
def generate_base_batch(batch_size, size, device):
    """Generate base images using batched generators."""
    result = torch.zeros(batch_size, 3, size, size, device=device)
    assignments = [random.choice(BASE_CLASSES) for _ in range(batch_size)]

    for cls in set(assignments):
        indices = [i for i, c in enumerate(assignments) if c == cls]
        n = len(indices)

        if cls in BATCH_FUNCS:
            batch = BATCH_FUNCS[cls](n, size, device)
        else:
            # Fallback to sequential
            batch = torch.stack([cls().generate(size, device) for _ in range(n)])

        for j, idx in enumerate(indices):
            result[idx] = batch[j]

    return result


@torch.no_grad()
def generate_effect_batch(batch_size, size, device):
    """Generate effect images using batched generators."""
    result = torch.zeros(batch_size, 3, size, size, device=device)
    assignments = [random.choice(EFFECT_CLASSES) for _ in range(batch_size)]

    for cls in set(assignments):
        indices = [i for i, c in enumerate(assignments) if c == cls]
        n = len(indices)

        if cls in BATCH_FUNCS:
            batch = BATCH_FUNCS[cls](n, size, device)
        else:
            batch = torch.stack([cls().generate(size, device) for _ in range(n)])

        for j, idx in enumerate(indices):
            result[idx] = batch[j]

    return result


@torch.no_grad()
def generate_batch_varied(batch_size, size, device):
    """Generate a full batch with 1-4 effect layers each."""
    images = generate_base_batch(batch_size, size, device)

    n_layers = torch.randint(MIN_LAYERS, MAX_LAYERS + 1, (batch_size,))
    max_layers = n_layers.max().item()

    bulge = BulgeDistortion(min_regions=1, max_regions=3,
                            min_strength=-1.0, max_strength=1.0,
                            min_size=0.15, max_size=1.2,
                            min_twist=-2.0, max_twist=2.0)

    for layer in range(max_layers):
        active = n_layers > layer
        n_active = active.sum().item()
        if n_active == 0:
            break

        active_indices = torch.where(active)[0]
        is_effect = torch.rand(n_active) < 0.7

        # Effects
        effect_indices = active_indices[is_effect]
        if len(effect_indices) > 0:
            effects = generate_effect_batch(len(effect_indices), size, device)
            modes = [random.choice(BLEND_MODES) for _ in range(len(effect_indices))]
            strengths = torch.empty(len(effect_indices), 1, 1, 1, device=device).uniform_(0.2, 0.8)

            for i, idx in enumerate(effect_indices):
                images[idx] = apply_blend_batched(
                    images[idx:idx+1], effects[i:i+1], modes[i], strengths[i:i+1]
                ).squeeze(0)

        # Distortions
        distort_indices = active_indices[~is_effect]
        for idx in distort_indices:
            images[idx] = bulge.apply_to(images[idx], device)

    return images.clamp(0, 1)


@torch.no_grad()
def main():
    import time

    print(f"Device: {DEVICE}")

    # Warmup
    print("Warmup...")
    _ = generate_batch_varied(32, IMAGE_SIZE, DEVICE)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()

    # Generate 256
    print("Generating 256 images...")
    start = time.perf_counter()
    images = generate_batch_varied(256, IMAGE_SIZE, DEVICE)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"\n=== RESULTS ===")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Per image: {elapsed/256*1000:.1f}ms")
    print(f"Throughput: {256/elapsed:.1f} images/sec")

    # Save grid
    n_rows, n_cols = 8, 8
    images_np = images[:n_rows*n_cols].cpu().numpy()
    rows = []
    for r in range(n_rows):
        row_imgs = [images_np[r*n_cols + c].transpose(1, 2, 0) for c in range(n_cols)]
        rows.append(np.concatenate(row_imgs, axis=1))
    grid = np.concatenate(rows, axis=0)
    final = Image.fromarray((grid * 255).astype(np.uint8))
    final.save("batched_output.png")
    print(f"Saved batched_output.png")


if __name__ == "__main__":
    main()
