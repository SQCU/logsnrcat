# src/fractal.py - Procedural fractal image generation
"""
Procedural fractal image generation for training.
Supports Mandelbrot, Julia, Burning Ship, and other fractals.

CUDA-native implementation for async non-blocking generation.
"""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class CUDAFractalGenerator:
    """
    CUDA-native fractal generator - generates directly on GPU.

    No CPU-GPU transfers, no multiprocessing overhead, fully async.
    Uses vectorized torch operations that saturate GPU compute.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32, seed: int = 42):
        self.device = device
        self.dtype = dtype
        # Use CPU generator for parameters (small scalars, negligible overhead)
        # This avoids GPU sync from .item() calls
        self._cpu_rng = torch.Generator()
        self._cpu_rng.manual_seed(seed)
        self._call_count = 0

    def _rand(self, low: float, high: float) -> float:
        """Generate random float - CPU for param selection (no GPU sync)."""
        return torch.rand(1, generator=self._cpu_rng).item() * (high - low) + low

    def _randint(self, low: int, high: int) -> int:
        """Generate random int - CPU for param selection (no GPU sync)."""
        return int(torch.randint(low, high, (1,), generator=self._cpu_rng).item())

    def _choice(self, options: List[str]) -> str:
        """Random choice from list."""
        idx = self._randint(0, len(options))
        return options[idx]

    def generate_mandelbrot(self, size: int, center_real: float, center_imag: float,
                           zoom: float, max_iter: int) -> torch.Tensor:
        """Generate Mandelbrot set on GPU - fully vectorized."""
        # Create coordinate grids on GPU
        x = torch.linspace(center_real - 2/zoom, center_real + 2/zoom, size,
                          device=self.device, dtype=self.dtype)
        y = torch.linspace(center_imag - 2/zoom, center_imag + 2/zoom, size,
                          device=self.device, dtype=self.dtype)
        Y, X = torch.meshgrid(y, x, indexing='ij')

        # C = X + iY (store as real, imag pairs)
        C_real = X
        C_imag = Y
        Z_real = torch.zeros_like(C_real)
        Z_imag = torch.zeros_like(C_imag)
        M = torch.zeros_like(C_real)

        # Vectorized iteration - no Python loop per pixel
        for i in range(max_iter):
            # |Z|^2 <= 4
            mag_sq = Z_real * Z_real + Z_imag * Z_imag
            mask = mag_sq <= 4.0

            # Z = Z^2 + C
            Z_real_new = Z_real * Z_real - Z_imag * Z_imag + C_real
            Z_imag_new = 2 * Z_real * Z_imag + C_imag

            Z_real = torch.where(mask, Z_real_new, Z_real)
            Z_imag = torch.where(mask, Z_imag_new, Z_imag)
            M = torch.where(mask, torch.full_like(M, i), M)

        return M / max_iter

    def generate_julia(self, size: int, c_real: float, c_imag: float,
                      zoom: float, max_iter: int) -> torch.Tensor:
        """Generate Julia set on GPU."""
        x = torch.linspace(-2/zoom, 2/zoom, size, device=self.device, dtype=self.dtype)
        y = torch.linspace(-2/zoom, 2/zoom, size, device=self.device, dtype=self.dtype)
        Y, X = torch.meshgrid(y, x, indexing='ij')

        Z_real = X
        Z_imag = Y
        M = torch.zeros_like(Z_real)

        for i in range(max_iter):
            mag_sq = Z_real * Z_real + Z_imag * Z_imag
            mask = mag_sq <= 4.0

            Z_real_new = Z_real * Z_real - Z_imag * Z_imag + c_real
            Z_imag_new = 2 * Z_real * Z_imag + c_imag

            Z_real = torch.where(mask, Z_real_new, Z_real)
            Z_imag = torch.where(mask, Z_imag_new, Z_imag)
            M = torch.where(mask, torch.full_like(M, i), M)

        return M / max_iter

    def generate_burning_ship(self, size: int, center_real: float, center_imag: float,
                              zoom: float, max_iter: int) -> torch.Tensor:
        """Generate Burning Ship fractal on GPU."""
        x = torch.linspace(center_real - 2/zoom, center_real + 2/zoom, size,
                          device=self.device, dtype=self.dtype)
        y = torch.linspace(center_imag - 2/zoom, center_imag + 2/zoom, size,
                          device=self.device, dtype=self.dtype)
        Y, X = torch.meshgrid(y, x, indexing='ij')

        C_real = X
        C_imag = Y
        Z_real = torch.zeros_like(C_real)
        Z_imag = torch.zeros_like(C_imag)
        M = torch.zeros_like(C_real)

        for i in range(max_iter):
            mag_sq = Z_real * Z_real + Z_imag * Z_imag
            mask = mag_sq <= 4.0

            # Burning ship: take abs before squaring
            abs_real = torch.abs(Z_real)
            abs_imag = torch.abs(Z_imag)
            Z_real_new = abs_real * abs_real - abs_imag * abs_imag + C_real
            Z_imag_new = 2 * abs_real * abs_imag + C_imag

            Z_real = torch.where(mask, Z_real_new, Z_real)
            Z_imag = torch.where(mask, Z_imag_new, Z_imag)
            M = torch.where(mask, torch.full_like(M, i), M)

        return M / max_iter

    def generate_tricorn(self, size: int, center_real: float, center_imag: float,
                        zoom: float, max_iter: int) -> torch.Tensor:
        """Generate Tricorn (Mandelbar) fractal on GPU."""
        x = torch.linspace(center_real - 2/zoom, center_real + 2/zoom, size,
                          device=self.device, dtype=self.dtype)
        y = torch.linspace(center_imag - 2/zoom, center_imag + 2/zoom, size,
                          device=self.device, dtype=self.dtype)
        Y, X = torch.meshgrid(y, x, indexing='ij')

        C_real = X
        C_imag = Y
        Z_real = torch.zeros_like(C_real)
        Z_imag = torch.zeros_like(C_imag)
        M = torch.zeros_like(C_real)

        for i in range(max_iter):
            mag_sq = Z_real * Z_real + Z_imag * Z_imag
            mask = mag_sq <= 4.0

            # Tricorn: conjugate before squaring (negate imag)
            Z_real_new = Z_real * Z_real - Z_imag * Z_imag + C_real
            Z_imag_new = -2 * Z_real * Z_imag + C_imag  # Note: negative

            Z_real = torch.where(mask, Z_real_new, Z_real)
            Z_imag = torch.where(mask, Z_imag_new, Z_imag)
            M = torch.where(mask, torch.full_like(M, i), M)

        return M / max_iter

    def apply_colormap(self, fractal: torch.Tensor) -> torch.Tensor:
        """Apply random colormap on GPU. Returns [3, H, W]."""
        # Random HSV parameters
        hue_offset = self._rand(0.0, 1.0)
        hue_scale = self._rand(0.3, 1.0)

        h = (fractal * hue_scale + hue_offset) % 1.0
        s = torch.clamp(fractal * 2, 0.5, 1.0)
        v = torch.clamp(1 - fractal * 0.5, 0.3, 1.0)

        # HSV to RGB (vectorized on GPU)
        i = (h * 6).long() % 6
        f = (h * 6) - i.float()
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        # Build RGB using gather/scatter
        r = torch.where(i == 0, v, torch.where(i == 1, q, torch.where(i == 2, p,
            torch.where(i == 3, p, torch.where(i == 4, t, v)))))
        g = torch.where(i == 0, t, torch.where(i == 1, v, torch.where(i == 2, v,
            torch.where(i == 3, q, torch.where(i == 4, p, p)))))
        b = torch.where(i == 0, p, torch.where(i == 1, p, torch.where(i == 2, t,
            torch.where(i == 3, v, torch.where(i == 4, v, q)))))

        return torch.stack([r, g, b], dim=0)  # [3, H, W]

    def generate_batch(self, batch_size: int, size: int = 256,
                       fractal_types: Optional[List[str]] = None,
                       max_iter: int = 128) -> torch.Tensor:
        """
        Generate a batch of random fractals directly on GPU.

        Returns [B, 3, H, W] tensor on self.device.
        """
        if fractal_types is None:
            fractal_types = ["mandelbrot", "julia", "burning_ship", "tricorn"]

        batch = []
        for _ in range(batch_size):
            fractal_type = self._choice(fractal_types)
            zoom = self._rand(0.5, 4.0)

            if fractal_type == "mandelbrot":
                center_real = self._rand(-0.8, 0.4)
                center_imag = self._rand(-0.5, 0.5)
                fractal = self.generate_mandelbrot(size, center_real, center_imag, zoom, max_iter)
            elif fractal_type == "julia":
                c_real = self._rand(-1.0, 0.5)
                c_imag = self._rand(-0.5, 0.5)
                fractal = self.generate_julia(size, c_real, c_imag, zoom, max_iter)
            elif fractal_type == "burning_ship":
                center_real = self._rand(-1.8, -1.6)
                center_imag = self._rand(-0.1, 0.1)
                fractal = self.generate_burning_ship(size, center_real, center_imag, zoom, max_iter)
            else:  # tricorn
                center_real = self._rand(-0.5, 0.5)
                center_imag = self._rand(-0.5, 0.5)
                fractal = self.generate_tricorn(size, center_real, center_imag, zoom, max_iter)

            rgb = self.apply_colormap(fractal)
            batch.append(rgb)

        return torch.stack(batch, dim=0)  # [B, 3, H, W]


# Keep old numpy-based generator as fallback for CPU-only environments
class FractalGenerator:
    """CPU fallback fractal generator using numpy."""

    def __init__(self, size: int = 256, seed: Optional[int] = None):
        import numpy as np
        import random
        self.size = size
        self.rng = random.Random(seed)
        self.np = np

    def generate_random(self, fractal_types: Optional[list] = None,
                       palette: str = "random") -> torch.Tensor:
        """Generate a random fractal on CPU. Returns [C, H, W]."""
        np = self.np
        if fractal_types is None:
            fractal_types = ["mandelbrot", "julia"]

        fractal_type = self.rng.choice(fractal_types)
        zoom = self.rng.uniform(0.5, 4.0)
        max_iter = self.rng.randint(64, 128)  # Lower iter for CPU

        # Create grids
        x = np.linspace(-2/zoom, 2/zoom, self.size)
        y = np.linspace(-2/zoom, 2/zoom, self.size)
        X, Y = np.meshgrid(x, y)

        if fractal_type == "mandelbrot":
            C = X + 1j * Y
            Z = np.zeros_like(C)
        else:  # julia
            c = complex(self.rng.uniform(-1.0, 0.5), self.rng.uniform(-0.5, 0.5))
            Z = X + 1j * Y
            C = np.full_like(Z, c)

        M = np.zeros(Z.shape)
        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask] ** 2 + C[mask]
            M[mask] = i
        M = M / max_iter

        # Simple colormap
        rgb = np.stack([M, M * 0.8, M * 0.6], axis=-1)
        return torch.from_numpy(rgb).permute(2, 0, 1).float()


class FractalIterator:
    """
    Iterator for procedural fractal images.
    Integrates with CompositeIterator via standard interface.

    Uses CUDA-native fractal generation - no CPU blocking, no multiprocessing.
    All computation happens on GPU in the CUDA stream, fully async.
    """

    def __init__(self, device: torch.device, config: dict):
        from .model import ContextBlock

        self.device = device
        self.config = config
        self.seed = config['seed']
        self.text_pos = config['text_position']
        self.fractal_types = config['fractal_types']
        self.ContextBlock = ContextBlock

        # CUDA-native generator - all ops on GPU, no CPU-GPU sync
        self.generator = CUDAFractalGenerator(
            device=device,
            dtype=torch.float32,  # Fractals computed in fp32 for precision
            seed=self.seed
        )

    def generate_batch_list(self, batch_size: int, resolution: int = 256,
                           **kwargs) -> list:
        """Generate a batch of fractal images as ContextBlocks.

        All generation happens on GPU - no CPU blocking.

        Args:
            batch_size: Number of images to generate.
            resolution: Target resolution.
            **kwargs: Additional arguments (start_group_id, etc.)

        Returns:
            List of ContextBlock objects.
        """
        start_group_id = kwargs.get('start_group_id', 0)

        # Generate entire batch on GPU (non-blocking, async in CUDA stream)
        # Use lower max_iter for speed (128 is enough for good detail)
        imgs = self.generator.generate_batch(
            batch_size=batch_size,
            size=resolution,  # Generate at target resolution directly
            fractal_types=self.fractal_types,
            max_iter=128
        )  # [B, 3, H, W] on GPU

        # Build ContextBlocks
        blocks = []
        for i in range(batch_size):
            curr_gid = start_group_id + i
            img_block = self.ContextBlock(
                content=imgs[i],  # Already on GPU
                type='latent',
                causal=False,
                shape_meta=(resolution, resolution),
                group_id=curr_gid,
                id=f"fractal_{curr_gid}",
                source="fractal"
            )
            blocks.append(img_block)

        return blocks

    def shutdown(self):
        """No cleanup needed - CUDA generator has no external resources."""
        pass
