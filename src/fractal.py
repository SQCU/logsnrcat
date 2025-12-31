# src/fractal.py - Procedural fractal image generation
"""
Procedural fractal image generation for training.
Supports Mandelbrot, Julia, Burning Ship, and other fractals.

CUDA-native implementation with explicit CUDA Graph capture/replay.
Graph capture eliminates per-kernel launch overhead - the entire fractal
iteration loop is recorded once and replayed with a single CPU call.
"""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math


class CUDAGraphFractalGenerator:
    """
    CUDA Graph-accelerated fractal generator.

    Uses explicit torch.cuda.CUDAGraph capture to record the iteration loop
    once, then replays it with different parameters. This eliminates:
    - Per-iteration kernel launch overhead (128 launches → 1 graph replay)
    - CPU-GPU synchronization between iterations
    - Python interpreter overhead in the hot loop

    The graph is captured per (size, max_iter) combination and cached.
    Input parameters (center, zoom, c) are updated via pre-allocated tensors.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32, seed: int = 42):
        self.device = device
        self.dtype = dtype
        self._cpu_rng = torch.Generator()
        self._cpu_rng.manual_seed(seed)

        # Graph cache: (size, max_iter, fractal_type) -> (graph, input_buffers, output_buffer)
        self._graph_cache: Dict[Tuple, Tuple] = {}

        # Warmup stream for graph capture
        self._capture_stream = torch.cuda.Stream(device=device)

    def _rand(self, low: float, high: float) -> float:
        return torch.rand(1, generator=self._cpu_rng).item() * (high - low) + low

    def _randint(self, low: int, high: int) -> int:
        return int(torch.randint(low, high, (1,), generator=self._cpu_rng).item())

    def _choice(self, options: List[str]) -> str:
        return options[self._randint(0, len(options))]

    def _get_or_capture_mandelbrot_graph(self, size: int, max_iter: int):
        """Get cached graph or capture new one for Mandelbrot."""
        cache_key = (size, max_iter, 'mandelbrot')

        if cache_key not in self._graph_cache:
            # Pre-allocate all buffers (these persist across replays)
            # Input parameters (updated in-place before replay)
            center_real = torch.zeros(1, device=self.device, dtype=self.dtype)
            center_imag = torch.zeros(1, device=self.device, dtype=self.dtype)
            zoom = torch.ones(1, device=self.device, dtype=self.dtype)

            # Working buffers
            C_real = torch.zeros(size, size, device=self.device, dtype=self.dtype)
            C_imag = torch.zeros(size, size, device=self.device, dtype=self.dtype)
            Z_real = torch.zeros(size, size, device=self.device, dtype=self.dtype)
            Z_imag = torch.zeros(size, size, device=self.device, dtype=self.dtype)
            M = torch.zeros(size, size, device=self.device, dtype=self.dtype)

            # Output buffer
            output = torch.zeros(size, size, device=self.device, dtype=self.dtype)

            # Warmup run (required before capture)
            s = self._capture_stream
            s.wait_stream(torch.cuda.current_stream(self.device))

            with torch.cuda.stream(s):
                # Compute coordinate grids from parameters
                inv_zoom = 2.0 / zoom
                x_min = center_real - inv_zoom
                x_max = center_real + inv_zoom
                y_min = center_imag - inv_zoom
                y_max = center_imag + inv_zoom

                # Create grids (these ops will be captured)
                x_coords = torch.linspace(0, 1, size, device=self.device, dtype=self.dtype)
                y_coords = torch.linspace(0, 1, size, device=self.device, dtype=self.dtype)

                C_real.copy_(x_min + (x_max - x_min) * x_coords.unsqueeze(0))
                C_imag.copy_(y_min + (y_max - y_min) * y_coords.unsqueeze(1))

                Z_real.zero_()
                Z_imag.zero_()
                M.zero_()

                for i in range(max_iter):
                    mag_sq = Z_real * Z_real + Z_imag * Z_imag
                    mask = mag_sq <= 4.0

                    Z_real_new = Z_real * Z_real - Z_imag * Z_imag + C_real
                    Z_imag_new = 2 * Z_real * Z_imag + C_imag

                    Z_real.copy_(torch.where(mask, Z_real_new, Z_real))
                    Z_imag.copy_(torch.where(mask, Z_imag_new, Z_imag))
                    M.copy_(torch.where(mask, i / max_iter, M))

                output.copy_(M)

            torch.cuda.current_stream(self.device).wait_stream(s)

            # Now capture the graph
            graph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(graph, stream=s):
                inv_zoom = 2.0 / zoom
                x_min = center_real - inv_zoom
                x_max = center_real + inv_zoom
                y_min = center_imag - inv_zoom
                y_max = center_imag + inv_zoom

                x_coords = torch.linspace(0, 1, size, device=self.device, dtype=self.dtype)
                y_coords = torch.linspace(0, 1, size, device=self.device, dtype=self.dtype)

                C_real.copy_(x_min + (x_max - x_min) * x_coords.unsqueeze(0))
                C_imag.copy_(y_min + (y_max - y_min) * y_coords.unsqueeze(1))

                Z_real.zero_()
                Z_imag.zero_()
                M.zero_()

                for i in range(max_iter):
                    mag_sq = Z_real * Z_real + Z_imag * Z_imag
                    mask = mag_sq <= 4.0

                    Z_real_new = Z_real * Z_real - Z_imag * Z_imag + C_real
                    Z_imag_new = 2 * Z_real * Z_imag + C_imag

                    Z_real.copy_(torch.where(mask, Z_real_new, Z_real))
                    Z_imag.copy_(torch.where(mask, Z_imag_new, Z_imag))
                    M.copy_(torch.where(mask, i / max_iter, M))

                output.copy_(M)

            self._graph_cache[cache_key] = (
                graph,
                {'center_real': center_real, 'center_imag': center_imag, 'zoom': zoom},
                output
            )

        return self._graph_cache[cache_key]

    def _get_or_capture_julia_graph(self, size: int, max_iter: int):
        """Get cached graph or capture new one for Julia set."""
        cache_key = (size, max_iter, 'julia')

        if cache_key not in self._graph_cache:
            # Input parameters
            c_real = torch.zeros(1, device=self.device, dtype=self.dtype)
            c_imag = torch.zeros(1, device=self.device, dtype=self.dtype)
            zoom = torch.ones(1, device=self.device, dtype=self.dtype)

            # Working buffers
            Z_real = torch.zeros(size, size, device=self.device, dtype=self.dtype)
            Z_imag = torch.zeros(size, size, device=self.device, dtype=self.dtype)
            M = torch.zeros(size, size, device=self.device, dtype=self.dtype)
            output = torch.zeros(size, size, device=self.device, dtype=self.dtype)

            s = self._capture_stream
            s.wait_stream(torch.cuda.current_stream(self.device))

            # Warmup
            with torch.cuda.stream(s):
                inv_zoom = 2.0 / zoom
                x_coords = torch.linspace(-1, 1, size, device=self.device, dtype=self.dtype) * inv_zoom
                y_coords = torch.linspace(-1, 1, size, device=self.device, dtype=self.dtype) * inv_zoom

                Z_real.copy_(x_coords.unsqueeze(0).expand(size, size))
                Z_imag.copy_(y_coords.unsqueeze(1).expand(size, size))
                M.zero_()

                for i in range(max_iter):
                    mag_sq = Z_real * Z_real + Z_imag * Z_imag
                    mask = mag_sq <= 4.0
                    Z_real_new = Z_real * Z_real - Z_imag * Z_imag + c_real
                    Z_imag_new = 2 * Z_real * Z_imag + c_imag
                    Z_real.copy_(torch.where(mask, Z_real_new, Z_real))
                    Z_imag.copy_(torch.where(mask, Z_imag_new, Z_imag))
                    M.copy_(torch.where(mask, i / max_iter, M))
                output.copy_(M)

            torch.cuda.current_stream(self.device).wait_stream(s)

            # Capture
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=s):
                inv_zoom = 2.0 / zoom
                x_coords = torch.linspace(-1, 1, size, device=self.device, dtype=self.dtype) * inv_zoom
                y_coords = torch.linspace(-1, 1, size, device=self.device, dtype=self.dtype) * inv_zoom
                Z_real.copy_(x_coords.unsqueeze(0).expand(size, size))
                Z_imag.copy_(y_coords.unsqueeze(1).expand(size, size))
                M.zero_()
                for i in range(max_iter):
                    mag_sq = Z_real * Z_real + Z_imag * Z_imag
                    mask = mag_sq <= 4.0
                    Z_real_new = Z_real * Z_real - Z_imag * Z_imag + c_real
                    Z_imag_new = 2 * Z_real * Z_imag + c_imag
                    Z_real.copy_(torch.where(mask, Z_real_new, Z_real))
                    Z_imag.copy_(torch.where(mask, Z_imag_new, Z_imag))
                    M.copy_(torch.where(mask, i / max_iter, M))
                output.copy_(M)

            self._graph_cache[cache_key] = (
                graph,
                {'c_real': c_real, 'c_imag': c_imag, 'zoom': zoom},
                output
            )

        return self._graph_cache[cache_key]

    def generate_mandelbrot(self, size: int, center_real: float, center_imag: float,
                           zoom: float, max_iter: int) -> torch.Tensor:
        """Generate Mandelbrot via graph replay."""
        graph, inputs, output = self._get_or_capture_mandelbrot_graph(size, max_iter)

        # Update input parameters in-place (no allocation)
        inputs['center_real'].fill_(center_real)
        inputs['center_imag'].fill_(center_imag)
        inputs['zoom'].fill_(zoom)

        # Single graph replay instead of 128 kernel launches
        graph.replay()

        return output.clone()  # Clone to avoid aliasing issues

    def generate_julia(self, size: int, c_real: float, c_imag: float,
                      zoom: float, max_iter: int) -> torch.Tensor:
        """Generate Julia set via graph replay."""
        graph, inputs, output = self._get_or_capture_julia_graph(size, max_iter)

        inputs['c_real'].fill_(c_real)
        inputs['c_imag'].fill_(c_imag)
        inputs['zoom'].fill_(zoom)

        graph.replay()
        return output.clone()

    def apply_colormap(self, fractal: torch.Tensor) -> torch.Tensor:
        """Apply random colormap on GPU. Returns [3, H, W]."""
        hue_offset = self._rand(0.0, 1.0)
        hue_scale = self._rand(0.3, 1.0)

        h = (fractal * hue_scale + hue_offset) % 1.0
        s = torch.clamp(fractal * 2, 0.5, 1.0)
        v = torch.clamp(1 - fractal * 0.5, 0.3, 1.0)

        i = (h * 6).long() % 6
        f = (h * 6) - i.float()
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        r = torch.where(i == 0, v, torch.where(i == 1, q, torch.where(i == 2, p,
            torch.where(i == 3, p, torch.where(i == 4, t, v)))))
        g = torch.where(i == 0, t, torch.where(i == 1, v, torch.where(i == 2, v,
            torch.where(i == 3, q, torch.where(i == 4, p, p)))))
        b = torch.where(i == 0, p, torch.where(i == 1, p, torch.where(i == 2, t,
            torch.where(i == 3, v, torch.where(i == 4, v, q)))))

        return torch.stack([r, g, b], dim=0)

    def generate_batch(self, batch_size: int, size: int = 256,
                       fractal_types: Optional[List[str]] = None,
                       max_iter: int = 128) -> torch.Tensor:
        """Generate batch of fractals using graph replay."""
        if fractal_types is None:
            fractal_types = ["mandelbrot", "julia"]

        batch = []
        for _ in range(batch_size):
            fractal_type = self._choice(fractal_types)
            zoom = self._rand(0.5, 4.0)

            if fractal_type == "mandelbrot":
                center_real = self._rand(-0.8, 0.4)
                center_imag = self._rand(-0.5, 0.5)
                fractal = self.generate_mandelbrot(size, center_real, center_imag, zoom, max_iter)
            else:  # julia
                c_real = self._rand(-1.0, 0.5)
                c_imag = self._rand(-0.5, 0.5)
                fractal = self.generate_julia(size, c_real, c_imag, zoom, max_iter)

            rgb = self.apply_colormap(fractal)
            batch.append(rgb)

        return torch.stack(batch, dim=0)


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


# ==============================================================================
# Query Generation & Rendering (Matches torus/checkerboard pattern)
# ==============================================================================

def generate_fractal_query(seed: int, config: dict) -> dict:
    """
    Generate JSON-serializable fractal query parameters.

    Matches the pattern used by generate_torus_query/generate_checkerboard_query.
    The query dict is converted to text tokens via serialize_query().

    Args:
        seed: Random seed for reproducibility.
        config: Dataset config dict with fractal_types, etc.

    Returns:
        Dict with all parameters needed to reproduce the fractal.
    """
    import random
    rng = random.Random(seed)

    fractal_types = config.get('fractal_types', ['mandelbrot', 'julia'])
    fractal_type = rng.choice(fractal_types)

    # Base parameters common to all types
    zoom = round(rng.uniform(0.5, 4.0), 3)
    max_iter = config.get('max_iterations', 128)

    # Colormap parameters
    hue_offset = round(rng.uniform(0.0, 1.0), 3)
    hue_scale = round(rng.uniform(0.3, 1.0), 3)

    query = {
        "type": fractal_type,
        "zoom": zoom,
        "max_iter": max_iter,
        "hue_offset": hue_offset,
        "hue_scale": hue_scale,
    }

    # Type-specific parameters
    if fractal_type == "mandelbrot":
        query["center_real"] = round(rng.uniform(-0.8, 0.4), 4)
        query["center_imag"] = round(rng.uniform(-0.5, 0.5), 4)
    elif fractal_type == "julia":
        query["c_real"] = round(rng.uniform(-1.0, 0.5), 4)
        query["c_imag"] = round(rng.uniform(-0.5, 0.5), 4)
    elif fractal_type == "burning_ship":
        query["center_real"] = round(rng.uniform(-1.8, -1.6), 4)
        query["center_imag"] = round(rng.uniform(-0.1, 0.1), 4)
    elif fractal_type == "tricorn":
        query["center_real"] = round(rng.uniform(-0.5, 0.5), 4)
        query["center_imag"] = round(rng.uniform(-0.5, 0.5), 4)

    return query


def render_fractal(query: dict, resolution: int, device: torch.device) -> torch.Tensor:
    """
    Render fractal image from query parameters.

    Matches the pattern used by render_torus/render_checkerboard.
    Uses CUDA-native computation for maximum throughput.

    Args:
        query: Dict from generate_fractal_query().
        resolution: Target image resolution.
        device: Target CUDA device.

    Returns:
        [3, H, W] tensor on specified device.
    """
    dtype = torch.float32
    fractal_type = query["type"]
    zoom = query["zoom"]
    max_iter = query["max_iter"]
    hue_offset = query["hue_offset"]
    hue_scale = query["hue_scale"]

    # Generate escape-time field based on fractal type
    if fractal_type == "mandelbrot":
        center_real = query["center_real"]
        center_imag = query["center_imag"]
        fractal = _render_mandelbrot_cuda(resolution, center_real, center_imag, zoom, max_iter, device, dtype)
    elif fractal_type == "julia":
        c_real = query["c_real"]
        c_imag = query["c_imag"]
        fractal = _render_julia_cuda(resolution, c_real, c_imag, zoom, max_iter, device, dtype)
    elif fractal_type == "burning_ship":
        center_real = query["center_real"]
        center_imag = query["center_imag"]
        fractal = _render_burning_ship_cuda(resolution, center_real, center_imag, zoom, max_iter, device, dtype)
    elif fractal_type == "tricorn":
        center_real = query["center_real"]
        center_imag = query["center_imag"]
        fractal = _render_tricorn_cuda(resolution, center_real, center_imag, zoom, max_iter, device, dtype)
    else:
        # Fallback to mandelbrot for unknown types
        fractal = _render_mandelbrot_cuda(resolution, 0.0, 0.0, zoom, max_iter, device, dtype)

    # Apply deterministic colormap (same params = same colors)
    return _apply_colormap_cuda(fractal, hue_offset, hue_scale)


def _render_mandelbrot_cuda(size: int, center_real: float, center_imag: float,
                            zoom: float, max_iter: int, device: torch.device,
                            dtype: torch.dtype) -> torch.Tensor:
    """CUDA-native Mandelbrot generation."""
    x = torch.linspace(center_real - 2/zoom, center_real + 2/zoom, size, device=device, dtype=dtype)
    y = torch.linspace(center_imag - 2/zoom, center_imag + 2/zoom, size, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    C_real, C_imag = X, Y
    Z_real = torch.zeros_like(C_real)
    Z_imag = torch.zeros_like(C_imag)
    M = torch.zeros_like(C_real)

    for i in range(max_iter):
        mag_sq = Z_real * Z_real + Z_imag * Z_imag
        mask = mag_sq <= 4.0
        Z_real_new = Z_real * Z_real - Z_imag * Z_imag + C_real
        Z_imag_new = 2 * Z_real * Z_imag + C_imag
        Z_real = torch.where(mask, Z_real_new, Z_real)
        Z_imag = torch.where(mask, Z_imag_new, Z_imag)
        M = torch.where(mask, torch.full_like(M, i), M)

    return M / max_iter


def _render_julia_cuda(size: int, c_real: float, c_imag: float,
                       zoom: float, max_iter: int, device: torch.device,
                       dtype: torch.dtype) -> torch.Tensor:
    """CUDA-native Julia set generation."""
    x = torch.linspace(-2/zoom, 2/zoom, size, device=device, dtype=dtype)
    y = torch.linspace(-2/zoom, 2/zoom, size, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    Z_real, Z_imag = X, Y
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


def _render_burning_ship_cuda(size: int, center_real: float, center_imag: float,
                               zoom: float, max_iter: int, device: torch.device,
                               dtype: torch.dtype) -> torch.Tensor:
    """CUDA-native Burning Ship generation."""
    x = torch.linspace(center_real - 2/zoom, center_real + 2/zoom, size, device=device, dtype=dtype)
    y = torch.linspace(center_imag - 2/zoom, center_imag + 2/zoom, size, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    C_real, C_imag = X, Y
    Z_real = torch.zeros_like(C_real)
    Z_imag = torch.zeros_like(C_imag)
    M = torch.zeros_like(C_real)

    for i in range(max_iter):
        mag_sq = Z_real * Z_real + Z_imag * Z_imag
        mask = mag_sq <= 4.0
        abs_real = torch.abs(Z_real)
        abs_imag = torch.abs(Z_imag)
        Z_real_new = abs_real * abs_real - abs_imag * abs_imag + C_real
        Z_imag_new = 2 * abs_real * abs_imag + C_imag
        Z_real = torch.where(mask, Z_real_new, Z_real)
        Z_imag = torch.where(mask, Z_imag_new, Z_imag)
        M = torch.where(mask, torch.full_like(M, i), M)

    return M / max_iter


def _render_tricorn_cuda(size: int, center_real: float, center_imag: float,
                         zoom: float, max_iter: int, device: torch.device,
                         dtype: torch.dtype) -> torch.Tensor:
    """CUDA-native Tricorn generation."""
    x = torch.linspace(center_real - 2/zoom, center_real + 2/zoom, size, device=device, dtype=dtype)
    y = torch.linspace(center_imag - 2/zoom, center_imag + 2/zoom, size, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    C_real, C_imag = X, Y
    Z_real = torch.zeros_like(C_real)
    Z_imag = torch.zeros_like(C_imag)
    M = torch.zeros_like(C_real)

    for i in range(max_iter):
        mag_sq = Z_real * Z_real + Z_imag * Z_imag
        mask = mag_sq <= 4.0
        Z_real_new = Z_real * Z_real - Z_imag * Z_imag + C_real
        Z_imag_new = -2 * Z_real * Z_imag + C_imag  # Conjugate
        Z_real = torch.where(mask, Z_real_new, Z_real)
        Z_imag = torch.where(mask, Z_imag_new, Z_imag)
        M = torch.where(mask, torch.full_like(M, i), M)

    return M / max_iter


def _apply_colormap_cuda(fractal: torch.Tensor, hue_offset: float, hue_scale: float) -> torch.Tensor:
    """Apply HSV colormap on GPU. Returns [3, H, W]."""
    h = (fractal * hue_scale + hue_offset) % 1.0
    s = torch.clamp(fractal * 2, 0.5, 1.0)
    v = torch.clamp(1 - fractal * 0.5, 0.3, 1.0)

    # HSV to RGB (vectorized)
    i = (h * 6).long() % 6
    f = (h * 6) - i.float()
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    r = torch.where(i == 0, v, torch.where(i == 1, q, torch.where(i == 2, p,
        torch.where(i == 3, p, torch.where(i == 4, t, v)))))
    g = torch.where(i == 0, t, torch.where(i == 1, v, torch.where(i == 2, v,
        torch.where(i == 3, q, torch.where(i == 4, p, p)))))
    b = torch.where(i == 0, p, torch.where(i == 1, p, torch.where(i == 2, t,
        torch.where(i == 3, v, torch.where(i == 4, v, q)))))

    return torch.stack([r, g, b], dim=0)


# ==============================================================================
# FractalIterator - Uses FunctionalIterator pattern with text tokens
# ==============================================================================

class FractalIterator:
    """
    Iterator for procedural fractal images with JSON text token representation.

    Follows the same pattern as torus/checkerboard iterators:
    - generate_fractal_query() creates JSON-serializable params
    - serialize_query() converts to text tokens
    - render_fractal() renders from query dict
    - Supports text_position: prefix, suffix, none, random

    Uses CUDA-native computation for maximum throughput.
    """

    def __init__(self, device: torch.device, config: dict):
        from .model import ContextBlock
        from .data_functional import serialize_query
        import random

        self.device = device
        self.config = config
        self.seed = config['seed']
        self.resolution_override = config.get('resolution', None)
        self.text_pos = config['text_position']
        self.ContextBlock = ContextBlock
        self.serialize_query = serialize_query
        self._random = random

    def generate_batch_list(self, batch_size: int, resolution: int = 256,
                           **kwargs) -> list:
        """
        Generate batch of fractal images as ContextBlocks with text tokens.

        Each fractal gets a JSON text representation matching the query format
        used by torus/checkerboard datasets. This enables:
        - Text-to-image generation (query -> fractal)
        - Image-to-text (fractal -> reconstruct query)
        - Consistent multimodal training

        Args:
            batch_size: Number of images to generate.
            resolution: Target resolution (overridden by config if set).
            **kwargs: Additional arguments (start_group_id, etc.)

        Returns:
            List of ContextBlock objects (text + latent pairs).
        """
        start_group_id = kwargs.get('start_group_id', 0)
        res = self.resolution_override if self.resolution_override else resolution

        blocks = []
        for i in range(batch_size):
            # 1. Generate query (JSON-serializable params)
            query = generate_fractal_query(self.seed, self.config)
            self.seed += 1
            curr_gid = start_group_id + i

            # 2. Determine layout (prefix/suffix/none/random)
            layout = self.text_pos
            if layout == 'random':
                layout = self._random.choice(['prefix', 'suffix', 'none'])

            # 3. Create text block from serialized query
            text_content = self.serialize_query(query).to(self.device)
            text_block = self.ContextBlock(
                content=text_content,
                type='text',
                causal=True,
                shape_meta=(text_content.shape[0],),
                group_id=curr_gid,
                id=f"txt_{curr_gid}"
            )

            # 4. Render image from query
            img_content = render_fractal(query, res, self.device)
            img_block = self.ContextBlock(
                content=img_content,
                type='latent',
                causal=False,
                shape_meta=(res, res),
                group_id=curr_gid,
                id=f"fractal_{curr_gid}"
            )

            # 5. Assemble based on layout
            if layout == 'prefix':
                blocks.extend([text_block, img_block])
            elif layout == 'suffix':
                blocks.extend([img_block, text_block])
            else:  # none
                blocks.append(img_block)

        return blocks

    def shutdown(self):
        """No cleanup needed - CUDA computation has no external resources."""
        pass
