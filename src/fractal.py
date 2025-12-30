# src/fractal.py - Procedural fractal image generation
"""
Procedural fractal image generation for training.
Supports Mandelbrot, Julia, Burning Ship, and other fractals.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import random
import math
from multiprocessing import Process, Queue
import time


class FractalGenerator:
    """Single-threaded fractal generator."""

    def __init__(self, size: int = 256, seed: Optional[int] = None):
        self.size = size
        self.rng = random.Random(seed)

    def generate_mandelbrot(self, center: complex, zoom: float,
                           max_iter: int = 256) -> np.ndarray:
        """Generate Mandelbrot set image."""
        x = np.linspace(center.real - 2/zoom, center.real + 2/zoom, self.size)
        y = np.linspace(center.imag - 2/zoom, center.imag + 2/zoom, self.size)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y

        Z = np.zeros_like(C)
        M = np.zeros(C.shape)

        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask] ** 2 + C[mask]
            M[mask] = i

        return M / max_iter

    def generate_julia(self, c: complex, zoom: float = 1.0,
                      max_iter: int = 256) -> np.ndarray:
        """Generate Julia set for parameter c."""
        x = np.linspace(-2/zoom, 2/zoom, self.size)
        y = np.linspace(-2/zoom, 2/zoom, self.size)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y

        M = np.zeros(Z.shape)

        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask] ** 2 + c
            M[mask] = i

        return M / max_iter

    def generate_burning_ship(self, center: complex, zoom: float,
                              max_iter: int = 256) -> np.ndarray:
        """Generate Burning Ship fractal."""
        x = np.linspace(center.real - 2/zoom, center.real + 2/zoom, self.size)
        y = np.linspace(center.imag - 2/zoom, center.imag + 2/zoom, self.size)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y

        Z = np.zeros_like(C)
        M = np.zeros(C.shape)

        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = (np.abs(Z[mask].real) + 1j * np.abs(Z[mask].imag)) ** 2 + C[mask]
            M[mask] = i

        return M / max_iter

    def generate_tricorn(self, center: complex, zoom: float,
                        max_iter: int = 256) -> np.ndarray:
        """Generate Tricorn (Mandelbar) fractal."""
        x = np.linspace(center.real - 2/zoom, center.real + 2/zoom, self.size)
        y = np.linspace(center.imag - 2/zoom, center.imag + 2/zoom, self.size)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y

        Z = np.zeros_like(C)
        M = np.zeros(C.shape)

        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = np.conj(Z[mask]) ** 2 + C[mask]
            M[mask] = i

        return M / max_iter

    def apply_colormap(self, fractal: np.ndarray, palette: str = "random") -> np.ndarray:
        """Apply colormap to scalar fractal image."""
        if palette == "random":
            # Generate random smooth colormap
            hue_offset = self.rng.random()
            hue_scale = self.rng.uniform(0.3, 1.0)

            h = (fractal * hue_scale + hue_offset) % 1.0
            s = np.clip(fractal * 2, 0.5, 1.0)
            v = np.clip(1 - fractal * 0.5, 0.3, 1.0)

            # HSV to RGB
            rgb = self._hsv_to_rgb(h, s, v)
        elif palette == "fire":
            r = np.clip(fractal * 3, 0, 1)
            g = np.clip(fractal * 1.5 - 0.5, 0, 1)
            b = np.clip(fractal - 0.7, 0, 1)
            rgb = np.stack([r, g, b], axis=-1)
        elif palette == "ice":
            r = np.clip(fractal - 0.5, 0, 1)
            g = np.clip(fractal * 1.5, 0, 1)
            b = np.clip(fractal * 2, 0, 1)
            rgb = np.stack([r, g, b], axis=-1)
        elif palette == "earth":
            r = np.clip(fractal * 1.5, 0, 1)
            g = np.clip(fractal * 1.2 - 0.2, 0, 1)
            b = np.clip(fractal * 0.5, 0, 1)
            rgb = np.stack([r, g, b], axis=-1)
        else:  # grayscale fallback
            rgb = np.stack([fractal] * 3, axis=-1)

        return rgb

    def _hsv_to_rgb(self, h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Vectorized HSV to RGB conversion."""
        i = (h * 6).astype(int) % 6
        f = (h * 6) - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        conditions = [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5]
        r = np.select(conditions, [v, q, p, p, t, v])
        g = np.select(conditions, [t, v, v, q, p, p])
        b = np.select(conditions, [p, p, t, v, v, q])

        return np.stack([r, g, b], axis=-1)

    def generate_random(self, fractal_types: Optional[list] = None,
                       palette: str = "random") -> torch.Tensor:
        """Generate a random fractal image.

        Args:
            fractal_types: List of allowed fractal types. Defaults to all.
            palette: Color palette to use.

        Returns:
            Tensor of shape [C, H, W] with values in [0, 1].
        """
        if fractal_types is None:
            fractal_types = ["mandelbrot", "julia", "burning_ship", "tricorn"]

        fractal_type = self.rng.choice(fractal_types)
        zoom = self.rng.uniform(0.5, 4.0)
        max_iter = self.rng.randint(128, 512)

        if fractal_type == "mandelbrot":
            center = complex(
                self.rng.uniform(-0.8, 0.4),
                self.rng.uniform(-0.5, 0.5)
            )
            fractal = self.generate_mandelbrot(center, zoom, max_iter)
        elif fractal_type == "julia":
            c = complex(
                self.rng.uniform(-1.0, 0.5),
                self.rng.uniform(-0.5, 0.5)
            )
            fractal = self.generate_julia(c, zoom, max_iter)
        elif fractal_type == "burning_ship":
            center = complex(
                self.rng.uniform(-1.8, -1.6),
                self.rng.uniform(-0.1, 0.1)
            )
            fractal = self.generate_burning_ship(center, zoom, max_iter)
        else:  # tricorn
            center = complex(
                self.rng.uniform(-0.5, 0.5),
                self.rng.uniform(-0.5, 0.5)
            )
            fractal = self.generate_tricorn(center, zoom, max_iter)

        rgb = self.apply_colormap(fractal, palette)

        # Convert to tensor [C, H, W]
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        return tensor


def _worker_loop(queue: Queue, size: int, worker_id: int, fractal_types: list, palette: str):
    """Worker process main loop."""
    gen = FractalGenerator(size, seed=worker_id * 1000 + int(time.time()))
    while True:
        try:
            img = gen.generate_random(fractal_types=fractal_types, palette=palette)
            queue.put(img.numpy(), block=True)
        except Exception:
            break


class FractalQueue:
    """Multi-process fractal generation queue."""

    def __init__(self, size: int = 256, n_workers: int = 4, queue_size: int = 256,
                 fractal_types: Optional[list] = None, palette: str = "random"):
        self.size = size
        self.queue = Queue(maxsize=queue_size)
        self.workers = []
        self.running = True

        if fractal_types is None:
            fractal_types = ["mandelbrot", "julia", "burning_ship"]

        for i in range(n_workers):
            p = Process(
                target=_worker_loop,
                args=(self.queue, size, i, fractal_types, palette)
            )
            p.daemon = True
            p.start()
            self.workers.append(p)

    def get_batch(self, batch_size: int) -> torch.Tensor:
        """Get a batch of fractal images.

        Args:
            batch_size: Number of images to retrieve.

        Returns:
            Tensor of shape [B, C, H, W].
        """
        batch = []
        for _ in range(batch_size):
            img_np = self.queue.get(block=True)
            batch.append(torch.from_numpy(img_np))
        return torch.stack(batch)

    def shutdown(self):
        """Terminate all worker processes."""
        self.running = False
        for p in self.workers:
            p.terminate()
            p.join(timeout=1)


class FractalIterator:
    """
    Iterator for procedural fractal images.
    Integrates with CompositeIterator via standard interface.
    """

    def __init__(self, device: torch.device, config: dict):
        from .model import ContextBlock

        self.device = device
        self.config = config
        self.seed = config['seed']
        self.generator = FractalGenerator(size=256, seed=self.seed)
        self.text_pos = config['text_position']
        self.fractal_types = config['fractal_types']
        self.palette = config['color_palette']
        self.ContextBlock = ContextBlock

    def generate_batch_list(self, batch_size: int, resolution: int = 256,
                           **kwargs) -> list:
        """Generate a batch of fractal images as ContextBlocks.

        Args:
            batch_size: Number of images to generate.
            resolution: Target resolution (images will be resized if needed).
            **kwargs: Additional arguments (start_group_id, etc.)

        Returns:
            List of ContextBlock objects.
        """
        start_group_id = kwargs.get('start_group_id', 0)
        blocks = []

        for i in range(batch_size):
            curr_gid = start_group_id + i

            # Generate fractal at native resolution
            img = self.generator.generate_random(
                fractal_types=self.fractal_types,
                palette=self.palette
            )

            # Resize if needed
            if resolution != self.generator.size:
                img = F.interpolate(
                    img.unsqueeze(0),
                    size=(resolution, resolution),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

            img = img.to(self.device)

            img_block = self.ContextBlock(
                content=img,
                type='latent',
                causal=False,
                shape_meta=(resolution, resolution),
                group_id=curr_gid,
                id=f"fractal_{curr_gid}",
                source="fractal"
            )

            blocks.append(img_block)

        return blocks
