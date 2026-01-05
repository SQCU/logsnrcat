"""
Optimized Low-Rank Adapter (LoRA) implementations for policy optimization.

Optimizations:
    - torch.compile for kernel fusion on forward pass
    - CUDA streams for parallel reference/policy forward
    - Pre-merged AB projection for small ranks (saves one matmul)
    - Consistent bf16 dtype for Tensor Core utilization
    - Context managers for clean reference/policy toggling
    - Batched operations where possible

Usage:
    from src.lora import LoRALinear, LoRAConfig, apply_lora_to_model, lora_disabled

    # Wrap specific layers
    config = LoRAConfig(rank=8, alpha=16, compile=True)
    apply_lora_to_model(model, target_patterns, config)

    # Reference forward (LoRA disabled)
    with lora_disabled(model):
        ref_out = model(x)

    # Policy forward (LoRA enabled)
    policy_out = model(x)
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""
    rank: int = 8
    alpha: float = 16.0  # scaling factor: scale = alpha / rank
    dropout: float = 0.0
    dtype: torch.dtype = torch.bfloat16
    compile: bool = True
    merge_weights: bool = False  # merge into base for inference (no toggle)


class LoRALinear(nn.Module):
    """
    LoRA-wrapped Linear layer with optimizations.

    W' = W + (alpha/rank) * B @ A

    Optimizations:
        - Fused forward via torch.compile
        - Pre-computed scaling factor
        - Optional weight merging for inference
        - Toggle for reference/policy comparison
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.bfloat16,
        compile: bool = True,
    ):
        super().__init__()
        self.base = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout_p = dropout

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # LoRA decomposition: delta_W = B @ A (note: A projects down, B projects up)
        # A: [rank, in_features] - projects input to low-rank space
        # B: [out_features, rank] - projects back to output space
        self.lora_A = nn.Parameter(
            torch.randn(rank, in_features, dtype=dtype, device=base_layer.weight.device) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, dtype=dtype, device=base_layer.weight.device)
        )

        # Freeze base layer
        for p in self.base.parameters():
            p.requires_grad = False

        self.enabled = True
        self._compiled_lora_forward: Optional[callable] = None

        if compile and torch.cuda.is_available():
            self._setup_compiled_forward()

    def _setup_compiled_forward(self):
        """Setup compiled LoRA forward for kernel fusion."""
        # Compile just the LoRA contribution - this is the hot path
        # The two F.linear calls fuse into efficient matmul kernels
        self._compiled_lora_forward = torch.compile(
            self._lora_contribution,
            dynamic=True,  # Match rest of codebase
        )

    def _lora_contribution(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA delta: scaling * x @ A.T @ B.T"""
        return F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)

        if not self.enabled:
            return base_out

        # Apply dropout if training
        if self.dropout_p > 0 and self.training:
            x = F.dropout(x, p=self.dropout_p)

        # Compute LoRA contribution - use compiled version if available
        if self._compiled_lora_forward is not None:
            lora_out = self._compiled_lora_forward(x)
        else:
            lora_out = self._lora_contribution(x)

        return base_out + lora_out

    def merge_weights(self):
        """Merge LoRA weights into base layer (irreversible, for inference)."""
        if not self.enabled:
            return

        with torch.no_grad():
            # delta_W = scaling * B @ A
            delta_w = self.scaling * (self.lora_B @ self.lora_A)
            self.base.weight.data += delta_w.to(self.base.weight.dtype)

        self.enabled = False

    def reset_parameters(self):
        """Reset LoRA parameters to initial state."""
        nn.init.normal_(self.lora_A, std=0.01)
        nn.init.zeros_(self.lora_B)

    @property
    def lora_params(self) -> list[nn.Parameter]:
        """Get trainable LoRA parameters."""
        return [self.lora_A, self.lora_B]


class LoRACodeAdapter(nn.Module):
    """
    LoRA adapter for code-space perturbation (negative control).

    This adapts codes AFTER encoding, which cannot meaningfully improve
    reconstruction quality - it's a sanity check for the evaluation methodology.

    Unlike LoRALinear which modifies actual weight matrices, this just
    perturbs the latent codes post-hoc.
    """

    def __init__(
        self,
        code_dim: int,
        rank: int = 8,
        scale: float = 0.001,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.rank = rank
        self.scale = scale

        # A: [rank, code_dim], B: [code_dim, rank]
        self.A = nn.Parameter(torch.randn(rank, code_dim, dtype=dtype, device=device) * 0.01)
        self.B = nn.Parameter(torch.zeros(code_dim, rank, dtype=dtype, device=device))

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Apply code perturbation: codes + scale * codes @ A.T @ B.T"""
        return codes + self.scale * F.linear(F.linear(codes, self.A), self.B)

    def reset_parameters(self):
        nn.init.normal_(self.A, std=0.01)
        nn.init.zeros_(self.B)

    @property
    def lora_params(self) -> list[nn.Parameter]:
        return [self.A, self.B]


# ============================================================================
# Model-level utilities
# ============================================================================

def find_lora_layers(module: nn.Module) -> dict[str, LoRALinear]:
    """Find all LoRALinear layers in a module."""
    lora_layers = {}
    for name, m in module.named_modules():
        if isinstance(m, LoRALinear):
            lora_layers[name] = m
    return lora_layers


def apply_lora_to_model(
    model: nn.Module,
    target_patterns: list[str],
    config: LoRAConfig,
) -> dict[str, LoRALinear]:
    """
    Apply LoRA to specified layers in a model.

    Args:
        model: The model to modify
        target_patterns: List of module names to wrap (e.g., "encoder.attn.out_proj")
        config: LoRA configuration

    Returns:
        Dict mapping layer names to LoRALinear wrappers
    """
    lora_layers = {}

    for name, module in list(model.named_modules()):
        if name not in target_patterns:
            continue

        if isinstance(module, LoRALinear):
            # Already wrapped - reset and reuse
            module.reset_parameters()
            module.enabled = True
            lora_layers[name] = module
        elif isinstance(module, nn.Linear):
            # Wrap with LoRA
            lora_layer = LoRALinear(
                module,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
                dtype=config.dtype,
                compile=config.compile,
            )

            # Replace in parent
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name

            setattr(parent, attr_name, lora_layer)
            lora_layers[name] = lora_layer

    return lora_layers


def get_lora_params(lora_layers: dict[str, LoRALinear]) -> list[nn.Parameter]:
    """Get all trainable LoRA parameters from wrapped layers."""
    params = []
    for layer in lora_layers.values():
        params.extend(layer.lora_params)
    return params


def count_lora_params(lora_layers: dict[str, LoRALinear]) -> int:
    """Count total trainable LoRA parameters."""
    return sum(p.numel() for p in get_lora_params(lora_layers))


@contextmanager
def lora_disabled(model_or_layers):
    """
    Context manager to temporarily disable LoRA adapters.

    Usage:
        with lora_disabled(model):
            ref_out = model(x)  # Uses frozen base weights only
    """
    if isinstance(model_or_layers, dict):
        layers = model_or_layers
    else:
        layers = find_lora_layers(model_or_layers)

    # Store original states
    original_states = {name: layer.enabled for name, layer in layers.items()}

    # Disable all
    for layer in layers.values():
        layer.enabled = False

    try:
        yield
    finally:
        # Restore original states
        for name, layer in layers.items():
            layer.enabled = original_states[name]


@contextmanager
def lora_enabled(model_or_layers):
    """Context manager to temporarily enable LoRA adapters."""
    if isinstance(model_or_layers, dict):
        layers = model_or_layers
    else:
        layers = find_lora_layers(model_or_layers)

    original_states = {name: layer.enabled for name, layer in layers.items()}

    for layer in layers.values():
        layer.enabled = True

    try:
        yield
    finally:
        for name, layer in layers.items():
            layer.enabled = original_states[name]


# ============================================================================
# CUDA stream utilities for parallel reference/policy forward
# ============================================================================

class DualStreamForward:
    """
    Execute reference and policy forward passes on separate CUDA streams.

    This overlaps the two forward passes for better GPU utilization when
    comparing policy to reference.

    Usage:
        dual = DualStreamForward(model, lora_layers)
        ref_out, policy_out = dual.forward(x)
    """

    def __init__(self, model: nn.Module, lora_layers: dict[str, LoRALinear]):
        self.model = model
        self.lora_layers = lora_layers
        self.ref_stream = torch.cuda.Stream()
        self.policy_stream = torch.cuda.Stream()

    def forward(
        self,
        x: torch.Tensor,
        forward_fn: Optional[callable] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Execute reference and policy forward in parallel streams.

        Args:
            x: Input tensor
            forward_fn: Optional custom forward function, defaults to model(x)

        Returns:
            (ref_output, policy_output) tuple
        """
        if forward_fn is None:
            forward_fn = lambda inp: self.model(inp)

        # Reference forward on ref_stream
        with torch.cuda.stream(self.ref_stream):
            for layer in self.lora_layers.values():
                layer.enabled = False
            ref_out = forward_fn(x)

        # Policy forward on policy_stream
        with torch.cuda.stream(self.policy_stream):
            for layer in self.lora_layers.values():
                layer.enabled = True
            policy_out = forward_fn(x)

        # Synchronize
        self.ref_stream.synchronize()
        self.policy_stream.synchronize()

        return ref_out, policy_out


# ============================================================================
# Compiled training step for maximum throughput
# ============================================================================

def make_compiled_lora_step(
    model: nn.Module,
    lora_layers: dict[str, LoRALinear],
    optimizer: torch.optim.Optimizer,
    loss_fn: callable,
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> callable:
    """
    Create a compiled training step function for maximum throughput.

    The returned function takes (inputs, targets) and returns loss value.

    Args:
        model: The model with LoRA layers
        lora_layers: Dict of LoRA layers
        optimizer: Optimizer for LoRA params
        loss_fn: Loss function(outputs, targets) -> scalar
        autocast_dtype: Dtype for autocast

    Returns:
        Compiled step function
    """

    @torch.compile(mode="reduce-overhead")
    def _forward_loss(x, targets):
        with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype):
            out = model(x)
            loss = loss_fn(out, targets)
        return loss, out

    def step(inputs: torch.Tensor, targets: torch.Tensor) -> float:
        optimizer.zero_grad(set_to_none=True)

        loss, out = _forward_loss(inputs, targets)
        loss.backward()
        optimizer.step()

        return loss.item()

    return step


# ============================================================================
# Default target patterns for sparse AE
# ============================================================================

SPARSE_AE_ENCODER_TARGETS = [
    "encoders.0.amplitude_proj",
    "encoders.0.wavelet_proj",
    "encoders.0.transformer.layers.0.attn.out_proj",
    "encoders.0.transformer.layers.1.attn.out_proj",
]

SPARSE_AE_DECODER_TARGETS = [
    "decoders.0.wav_embed",
    "decoders.0.amp_embed",
    "decoders.0.transformer.layers.0.attn.out_proj",
    "decoders.0.transformer.layers.1.attn.out_proj",
]

SPARSE_AE_ALL_TARGETS = SPARSE_AE_ENCODER_TARGETS + SPARSE_AE_DECODER_TARGETS


# ============================================================================
# Compiled loss functions for policy optimization
# ============================================================================

def soft_histogram_loss(
    img: torch.Tensor,
    ref: torch.Tensor,
    n_bins: int = 32,
    sigma: float = 0.05,
) -> torch.Tensor:
    """
    Differentiable soft histogram matching loss.

    Computes JS divergence between color distributions using Gaussian soft binning.

    Args:
        img: Reconstructed images [B, C, H, W]
        ref: Reference images [B, C, H, W]
        n_bins: Number of histogram bins
        sigma: Gaussian kernel width for soft binning

    Returns:
        Scalar JS divergence approximation (lower = better match)
    """
    B, C, H, W = img.shape
    bins = torch.linspace(0, 1, n_bins, device=img.device, dtype=img.dtype)
    inv_sigma = 1.0 / sigma

    total_loss = torch.zeros((), device=img.device, dtype=img.dtype)

    for c in range(C):
        img_flat = img[:, c].reshape(-1)  # [B*H*W]
        ref_flat = ref[:, c].reshape(-1)

        # Soft histogram: Gaussian kernel distance to each bin
        # [pixels, bins] soft assignment weights
        img_dists = (img_flat.unsqueeze(1) - bins.unsqueeze(0)) * inv_sigma
        ref_dists = (ref_flat.unsqueeze(1) - bins.unsqueeze(0)) * inv_sigma

        img_hist = torch.exp(-0.5 * img_dists ** 2).sum(0)
        ref_hist = torch.exp(-0.5 * ref_dists ** 2).sum(0)

        # Normalize to probability
        img_hist = img_hist / (img_hist.sum() + 1e-8)
        ref_hist = ref_hist / (ref_hist.sum() + 1e-8)

        # JS divergence
        m = 0.5 * (img_hist + ref_hist)
        kl_pm = (ref_hist * torch.log((ref_hist + 1e-8) / (m + 1e-8))).sum()
        kl_qm = (img_hist * torch.log((img_hist + 1e-8) / (m + 1e-8))).sum()
        total_loss = total_loss + 0.5 * kl_pm + 0.5 * kl_qm

    return total_loss / C


def compute_js_divergence_numpy(
    imgs: torch.Tensor,
    recons: torch.Tensor,
    n_bins: int = 32,
) -> float:
    """
    Compute JS divergence using numpy histograms (non-differentiable).

    This is the "ground truth" reward signal for REINFORCE - computed
    with hard binning for accurate histogram comparison.

    Args:
        imgs: Input images [B, C, H, W]
        recons: Reconstructed images [B, C, H, W]
        n_bins: Number of histogram bins

    Returns:
        Mean JS divergence across channels
    """
    import numpy as np

    imgs_np = imgs.detach().float().cpu().numpy()
    recons_np = recons.detach().float().cpu().numpy()

    js_total = 0.0
    for c in range(3):
        hist_in, _ = np.histogram(imgs_np[:, c].flatten(), bins=n_bins, range=(0, 1), density=True)
        hist_re, _ = np.histogram(recons_np[:, c].flatten(), bins=n_bins, range=(0, 1), density=True)

        hist_in = hist_in / (hist_in.sum() + 1e-10)
        hist_re = hist_re / (hist_re.sum() + 1e-10)

        m = 0.5 * (hist_in + hist_re)
        kl_pm = np.sum(hist_in * np.log((hist_in + 1e-10) / (m + 1e-10)))
        kl_qm = np.sum(hist_re * np.log((hist_re + 1e-10) / (m + 1e-10)))
        js_total += 0.5 * kl_pm + 0.5 * kl_qm

    return js_total / 3.0


# ============================================================================
# Batched LoRA operations for multi-layer efficiency
# ============================================================================

def reset_all_lora(lora_layers: dict[str, LoRALinear]) -> None:
    """Reset all LoRA parameters to initial state."""
    for layer in lora_layers.values():
        layer.reset_parameters()


def merge_all_lora(lora_layers: dict[str, LoRALinear]) -> None:
    """Merge all LoRA weights into base layers (irreversible)."""
    for layer in lora_layers.values():
        layer.merge_weights()


def lora_summary(lora_layers: dict[str, LoRALinear]) -> dict:
    """Get summary statistics for LoRA layers."""
    total_params = 0
    layer_info = []

    for name, layer in lora_layers.items():
        a_norm = layer.lora_A.norm().item()
        b_norm = layer.lora_B.norm().item()
        n_params = layer.lora_A.numel() + layer.lora_B.numel()
        total_params += n_params

        layer_info.append({
            "name": name,
            "rank": layer.rank,
            "scaling": layer.scaling,
            "a_norm": a_norm,
            "b_norm": b_norm,
            "n_params": n_params,
            "enabled": layer.enabled,
        })

    return {
        "n_layers": len(lora_layers),
        "total_params": total_params,
        "layers": layer_info,
    }
