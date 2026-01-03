# src/optim_utils.py
"""
Heterogeneous optimizer utilities for mixed-precision training.

Supports:
- Muon optimizer for transformer layers (momentum-orthogonalized updates)
- AdamW for embedding/unembedding layers (needs stable gradients)
- Coordinated scheduling across multiple optimizer groups
- fp8 weights/activations for transformer, bf16/fp32 for embeddings

Usage:
    optimizer_group = build_optimizer_group(model, config)

    for step in range(steps):
        optimizer_group.zero_grad()
        loss.backward()
        optimizer_group.step()
        optimizer_group.schedule_step()
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Dict, List, Optional, Tuple, Any, Callable, Iterable
from dataclasses import dataclass, field
import math

# FP8 dtypes (requires PyTorch 2.1+ and Hopper/Ada GPU for native support)
# E4M3: 4 exponent bits, 3 mantissa - better for weights (more precision)
# E5M2: 5 exponent bits, 2 mantissa - better for gradients (more range)
FP8_E4M3 = getattr(torch, 'float8_e4m3fn', None)
FP8_E5M2 = getattr(torch, 'float8_e5m2', None)
HAS_FP8 = FP8_E4M3 is not None

# Try to import triton for fused FP8 kernels
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# =============================================================================
# Fused FP8 Triton Kernels - Row-wise Quantization
# =============================================================================

if HAS_TRITON and HAS_FP8:
    @triton.jit
    def _fp8_rowwise_quantize_kernel(
        x_ptr, out_ptr, scale_ptr,
        M, K,
        stride_xm, stride_xk,
        stride_om, stride_ok,
        BLOCK_K: tl.constexpr,
    ):
        """
        Single-pass row-wise FP8 quantization.
        Each row gets its own scale factor = 448 / row_amax.
        No global reduction needed - fully parallelized.
        """
        row_idx = tl.program_id(0)

        # Compute row amax (scalar accumulator)
        row_amax = 0.0

        # First pass: compute row amax
        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            mask = k_offs < K
            x = tl.load(x_ptr + row_idx * stride_xm + k_offs * stride_xk, mask=mask, other=0.0)
            block_max = tl.max(tl.abs(x))
            row_amax = tl.maximum(row_amax, block_max)

        # Compute scale for this row
        row_amax = tl.maximum(row_amax, 1e-12)
        scale = 448.0 / row_amax
        inv_scale = row_amax / 448.0

        # Store inverse scale (for _scaled_mm)
        tl.store(scale_ptr + row_idx, inv_scale)

        # Second pass: apply scale and store
        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            mask = k_offs < K
            x = tl.load(x_ptr + row_idx * stride_xm + k_offs * stride_xk, mask=mask, other=0.0)
            x_scaled = x * scale
            tl.store(out_ptr + row_idx * stride_om + k_offs * stride_ok, x_scaled, mask=mask)

    @triton.jit
    def _fp8_colwise_quantize_kernel(
        x_ptr, out_ptr, scale_ptr,
        K, N,
        stride_xk, stride_xn,
        stride_ok, stride_on,
        BLOCK_K: tl.constexpr,
    ):
        """
        Single-pass column-wise FP8 quantization for weight matrix.
        Each column gets its own scale factor.
        """
        col_idx = tl.program_id(0)

        # Compute column amax (scalar accumulator)
        col_amax = 0.0
        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            mask = k_offs < K
            x = tl.load(x_ptr + k_offs * stride_xk + col_idx * stride_xn, mask=mask, other=0.0)
            block_max = tl.max(tl.abs(x))
            col_amax = tl.maximum(col_amax, block_max)

        col_amax = tl.maximum(col_amax, 1e-12)
        scale = 448.0 / col_amax
        inv_scale = col_amax / 448.0
        tl.store(scale_ptr + col_idx, inv_scale)

        # Apply scale
        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            mask = k_offs < K
            x = tl.load(x_ptr + k_offs * stride_xk + col_idx * stride_xn, mask=mask, other=0.0)
            tl.store(out_ptr + k_offs * stride_ok + col_idx * stride_on, x * scale, mask=mask)

    def _triton_rowwise_quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Row-wise quantize (M, K) tensor. Returns FP8 tensor and (M, 1) scales."""
        M, K = x.shape
        BLOCK_K = min(1024, triton.next_power_of_2(K))

        out_f32 = torch.empty_like(x, dtype=torch.float32)
        scales = torch.empty(M, 1, device=x.device, dtype=torch.float32)

        _fp8_rowwise_quantize_kernel[(M,)](
            x, out_f32, scales,
            M, K,
            x.stride(0), x.stride(1),
            out_f32.stride(0), out_f32.stride(1),
            BLOCK_K,
        )

        return out_f32.to(FP8_E4M3), scales

    def _triton_colwise_quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Column-wise quantize (K, N) tensor. Returns FP8 tensor and (1, N) scales."""
        K, N = x.shape
        BLOCK_K = min(1024, triton.next_power_of_2(K))

        out_f32 = torch.empty_like(x, dtype=torch.float32)
        scales = torch.empty(1, N, device=x.device, dtype=torch.float32)

        _fp8_colwise_quantize_kernel[(N,)](
            x, out_f32, scales,
            K, N,
            x.stride(0), x.stride(1),
            out_f32.stride(0), out_f32.stride(1),
            BLOCK_K,
        )

        return out_f32.to(FP8_E4M3), scales

    @triton.jit
    def _fp8_amax_kernel(
        x_ptr, amax_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Parallel block-wise amax reduction. Stores block maxes for final reduction."""
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        block_max = tl.max(tl.abs(x))

        # Atomic max to global amax
        tl.atomic_max(amax_ptr, block_max)

    @triton.jit
    def _fp8_scale_kernel(
        x_ptr, out_ptr, amax_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Apply computed scale to quantize tensor."""
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        amax = tl.load(amax_ptr)
        scale = 448.0 / tl.maximum(amax, 1e-12)
        x_scaled = x * scale

        tl.store(out_ptr + offsets, x_scaled, mask=mask)

    def _triton_tensor_quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-tensor FP8 quantization using fused Triton kernels."""
        x_flat = x.reshape(-1)
        n = x_flat.numel()
        BLOCK_SIZE = 1024

        # Phase 1: parallel amax reduction
        amax = torch.zeros(1, device=x.device, dtype=torch.float32)
        grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        _fp8_amax_kernel[grid](x_flat, amax, n, BLOCK_SIZE)

        # Phase 2: apply scale
        out_f32 = torch.empty_like(x_flat, dtype=torch.float32)
        _fp8_scale_kernel[grid](x_flat, out_f32, amax, n, BLOCK_SIZE)

        # Convert to FP8 and compute inverse scale
        x_fp8 = out_f32.view(x.shape).to(FP8_E4M3)
        inv_scale = (amax.clamp(min=1e-12) / 448.0).squeeze()

        return x_fp8, inv_scale


# =============================================================================
# FP8 Matmul with Per-Tensor Scaling (Ada compatible)
# =============================================================================

class _FP8MatmulFunc(torch.autograd.Function):
    """
    FP8 matmul using _scaled_mm with per-tensor scaling.

    Ada (SM89) only supports TensorWise scaling.
    Uses fused Triton kernel for amax + quantization when available.
    """

    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        ctx.input_shape = input.shape

        input_2d = input.view(-1, input.shape[-1])
        N = weight.shape[0]  # weight is (out, in) = (N, K)

        # Fused amax + scale + quantize for input
        if HAS_TRITON:
            input_fp8, input_inv_scale = _triton_tensor_quantize(input_2d)
            weight_fp8, weight_inv_scale = _triton_tensor_quantize(weight)
        else:
            input_amax = input_2d.abs().amax().clamp(min=1e-12)
            input_scale = (448.0 / input_amax).float()
            input_fp8 = (input_2d.float() * input_scale).to(FP8_E4M3)
            input_inv_scale = (1.0 / input_scale)

            weight_amax = weight.abs().amax().clamp(min=1e-12)
            weight_scale = (448.0 / weight_amax).float()
            weight_fp8 = (weight.float() * weight_scale).to(FP8_E4M3)
            weight_inv_scale = (1.0 / weight_scale)

        # _scaled_mm: row-major @ column-major
        # weight.t() gives column-major when weight is contiguous
        output = torch._scaled_mm(
            input_fp8,
            weight_fp8.t(),
            scale_a=input_inv_scale,
            scale_b=weight_inv_scale,
            out_dtype=torch.bfloat16
        )

        output = output.view(*ctx.input_shape[:-1], N)
        if bias is not None:
            output = output + bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        input_2d = input.view(-1, input.shape[-1])
        grad_output_2d = grad_output.view(-1, grad_output.shape[-1])

        # Gradients in bf16 for numerical stability
        grad_input = (grad_output_2d @ weight).view(ctx.input_shape)
        grad_weight = grad_output_2d.t() @ input_2d
        grad_bias = grad_output_2d.sum(0) if bias is not None else None

        return grad_input, grad_weight, grad_bias


# Global flag to control FP8 usage
# On Ada (SM89), per-tensor scaling overhead often exceeds tensor core gains
# Enable only if you have very large matrices or cached weight quantization
FP8_ENABLED = False


def _fp8_linear(input, weight, bias):
    """FP8 linear - uses tensor cores on Ada/Hopper when enabled."""
    if FP8_ENABLED and HAS_FP8 and input.is_cuda:
        return _FP8MatmulFunc.apply(input, weight, bias)
    else:
        return nn.functional.linear(input, weight, bias)


def set_fp8_enabled(enabled: bool):
    """Enable/disable FP8 matmul globally. Default is disabled on Ada due to overhead."""
    global FP8_ENABLED
    FP8_ENABLED = enabled


class _FP8DequantSTE(torch.autograd.Function):
    """
    Straight-Through Estimator for FP8 dequantization with gradient capture.

    Forward: dequant FP8 -> compute dtype
    Backward: captures gradient (detached) for manual application to FP8
    """
    @staticmethod
    def forward(ctx, weight_fp8: torch.Tensor, scale: torch.Tensor, grad_holder: list, target_dtype: torch.dtype):
        ctx.grad_holder = grad_holder
        return weight_fp8.to(target_dtype) * scale.to(target_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # Capture gradient DETACHED - critical to avoid graph retention issues
        ctx.grad_holder.append(grad_output.detach().clone())
        return None, None, None, None


class FP8Linear(nn.Module):
    """
    W8A16 Linear: 8-bit weights, 16-bit activations. TRUE NO-MASTER-WEIGHT TRAINING.

    Memory profile:
    - Weight storage: 1 byte/param (FP8 only, no bf16 shadow)
    - Scale: 4 bytes per tensor (negligible)
    - Gradient: captured lazily, freed after step

    Architecture:
    - Weights stored ONLY as FP8 + per-tensor scale
    - Forward: dequant to bf16 via STE, regular matmul
    - Backward: STE captures gradient into holder (not into a Parameter)
    - Optimizer: external FP8-aware optimizer applies update directly

    Bias is bf16 Parameter (small, not worth quantizing).

    Usage:
        # After loss.backward():
        fp8_optimizer.step()  # Applies captured gradients to FP8 weights
        fp8_optimizer.zero_grad()  # Clears gradient holders
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._compute_dtype = dtype or torch.bfloat16

        # FP8 is the ONLY weight storage (1 byte/param)
        if HAS_FP8:
            self.register_buffer('weight_fp8', torch.empty(out_features, in_features, device=device, dtype=FP8_E4M3))
            self.register_buffer('weight_scale', torch.ones(1, device=device, dtype=torch.float32))
        else:
            # Fallback: bf16 weight as Parameter
            self.weight_fp8 = None
            self.weight_scale = None
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features, device=device, dtype=dtype or torch.bfloat16)
            )

        # Gradient holder - captured by STE backward, consumed by optimizer
        self._grad_holder: list = []

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype or torch.bfloat16))
        else:
            self.register_parameter('bias', None)

        self._init_weights()

    def _init_weights(self):
        # Initialize weights
        temp = torch.empty(self.out_features, self.in_features, device=self._get_device(), dtype=torch.float32)
        nn.init.kaiming_uniform_(temp, a=math.sqrt(5))

        if self.weight_fp8 is not None:
            self._quantize_to_fp8(temp)
        else:
            self.weight.data.copy_(temp.to(self.weight.dtype))

        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _get_device(self):
        if self.weight_fp8 is not None:
            return self.weight_fp8.device
        return self.weight.device if hasattr(self, 'weight') else 'cuda'

    def _quantize_to_fp8(self, tensor: torch.Tensor):
        """Quantize tensor to FP8 storage."""
        amax = tensor.abs().amax().clamp(min=1e-12)
        self.weight_fp8.copy_((tensor.float() * (448.0 / amax)).to(FP8_E4M3))
        self.weight_scale.fill_(amax / 448.0)

    def _dequantize(self, dtype: torch.dtype) -> torch.Tensor:
        """Dequantize FP8 to target dtype."""
        return self.weight_fp8.to(dtype) * self.weight_scale.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_fp8 is not None:
            # Dequant with STE - gradient captured to holder
            w = _FP8DequantSTE.apply(self.weight_fp8, self.weight_scale, self._grad_holder, x.dtype)
            return nn.functional.linear(x, w, self.bias)
        else:
            return nn.functional.linear(x, self.weight, self.bias)

    def get_grad(self) -> Optional[torch.Tensor]:
        """Get captured gradient (sum if multiple backward passes)."""
        if not self._grad_holder:
            return None
        if len(self._grad_holder) == 1:
            return self._grad_holder[0]
        return sum(self._grad_holder)

    def clear_grad(self):
        """Clear captured gradients."""
        self._grad_holder.clear()

    def apply_update(self, update: torch.Tensor):
        """Apply weight update directly to FP8 storage."""
        if self.weight_fp8 is None:
            self.weight.data.add_(update.to(self.weight.dtype))
            return

        # Dequant -> apply -> requant cycle
        w = self._dequantize(torch.float32)
        w_new = w + update.float()
        self._quantize_to_fp8(w_new)

    # Legacy compatibility
    def sync_fp8(self):
        pass

    def mark_dirty(self):
        pass  # No-op - updates are applied directly via apply_update

    def apply_gradient_update(self):
        pass  # No-op - use FP8SGD/FP8Muon optimizer instead

    def to_inference(self):
        """Already minimal - just clear grad holder."""
        self._grad_holder.clear()
        return self


class FP8SGD:
    """
    SGD optimizer for FP8Linear layers. No master weights, no state overhead.

    Memory: 0 bytes optimizer state (just lr scalar)
    """
    def __init__(self, fp8_modules: List[FP8Linear], lr: float = 1e-3):
        self.fp8_modules = fp8_modules
        self.lr = lr

    def step(self):
        for m in self.fp8_modules:
            grad = m.get_grad()
            if grad is None:
                continue
            m.apply_update(-self.lr * grad)

    def zero_grad(self):
        for m in self.fp8_modules:
            m.clear_grad()


class FP8Muon:
    """
    Muon optimizer for FP8Linear layers. Momentum stored as bf16 (2 bytes/param).

    Memory: 2 bytes/param for momentum (vs 8 bytes for AdamW state)

    Total with FP8 weights: 1 + 2 = 3 bytes/param
    (vs bf16 + AdamW: 2 + 8 = 10 bytes/param)

    IMPORTANT: This implementation matches the standard Muon optimizer exactly:
    - Momentum accumulates gradients: buf = momentum * buf + grad (NOT EMA style)
    - Orthogonalization uses proper spectral norm estimate and restores scale
    - Nesterov: update = grad + momentum * orthogonalized_buf
    """
    def __init__(
        self,
        fp8_modules: List[FP8Linear],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        self.fp8_modules = fp8_modules
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps

        # Momentum buffers stored as bf16 (2 bytes/param)
        self.momentum_buffers: Dict[int, torch.Tensor] = {}
        for m in fp8_modules:
            self.momentum_buffers[id(m)] = torch.zeros(
                m.out_features, m.in_features,
                device=m._get_device(), dtype=torch.bfloat16
            )

    def _orthogonalize(self, m: torch.Tensor) -> torch.Tensor:
        """
        Newton-Schulz orthogonalization matching standard Muon.

        Key steps:
        1. Compute in fp32 for numerical stability
        2. Estimate spectral norm using Frobenius/sqrt(min(n,d))
        3. Scale to unit spectral norm for convergence
        4. Run Newton-Schulz iterations
        5. Restore original scale
        """
        original_dtype = m.dtype
        m = m.float()

        if m.ndim < 2:
            return m.to(original_dtype)

        # Reshape to 2D if needed
        original_shape = m.shape
        if m.ndim > 2:
            m = m.reshape(m.shape[0], -1)

        n, d = m.shape

        # Transpose if more columns than rows (want tall matrix)
        transposed = n < d
        if transposed:
            m = m.T
            n, d = m.shape

        # Spectral norm estimate: Frobenius / sqrt(min(n,d))
        frob = m.norm()
        spectral_est = frob / math.sqrt(min(n, d))
        scale = spectral_est + 1e-7

        # Scale to approx unit spectral norm
        X = m / scale

        # Newton-Schulz iteration: X = 1.5*X - 0.5*X@X^T@X
        for _ in range(self.ns_steps):
            A = X @ X.T
            X = 1.5 * X - 0.5 * A @ X

        # Restore scale
        X = X * scale

        if transposed:
            X = X.T

        return X.reshape(original_shape).to(original_dtype)

    def step(self):
        for m in self.fp8_modules:
            grad = m.get_grad()
            if grad is None:
                continue

            # Get momentum buffer
            buf = self.momentum_buffers[id(m)]

            # Standard momentum: buf = momentum * buf + grad (NOT EMA style!)
            buf.mul_(self.momentum).add_(grad.to(torch.bfloat16))

            # Orthogonalize momentum buffer
            update = self._orthogonalize(buf)

            # Nesterov: update = grad + momentum * orthogonalized_momentum
            if self.nesterov:
                update = grad.to(torch.bfloat16) + self.momentum * update

            # Apply update to FP8 weight
            m.apply_update(-self.lr * update)

    def zero_grad(self):
        for m in self.fp8_modules:
            m.clear_grad()


class FP8AdamW:
    """
    AdamW optimizer for FP8Linear layers. Maintains proper m/v state.

    Memory: 8 bytes/param for optimizer state (m in bf16, v in fp32)
    Total with FP8 weights: 1 + 2 + 4 = 7 bytes/param
    (vs bf16 + AdamW: 2 + 8 = 10 bytes/param)

    This ensures FP8 weights get the same adaptive learning rate behavior
    as regular AdamW, avoiding the training instability from naive SGD.
    """
    def __init__(
        self,
        fp8_modules: List[FP8Linear],
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.1,
        eps: float = 1e-8,
    ):
        self.fp8_modules = fp8_modules
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.eps = eps
        self.step_count = 0

        # State buffers: m in bf16 (momentum), v in fp32 (variance needs precision)
        self.m_buffers: Dict[int, torch.Tensor] = {}
        self.v_buffers: Dict[int, torch.Tensor] = {}
        for m in fp8_modules:
            device = m._get_device()
            shape = (m.out_features, m.in_features)
            self.m_buffers[id(m)] = torch.zeros(shape, device=device, dtype=torch.bfloat16)
            self.v_buffers[id(m)] = torch.zeros(shape, device=device, dtype=torch.float32)

    def step(self):
        self.step_count += 1
        bias_correction1 = 1 - self.beta1 ** self.step_count
        bias_correction2 = 1 - self.beta2 ** self.step_count

        for m in self.fp8_modules:
            grad = m.get_grad()
            if grad is None:
                continue

            # Get state buffers
            m_buf = self.m_buffers[id(m)]
            v_buf = self.v_buffers[id(m)]

            # Get current weight for decoupled weight decay
            # FP8Linear stores weight_fp8 + scale, we need to get the effective weight
            weight = m.weight_fp8.to(torch.float32) * m.weight_scale

            # AdamW: decoupled weight decay (applied to weight, not gradient)
            if self.weight_decay > 0:
                weight_update = -self.lr * self.weight_decay * weight

            # Update biased first moment estimate (m = beta1 * m + (1 - beta1) * grad)
            grad_f32 = grad.float()
            m_buf.mul_(self.beta1).add_(grad_f32.to(torch.bfloat16), alpha=1 - self.beta1)

            # Update biased second moment estimate (v = beta2 * v + (1 - beta2) * grad^2)
            v_buf.mul_(self.beta2).addcmul_(grad_f32, grad_f32, value=1 - self.beta2)

            # Bias correction
            m_hat = m_buf.float() / bias_correction1
            v_hat = v_buf / bias_correction2

            # Compute update: -lr * m_hat / (sqrt(v_hat) + eps)
            update = -self.lr * m_hat / (v_hat.sqrt() + self.eps)

            # Add weight decay update
            if self.weight_decay > 0:
                update = update + weight_update

            # Apply update to FP8 weight
            m.apply_update(update)

    def zero_grad(self):
        for m in self.fp8_modules:
            m.clear_grad()


def collect_fp8_modules(model: nn.Module) -> List[FP8Linear]:
    """Collect all FP8Linear modules from a model."""
    return [m for m in model.modules() if isinstance(m, FP8Linear)]


def mark_fp8_dirty(model: nn.Module):
    """Legacy compat - no-op for new FP8Linear (updates applied via optimizer)."""
    pass  # Gradients are now applied via FP8SGD/FP8Muon.step()


def convert_to_fp8(
    module: nn.Module,
    skip_patterns: List[str] = None,
    device=None,
) -> nn.Module:
    """
    Convert Linear layers to FP8Linear, skipping embeddings/norms.

    Args:
        module: Model to convert
        skip_patterns: Name patterns to skip (default: embed, norm, head)
        device: Target device

    Returns:
        Converted module (in-place modification)
    """
    if skip_patterns is None:
        skip_patterns = ['embed', 'head', 'lm_head', 'wte', 'wpe', 'norm', 'ln', 'layernorm']

    skip_patterns = [p.lower() for p in skip_patterns]

    def should_skip(name: str) -> bool:
        name_lower = name.lower()
        return any(p in name_lower for p in skip_patterns)

    # Collect replacements (can't modify during iteration)
    replacements = []

    for name, child in module.named_modules():
        if isinstance(child, nn.Linear) and not should_skip(name):
            replacements.append((name, child))

    # Apply replacements
    for name, old_linear in replacements:
        # Navigate to parent
        parts = name.split('.')
        parent = module
        for part in parts[:-1]:
            parent = getattr(parent, part)

        # Create FP8Linear with same config
        new_linear = FP8Linear(
            old_linear.in_features,
            old_linear.out_features,
            bias=old_linear.bias is not None,
            device=device or old_linear.weight.device,
            dtype=old_linear.weight.dtype,
        )

        # Copy weights - FP8Linear stores weights differently based on HAS_FP8
        if new_linear.weight_fp8 is not None:
            # HAS_FP8=True: quantize directly to FP8 storage (no master weight)
            new_linear._quantize_to_fp8(old_linear.weight.data)
        else:
            # Fallback: copy to bf16 Parameter
            new_linear.weight.data.copy_(old_linear.weight.data)

        if old_linear.bias is not None:
            new_linear.bias.data.copy_(old_linear.bias.data)

        # Replace
        setattr(parent, parts[-1], new_linear)

    return module


def get_fp8_param_groups(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """
    Separate parameters into FP8 and non-FP8 groups.

    Returns:
        (fp8_params, bf16_params) - FP8Linear params vs regular params
    """
    fp8_params = []
    bf16_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if this param belongs to an FP8Linear
        is_fp8 = False
        parts = name.split('.')
        parent = model
        try:
            for part in parts[:-1]:
                parent = getattr(parent, part)
            is_fp8 = isinstance(parent, FP8Linear)
        except AttributeError:
            pass

        if is_fp8:
            fp8_params.append(param)
        else:
            bf16_params.append(param)

    return fp8_params, bf16_params


# =============================================================================
# Muon Optimizer
# =============================================================================

class Muon(Optimizer):
    """
    Muon: Momentum-Orthogonalized Updates.

    Key insight: Standard momentum can cause updates to collapse into a
    low-dimensional subspace. Muon orthogonalizes the momentum against
    previous update directions, maintaining diversity in the update space.

    For transformer layers, this helps with:
    - Better gradient utilization across all dimensions
    - Reduced sensitivity to learning rate
    - More stable training at lower precision (fp8)

    Numerical properties:
    - Only maintains momentum buffer (no second moment like AdamW's v)
    - No division by small numbers (orthogonalization replaces adaptive LR)
    - Safe to use bf16 for optimizer state (unlike AdamW which needs fp32 for v)

    Reference: https://github.com/KellerJordan/modded-nanogpt

    Args:
        params: Parameters to optimize (should be transformer weights, NOT embeddings)
        lr: Learning rate (default: 0.02, higher than AdamW due to orthogonalization)
        momentum: Momentum coefficient (default: 0.95)
        nesterov: Use Nesterov momentum (default: True)
        ns_steps: Newton-Schulz iteration steps for orthogonalization (default: 5)
        state_dtype: Dtype for momentum buffer (default: bf16, safe due to no v term)
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        state_dtype: torch.dtype = torch.bfloat16,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, state_dtype=state_dtype)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            state_dtype = group['state_dtype']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Get or initialize momentum buffer in bf16
                # Safe because Muon has no second moment - just simple EMA
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p, dtype=state_dtype)

                buf = state['momentum_buffer']

                # Update momentum buffer (cast grad to state dtype)
                # buf = momentum * buf + grad
                buf.mul_(momentum).add_(grad.to(state_dtype))

                # Orthogonalize for 2D+ parameters (transformer weights)
                # Work in bf16 - orthogonalization is numerically stable
                if p.dim() >= 2:
                    update = self._orthogonalize(buf, ns_steps)
                else:
                    # 1D params (biases, norms) - just use momentum directly
                    update = buf

                # Nesterov momentum
                if nesterov:
                    update = grad.to(state_dtype) + momentum * update

                # Apply update (cast back to param dtype)
                p.add_(update.to(p.dtype), alpha=-lr)

        return loss

    def _orthogonalize(self, m: torch.Tensor, ns_steps: int) -> torch.Tensor:
        """
        Orthogonalize momentum tensor using Newton-Schulz iteration.

        For a matrix M, computes M @ (M^T @ M)^{-1/2} which orthogonalizes
        the columns of M. This is done iteratively without explicit inverse.

        NUMERICAL STABILITY:
        - All computation done in fp32 (bf16 matmuls accumulate error over ns_steps)
        - Uses spectral norm estimate for scaling (Frobenius/sqrt(min(n,d)) approximates max singular value)
        - Result cast back to input dtype at the end
        """
        original_dtype = m.dtype
        original_shape = m.shape

        # Reshape to 2D for matrix operations
        if m.dim() > 2:
            m = m.view(m.shape[0], -1)
        elif m.dim() == 1:
            return m  # Can't orthogonalize 1D

        # Transpose if more columns than rows (want tall matrix for efficiency)
        transposed = m.shape[0] < m.shape[1]
        if transposed:
            m = m.T

        n, d = m.shape

        # === CRITICAL: Do iteration in fp32 ===
        # bf16 has ~3 decimal digits of precision; 5 iterations of matmuls
        # compounds rounding errors, leading to NaN/divergence
        X = m.float()

        # Scale to approximate unit spectral norm
        # Frobenius / sqrt(min(n,d)) estimates the largest singular value
        # (exact for rank-1, good approximation for well-conditioned matrices)
        frob = X.norm()
        spectral_est = frob / math.sqrt(min(n, d))
        scale = spectral_est + 1e-7
        X = X / scale

        # Newton-Schulz iteration: X_{k+1} = 1.5*X - 0.5*X@X^T@X
        # Converges to orthogonal polar factor when spectral_norm(X) < sqrt(3)
        for _ in range(ns_steps):
            A = X @ X.T
            X = 1.5 * X - 0.5 * A @ X

        # Rescale and cast back to original dtype
        X = X * scale
        X = X.to(original_dtype)

        if transposed:
            X = X.T

        return X.view(original_shape)


# =============================================================================
# Optimizer Group - Coordinates Multiple Optimizers
# =============================================================================

@dataclass
class OptimizerSpec:
    """Specification for a single optimizer in the group."""
    name: str
    optimizer: Optimizer
    scheduler: Optional[LRScheduler] = None
    param_names: List[str] = field(default_factory=list)


class OptimizerGroup:
    """
    Coordinates multiple optimizers for heterogeneous training.

    Provides unified interface for:
    - Zero grad across all optimizers
    - Step all optimizers
    - Step all schedulers
    - Gradient clipping per group
    - State dict save/load
    """

    def __init__(self, specs: List[OptimizerSpec]):
        self.specs = {s.name: s for s in specs}
        self._step_count = 0

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients for all optimizers."""
        for spec in self.specs.values():
            spec.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Optional[Callable] = None):
        """Step all optimizers."""
        for spec in self.specs.values():
            spec.optimizer.step(closure)
        self._step_count += 1

    def schedule_step(self):
        """Step all schedulers."""
        for spec in self.specs.values():
            if spec.scheduler is not None:
                spec.scheduler.step()

    def clip_grad_norm(self, max_norm: float, norm_type: float = 2.0) -> Dict[str, float]:
        """Clip gradients per optimizer group. Returns dict of norms."""
        norms = {}
        for name, spec in self.specs.items():
            params = [p for g in spec.optimizer.param_groups for p in g['params'] if p.grad is not None]
            if params:
                norm = torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type)
                norms[name] = norm.item()
        return norms

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for all optimizers and schedulers."""
        return {
            'optimizers': {name: spec.optimizer.state_dict() for name, spec in self.specs.items()},
            'schedulers': {
                name: spec.scheduler.state_dict()
                for name, spec in self.specs.items()
                if spec.scheduler is not None
            },
            'step_count': self._step_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict for all optimizers and schedulers."""
        for name, opt_state in state_dict['optimizers'].items():
            if name in self.specs:
                self.specs[name].optimizer.load_state_dict(opt_state)
        for name, sched_state in state_dict.get('schedulers', {}).items():
            if name in self.specs and self.specs[name].scheduler is not None:
                self.specs[name].scheduler.load_state_dict(sched_state)
        self._step_count = state_dict.get('step_count', 0)

    def get_lr(self) -> Dict[str, float]:
        """Get current learning rate for each optimizer."""
        return {
            name: spec.optimizer.param_groups[0]['lr']
            for name, spec in self.specs.items()
        }

    @property
    def step_count(self) -> int:
        return self._step_count


# =============================================================================
# Parameter Classification
# =============================================================================

def classify_parameters(
    model: nn.Module,
    embedding_patterns: List[str] = None,
    norm_patterns: List[str] = None,
    fsq_patterns: List[str] = None,
    ae_patterns: List[str] = None,
) -> Dict[str, List[Tuple[str, nn.Parameter]]]:
    """
    Classify model parameters into groups for heterogeneous optimization.

    Groups:
    - 'embedding': Embedding and unembedding layers (use AdamW)
    - 'norm': LayerNorm, RMSNorm parameters (use AdamW, no weight decay)
    - 'fsq_adjacent': Parameters feeding into FSQ/sparsity (use AdamW)
        These have attenuated gradients due to sigmoid STE or sparse masking.
        Muon's orthogonalization assumes uniform gradient magnitudes.
    - 'ae': Autoencoder components (use AdamW)
        All encoder/decoder layers from the AE should use AdamW.
        FSQ, sparsity, and the transformer layers inside AE have heterogeneous gradients.
    - 'transformer': Everything else (use Muon)
        This should primarily be the denoiser's uniform transformer blocks.

    Args:
        model: The model to classify
        embedding_patterns: Name patterns for embedding params
        norm_patterns: Name patterns for norm params
        fsq_patterns: Name patterns for FSQ-adjacent params
        ae_patterns: Name patterns for AE components (all go to AdamW)

    Returns:
        Dict mapping group name to list of (param_name, param) tuples
    """
    if embedding_patterns is None:
        embedding_patterns = ['embed', 'head', 'lm_head', 'wte', 'wpe', 'embedder', 'unembedder']
    if norm_patterns is None:
        norm_patterns = ['norm', 'ln', 'layernorm', 'rmsnorm']
    if fsq_patterns is None:
        # code_proj: feeds into sigmoid FSQ (gradient attenuated by sigmoid')
        # fsq: the quantization module itself
        # sparsity: sparse masking zeros most gradients
        # dim_logits: topk selection has zero gradient (now has MoE STE but still heterogeneous)
        # attn_gate: post-attention sigmoid gate
        # logsnr: small auxiliary heads, not worth orthogonalizing
        fsq_patterns = ['code_proj', 'code_unproj', 'fsq', 'sparsity', 'dim_logits',
                        'attn_gate', 'logsnr']
    if ae_patterns is None:
        # All AE components should use AdamW, not Muon
        # - sparse_ae: the full autoencoder (encoders, decoders)
        # - encoders/decoders: hierarchical level modules
        # - level_logsnr: per-level logsnr estimators
        # Note: denoiser should NOT match these patterns (it's a separate module)
        ae_patterns = ['sparse_ae', 'encoders', 'decoders', 'level_logsnr']

    embedding_patterns = [p.lower() for p in embedding_patterns]
    norm_patterns = [p.lower() for p in norm_patterns]
    fsq_patterns = [p.lower() for p in fsq_patterns]
    ae_patterns = [p.lower() for p in ae_patterns]

    groups = {
        'embedding': [],
        'norm': [],
        'fsq_adjacent': [],
        'ae': [],
        'transformer': [],
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        name_lower = name.lower()

        # Check embedding patterns first (highest priority)
        if any(p in name_lower for p in embedding_patterns):
            groups['embedding'].append((name, param))
        # Check norm patterns
        elif any(p in name_lower for p in norm_patterns):
            groups['norm'].append((name, param))
        # Check AE patterns (before FSQ - AE is more specific)
        elif any(p in name_lower for p in ae_patterns):
            groups['ae'].append((name, param))
        # Check FSQ-adjacent patterns (sigmoid STE, sparse masks = bad for Muon)
        elif any(p in name_lower for p in fsq_patterns):
            groups['fsq_adjacent'].append((name, param))
        # Everything else is transformer (Muon territory) - should be denoiser layers
        else:
            groups['transformer'].append((name, param))

    return groups


# =============================================================================
# Builder Functions
# =============================================================================

def build_optimizer_group(
    model: nn.Module,
    config: Dict[str, Any],
    total_steps: int,
) -> OptimizerGroup:
    """
    Build heterogeneous optimizer group from config.

    Config structure (defined in config.py OptimizerConfig):
        training.optimizer:
            type: "heterogeneous"  # or "adamw" for single optimizer

            # Muon settings for transformer layers
            muon:
                lr: 0.02
                momentum: 0.95

            # AdamW settings for embeddings
            adamw:
                lr: 3e-4
                weight_decay: 0.1
                betas: [0.9, 0.95]

            # Scheduler (applied to all)
            scheduler:
                type: "onecycle"
                pct_start: 0.1

    All fields must be present in config after Pydantic validation.
    Missing fields = schema bug in config.py, crash is correct behavior.
    """
    opt_cfg = config['training']['optimizer']
    opt_type = opt_cfg['type']

    if opt_type == 'adamw':
        return _build_single_adamw(model, opt_cfg, total_steps)
    elif opt_type == 'heterogeneous':
        return _build_heterogeneous(model, opt_cfg, total_steps)
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")


def _build_single_adamw(
    model: nn.Module,
    opt_cfg: Dict[str, Any],
    total_steps: int,
) -> OptimizerGroup:
    """Build single AdamW optimizer (backwards compatible).

    Uses top-level fields from OptimizerConfig (lr, weight_decay, betas).
    All fields guaranteed present by Pydantic schema.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg['lr'],
        weight_decay=opt_cfg['weight_decay'],
        betas=tuple(opt_cfg['betas']),
        fused=True,
    )

    scheduler = _build_scheduler(optimizer, opt_cfg, total_steps)

    return OptimizerGroup([
        OptimizerSpec(
            name='adamw',
            optimizer=optimizer,
            scheduler=scheduler,
            param_names=[n for n, _ in model.named_parameters()],
        )
    ])


def _parse_dtype(dtype_str: str) -> torch.dtype:
    """Parse dtype string to torch.dtype."""
    dtype_map = {
        'fp32': torch.float32,
        'float32': torch.float32,
        'bf16': torch.bfloat16,
        'bfloat16': torch.bfloat16,
        'fp16': torch.float16,
        'float16': torch.float16,
    }
    return dtype_map.get(dtype_str.lower(), torch.bfloat16)


def _build_heterogeneous(
    model: nn.Module,
    opt_cfg: Dict[str, Any],
    total_steps: int,
) -> OptimizerGroup:
    """Build heterogeneous optimizer group with Muon + AdamW.

    Requires muon and adamw subconfigs to be present in opt_cfg.
    Pydantic validator in config.py ensures this when type == "heterogeneous".
    """
    # Get targeting config if present, otherwise use defaults
    targeting = opt_cfg.get('targeting')
    if targeting is not None:
        param_groups = classify_parameters(
            model,
            embedding_patterns=targeting.get('embedding_patterns'),
            norm_patterns=targeting.get('norm_patterns'),
            fsq_patterns=targeting.get('fsq_patterns'),
            ae_patterns=targeting.get('ae_patterns'),
        )
    else:
        param_groups = classify_parameters(model)

    specs = []

    # muon and adamw subconfigs MUST exist when type == "heterogeneous"
    # Schema validation in config.py guarantees this - crash here = schema bug
    muon_cfg = opt_cfg['muon']
    adamw_cfg = opt_cfg['adamw']

    # Muon for transformer layers
    # Uses bf16 state by default (safe - no second moment accumulator)
    transformer_params = [p for _, p in param_groups['transformer']]
    if transformer_params:
        state_dtype = _parse_dtype(muon_cfg['state_dtype'])
        muon_lr = muon_cfg['lr']
        muon_opt = Muon(
            transformer_params,
            lr=muon_lr,
            momentum=muon_cfg['momentum'],
            nesterov=muon_cfg['nesterov'],
            ns_steps=muon_cfg['ns_steps'],
            state_dtype=state_dtype,
        )
        # Pass Muon's lr as max_lr for scheduler
        muon_sched = _build_scheduler(muon_opt, opt_cfg, total_steps, max_lr_override=muon_lr)
        specs.append(OptimizerSpec(
            name='muon',
            optimizer=muon_opt,
            scheduler=muon_sched,
            param_names=[n for n, _ in param_groups['transformer']],
        ))

    # AdamW for embeddings (needs fp32 state due to v accumulator numerical sensitivity)
    # The m/sqrt(v) ratio is numerically unstable - v can get very small, division explodes
    # PyTorch AdamW always uses fp32 for state internally
    embedding_params = [p for _, p in param_groups['embedding']]
    if embedding_params:
        adamw_lr = adamw_cfg['lr']
        adamw_opt = torch.optim.AdamW(
            embedding_params,
            lr=adamw_lr,
            weight_decay=adamw_cfg['weight_decay'],
            betas=tuple(adamw_cfg['betas']),
            fused=True,
        )
        # Pass AdamW's lr as max_lr for scheduler
        adamw_sched = _build_scheduler(adamw_opt, opt_cfg, total_steps, max_lr_override=adamw_lr)
        specs.append(OptimizerSpec(
            name='adamw_embed',
            optimizer=adamw_opt,
            scheduler=adamw_sched,
            param_names=[n for n, _ in param_groups['embedding']],
        ))

    # AdamW for norms (no weight decay)
    norm_params = [p for _, p in param_groups['norm']]
    if norm_params:
        adamw_lr = adamw_cfg['lr']  # Same lr as embeddings
        norm_opt = torch.optim.AdamW(
            norm_params,
            lr=adamw_lr,
            weight_decay=0.0,  # Never decay norms
            betas=tuple(adamw_cfg['betas']),
            fused=True,
        )
        norm_sched = _build_scheduler(norm_opt, opt_cfg, total_steps, max_lr_override=adamw_lr)
        specs.append(OptimizerSpec(
            name='adamw_norm',
            optimizer=norm_opt,
            scheduler=norm_sched,
            param_names=[n for n, _ in param_groups['norm']],
        ))

    # AdamW for FSQ-adjacent params (sigmoid STE attenuates gradients, sparse masks zero most)
    # These params have heterogeneous gradient magnitudes incompatible with Muon's orthogonalization
    fsq_params = [p for _, p in param_groups['fsq_adjacent']]
    if fsq_params:
        adamw_lr = adamw_cfg['lr']
        fsq_opt = torch.optim.AdamW(
            fsq_params,
            lr=adamw_lr,
            weight_decay=adamw_cfg['weight_decay'],
            betas=tuple(adamw_cfg['betas']),
            fused=True,
        )
        fsq_sched = _build_scheduler(fsq_opt, opt_cfg, total_steps, max_lr_override=adamw_lr)
        specs.append(OptimizerSpec(
            name='adamw_fsq',
            optimizer=fsq_opt,
            scheduler=fsq_sched,
            param_names=[n for n, _ in param_groups['fsq_adjacent']],
        ))

    # AdamW for AE components (encoders, decoders, sparse_ae)
    # The AE has FSQ with sigmoid STE and level-global sparsity, creating heterogeneous
    # gradient magnitudes. Use AdamW for all AE params, only use Muon for denoiser.
    ae_params = [p for _, p in param_groups['ae']]
    if ae_params:
        adamw_lr = adamw_cfg['lr']
        ae_opt = torch.optim.AdamW(
            ae_params,
            lr=adamw_lr,
            weight_decay=adamw_cfg['weight_decay'],
            betas=tuple(adamw_cfg['betas']),
            fused=True,
        )
        ae_sched = _build_scheduler(ae_opt, opt_cfg, total_steps, max_lr_override=adamw_lr)
        specs.append(OptimizerSpec(
            name='adamw_ae',
            optimizer=ae_opt,
            scheduler=ae_sched,
            param_names=[n for n, _ in param_groups['ae']],
        ))

    return OptimizerGroup(specs)


def _build_scheduler(
    optimizer: Optimizer,
    opt_cfg: Dict[str, Any],
    total_steps: int,
    max_lr_override: Optional[float] = None,
) -> Optional[LRScheduler]:
    """Build learning rate scheduler from config.

    Args:
        optimizer: The optimizer to schedule
        opt_cfg: Full optimizer config dict
        total_steps: Total training steps
        max_lr_override: If provided, use this instead of opt_cfg['max_lr']
                        (used for heterogeneous optimizer where each has its own lr)

    scheduler subconfig guaranteed present by Pydantic schema.
    """
    sched_cfg = opt_cfg['scheduler']
    max_lr = max_lr_override if max_lr_override is not None else opt_cfg['max_lr']

    # scheduler can be None if not specified in TOML (uses default from schema)
    if sched_cfg is None:
        # Fallback to top-level pct_start/div_factor for backwards compat
        from torch.optim.lr_scheduler import OneCycleLR
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=opt_cfg['pct_start'],
            div_factor=opt_cfg['div_factor'],
            final_div_factor=opt_cfg['final_div_factor'],
        )

    sched_type = sched_cfg['type']

    if sched_type == 'none':
        return None
    elif sched_type == 'onecycle':
        from torch.optim.lr_scheduler import OneCycleLR
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=sched_cfg['pct_start'],
            div_factor=sched_cfg['div_factor'],
            final_div_factor=sched_cfg['final_div_factor'],
        )
    elif sched_type == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=sched_cfg['min_lr'],
        )
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")


# =============================================================================
# FP8 Model Preparation
# =============================================================================

def prepare_model_for_training(
    model: nn.Module,
    config: Dict[str, Any],
    device=None,
) -> nn.Module:
    """
    Prepare model for training with optional FP8 conversion.

    Config structure (defined in config.py PrecisionConfig):
        training.precision_config:
            weights: "fp8"  # or "bf16" (default)
            activations: "bf16"  # always bf16 for now
            skip_patterns: ["embed", "head", "norm"]  # layers to keep in bf16

    All fields guaranteed present by Pydantic schema.
    """
    prec_cfg = config['training']['precision_config']
    weight_dtype = prec_cfg['weights']

    if weight_dtype == 'fp8':
        skip_patterns = prec_cfg['skip_patterns']
        print(f"[FP8] Converting model weights to FP8 (HAS_FP8={HAS_FP8})")
        model = convert_to_fp8(model, skip_patterns=skip_patterns, device=device)

        # Count converted layers
        n_fp8 = sum(1 for m in model.modules() if isinstance(m, FP8Linear))
        n_linear = sum(1 for m in model.modules() if isinstance(m, (nn.Linear, FP8Linear)))
        print(f"[FP8] Converted {n_fp8}/{n_linear} Linear layers to FP8")

    return model


def count_fp8_layers(model: nn.Module) -> Tuple[int, int]:
    """Count FP8 vs regular Linear layers."""
    n_fp8 = sum(1 for m in model.modules() if isinstance(m, FP8Linear))
    n_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    return n_fp8, n_linear


def get_linear_weight_dtype(module: nn.Module) -> torch.dtype:
    """
    Get the weight dtype from a Linear-like module.

    Works with:
    - nn.Linear: has .weight attribute
    - FP8Linear: has .weight attribute (bf16 Parameter, always trainable)

    Both nn.Linear and FP8Linear now always have .weight as a Parameter.
    """
    if hasattr(module, 'weight') and module.weight is not None:
        return module.weight.dtype
    else:
        # Fallback: try to find any parameter
        for p in module.parameters():
            return p.dtype
        return torch.bfloat16  # Ultimate fallback


# =============================================================================
# Role-Based Optimizer Building
# =============================================================================

def build_optimizer_for_role(
    model: nn.Module,
    role: str,
    config: Dict[str, Any],
    total_steps: int,
) -> OptimizerGroup:
    """
    Build optimizer group based on model role.

    This provides unified, config-driven optimizer targeting for different
    model types in the training pipeline.

    Roles:
    - "ae": Sparse AE - uses simple AdamW (no Muon)
        - FSQ gradients are heterogeneous (sigmoid STE attenuates)
        - Hierarchical residual scaling creates different gradient magnitudes
        - Keep it simple until joint system works

    - "denoiser": Code denoiser transformer - uses heterogeneous Muon+AdamW
        - Uniform transformer stack (Muon's target architecture)
        - Only the code-denoising layers, not the full main model

    - "main": Main diffusion model - uses config-specified optimizer
        - Falls back to existing build_optimizer_group behavior

    Args:
        model: The model to create optimizer for
        role: One of "ae", "denoiser", "main"
        config: Full training config dict
        total_steps: Total training steps for scheduler

    Returns:
        OptimizerGroup with appropriate optimizer setup
    """
    if role == "ae":
        return _build_ae_optimizer(model, config, total_steps)
    elif role == "denoiser":
        return _build_denoiser_optimizer(model, config, total_steps)
    elif role == "main":
        return build_optimizer_group(model, config, total_steps)
    else:
        raise ValueError(f"Unknown model role: {role}. Expected 'ae', 'denoiser', or 'main'")


def _build_ae_optimizer(
    model: nn.Module,
    config: Dict[str, Any],
    total_steps: int,
) -> OptimizerGroup:
    """
    Build simple AdamW optimizer for sparse AE.

    The AE has FSQ quantization with sigmoid STE and level-global sparsity,
    both of which create heterogeneous gradient magnitudes incompatible with
    Muon's orthogonalization assumptions. Use simple AdamW until the joint
    system is working.

    Uses training.ae_optimizer config section.
    """
    ae_cfg = config['training']['ae_optimizer']

    # Simple AdamW for all AE params
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ae_cfg['lr'],
        weight_decay=ae_cfg['weight_decay'],
        fused=True,
    )

    # OneCycle scheduler
    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(
        optimizer,
        max_lr=ae_cfg['max_lr'],
        total_steps=total_steps,
        pct_start=ae_cfg['pct_start'],
    )

    param_names = [n for n, _ in model.named_parameters()]

    return OptimizerGroup([
        OptimizerSpec(
            name='adamw_ae',
            optimizer=optimizer,
            scheduler=scheduler,
            param_names=param_names,
        )
    ])


def _build_denoiser_optimizer(
    model: nn.Module,
    config: Dict[str, Any],
    total_steps: int,
) -> OptimizerGroup:
    """
    Build heterogeneous Muon+AdamW for denoiser transformer.

    The denoiser is a uniform transformer stack (GPT-2 style) which is
    exactly what Muon was designed for. Use Muon for transformer layers,
    AdamW for embeddings/norms.

    Uses training.optimizer config section (muon and adamw subconfigs).
    """
    opt_cfg = config['training']['optimizer']

    # If config says adamw-only or doesn't have muon config, fall back
    if opt_cfg['type'] != 'heterogeneous' or opt_cfg.get('muon') is None:
        return _build_single_adamw(model, opt_cfg, total_steps)

    # Classify parameters for denoiser
    # Denoiser is simpler than main model - no FSQ, just transformer + embeddings
    param_groups = classify_parameters(
        model,
        # Denoiser has simpler patterns - just embeddings and norms
        embedding_patterns=['embed', 'proj', 'input_proj', 'output_proj'],
        norm_patterns=['norm', 'ln', 'layernorm', 'rmsnorm'],
        fsq_patterns=[],  # No FSQ in denoiser
    )

    specs = []
    muon_cfg = opt_cfg['muon']
    adamw_cfg = opt_cfg['adamw']

    # Muon for transformer layers (the bulk of the denoiser)
    transformer_params = [p for _, p in param_groups['transformer']]
    if transformer_params:
        state_dtype = _parse_dtype(muon_cfg['state_dtype'])
        muon_lr = muon_cfg['lr']
        muon_opt = Muon(
            transformer_params,
            lr=muon_lr,
            momentum=muon_cfg['momentum'],
            nesterov=muon_cfg['nesterov'],
            ns_steps=muon_cfg['ns_steps'],
            state_dtype=state_dtype,
        )
        muon_sched = _build_scheduler(muon_opt, opt_cfg, total_steps, max_lr_override=muon_lr)
        specs.append(OptimizerSpec(
            name='muon_denoiser',
            optimizer=muon_opt,
            scheduler=muon_sched,
            param_names=[n for n, _ in param_groups['transformer']],
        ))

    # AdamW for embedding/input/output projections
    embedding_params = [p for _, p in param_groups['embedding']]
    if embedding_params:
        adamw_lr = adamw_cfg['lr']
        adamw_opt = torch.optim.AdamW(
            embedding_params,
            lr=adamw_lr,
            weight_decay=adamw_cfg['weight_decay'],
            betas=tuple(adamw_cfg['betas']),
            fused=True,
        )
        adamw_sched = _build_scheduler(adamw_opt, opt_cfg, total_steps, max_lr_override=adamw_lr)
        specs.append(OptimizerSpec(
            name='adamw_denoiser_embed',
            optimizer=adamw_opt,
            scheduler=adamw_sched,
            param_names=[n for n, _ in param_groups['embedding']],
        ))

    # AdamW for norms (no weight decay)
    norm_params = [p for _, p in param_groups['norm']]
    if norm_params:
        norm_opt = torch.optim.AdamW(
            norm_params,
            lr=adamw_cfg['lr'],
            weight_decay=0.0,
            betas=tuple(adamw_cfg['betas']),
            fused=True,
        )
        norm_sched = _build_scheduler(norm_opt, opt_cfg, total_steps, max_lr_override=adamw_cfg['lr'])
        specs.append(OptimizerSpec(
            name='adamw_denoiser_norm',
            optimizer=norm_opt,
            scheduler=norm_sched,
            param_names=[n for n, _ in param_groups['norm']],
        ))

    if not specs:
        # Fallback if no params matched (shouldn't happen)
        return _build_single_adamw(model, opt_cfg, total_steps)

    return OptimizerGroup(specs)


def print_role_optimizer_summary(
    optimizer_group: OptimizerGroup,
    model: nn.Module,
    role: str,
):
    """Print summary of role-based optimizer configuration."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'=' * 60}")
    print(f"Optimizer Configuration [{role.upper()}]")
    print(f"{'=' * 60}")

    for name, spec in optimizer_group.specs.items():
        opt = spec.optimizer
        opt_type = type(opt).__name__
        lr = opt.param_groups[0]['lr']
        n_params = sum(p.numel() for g in opt.param_groups for p in g['params'])
        pct = 100 * n_params / total_params if total_params > 0 else 0

        print(f"\n{name}:")
        print(f"  Type: {opt_type}")
        print(f"  LR: {lr:.2e}")
        print(f"  Params: {n_params:,} ({pct:.1f}%)")

        if isinstance(opt, Muon):
            state_dtype = opt.defaults.get('state_dtype', torch.bfloat16)
            dtype_name = {torch.float32: 'fp32', torch.bfloat16: 'bf16', torch.float16: 'fp16'}.get(state_dtype, str(state_dtype))
            print(f"  State dtype: {dtype_name}")

    print(f"\n{'=' * 60}")


# =============================================================================
# Convenience Functions
# =============================================================================

def get_param_count_by_group(model: nn.Module) -> Dict[str, int]:
    """Get parameter count per group for logging."""
    groups = classify_parameters(model)
    return {
        name: sum(p.numel() for _, p in params)
        for name, params in groups.items()
    }


def print_optimizer_summary(optimizer_group: OptimizerGroup, model: nn.Module):
    """Print summary of optimizer configuration."""
    param_counts = get_param_count_by_group(model)
    total = sum(param_counts.values())

    # Check FP8 status
    n_fp8, n_linear = count_fp8_layers(model)

    print("\n" + "=" * 60)
    print("Optimizer Configuration")
    print("=" * 60)

    if n_fp8 > 0:
        print(f"\nFP8 Status: {n_fp8} FP8Linear, {n_linear} Linear (native={HAS_FP8})")

    for name, spec in optimizer_group.specs.items():
        opt = spec.optimizer
        opt_type = type(opt).__name__
        lr = opt.param_groups[0]['lr']
        n_params = sum(p.numel() for g in opt.param_groups for p in g['params'])
        pct = 100 * n_params / total if total > 0 else 0

        print(f"\n{name}:")
        print(f"  Type: {opt_type}")
        print(f"  LR: {lr:.2e}")
        print(f"  Params: {n_params:,} ({pct:.1f}%)")

        # Show state dtype for Muon
        if isinstance(opt, Muon):
            state_dtype = opt.defaults.get('state_dtype', torch.bfloat16)
            dtype_name = {torch.float32: 'fp32', torch.bfloat16: 'bf16', torch.float16: 'fp16'}.get(state_dtype, str(state_dtype))
            print(f"  State dtype: {dtype_name} (momentum only, no v accumulator)")
        elif 'Adam' in opt_type:
            print(f"  State dtype: fp32 (m + v accumulators)")

        if hasattr(opt, 'defaults'):
            for k, v in opt.defaults.items():
                if k not in ('lr', 'state_dtype'):
                    print(f"  {k}: {v}")

    print("\n" + "=" * 60)
