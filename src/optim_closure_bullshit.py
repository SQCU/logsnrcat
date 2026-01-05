# src/optim_closure_bullshit.py
"""
Optimizer closure factory for training loops.

Encapsulates all the dtype-dependent, FP8-aware, scaler-dependent optimizer
boilerplate into clean closures that training loops can call without caring
about the underlying complexity.

Usage:
    ctx = build_training_context(module, config, role="denoiser")

    for step in range(steps):
        ctx.zero_grad()
        with ctx.autocast():
            loss = compute_loss(...)
        ctx.backward(loss)
        ctx.step()

The closures handle:
- FP8 vs standard weights
- bf16 vs fp16 vs fp32 precision
- GradScaler for fp16
- Heterogeneous vs single optimizer
- Scheduler stepping
- All the if/else branching that otherwise pollutes training loops
"""
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from contextlib import contextmanager

import torch
import torch.nn as nn

from .optim_utils import (
    build_optimizer_group, build_optimizer_for_role,
    prepare_model_for_training, collect_fp8_modules,
    print_optimizer_summary, print_role_optimizer_summary,
    FP8Muon, FP8AdamW, OptimizerGroup,
)


@dataclass
class TrainingContext:
    """
    Container for training loop closures.

    All the optimizer/scaler/fp8 complexity is hidden behind these methods.
    Training loops just call zero_grad(), backward(), step() without branching.
    """
    zero_grad: Callable[[], None]
    backward: Callable[[torch.Tensor], None]
    step: Callable[[], None]
    autocast: Callable[[], Any]  # Returns context manager

    # Underlying objects (for introspection/debugging only)
    optimizer_group: OptimizerGroup
    fp8_optimizer: Optional[Any]
    scaler: Optional[torch.amp.GradScaler]
    dtype: torch.dtype
    device: torch.device


def build_training_context(
    module: nn.Module,
    config: Dict[str, Any],
    total_steps: int,
    role: Optional[str] = None,
    device: Optional[torch.device] = None,
    print_summary: bool = True,
) -> TrainingContext:
    """
    Build a complete training context with closures for optimizer operations.

    Args:
        module: The nn.Module to optimize (already on device)
        config: Full experiment config dict
        total_steps: Total training steps (for scheduler)
        role: Optional role for role-based optimizer ("ae", "denoiser", etc.)
              If None, uses full heterogeneous optimizer group.
        device: Device (defaults to config['device'])
        print_summary: Whether to print optimizer summary

    Returns:
        TrainingContext with closures: zero_grad, backward, step, autocast
    """
    device = device or config['device']
    dtype = config['dtype']

    # 1. Apply FP8 conversion if configured
    module = prepare_model_for_training(module, config, device=device)

    # 2. Build FP8 optimizer for FP8Linear modules (if any)
    fp8_modules = collect_fp8_modules(module)
    fp8_optimizer = None

    if fp8_modules:
        fp8_optimizer = _build_fp8_optimizer(fp8_modules, config, role)
        n_fp8_params = sum(m.out_features * m.in_features for m in fp8_modules)
        opt_name = type(fp8_optimizer).__name__
        print(f"[FP8] Created {opt_name} for {len(fp8_modules)} modules ({n_fp8_params:,} params)")

    # 3. Build standard optimizer group
    if role is not None:
        optimizer_group = build_optimizer_for_role(module, role, config, total_steps=total_steps)
        if print_summary:
            print_role_optimizer_summary(optimizer_group, module, role)
    else:
        optimizer_group = build_optimizer_group(module, config, total_steps=total_steps)
        if print_summary:
            print_optimizer_summary(optimizer_group, module)

    # 4. Build GradScaler (only for fp16)
    use_fp16_scaler = (dtype == torch.float16)
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16_scaler)

    # 5. Build closures
    use_amp = dtype in (torch.bfloat16, torch.float16)

    def zero_grad_fn():
        optimizer_group.zero_grad()

    if use_fp16_scaler:
        # FP16 path: use scaler for backward/step
        def backward_fn(loss: torch.Tensor):
            scaler.scale(loss).backward()

        def step_fn():
            for spec in optimizer_group.specs.values():
                scaler.step(spec.optimizer)
            scaler.update()
            optimizer_group.schedule_step()
            # FP8 step after scaler update
            if fp8_optimizer is not None:
                fp8_optimizer.step()
                fp8_optimizer.zero_grad()
    else:
        # BF16/FP32 path: direct backward/step
        def backward_fn(loss: torch.Tensor):
            loss.backward()

        def step_fn():
            optimizer_group.step()
            optimizer_group.schedule_step()
            # FP8 step
            if fp8_optimizer is not None:
                fp8_optimizer.step()
                fp8_optimizer.zero_grad()

    @contextmanager
    def autocast_fn():
        with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
            yield

    return TrainingContext(
        zero_grad=zero_grad_fn,
        backward=backward_fn,
        step=step_fn,
        autocast=autocast_fn,
        optimizer_group=optimizer_group,
        fp8_optimizer=fp8_optimizer,
        scaler=scaler if use_fp16_scaler else None,
        dtype=dtype,
        device=device,
    )


def _build_fp8_optimizer(
    fp8_modules: list,
    config: Dict[str, Any],
    role: Optional[str] = None,
) -> Any:
    """
    Build the appropriate FP8 optimizer based on config.

    For role="ae": Always use FP8AdamW (AE uses simple AdamW)
    For role=None or "denoiser": Match the main optimizer type
    """
    opt_cfg = config['training']['optimizer']

    # AE role always uses AdamW
    if role == "ae":
        ae_opt_cfg = config['training']['ae_optimizer']
        adamw_cfg = opt_cfg['adamw']
        return FP8AdamW(
            fp8_modules,
            lr=ae_opt_cfg['lr'],
            betas=tuple(adamw_cfg['betas']),
            weight_decay=ae_opt_cfg['weight_decay'],
        )

    # Denoiser/full model: match main optimizer type
    opt_type = opt_cfg['type']

    if opt_type == "heterogeneous":
        # Muon for transformer weights in heterogeneous mode
        muon_cfg = opt_cfg['muon']
        return FP8Muon(
            fp8_modules,
            lr=muon_cfg['lr'],
            momentum=muon_cfg['momentum'],
            nesterov=muon_cfg['nesterov'],
            ns_steps=muon_cfg['ns_steps'],
        )
    else:
        # AdamW for all weights when using single optimizer
        adamw_cfg = opt_cfg['adamw']
        return FP8AdamW(
            fp8_modules,
            lr=adamw_cfg['lr'],
            betas=tuple(adamw_cfg['betas']),
            weight_decay=adamw_cfg['weight_decay'],
        )


# Convenience function for modules that need wrapping before optimization
def wrap_for_optimization(
    *submodules: nn.Module,
    name: str = "TrainingModule",
) -> nn.Module:
    """
    Wrap multiple submodules into a single nn.Module for optimization.

    This is useful when you want to jointly optimize components that aren't
    already under a single parent module.

    Usage:
        wrapped = wrap_for_optimization(encoder, decoder, projections)
        ctx = build_training_context(wrapped, config, total_steps=1000)
    """
    class WrappedModule(nn.Module):
        def __init__(self, submodules):
            super().__init__()
            for i, m in enumerate(submodules):
                # Use module's class name if available, else index
                attr_name = getattr(m, '_get_name', lambda: f'module_{i}')()
                attr_name = attr_name.lower().replace(' ', '_')
                setattr(self, f'{attr_name}_{i}', m)

    wrapped = WrappedModule(submodules)
    wrapped.__class__.__name__ = name
    return wrapped
