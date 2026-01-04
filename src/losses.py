# src/losses.py - Stateless loss functions for AE and denoising training
"""
Decoupled loss computation for:
- AE reconstruction (sparse_dim, swiglu, contextual patch embed)
- Denoising (pixel-space velocity, latent-space velocity)

All functions are stateless: (output, target, config) -> (loss, stats)
Same functions work in train_autoembed, train_denoise, or joint training.
"""

from typing import Dict, Any, Tuple, List, Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# AE Loss Functions
# =============================================================================

def cumulative_mse_loss(
    output: Dict[str, Any],
    target: torch.Tensor,
    **kwargs
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Average MSE across all level reconstructions.

    Matches reference implementation - trains each level to produce
    a good reconstruction at that stage, not just the final output.

    Args:
        output: Dict with 'level_recons' (list of [B, C, H, W])
        target: [B, C, H, W] original images

    Returns:
        loss: scalar tensor
        stats: dict with per-level losses
    """
    level_recons = output['level_recons']
    losses = [F.mse_loss(recon, target) for recon in level_recons]
    loss = sum(losses) / len(losses)

    stats = {
        'recon_loss': loss.detach(),
        'per_level': [l.detach() for l in losses],
        'sparsity': output['sparsity'].detach()
    }
    return loss, stats


def final_mse_loss(
    output: Dict[str, Any],
    target: torch.Tensor,
    **kwargs
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    MSE only on final reconstruction.

    Args:
        output: Dict with 'recon' ([B, C, H, W])
        target: [B, C, H, W] original images

    Returns:
        loss: scalar tensor
        stats: dict
    """
    loss = F.mse_loss(output['recon'], target)

    stats = {
        'recon_loss': loss.detach(),
        'sparsity': output['sparsity'].detach()
    }
    return loss, stats


def cumulative_mse_with_contribution(
    output: Dict[str, Any],
    target: torch.Tensor,
    contrib_weight: float = 0.1,
    **kwargs
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    MSE + penalty for levels that contribute too little.
    Encourages each level to meaningfully change the reconstruction.

    Args:
        output: Dict with 'level_recons' (list of [B, C, H, W])
        target: [B, C, H, W] original images
        contrib_weight: Weight for contribution penalty

    Returns:
        loss: scalar tensor
        stats: dict
    """
    level_recons = output['level_recons']
    n_levels = len(level_recons)

    # Base MSE
    mse_losses = [F.mse_loss(recon, target) for recon in level_recons]
    mse_loss = sum(mse_losses) / n_levels

    # Contribution: measure how much each level changes reconstruction
    contrib_losses = []
    for i in range(n_levels):
        if i == 0:
            delta = level_recons[0]
        else:
            delta = level_recons[i] - level_recons[i-1]

        # Penalize small contributions (negative because we want to maximize)
        contrib = delta.abs().mean()
        contrib_losses.append(-contrib)

    contrib_loss = sum(contrib_losses) / n_levels
    loss = mse_loss + contrib_weight * contrib_loss

    stats = {
        'mse_loss': mse_loss.detach(),
        'contrib': (-contrib_loss).detach(),  # Flip sign for logging
        'recon_loss': loss.detach(),
        'sparsity': output['sparsity'].detach()
    }
    return loss, stats


# =============================================================================
# K-Annealing Schedule (for sparsity curriculum)
# =============================================================================

def get_k_for_step(
    step: int,
    k_start: int,
    k_end: int,
    anneal_steps: int
) -> int:
    """
    Compute current k for sparsity annealing.

    Exponential decay from k_start to k_end over anneal_steps.
    Curriculum: start with more active dims (easier task), progressively constrain.

    From reference: k = k_start * ((k_end / k_start) ** t)

    Args:
        step: Current training step
        k_start: Starting k (more active dims)
        k_end: Final k (target sparsity)
        anneal_steps: Steps over which to anneal

    Returns:
        Current k value (integer, clamped to k_end minimum)
    """
    if k_start is None or k_start <= k_end:
        return k_end

    t = min(step / max(anneal_steps, 1), 1.0)
    # Exponential interpolation: k_start * (k_end/k_start)^t
    k = k_start * ((k_end / k_start) ** t)
    return max(int(round(k)), k_end)


# =============================================================================
# Scheduled MSE+BCE Loss
# =============================================================================

def _compute_schedule_weights(
    step: int,
    total_steps: int,
    schedule_cfg: Dict[str, Any]
) -> Tuple[float, float]:
    """
    Compute MSE and BCE weights from schedule config.

    Extracted helper to avoid duplication between final and cumulative variants.
    """
    mse_start = schedule_cfg.get('mse_start', 1.0)
    mse_end = schedule_cfg.get('mse_end', 0.1)
    bce_start = schedule_cfg.get('bce_start', 0.0)
    bce_end = schedule_cfg.get('bce_end', 0.9)
    schedule_type = schedule_cfg.get('schedule', 'linear')
    pct_switch = schedule_cfg.get('pct_switch', 0.8)

    # Compute progress [0, 1]
    progress = min(step / max(total_steps, 1), 1.0)

    # Compute lerp factor based on schedule type
    if schedule_type == 'linear':
        t = progress
    elif schedule_type == 'cosine':
        import math
        t = 0.5 * (1 - math.cos(math.pi * progress))
    elif schedule_type == 'step':
        t = 1.0 if progress >= pct_switch else 0.0
    else:
        t = progress  # fallback to linear

    # Lerp weights
    mse_weight = mse_start + t * (mse_end - mse_start)
    bce_weight = bce_start + t * (bce_end - bce_start)

    return mse_weight, bce_weight


def scheduled_mse_bce_loss(
    output: Dict[str, Any],
    target: torch.Tensor,
    step: int = 0,
    total_steps: int = 1,
    schedule_cfg: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Scheduled MSE+BCE loss that lerps from pure MSE to mostly BCE over training.

    FINAL-ONLY variant: applies loss only to output['recon'].

    Early training: MSE provides smooth gradients for coarse structure.
    Late training: BCE pushes for sharp, committed predictions.

    Args:
        output: Dict with 'recon' ([B, C, H, W]) in [0, 1]
        target: [B, C, H, W] in [0, 1]
        step: Current training step
        total_steps: Total training steps
        schedule_cfg: Dict with mse_start, mse_end, bce_start, bce_end, schedule type
    """
    if schedule_cfg is None:
        schedule_cfg = {}

    mse_weight, bce_weight = _compute_schedule_weights(step, total_steps, schedule_cfg)

    recon = output['recon']

    # MSE loss
    mse_loss = F.mse_loss(recon, target)

    # BCE loss - must compute outside autocast (not autocast-safe)
    # Clamp to avoid log(0)
    with torch.amp.autocast(device_type='cuda', enabled=False):
        recon_clamped = recon.float().clamp(1e-7, 1 - 1e-7)
        target_float = target.float()
        bce_loss = F.binary_cross_entropy(recon_clamped, target_float)

    # Combined loss
    loss = mse_weight * mse_loss + bce_weight * bce_loss

    stats = {
        'recon_loss': loss.detach(),
        'mse_loss': mse_loss.detach(),
        'bce_loss': bce_loss.detach(),
        'mse_weight': mse_weight,
        'bce_weight': bce_weight,
        'sparsity': output['sparsity'].detach()
    }
    return loss, stats


def cumulative_scheduled_mse_bce_loss(
    output: Dict[str, Any],
    target: torch.Tensor,
    step: int = 0,
    total_steps: int = 1,
    schedule_cfg: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Scheduled MSE+BCE loss applied to each level reconstruction independently.

    CUMULATIVE variant: applies loss to each level_recons[i] vs target, then averages.

    This gives clean gradient signals to each level rather than coupling all levels
    through the final output. BCE's nonlinear gradient (∝ 1/(pred*(1-pred))) creates
    interference when applied only to the final output because all levels contribute
    through the residual chain. By applying BCE per-level, each level gets direct
    feedback on its cumulative reconstruction quality.

    Same schedule semantics as scheduled_mse_bce_loss:
    - Early training: MSE provides smooth gradients for coarse structure
    - Late training: BCE pushes for sharp, committed predictions at ALL levels

    Args:
        output: Dict with 'level_recons' (list of [B, C, H, W]) in [0, 1]
        target: [B, C, H, W] in [0, 1]
        step: Current training step
        total_steps: Total training steps
        schedule_cfg: Dict with mse_start, mse_end, bce_start, bce_end, schedule type
    """
    if schedule_cfg is None:
        schedule_cfg = {}

    mse_weight, bce_weight = _compute_schedule_weights(step, total_steps, schedule_cfg)

    level_recons = output['level_recons']
    n_levels = len(level_recons)

    mse_losses = []
    bce_losses = []

    # Compute loss for each level's cumulative reconstruction
    with torch.amp.autocast(device_type='cuda', enabled=False):
        target_float = target.float()

        for recon in level_recons:
            # MSE for this level
            mse_losses.append(F.mse_loss(recon, target))

            # BCE for this level - clamp to avoid log(0)
            recon_clamped = recon.float().clamp(1e-7, 1 - 1e-7)
            bce_losses.append(F.binary_cross_entropy(recon_clamped, target_float))

    # Average across levels
    avg_mse = sum(mse_losses) / n_levels
    avg_bce = sum(bce_losses) / n_levels

    # Combined loss with schedule weights
    loss = mse_weight * avg_mse + bce_weight * avg_bce

    stats = {
        'recon_loss': loss.detach(),
        'mse_loss': avg_mse.detach(),
        'bce_loss': avg_bce.detach(),
        'mse_weight': mse_weight,
        'bce_weight': bce_weight,
        'per_level_mse': [l.detach() for l in mse_losses],
        'per_level_bce': [l.detach() for l in bce_losses],
        'sparsity': output['sparsity'].detach()
    }
    return loss, stats


# =============================================================================
# Scheduled MSE+BCE Velocity Loss (for diffusion v-field training)
# =============================================================================

def scheduled_mse_bce_velocity_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    step: int = 0,
    total_steps: int = 1,
    schedule_cfg: Optional[Dict[str, Any]] = None,
    variance_tracker=None,
    **kwargs
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Scheduled MSE+BCE loss for v-field prediction.

    Unlike image reconstruction where targets are in [0,1], v-field targets
    are continuous (unbounded). We apply sigmoid to both pred and target
    before BCE, treating the v-field as logits.

    This tests whether BCE gradients on sigmoid(v) find better v-fields than MSE.

    Args:
        v_pred: [B, L, D] or [B, C, H, W] predicted velocity field
        v_target: Same shape as v_pred, target velocity field
        step: Current training step
        total_steps: Total training steps
        schedule_cfg: Dict with mse_start, mse_end, bce_start, bce_end, schedule type
        variance_tracker: Optional for adaptive weighting (applied to MSE component)
    """
    if schedule_cfg is None:
        schedule_cfg = {}

    mse_weight, bce_weight = _compute_schedule_weights(step, total_steps, schedule_cfg)

    # MSE loss (with optional variance tracking)
    sq_err = (v_pred - v_target) ** 2
    weights = None
    if variance_tracker is not None:
        # Need logsnr_map for variance tracker - get from kwargs if provided
        logsnr_map = kwargs.get('logsnr_map')
        if logsnr_map is not None:
            variance_tracker.update(logsnr_map, sq_err)
            weights = variance_tracker.get_weight_map(logsnr_map, sq_err.shape)
            mse_loss = (sq_err * weights).mean()
        else:
            mse_loss = sq_err.mean()
    else:
        mse_loss = sq_err.mean()

    # BCE loss - apply sigmoid to treat v as logits
    # Must compute outside autocast for numerical stability
    with torch.amp.autocast(device_type='cuda', enabled=False):
        v_pred_prob = torch.sigmoid(v_pred.float())
        v_target_prob = torch.sigmoid(v_target.float())
        # Clamp to avoid log(0)
        v_pred_clamped = v_pred_prob.clamp(1e-7, 1 - 1e-7)
        v_target_clamped = v_target_prob.clamp(1e-7, 1 - 1e-7)
        bce_loss = F.binary_cross_entropy(v_pred_clamped, v_target_clamped)

    # Combined loss
    loss = mse_weight * mse_loss + bce_weight * bce_loss

    stats = {
        'loss': loss.detach(),
        'mse_loss': mse_loss.detach(),
        'bce_loss': bce_loss.detach(),
        'mse_weight': mse_weight,
        'bce_weight': bce_weight,
        'loss_unweighted': sq_err.mean().detach(),
    }

    # Add variance tracker stats when active
    if weights is not None:
        with torch.no_grad():
            stats['weight_mean'] = weights.mean().detach()
            stats['weight_min'] = weights.min().detach()
            stats['weight_max'] = weights.max().detach()
            # loss_var = weight range as proxy for correction magnitude
            stats['loss_var'] = (weights.max() - weights.min()).detach()

    return loss, stats


# =============================================================================
# Denoising Loss Functions
# =============================================================================

def pixel_velocity_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    logsnr_map: torch.Tensor,
    variance_tracker=None,
    **kwargs
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Pixel-space velocity prediction loss with optional variance weighting.

    Args:
        v_pred: [B, C, H, W] predicted velocity
        v_target: [B, C, H, W] target velocity
        logsnr_map: [B, 1, H, W] per-pixel logsnr
        variance_tracker: Optional OnlineVarianceTracker for adaptive weighting

    Returns:
        loss: scalar tensor
        stats: dict with loss components
    """
    if variance_tracker is not None:
        # Use the existing compute_online_weighted_mse logic
        sq_err = (v_pred - v_target) ** 2
        variance_tracker.update(logsnr_map, sq_err)
        weights = variance_tracker.get_weight_map(logsnr_map, sq_err.shape)
        weighted_sq_err = sq_err * weights
        loss = weighted_sq_err.mean()
        loss_unweighted = sq_err.mean()

        stats = {
            'loss': loss.detach(),
            'loss_unweighted': loss_unweighted.detach(),
            'weight_mean': weights.mean().detach(),
            'weight_range': (weights.max() - weights.min()).detach(),
        }
    else:
        sq_err = (v_pred - v_target) ** 2
        loss = sq_err.mean()

        stats = {
            'loss': loss.detach(),
            'loss_var': sq_err.var().detach(),
        }

    return loss, stats


def latent_velocity_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    logsnr_map: torch.Tensor,
    variance_tracker=None,
    **kwargs
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Latent-space velocity prediction loss.

    Same interface as pixel_velocity_loss but operates in code space.
    Can have different weighting strategies for latent diffusion.
    """
    # For now, same logic as pixel - can be customized
    return pixel_velocity_loss(v_pred, v_target, logsnr_map, variance_tracker, **kwargs)


def logsnr_prediction_loss(
    pred_logsnr: torch.Tensor,
    target_logsnr: torch.Tensor,
    **kwargs
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Lambda (logsnr) prediction L1 loss.
    """
    loss = F.l1_loss(pred_logsnr, target_logsnr)
    return loss, {'logsnr_loss': loss.detach()}


# =============================================================================
# Registry and Factory
# =============================================================================

AE_LOSS_REGISTRY = {
    'cumulative_mse': cumulative_mse_loss,
    'final_mse': final_mse_loss,
    'cumulative_mse_contrib': cumulative_mse_with_contribution,
}

DENOISE_LOSS_REGISTRY = {
    'pixel_velocity': pixel_velocity_loss,
    'latent_velocity': latent_velocity_loss,
}


def get_ae_loss_fn(config: Dict[str, Any]) -> Callable:
    """
    Get AE loss function from config.

    Looks at config['training']['sparse_ae']['loss_type'] and ['loss_schedule'].

    If loss_schedule.enabled is True, returns a CUMULATIVE scheduled MSE+BCE loss
    that applies both MSE and BCE to each level_recons[i] independently before
    averaging. This gives clean gradient signals to each level rather than
    coupling all levels through the final output.

    NOTE: FSQ autoencoder is a pure image compression network.
    logsnr prediction is the LDTformer denoiser's job, not the AE's.
    """
    sparse_ae_cfg = config['training']['sparse_ae']
    loss_type = sparse_ae_cfg['loss_type']

    # Check if loss schedule is enabled
    loss_schedule_cfg = sparse_ae_cfg['loss_schedule']
    if loss_schedule_cfg['enabled']:
        # Use CUMULATIVE variant: applies MSE+BCE to each level independently
        # This avoids BCE gradient interference between residual levels
        def scheduled_loss_wrapper(output, target, **kwargs):
            # Extract step/total_steps before passing remaining kwargs
            step = kwargs.pop('step', 0)
            total_steps = kwargs.pop('total_steps', 1)
            return cumulative_scheduled_mse_bce_loss(
                output, target,
                step=step,
                total_steps=total_steps,
                schedule_cfg=loss_schedule_cfg,
                **kwargs
            )
        return scheduled_loss_wrapper

    # Fallback to standard loss registry
    if loss_type not in AE_LOSS_REGISTRY:
        raise ValueError(f"Unknown AE loss: {loss_type}. Available: {list(AE_LOSS_REGISTRY.keys())}")

    return AE_LOSS_REGISTRY[loss_type]


def get_denoise_loss_fn(config: Dict[str, Any]) -> Callable:
    """
    Get denoising loss function from config.

    Looks at config['training']['sparse_ae']['topology']['diffusion_space'].
    'pixel' -> pixel_velocity_loss
    'latent' -> latent_velocity_loss
    """
    diffusion_space = config['training']['sparse_ae']['topology']['diffusion_space']

    if diffusion_space == 'latent':
        return DENOISE_LOSS_REGISTRY['latent_velocity']
    else:
        return DENOISE_LOSS_REGISTRY['pixel_velocity']


# =============================================================================
# Batch Preparation Helpers
# =============================================================================

def group_blocks_by_grid(
    blocks: List,
    patch_size: int,
    device: torch.device
) -> Dict[Tuple[int, int], List]:
    """
    Group ContextBlocks by grid_shape for efficient batched processing.

    Args:
        blocks: List of ContextBlock with type='latent'
        patch_size: AE patch size for grid computation
        device: Target device

    Returns:
        Dict mapping grid_shape -> list of (block, img, logsnr)
    """
    groups = {}

    for b in blocks:
        if b.type != 'latent':
            continue

        img = b.content  # [C, H, W]
        lsnr = b.logsnr  # [1, H, W]

        # Compute grid_shape
        grid_shape = (img.shape[1] // patch_size, img.shape[2] // patch_size)

        if grid_shape not in groups:
            groups[grid_shape] = []
        groups[grid_shape].append((b, img, lsnr))

    return groups


def prepare_ae_batch(
    group: List[Tuple],
    ae_model: nn.Module,
    device: torch.device
) -> Dict[str, Any]:
    """
    Prepare a batched AE input from a group of same-grid-shape blocks.

    Args:
        group: List of (block, img, logsnr) tuples
        ae_model: The sparse AE model (for mask building)
        device: Target device

    Returns:
        Dict with:
            'images': [B, C, H, W] batched images
            'logsnr': [B, 1, H, W] batched logsnr maps
            'grid_shape': (GH, GW) tuple
            'masks': (encoder_masks, decoder_masks)
            'sources': list of source tags
    """
    imgs = torch.stack([g[1] for g in group], dim=0)
    logsnrs = torch.stack([g[2] for g in group], dim=0)
    sources = [getattr(g[0], 'source', 'unknown') for g in group]

    # Compute grid_shape from first image
    p = ae_model.patch_size
    grid_shape = (imgs.shape[2] // p, imgs.shape[3] // p)

    # Build masks (cached internally by AE)
    encoder_masks, decoder_masks = ae_model.build_masks(grid_shape, device)

    return {
        'images': imgs,
        'logsnr': logsnrs,
        'grid_shape': grid_shape,
        'masks': (encoder_masks, decoder_masks),
        'sources': sources,
    }


def compute_ae_forward(
    ae_model: nn.Module,
    prepared: Dict[str, Any],
    k_override: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run AE forward pass and return uniform output dict.

    Calls ae_model.forward() directly (NOT through wrapper projections).
    This is the correct path for AE training - no information bottleneck.

    NOTE: FSQ autoencoder is a pure image compression network.
    logsnr_map exists in prepared batch for pipeline compatibility but is ignored.

    Args:
        ae_model: The sparse AE model
        prepared: Dict from prepare_ae_batch()
        k_override: Optional override for k sparsity (for k-annealing during training)

    Returns:
        Dict with uniform keys: 'recon', 'level_recons', 'codes', 'sparsity', ...
    """
    encoder_masks, decoder_masks = prepared['masks']

    output = ae_model(
        prepared['images'],
        encoder_masks=encoder_masks,
        decoder_masks=decoder_masks,
        grid_shape=prepared['grid_shape'],
        k_override=k_override
    )

    return output
