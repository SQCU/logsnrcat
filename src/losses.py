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

    Looks at config['training']['sparse_ae']['loss_type'].

    NOTE: FSQ autoencoder is a pure image compression network.
    logsnr prediction is the LDTformer denoiser's job, not the AE's.
    """
    loss_type = config['training']['sparse_ae']['loss_type']

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
    prepared: Dict[str, Any]
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

    Returns:
        Dict with uniform keys: 'recon', 'level_recons', 'codes', 'sparsity', ...
    """
    encoder_masks, decoder_masks = prepared['masks']

    output = ae_model(
        prepared['images'],
        encoder_masks=encoder_masks,
        decoder_masks=decoder_masks,
        grid_shape=prepared['grid_shape']
    )

    return output
