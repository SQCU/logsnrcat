"""Modular loss functions for FSQ autoencoder."""
import torch
import torch.nn.functional as F


def cumulative_mse_loss(output, target):
    """
    Average MSE across all cumulative level reconstructions.

    Args:
        output: dict with 'level_recons' list of [B, C, H, W] tensors
        target: [B, C, H, W] original images

    Returns:
        loss: scalar tensor
        info: dict with per-level losses for logging
    """
    level_recons = output['level_recons']
    losses = [F.mse_loss(recon, target) for recon in level_recons]
    loss = sum(losses) / len(losses)

    info = {
        'recon_loss': loss.item(),
        'per_level': [l.item() for l in losses]
    }
    return loss, info


def final_mse_loss(output, target):
    """
    MSE only on final reconstruction.

    Args:
        output: dict with 'level_recons' list
        target: [B, C, H, W] original images

    Returns:
        loss: scalar tensor
        info: dict for logging
    """
    final_recon = output['level_recons'][-1]
    loss = F.mse_loss(final_recon, target)

    info = {'recon_loss': loss.item()}
    return loss, info


def cumulative_mse_with_contribution(output, target, contrib_weight=0.1):
    """
    MSE + penalty for levels that contribute too little.
    Encourages each level to meaningfully change the reconstruction.
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
            delta = level_recons[0]  # first level's full output
        else:
            delta = level_recons[i] - level_recons[i-1]  # what this level added

        # Penalize small contributions (negative because we want to maximize)
        contrib = delta.abs().mean()
        contrib_losses.append(-contrib)

    contrib_loss = sum(contrib_losses) / n_levels

    loss = mse_loss + contrib_weight * contrib_loss

    info = {
        'mse': mse_loss.item(),
        'contrib': -contrib_loss.item(),  # flip sign for logging (higher = more contribution)
        'total': loss.item()
    }
    return loss, info


# Registry of available losses
LOSS_REGISTRY = {
    'cumulative_mse': cumulative_mse_loss,
    'final_mse': final_mse_loss,
    'cumulative_mse_contrib': cumulative_mse_with_contribution,
}


def get_loss_fn(name):
    """Get loss function by name."""
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name]
