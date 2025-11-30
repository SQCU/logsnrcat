"""
diffusion_oracle_lab.py

Experiment A: The Oracle Probe.
Measuring Error Amplification in Diffusion Parameterizations.

Hypothesis: 
1. Naive models will show exploding Score Error at t=0 and exploding x0 Error at t=1.
2. Factorized models might stabilize these explosions by coupling the error 
   magnitudes to the predicted lambda (Gain Control).
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

# ============================================================================
# 1. Physics & Math
# ============================================================================

def get_logsnr_schedule(t: torch.Tensor, min_logsnr=-20.0, max_logsnr=20.0) -> torch.Tensor:
    return max_logsnr - (max_logsnr - min_logsnr) * t

def logsnr_to_alpha_sigma(logsnr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    alpha = torch.sqrt(torch.sigmoid(logsnr))
    sigma = torch.sqrt(torch.sigmoid(-logsnr))
    return alpha, sigma

def forward_diffusion(x0: torch.Tensor, logsnr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    alpha, sigma = logsnr_to_alpha_sigma(logsnr)
    eps = torch.randn_like(x0)
    
    while alpha.ndim < x0.ndim:
        alpha = alpha.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)
        
    z_t = alpha * x0 + sigma * eps
    v_true = alpha * eps - sigma * x0
    return z_t, eps, v_true

# ============================================================================
# 2. The Oracles (Hypothetical Networks)
# ============================================================================

def get_oracle_prediction(
    mode: str, 
    target_type: str, 
    targets: Dict[str, torch.Tensor],
    logsnr_true: torch.Tensor,
    noise_std: float = 0.01
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Simulates a network that is 99% accurate (1% error).
    """
    # 1. Determine True Target based on configuration
    if mode == "naive":
        y_true = targets[target_type]
        lambda_true = None
        
    elif mode == "factorized":
        # Factorized targets are normalized
        alpha, sigma = logsnr_to_alpha_sigma(logsnr_true)
        while alpha.ndim < targets['x0'].ndim:
            alpha = alpha.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
            
        if target_type == "epsilon":
            # Target: eps / sigma
            y_true = targets['epsilon'] / (sigma + 1e-6) # Avoid div0 in generation
        elif target_type == "x0":
            # Target: x0 * alpha
            y_true = targets['x0'] * alpha
        elif target_type == "v":
            # Target: v / sigma (as per spec)
            y_true = targets['v'] / (sigma + 1e-6)
            
        lambda_true = logsnr_true
    
    # 2. Add Gaussian Noise (Simulate Model Error)
    y_pred = y_true + torch.randn_like(y_true) * noise_std
    
    lambda_pred = None
    if mode == "factorized":
        # Factorized model also predicts lambda with some error
        lambda_pred = lambda_true + torch.randn_like(lambda_true) * noise_std
        
    return y_pred, lambda_pred

# ============================================================================
# 3. Reconstruction & Error Calculation
# ============================================================================

def reconstruct_physical(
    mode: str,
    target_type: str,
    y_pred: torch.Tensor,
    lambda_pred: Optional[torch.Tensor],
    logsnr_true: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Convert model output back to x0 and epsilon to measure physical error.
    """
    # Initialize returns to ensure scope safety
    x0_recon = None
    eps_recon = None
    v_recon = None

    # Get schedule for reconstruction
    # For Naive: Use ground truth schedule (model doesn't predict it)
    # For Factorized: Use predicted schedule (model controls its own gain)
    if mode == "naive":
        l_used = logsnr_true
    else:
        l_used = lambda_pred
        
    alpha, sigma = logsnr_to_alpha_sigma(l_used)
    while alpha.ndim < y_pred.ndim:
        alpha = alpha.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)

    # 1. Recover x0, eps, or v based on target type
    if target_type == "x0":
        if mode == "naive":
            x0_recon = y_pred
        else:
            # Factorized x0: y = x0 * alpha
            # x0 = y / alpha
            x0_recon = y_pred / (alpha + 1e-6)
            
    elif target_type == "epsilon":
        if mode == "naive":
            eps_recon = y_pred
        else:
            # Factorized eps: y = eps / sigma
            # eps = y * sigma
            eps_recon = y_pred * sigma

    elif target_type == "v":
        # v = alpha * eps - sigma * x0
        if mode == "naive":
            v_recon = y_pred
        else:
            # Factorized v: v = y * sigma
            v_recon = y_pred * sigma
        
    return {"x0": x0_recon, "eps": eps_recon, "v": v_recon}

# ============================================================================
# 4. The Probe
# ============================================================================

@dataclass
class OracleStats:
    mode: str
    target: str
    timestep: float
    raw_mse: float
    x0_mse: float     # Trajectory Error
    score_mse: float  # Score Error (proxy for eps/sigma)
    error_kurtosis: float

def compute_kurtosis(tensor: torch.Tensor) -> float:
    x = tensor.detach().flatten().float()
    std = x.std()
    if std < 1e-9: return 0.0
    return (((x - x.mean()) / std) ** 4).mean().item()

def run_oracle_probe(timesteps: np.ndarray, batch_size=1000) -> pd.DataFrame:
    stats_list = []
    
    # Fixed clean data
    x0 = torch.randn(batch_size, 4)
    
    for t_val in timesteps:
        t = torch.full((batch_size,), t_val)
        logsnr = get_logsnr_schedule(t)
        z_t, eps_true, v_true = forward_diffusion(x0, logsnr)
        
        targets_true = {"x0": x0, "epsilon": eps_true, "v": v_true}
        
        # Test Matrix: Modes x Targets
        for target_type in ["x0", "epsilon", "v"]:
            for mode in ["naive", "factorized"]:
                
                # 1. Oracle Prediction
                y_pred, lambda_pred = get_oracle_prediction(
                    mode, target_type, targets_true, logsnr, noise_std=0.01
                )
                
                # 2. Raw Error (Sanity Check)
                # Recalculate 'true' used for generation to verify 1% noise
                y_gen_true, _ = get_oracle_prediction(mode, target_type, targets_true, logsnr, 0.0)
                raw_mse = nn.functional.mse_loss(y_pred, y_gen_true).item()
                
                # 3. Reconstruction to Physical Space
                recon = reconstruct_physical(mode, target_type, y_pred, lambda_pred, logsnr)
                
                # 4. Compute Derived Errors
                
                # Error A: Implied x0 Error (Trajectory Error)
                
                alpha, sigma = logsnr_to_alpha_sigma(logsnr)
                while alpha.ndim < x0.ndim: alpha, sigma = alpha.unsqueeze(-1), sigma.unsqueeze(-1)

                if recon['x0'] is not None:
                    err_x0 = recon['x0'] - x0
                elif recon['eps'] is not None:
                    # Derived x0 error from eps error
                    # x0 = (z - sigma*eps)/alpha
                    # dx0 = -sigma/alpha * deps
                    err_eps = recon['eps'] - eps_true
                    err_x0 = - (sigma / alpha) * err_eps
                elif recon['v'] is not None:
                    # v = alpha*eps - sigma*x0
                    # z = alpha*x0 + sigma*eps
                    # solve for x0: x0 = alpha*z - sigma*v
                    # dx0 = -sigma * dv
                    err_v = recon['v'] - v_true
                    err_x0 = -sigma * err_v
                
                x0_mse = (err_x0 ** 2).mean().item()
                
                # Error B: Implied Score Error (Gradient Error)
                # Score ~ -eps/sigma
                # We want error in (eps/sigma)
                
                if recon['eps'] is not None:
                    err_eps = recon['eps'] - eps_true
                    # Score Error = err_eps / sigma
                    score_mse = ((err_eps / sigma) ** 2).mean().item()
                elif recon['x0'] is not None:
                    # eps = (z - alpha*x0)/sigma
                    # deps = -alpha/sigma * dx0
                    # Score Error = deps/sigma = -alpha/sigma^2 * dx0
                    err_x0 = recon['x0'] - x0
                    score_mse = ((-alpha / (sigma**2)) * err_x0).pow(2).mean().item()
                elif recon['v'] is not None:
                    # Score Error = deps/sigma = alpha/sigma * dv
                    err_v = recon['v'] - v_true
                    score_mse = ((alpha / sigma) * err_v).pow(2).mean().item()
                
                # Kurtosis of the Physical Error (x0 space)
                kurt = compute_kurtosis(err_x0)
                
                stats_list.append(OracleStats(
                    mode=mode,
                    target=target_type,
                    timestep=t_val,
                    raw_mse=raw_mse,
                    x0_mse=x0_mse,
                    score_mse=score_mse,
                    error_kurtosis=kurt
                ))
                
    return pd.DataFrame([vars(s) for s in stats_list])

# ============================================================================
# 5. Visualization
# ============================================================================

def plot_oracle_results(df):
    targets = ["x0", "epsilon", "v"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 15), sharex=True)
    
    for i, target in enumerate(targets):
        subset = df[df['target'] == target]
        naive = subset[subset['mode'] == 'naive']
        fact = subset[subset['mode'] == 'factorized']
        
        # Plot 1: Trajectory Error (x0 MSE)
        ax_x0 = axes[i, 0]
        ax_x0.plot(naive['timestep'], naive['x0_mse'], 'r--', label='Naive', linewidth=2)
        ax_x0.plot(fact['timestep'], fact['x0_mse'], 'b-', label='Factorized', linewidth=2)
        ax_x0.set_title(f'{target.upper()} Oracle: Trajectory Error (x0 MSE)')
        ax_x0.set_yscale('log')
        ax_x0.grid(True, alpha=0.3)
        ax_x0.legend()
        
        # Plot 2: Score Error (Gradient MSE)
        ax_score = axes[i, 1]
        ax_score.plot(naive['timestep'], naive['score_mse'], 'r--', label='Naive', linewidth=2)
        ax_score.plot(fact['timestep'], fact['score_mse'], 'b-', label='Factorized', linewidth=2)
        ax_score.set_title(f'{target.upper()} Oracle: Score Error (eps/sigma MSE)')
        ax_score.set_yscale('log')
        ax_score.grid(True, alpha=0.3)
        
        # Mark Danger Zones
        if target == 'x0':
            ax_x0.axvline(0.99, color='k', alpha=0.2)
        elif target == 'epsilon':
            ax_score.axvline(0.01, color='k', alpha=0.2)

    axes[2,0].set_xlabel('Timestep (0=Clean, 1=Noisy)')
    axes[2,1].set_xlabel('Timestep (0=Clean, 1=Noisy)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Sweep
    t_vals = np.concatenate([
        np.logspace(-4, -1.3, 20),           # t -> 0
        np.linspace(0.06, 0.94, 20),         # Mid
        1.0 - np.logspace(-4, -1.3, 20)      # t -> 1
    ])
    t_vals.sort()
    
    print("Running Oracle Probe (Error Amplification Test)...")
    df = run_oracle_probe(t_vals)
    
    print("\nVisualizing Error Propagation...")
    plot_oracle_results(df)
    
    # Print Key Stats
    print("\n=== Error Amplification Factors ===")
    for target in ["x0", "epsilon", "v"]:
        sub = df[df['target'] == target]
        # Check t=0.01 (Clean) Score Error
        s_naive_0 = sub[(sub['mode']=='naive') & (sub['timestep']<0.02)]['score_mse'].mean()
        s_fact_0 = sub[(sub['mode']=='factorized') & (sub['timestep']<0.02)]['score_mse'].mean()
        
        # Check t=0.99 (Noisy) x0 Error
        x_naive_1 = sub[(sub['mode']=='naive') & (sub['timestep']>0.98)]['x0_mse'].mean()
        x_fact_1 = sub[(sub['mode']=='factorized') & (sub['timestep']>0.98)]['x0_mse'].mean()
        
        print(f"\nTarget: {target.upper()}")
        print(f"  t->0 Score Error: Naive={s_naive_0:.2e} | Fact={s_fact_0:.2e}")
        print(f"  t->1 x0 Error:    Naive={x_naive_1:.2e} | Fact={x_fact_1:.2e}")