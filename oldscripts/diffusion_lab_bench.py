"""
diffusion_lab_bench.py

A "Literate Code" Lab Bench for analyzing the Gradient SNR and Stability 
of Diffusion Parameterizations (Epsilon, X0, Velocity).

Feature Matrix:
1. "Sandwich" Topology: Physics(Input) -> Network -> Physics(Output).
2. Learned Passthrough Lambda: Network must learn identity map for schedule.
3. Factorized Targets: 
   - Epsilon Factorized: target ~ eps / sigma
   - X0 Factorized:      target ~ x0 * alpha
   - V Factorized:       target ~ v / sigma (per user spec)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional

# ============================================================================
# 1. The Physics (Schedule & Transforms)
# ============================================================================

def get_logsnr_schedule(t: torch.Tensor, min_logsnr=-20.0, max_logsnr=20.0) -> torch.Tensor:
    """Linear schedule in LogSNR space."""
    return max_logsnr - (max_logsnr - min_logsnr) * t

def logsnr_to_alpha_sigma(logsnr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """VP Schedule: alpha = sqrt(sigmoid(lambda)), sigma = sqrt(sigmoid(-lambda))"""
    alpha = torch.sqrt(torch.sigmoid(logsnr))
    sigma = torch.sqrt(torch.sigmoid(-logsnr))
    return alpha, sigma

def forward_diffusion(x0: torch.Tensor, logsnr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: z_t, eps, v
    v = alpha * eps - sigma * x0
    """
    alpha, sigma = logsnr_to_alpha_sigma(logsnr)
    eps = torch.randn_like(x0)
    
    # Broadcast
    while alpha.ndim < x0.ndim:
        alpha = alpha.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)
        
    z_t = alpha * x0 + sigma * eps
    v_true = alpha * eps - sigma * x0
    
    return z_t, eps, v_true

# ============================================================================
# 2. The Network (MicroMLP with Passthrough Lambda)
# ============================================================================

class PassthroughMLP(nn.Module):
    """
    Input:  Concat[Data, Lambda]
    Output: Concat[Data_Pred, Lambda_Pred]
    Constraint: No residual connection on Lambda.
    """
    def __init__(self, data_dim: int, hidden_dim: int = 64):
        super().__init__()
        input_dim = data_dim + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Fixed Initialization
        torch.manual_seed(42)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z_t: torch.Tensor, logsnr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        l_in = logsnr.view(-1, 1)
        x_in = torch.cat([z_t, l_in], dim=1)
        out = self.net(x_in)
        return out[:, :-1], out[:, -1]

# ============================================================================
# 3. The Reconstruction (Factorized Decoders)
# ============================================================================

def reconstruct_epsilon(h_data: torch.Tensor, h_lambda: torch.Tensor) -> torch.Tensor:
    """
    Factorized Epsilon: h_data represents (eps / sigma).
    Reconstruction: eps = h_data * sigma(lambda)
    Singularity: t->0 (sigma->0). Target explodes.
    """
    sigma_pred = torch.sqrt(torch.sigmoid(-h_lambda))
    while sigma_pred.ndim < h_data.ndim: sigma_pred = sigma_pred.unsqueeze(-1)
    return h_data * sigma_pred

def reconstruct_x0(h_data: torch.Tensor, h_lambda: torch.Tensor) -> torch.Tensor:
    """
    Factorized X0: h_data represents (x0 * alpha).
    Reconstruction: x0 = h_data / alpha(lambda)
    Singularity: t->1 (alpha->0). Reconstruction divides by zero.
    """
    alpha_pred = torch.sqrt(torch.sigmoid(h_lambda))
    while alpha_pred.ndim < h_data.ndim: alpha_pred = alpha_pred.unsqueeze(-1)
    return h_data / (alpha_pred + 1e-6)

def reconstruct_v(h_data: torch.Tensor, h_lambda: torch.Tensor) -> torch.Tensor:
    """
    Factorized V (User Spec): h_data represents (v / sigma).
    Reconstruction: v = h_data * sigma(lambda)
    Singularity: t->0 (sigma->0). Target explodes.
    """
    sigma_pred = torch.sqrt(torch.sigmoid(-h_lambda))
    while sigma_pred.ndim < h_data.ndim: sigma_pred = sigma_pred.unsqueeze(-1)
    return h_data * sigma_pred

# ============================================================================
# 4. The Probe (Gradient Analysis)
# ============================================================================

@dataclass
class GradientStats:
    target: str
    timestep: float
    grad_norm: float
    grad_kurtosis: float
    loss_val: float

def compute_kurtosis(tensor: torch.Tensor) -> float:
    x = tensor.detach().flatten().float()
    std = x.std()
    if std < 1e-9: return 0.0
    return (((x - x.mean()) / std) ** 4).mean().item()

def run_gradient_probe(target_type: str, timesteps: np.ndarray) -> pd.DataFrame:
    model = PassthroughMLP(data_dim=4)
    x0 = torch.randn(128, 4)
    stats_list = []
    
    for t_val in timesteps:
        t = torch.full((128,), t_val)
        logsnr = get_logsnr_schedule(t)
        z_t, eps_true, v_true = forward_diffusion(x0, logsnr)
        
        model.zero_grad()
        h_data, h_lambda = model(z_t, logsnr)
        
        # Branching Physics
        if target_type == "epsilon":
            pred = reconstruct_epsilon(h_data, h_lambda)
            target = eps_true
        elif target_type == "x0":
            pred = reconstruct_x0(h_data, h_lambda)
            target = x0
        elif target_type == "v":
            pred = reconstruct_v(h_data, h_lambda)
            target = v_true
            
        loss = nn.functional.mse_loss(pred, target)
        loss.backward()
        
        grad = model.net[0].weight.grad
        
        stats_list.append(GradientStats(
            target=target_type,
            timestep=t_val,
            grad_norm=grad.norm().item(),
            grad_kurtosis=compute_kurtosis(grad),
            loss_val=loss.item()
        ))
        
    return pd.DataFrame([vars(s) for s in stats_list])

# ============================================================================
# 5. Visualization & Execution
# ============================================================================

def plot_combined_results(dfs):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
    targets = ["x0", "epsilon", "v"]
    
    for i, target in enumerate(targets):
        df = dfs[target]
        
        # Norm Plot
        ax_norm = axes[i, 0]
        ax_norm.plot(df['timestep'], df['grad_norm'], color='blue', linewidth=2)
        ax_norm.set_title(f'{target.upper()} Target: Gradient Norm')
        ax_norm.set_ylabel('L2 Norm')
        ax_norm.grid(True, alpha=0.3)
        ax_norm.set_yscale('log') # Log scale handles explosions better
        
        # Kurtosis Plot
        ax_kurt = axes[i, 1]
        ax_kurt.plot(df['timestep'], df['grad_kurtosis'], color='orange', linewidth=2)
        ax_kurt.set_title(f'{target.upper()} Target: Gradient Kurtosis')
        ax_kurt.set_ylabel('Kurtosis')
        ax_kurt.grid(True, alpha=0.3)
        
        # Highlight singularities
        if target == "x0":
            ax_norm.axvline(x=0.99, color='r', linestyle='--', alpha=0.5, label='Singularity')
        elif target in ["epsilon", "v"]:
            ax_norm.axvline(x=0.01, color='r', linestyle='--', alpha=0.5, label='Singularity')

    axes[2,0].set_xlabel('Timestep (0=Clean, 1=Noisy)')
    axes[2,1].set_xlabel('Timestep (0=Clean, 1=Noisy)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Sweep Setup: Dense near tails
    t_center = np.linspace(0.05, 0.95, 20)
    t_tails_0 = np.logspace(-4, -1.3, 15)
    t_tails_1 = 1.0 - np.logspace(-4, -1.3, 15)
    timesteps = np.sort(np.concatenate([t_tails_0, t_center, t_tails_1]))
    
    dfs = {}
    for target in ["x0", "epsilon", "v"]:
        print(f"Running probe for {target}...")
        dfs[target] = run_gradient_probe(target, timesteps)
        
    print("\nProbe Complete. Visualizing...")
    plot_combined_results(dfs)
    
    # Text Report on Tails
    print("\n=== Tail Analysis (Mean Kurtosis in Danger Zone) ===")
    for target in ["x0", "epsilon", "v"]:
        df = dfs[target]
        kurt_t0 = df[df['timestep'] < 0.05]['grad_kurtosis'].mean()
        kurt_t1 = df[df['timestep'] > 0.95]['grad_kurtosis'].mean()
        print(f"{target.upper()}: t->0 Kurt={kurt_t0:.1f} | t->1 Kurt={kurt_t1:.1f}")