"""
diffusion_lab_bench_v2.py

A "Literate Code" Lab Bench for analyzing the Gradient SNR and Stability 
of Diffusion Parameterizations.

COMPARISON MODE:
1. Naive: Direct prediction of target (x0, eps, v).
2. Factorized: Learned Passthrough Lambda + Normalized Target.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

# ============================================================================
# 1. The Physics (Schedule & Transforms)
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
# 2. The Networks
# ============================================================================

class NaiveMLP(nn.Module):
    """
    Standard Architecture:
    Input:  Concat[z_t, logsnr]
    Output: Prediction (x0, eps, or v) directly.
    """
    def __init__(self, data_dim: int, hidden_dim: int = 64):
        super().__init__()
        input_dim = data_dim + 1
        output_dim = data_dim 
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self._init_weights()

    def _init_weights(self):
        torch.manual_seed(42)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z_t: torch.Tensor, logsnr: torch.Tensor) -> Tuple[torch.Tensor, None]:
        l_in = logsnr.view(-1, 1)
        x_in = torch.cat([z_t, l_in], dim=1)
        return self.net(x_in), None  # No lambda output

class PassthroughMLP(nn.Module):
    """
    Factorized Architecture:
    Input:  Concat[z_t, logsnr]
    Output: [Prediction_Norm, Lambda_Pred]
    """
    def __init__(self, data_dim: int, hidden_dim: int = 64):
        super().__init__()
        input_dim = data_dim + 1
        output_dim = data_dim + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self._init_weights()

    def _init_weights(self):
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
# 3. Reconstruction Logic
# ============================================================================

def reconstruct(
    mode: str, 
    target_type: str, 
    h_data: torch.Tensor, 
    h_lambda: Optional[torch.Tensor]
) -> torch.Tensor:
    
    if mode == "naive":
        # Direct prediction, no transformation
        return h_data

    elif mode == "factorized":
        # Apply the scale factor derived from the learned lambda
        if target_type == "epsilon":
            # eps = h_data * sigma(lambda)
            sigma_pred = torch.sqrt(torch.sigmoid(-h_lambda))
            while sigma_pred.ndim < h_data.ndim: sigma_pred = sigma_pred.unsqueeze(-1)
            return h_data * sigma_pred
            
        elif target_type == "x0":
            # x0 = h_data / alpha(lambda)
            alpha_pred = torch.sqrt(torch.sigmoid(h_lambda))
            while alpha_pred.ndim < h_data.ndim: alpha_pred = alpha_pred.unsqueeze(-1)
            return h_data / (alpha_pred + 1e-6)
            
        elif target_type == "v":
            # v = h_data * sigma(lambda)  (User Spec)
            sigma_pred = torch.sqrt(torch.sigmoid(-h_lambda))
            while sigma_pred.ndim < h_data.ndim: sigma_pred = sigma_pred.unsqueeze(-1)
            return h_data * sigma_pred
            
    return h_data

# ============================================================================
# 4. The Gradient Probe
# ============================================================================

@dataclass
class GradientStats:
    mode: str
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

def run_gradient_probe(mode: str, target_type: str, timesteps: np.ndarray) -> pd.DataFrame:
    # Select Model
    if mode == "naive":
        model = NaiveMLP(data_dim=4)
    else:
        model = PassthroughMLP(data_dim=4)
        
    x0 = torch.randn(128, 4)
    stats_list = []
    
    for t_val in timesteps:
        t = torch.full((128,), t_val)
        logsnr = get_logsnr_schedule(t)
        z_t, eps_true, v_true = forward_diffusion(x0, logsnr)
        
        # Reset
        model.zero_grad()
        
        # Forward
        h_data, h_lambda = model(z_t, logsnr)
        
        # Reconstruct
        pred = reconstruct(mode, target_type, h_data, h_lambda)
        
        # Get Ground Truth
        if target_type == "epsilon": target = eps_true
        elif target_type == "x0": target = x0
        elif target_type == "v": target = v_true
            
        # Loss & Backward
        loss = nn.functional.mse_loss(pred, target)
        loss.backward()
        
        # Measure Gradients at the INPUT layer (weights)
        grad = model.net[0].weight.grad
        
        stats_list.append(GradientStats(
            mode=mode,
            target=target_type,
            timestep=t_val,
            grad_norm=grad.norm().item(),
            grad_kurtosis=compute_kurtosis(grad),
            loss_val=loss.item()
        ))
        
    return pd.DataFrame([vars(s) for s in stats_list])

# ============================================================================
# 5. Visualization
# ============================================================================

def plot_comparison(dfs: Dict[str, pd.DataFrame]):
    fig, axes = plt.subplots(3, 2, figsize=(14, 15), sharex=True)
    targets = ["x0", "epsilon", "v"]
    
    for i, target in enumerate(targets):
        # Filter for Naive vs Factorized
        df = dfs[target]
        df_naive = df[df['mode'] == 'naive']
        df_fact = df[df['mode'] == 'factorized']
        
        # --- Gradient Norm ---
        ax_norm = axes[i, 0]
        ax_norm.plot(df_naive['timestep'], df_naive['grad_norm'], 
                     color='red', linestyle='--', label='Naive', linewidth=2)
        ax_norm.plot(df_fact['timestep'], df_fact['grad_norm'], 
                     color='blue', label='Factorized', linewidth=2)
        
        ax_norm.set_title(f'{target.upper()} Target: Gradient Norm')
        ax_norm.set_ylabel('L2 Norm')
        ax_norm.set_yscale('log')
        ax_norm.grid(True, alpha=0.3)
        ax_norm.legend()
        
        # --- Kurtosis ---
        ax_kurt = axes[i, 1]
        ax_kurt.plot(df_naive['timestep'], df_naive['grad_kurtosis'], 
                     color='red', linestyle='--', label='Naive', linewidth=2)
        ax_kurt.plot(df_fact['timestep'], df_fact['grad_kurtosis'], 
                     color='blue', label='Factorized', linewidth=2)
        
        ax_kurt.set_title(f'{target.upper()} Target: Gradient Kurtosis')
        ax_kurt.set_ylabel('Kurtosis')
        ax_kurt.grid(True, alpha=0.3)
        
        # Highlight Singularities
        if target == "x0":
            ax_norm.axvline(x=0.99, color='k', alpha=0.2)
            ax_kurt.axvline(x=0.99, color='k', alpha=0.2)
        else:
            ax_norm.axvline(x=0.01, color='k', alpha=0.2)
            ax_kurt.axvline(x=0.01, color='k', alpha=0.2)

    axes[2,0].set_xlabel('Timestep (0=Clean, 1=Noisy)')
    axes[2,1].set_xlabel('Timestep (0=Clean, 1=Noisy)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define sweep
    t_center = np.linspace(0.05, 0.95, 20)
    t_tails_0 = np.logspace(-4, -1.3, 15)
    t_tails_1 = 1.0 - np.logspace(-4, -1.3, 15)
    timesteps = np.sort(np.concatenate([t_tails_0, t_center, t_tails_1]))
    
    all_results = {}
    
    for target in ["x0", "epsilon", "v"]:
        print(f"Running probes for {target}...")
        df_naive = run_gradient_probe("naive", target, timesteps)
        df_fact = run_gradient_probe("factorized", target, timesteps)
        
        # Combine
        all_results[target] = pd.concat([df_naive, df_fact])
        
    print("\nVisualizing Head-to-Head Comparison...")
    plot_comparison(all_results)
    
    # Statistical Summary
    print("\n=== Kurtosis Comparison at Singularities ===")
    for target in ["x0", "epsilon", "v"]:
        df = all_results[target]
        
        # Define the danger zone for this target
        if target == "x0":
            danger_zone = df[df['timestep'] > 0.98]
            zone_name = "t->1"
        else:
            danger_zone = df[df['timestep'] < 0.02]
            zone_name = "t->0"
            
        k_naive = danger_zone[danger_zone['mode'] == 'naive']['grad_kurtosis'].mean()
        k_fact = danger_zone[danger_zone['mode'] == 'factorized']['grad_kurtosis'].mean()
        
        print(f"{target.upper()} ({zone_name}): Naive={k_naive:.1f} vs Factorized={k_fact:.1f}")