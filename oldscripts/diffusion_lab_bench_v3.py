# diffusion_lab_bench_v3.py
#literally evil monkey patch
import inductor_cas_client
# Hook the ZMQ compiler backend immediately
inductor_cas_client.install_cas_client() 

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import Literal, List, Dict
import multiprocessing

# --- Configuration ---
# Threads: The Daemon handles the heavy lifting, but we still need threads 
# for the Inductor frontend (tracing/graphing)
import torch._inductor.config
torch._inductor.config.compile_threads = max(4, (multiprocessing.cpu_count() // 2) - 1)

from diffusion_utils import get_schedule, get_alpha_sigma
from model import HybridGemmaDiT, DynamicSpatialBuffer

RESOLUTIONS = [16, 32, 64] 
TARGETS = ['v', 'x0', 'epsilon']
MODES = ['naive', 'factorized']

# We trade granularity in T for statistical depth per T
STEPS_PER_SWEEP = 25 
BATCHES_PER_STEP = 8  # The "Multi-Pass" factor
BATCH_SIZE = {16: 64, 32: 32, 64: 8} 

def get_target_tensor(target_name, x0, eps, v_true):
    if target_name == 'x0': return x0
    if target_name == 'epsilon': return eps
    if target_name == 'v': return v_true
    raise ValueError(f"Unknown target: {target_name}")

class ModelPatcher:
    @staticmethod
    def patch_resolution(model, resolution, device='cuda'):
        # Unwrap if previously compiled
        if hasattr(model, '_orig_mod'):
            real_model = model._orig_mod
        else:
            real_model = model

        grid_size = resolution // 2
        if real_model.spatial.max_grid_size < grid_size:
            head_dim = real_model.layers[0].head_dim
            new_spatial = DynamicSpatialBuffer(max_grid_size=grid_size, head_dim=head_dim, device=device)
            real_model.spatial = new_spatial
            for layer in real_model.layers:
                layer.rn_rope.buffer = new_spatial
        return real_model

def compute_kurtosis(t):
    """Computes Kurtosis of a tensor."""
    x = t.detach().flatten().float()
    if x.numel() < 2: return 0.0
    std = x.std()
    if std < 1e-9: return 0.0
    x_norm = (x - x.mean()) / std
    kurt = (x_norm ** 4).mean().item()
    return kurt

def run_population_sweep():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')
    
    results = []
    print(f"--- Diffusion Lab Bench v5: Population Stability (Multi-Pass) ---")
    
    for mode in MODES:
        print(f"\n> Mode: {mode.upper()}")
        
        # Fresh Init
        base_model = HybridGemmaDiT(mode=mode, embed_dim=256, depth=4).to(device)
        base_model.train() 
        
        for res in RESOLUTIONS:
            ModelPatcher.patch_resolution(base_model, res, device)
            print(f"  [Compiling] Resolution {res}x{res}...")
            # The CAS Daemon makes this fast on the 2nd run
            model = torch.compile(base_model)
            
            bs = BATCH_SIZE[res]
            
            for target_name in TARGETS:
                # Distribution of T
                t_mid = torch.linspace(0.05, 0.95, STEPS_PER_SWEEP-6, device=device)
                t_ends = torch.tensor([0.001, 0.01, 0.02, 0.98, 0.99, 0.999], device=device)
                timesteps = torch.cat([t_ends, t_mid]).sort().values
                
                pbar = tqdm(timesteps, desc=f"    {target_name.upper()}", leave=False)
                
                for t_val in pbar:
                    
                    # Accumulators for Population Statistics
                    pop_loss = []
                    pop_grad_norm = []
                    pop_grad_kurt = []
                    
                    # --- The Multi-Pass Loop ---
                    for _ in range(BATCHES_PER_STEP):
                        # 1. Physics
                        x0 = torch.randn(bs, 3, res, res, device=device)
                        eps = torch.randn_like(x0)
                        
                        t_batch = torch.full((bs,), t_val, device=device)
                        logsnr = get_schedule(t_batch)
                        alpha, sigma = get_alpha_sigma(logsnr)
                        alpha, sigma = alpha.view(-1,1,1,1), sigma.view(-1,1,1,1)
                        
                        z_t = alpha * x0 + sigma * eps
                        v_true = alpha * eps - sigma * x0
                        
                        # 2. Forward
                        model.zero_grad()
                        raw, l_pred = model(z_t, logsnr)
                        
                        # 3. Interpret Prediction
                        if mode == 'factorized':
                            sigma_p = torch.sqrt(torch.sigmoid(-l_pred)).view(-1, 1, 1, 1)
                            pred = raw * sigma_p
                        else:
                            pred = raw
                        
                        target_tensor = get_target_tensor(target_name, x0, eps, v_true)
                        
                        # 4. Loss
                        loss = F.mse_loss(pred, target_tensor)
                        loss.backward()
                        
                        # 5. Capture Stats
                        pop_loss.append(loss.item())
                        
                        if hasattr(model, '_orig_mod'):
                            w = model._orig_mod.patch_in.weight.grad
                        else:
                            w = model.patch_in.weight.grad
                            
                        pop_grad_norm.append(w.norm().item())
                        pop_grad_kurt.append(compute_kurtosis(w))
                    
                    # --- Aggregate Population Stats ---
                    results.append({
                        'mode': mode,
                        'res': res,
                        'target': target_name,
                        't': t_val.item(),
                        # Loss
                        'loss_mean': np.mean(pop_loss),
                        'loss_std': np.std(pop_loss),
                        # Grad Norm
                        'grad_norm_mean': np.mean(pop_grad_norm),
                        'grad_norm_std': np.std(pop_grad_norm),
                        # Grad Kurtosis
                        'grad_kurt_mean': np.mean(pop_grad_kurt),
                        'grad_kurt_std': np.std(pop_grad_kurt),
                    })
    
    return pd.DataFrame(results)

def plot_population_stability(df, metric_base, title, filename):
    """
    Plots Mean line with Shaded region representing +/- 1 Std Dev of the population.
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    colors = {16: 'tab:red', 32: 'tab:green', 64: 'tab:blue'}
    
    mean_col = f"{metric_base}_mean"
    std_col = f"{metric_base}_std"
    
    for r, target in enumerate(TARGETS):
        for c, mode in enumerate(MODES):
            ax = axes[r, c]
            for res in RESOLUTIONS:
                sub = df[(df['target'] == target) & 
                         (df['mode'] == mode) & 
                         (df['res'] == res)].sort_values('t')
                
                t = sub['t'].values
                mu = sub[mean_col].values
                sigma = sub[std_col].values
                
                # Plot Mean
                ax.plot(t, mu, color=colors[res], label=f'{res}px', linewidth=1.5)
                
                # Plot Population Variance
                lower = np.maximum(mu - sigma, 1e-8)
                upper = mu + sigma
                ax.fill_between(t, lower, upper, color=colors[res], alpha=0.15)
                
            ax.set_title(f"{mode.title()} - {target.upper()}")
            
            # Scales
            if "loss" in metric_base or "norm" in metric_base:
                ax.set_yscale('log')
            
            ax.grid(True, alpha=0.3)
            
            if r == 2: ax.set_xlabel("Timestep (t)")
            if c == 0: ax.set_ylabel(title)
            if r == 0 and c == 0: ax.legend(loc='upper right')
            
    plt.suptitle(f"Figure: {title} vs Timestep (Population Stability)", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")

if __name__ == "__main__":
    df = run_population_sweep()
    
    print("\nplotting...")
    
    # 1. Loss Stability
    plot_population_stability(df, 'loss', 
                              r"MSE Loss (Mean $\pm$ Pop Std)", 
                              "fig_pop_loss.png")
    
    # 2. Gradient Stability
    plot_population_stability(df, 'grad_norm', 
                              r"Gradient Norm (Mean $\pm$ Pop Std)", 
                              "fig_pop_grad_norm.png")
    
    # 3. Kurtosis Stability
    plot_population_stability(df, 'grad_kurt', 
                              r"Gradient Kurtosis (Mean $\pm$ Pop Std)", 
                              "fig_pop_kurtosis.png")
    
    print("\n=== SUMMARY: Resolution Scaling (Mean Gradient Norm) ===")
    for mode in MODES:
        for target in TARGETS:
            sub = df[(df['mode']==mode) & (df['target']==target)]
            g16 = sub[sub['res']==16]['grad_norm_mean'].mean()
            g64 = sub[sub['res']==64]['grad_norm_mean'].mean()
            print(f"[{mode}-{target}] 16->64px Scaling: {g64/g16:.2f}x")