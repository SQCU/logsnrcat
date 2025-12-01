# diffusion_lab_bench_v3.py
#literally evil monkey patch
import inductor_cas_client
inductor_cas_client.install_cas_client() # Auto-discovers port, might even be 54321

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import Literal, List, Dict

    
# --- Performance Tuning ---
# Force Inductor to use all cores for compilation
import torch._inductor.config
# logic: use physical cores (usually half of logical on hyperthreaded cpus) or default to 16
import multiprocessing
torch._inductor.config.compile_threads = max(7, (multiprocessing.cpu_count() // 2) -1)
#torch._inductor.config.compile_threads = 7
# Set to your physical core count

# Import internal dependencies
from diffusion_utils import get_schedule, get_alpha_sigma
from model import HybridGemmaDiT, DynamicSpatialBuffer

# --- Configuration ---
RESOLUTIONS = [16, 32, 64] 
TARGETS = ['v', 'x0', 'epsilon']
MODES = ['naive', 'factorized']
STEPS_PER_SWEEP = 50  # Higher resolution sweep since it runs faster now
BATCH_SIZE = {16: 64, 32: 32, 64: 8} # Fit in VRAM

def get_target_tensor(target_name, x0, eps, v_true):
    if target_name == 'x0': return x0
    if target_name == 'epsilon': return eps
    if target_name == 'v': return v_true
    raise ValueError(f"Unknown target: {target_name}")

class ModelPatcher:
    @staticmethod
    def patch_resolution(model, resolution, device='cuda'):
        # Unwrap if compiled
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
    """Computes Kurtosis of a tensor in a single pass."""
    x = t.detach().flatten().float()
    if x.numel() < 2: return 0.0
    
    # We need robust variance, so standard calc:
    std = x.std()
    if std < 1e-9: return 0.0
    
    # 4th moment
    x_norm = (x - x.mean()) / std
    kurt = (x_norm ** 4).mean().item()
    return kurt

def run_physics_sweep():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')
    
    results = []
    print(f"--- Diffusion Lab Bench v4: Spatial Stats & Kurtosis ---")
    
    for mode in MODES:
        print(f"\n> Initializing Model Mode: {mode.upper()}")
        
        # Fresh Model Init
        base_model = HybridGemmaDiT(mode=mode, embed_dim=256, depth=4).to(device)
        base_model.train() 
        
        for res in RESOLUTIONS:
            # Patch & Compile
            ModelPatcher.patch_resolution(base_model, res, device)
            print(f"  [Compiling] Resolution {res}x{res}...")
            model = torch.compile(base_model)
            
            bs = BATCH_SIZE[res]
            
            for target_name in TARGETS:
                # Sweep schedule
                # We want density at the singularities
                t_mid = torch.linspace(0.05, 0.95, STEPS_PER_SWEEP-8, device=device)
                t_ends = torch.tensor([0.001, 0.005, 0.01, 0.02, 0.98, 0.99, 0.995, 0.999], device=device)
                timesteps = torch.cat([t_ends, t_mid]).sort().values
                
                pbar = tqdm(timesteps, desc=f"    {target_name.upper()}", leave=False)
                
                for t_val in pbar:
                    # 1. Generate Data
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
                    
                    # 3. Prediction Interpretation
                    if mode == 'factorized':
                        sigma_p = torch.sqrt(torch.sigmoid(-l_pred)).view(-1, 1, 1, 1)
                        pred = raw * sigma_p
                    else:
                        pred = raw
                        
                    target_tensor = get_target_tensor(target_name, x0, eps, v_true)
                    
                    # 4. Spatial/Batch Statistics (The "Image" Stats)
                    # Squared Error per pixel
                    error_sq = (pred - target_tensor) ** 2 
                    
                    # Mean Loss (Scalar)
                    loss_mean = error_sq.mean()
                    
                    # Loss Spatial Variance (Scalar)
                    # How much does the error vary across the batch/pixels?
                    # High std = Model fails on specific features/samples
                    loss_std = error_sq.std() 
                    
                    # Backward for Gradients
                    loss_mean.backward()
                    
                    # 5. Gradient Statistics (The "Weight" Stats)
                    if hasattr(model, '_orig_mod'):
                        w_grad = model._orig_mod.patch_in.weight.grad
                    else:
                        w_grad = model.patch_in.weight.grad
                        
                    grad_norm = w_grad.norm().item()
                    grad_kurt = compute_kurtosis(w_grad)
                    
                    results.append({
                        'mode': mode,
                        'res': res,
                        'target': target_name,
                        't': t_val.item(),
                        'loss_mean': loss_mean.item(),
                        'loss_std': loss_std.item(),
                        'grad_norm': grad_norm,
                        'grad_kurt': grad_kurt
                    })
    
    return pd.DataFrame(results)

def plot_metric(df, metric_mean, metric_std, title, filename):
    """Generic plotter for Mean +/- Std lines"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    colors = {16: 'tab:red', 32: 'tab:green', 64: 'tab:blue'}
    
    for r, target in enumerate(TARGETS):
        for c, mode in enumerate(MODES):
            ax = axes[r, c]
            for res in RESOLUTIONS:
                sub = df[(df['target'] == target) & 
                         (df['mode'] == mode) & 
                         (df['res'] == res)].sort_values('t')
                
                t = sub['t'].values
                mu = sub[metric_mean].values
                
                # If a std column is provided, use it for shading
                # If not (e.g. Kurtosis), just plot the line
                ax.plot(t, mu, color=colors[res], label=f'{res}px', linewidth=1.5)
                
                if metric_std is not None:
                    sigma = sub[metric_std].values
                    # For log scale plots, clamp bottom
                    lower = np.maximum(mu - sigma, 1e-8) 
                    upper = mu + sigma
                    ax.fill_between(t, lower, upper, color=colors[res], alpha=0.15)
                
            ax.set_title(f"{mode.title()} - {target.upper()}")
            
            # Heuristics for scaling
            if "loss" in metric_mean:
                ax.set_yscale('log')
            elif "norm" in metric_mean:
                ax.set_yscale('log')
            # Kurtosis is usually linear scale or log if crazy spikes
            elif "kurt" in metric_mean:
                ax.set_yscale('linear') 
                
            ax.grid(True, alpha=0.3)
            
            if r == 2: ax.set_xlabel("Timestep (t)")
            if c == 0: ax.set_ylabel(title)
            if r == 0 and c == 0: ax.legend()
            
    plt.suptitle(f"Figure: {title} vs Timestep", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")

if __name__ == "__main__":
    df = run_physics_sweep()
    
    print("\nplotting...")
    
    # 1. Loss Stability (Spatial Variance)
    plot_metric(df, 'loss_mean', 'loss_std', 
                r"MSE Loss (Mean $\pm$ Spatial Std)", "fig_loss_stability.png")
    
    # 2. Gradient Norm
    plot_metric(df, 'grad_norm', None, 
                "Gradient Norm", "fig_grad_norm.png")
    
    # 3. Gradient Kurtosis
    plot_metric(df, 'grad_kurt', None, 
                "Gradient Kurtosis", "fig_grad_kurtosis.png")
    
    print("\n=== SUMMARY: Resolution Scaling Factor (Mean Gradient Norm) ===")
    for mode in MODES:
        for target in TARGETS:
            sub = df[(df['mode']==mode) & (df['target']==target)]
            g16 = sub[sub['res']==16]['grad_norm'].mean()
            g64 = sub[sub['res']==64]['grad_norm'].mean()
            print(f"[{mode}-{target}] 16->64px Scaling: {g64/g16:.2f}x")