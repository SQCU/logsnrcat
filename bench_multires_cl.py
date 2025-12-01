# bench_multires_cl.py
import inductor_cas_client
# Hook the ZMQ compiler backend immediately
inductor_cas_client.install_cas_client() 
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import sys
from pathlib import Path

from diffusion_utils import get_schedule, get_alpha_sigma, BucketManager
from dataset import CheckerboardIterator
from model import HybridGemmaDiT
from sampler import sample_euler_v

class ExperimentLogger:
    """
    Handles figure naming and saving based on script name and run count.
    """
    def __init__(self, output_dir="."):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Parse script name (without .py extension)
        script_path = Path(sys.argv[0])
        self.script_name = script_path.stem
        
        # Find next run number
        existing = list(self.output_dir.glob(f"{self.script_name}_run_*"))
        if existing:
            run_nums = [int(p.stem.split('_run_')[1].split('_')[0]) for p in existing]
            self.run_id = max(run_nums) + 1
        else:
            self.run_id = 0
            
        self.figure_count = 0
        print(f"📊 Experiment: {self.script_name} | Run: {self.run_id}")
        
    def save_figure(self, fig, name=None):
        """Save figure with automatic naming."""
        if name is None:
            name = f"fig{self.figure_count}"
        filename = f"{self.script_name}_run_{self.run_id:03d}_{name}.png"
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"💾 Saved: {filepath}")
        plt.close(fig)
        self.figure_count += 1
        return filepath

def visualize_dataset_samples(iterator, resolutions, samples_per_res=8):
    """
    Generate and plot dataset samples to verify what we're training on.
    Returns figure showing ground truth data.
    """
    fig, axes = plt.subplots(len(resolutions), samples_per_res, 
                            figsize=(samples_per_res * 2, len(resolutions) * 2))
    
    if len(resolutions) == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, res in enumerate(resolutions):
        samples = iterator.generate_batch(samples_per_res, res, num_tiles=4.0)
        samples = samples.cpu()
        
        for col_idx in range(samples_per_res):
            ax = axes[row_idx, col_idx]
            ax.imshow(samples[col_idx].permute(1, 2, 0).clamp(0, 1))
            ax.axis('off')
            if col_idx == 0:
                ax.set_title(f"GT {res}×{res}", fontsize=10, loc='left')
                
    plt.suptitle("Dataset Ground Truth Samples (num_tiles=4.0)", fontsize=14)
    plt.tight_layout()
    return fig

def train_multires(mode, steps=1000, embed_dim=256, depth=12, logger=None, is_moe=True):
    device = torch.device('cuda')
    print(f"\n--- Training: {mode.upper()} | embed_dim={embed_dim} depth={depth} is_moe={is_moe}---")
    
    # Init Model
    model = HybridGemmaDiT(mode, embed_dim=embed_dim, depth=depth).to(device)
    model = torch.compile(model)
        
    # Group A: Geometry (Householder) -> Low LR
    # These define the coordinate system. Stability is key.
    householder_params = [p for n,p in model.named_parameters() if 'orthogonal.vs' in n]
    # Group B: Interface (Embed/Unembed/Decoders) -> High LR
    # These map raw pixels/noise to the latent space. They need to move fast early on.
    interface_names = ['patch_in', 'patch_out', 'scale_decoder', 'lambda_head', 'output_head']
    params_interface = [p for n,p in model.named_parameters() 
                        if any(x in n for x in interface_names) and 'orthogonal' not in n]
    # Group C: Backbone (Transformers) -> Medium LR
    # Everything else (MLPs, QKV projections, ~~Norms~~ these better not have params)
    other_params = [p for n,p in model.named_parameters() 
                       if not any(x in n for x in interface_names) and 'orthogonal' not in n]
    # Option B: Don't schedule Householder
    opt_main = torch.optim.AdamW(other_params, lr=5e-4, weight_decay=0.1)
    opt_house = torch.optim.AdamW(householder_params, lr=1e-4, weight_decay=0.0)
    opt_interface = torch.optim.AdamW(params_interface, lr=0.1, weight_decay=0.0)
    # Learning rate schedule with warmup + decay
    from torch.optim.lr_scheduler import OneCycleLR
    scheduler_main = OneCycleLR(opt_main, max_lr=1e-4, total_steps=steps, 
                        pct_start=0.1, div_factor=10, final_div_factor=100)
    scheduler_house = OneCycleLR(opt_house, max_lr=1e-4, total_steps=steps, 
                        pct_start=0.1, div_factor=10, final_div_factor=100)
    scheduler_interface = OneCycleLR(opt_interface, max_lr=1e-2, total_steps=steps, 
                        pct_start=0.1, div_factor=10, final_div_factor=100)
    # Init Data
    iterator = CheckerboardIterator(device)
    
    # Buckets: (Resolution, BatchSize)
    # for depth:4 a3w8
    #buckets = [(16, 256), (32, 64)]
    # for depth:8 a3w8
    buckets = [(16, 128), (32, 32)]
    manager = BucketManager(buckets)
    
    history = []
    
    pbar = tqdm(range(steps), desc=f"{mode}")
    for i in pbar:
        opt_main.zero_grad()
        opt_house.zero_grad()
        opt_interface.zero_grad()
        aux_loss = None
        
        res, bs = manager.next_bucket()
        x0 = iterator.generate_batch(bs, res, num_tiles=4.0)
        
        t = torch.rand(bs, device=device).clamp(0.001, 0.999)
        logsnr = get_schedule(t)
        alpha, sigma = get_alpha_sigma(logsnr)
        alpha, sigma = alpha.view(-1,1,1,1), sigma.view(-1,1,1,1)
        
        eps = torch.randn_like(x0)
        z_t = x0 * alpha + eps * sigma
        v_true = alpha * eps - sigma * x0
        
        # Forward
        raw, l_pred, route_loss = model(z_t, logsnr)
        
        # Reconstruction
        if mode == 'factorized':
            sigma_p = torch.sqrt(torch.sigmoid(-l_pred)).view(-1, 1, 1, 1)
            v_pred = raw * sigma_p
        else:
            v_pred = raw
        loss = F.mse_loss(v_pred, v_true)
        if is_moe:
            aux_loss = 0.01*route_loss
        if aux_loss is not None:
            total_loss = loss + aux_loss
        else:
            total_loss = loss
        total_loss.backward()
        opt_main.step()
        opt_house.step()
        opt_interface.step()
        scheduler_main.step()
        scheduler_house.step()
        scheduler_interface.step()
        history.append({'step': i, 'res': res, 'loss': loss.item()})
        
        if i % 100 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'res': res})
            
    return pd.DataFrame(history), model

@torch.no_grad()
def sample_viz(model, res, num_samples=8):
    model.eval()
    z = torch.randn(num_samples, 3, res, res, device='cuda')
    ts = torch.linspace(1.0, 0.001, 50, device='cuda')
    
    for i in range(49):
        t = ts[i]; t_n = ts[i+1]
        logsnr = get_schedule(torch.full((num_samples,), t, device='cuda'))
        
        raw, l_pred, _ = model(z, logsnr)
        
        if model.mode == 'factorized':
            sigma_p = torch.sqrt(torch.sigmoid(-l_pred)).view(-1,1,1,1)
            v_pred = raw * sigma_p
        else:
            v_pred = raw
        
        logsnr_n = get_schedule(torch.full((num_samples,), t_n, device='cuda'))
        alpha, sigma = get_alpha_sigma(logsnr)
        alpha, sigma = alpha.view(-1,1,1,1), sigma.view(-1,1,1,1)
        
        x0 = alpha * z - sigma * v_pred
        eps = sigma * z + alpha * v_pred
        
        alpha_n, sigma_n = get_alpha_sigma(logsnr_n)
        z = alpha_n.view(-1,1,1,1) * x0 + sigma_n.view(-1,1,1,1) * eps
        
    return z.cpu().clamp(0, 1)

def plot_loss_curves(df_naive, df_fact, logger):
    """Plot loss curves for both resolutions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Res 16
    n16 = df_naive[df_naive['res'] == 16]
    f16 = df_fact[df_fact['res'] == 16]
    axes[0].plot(n16['step'], n16['loss'], label='Naive', alpha=0.7, linewidth=1)
    axes[0].plot(f16['step'], f16['loss'], label='Factorized', alpha=0.7, linewidth=1)
    axes[0].set_title("Resolution 16×16 Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot Res 32
    n32 = df_naive[df_naive['res'] == 32]
    f32 = df_fact[df_fact['res'] == 32]
    axes[1].plot(n32['step'], n32['loss'], label='Naive', alpha=0.7, linewidth=1)
    axes[1].plot(f32['step'], f32['loss'], label='Factorized', alpha=0.7, linewidth=1)
    axes[1].set_title("Resolution 32×32 Loss")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    logger.save_figure(fig, "loss_curves")

def plot_sample_comparison(n_s16, n_s32, f_s16, f_s32, logger):
    """Plot generated samples in grid layout."""
    fig, axes = plt.subplots(4, 8, figsize=(16, 9))
    
    labels = ["Naive (16px)", "Factorized (16px)", "Naive (32px)", "Factorized (32px)"]
    samples = [n_s16, f_s16, n_s32, f_s32]
    
    for row_idx, (label, samp) in enumerate(zip(labels, samples)):
        for col_idx in range(8):
            axes[row_idx, col_idx].imshow(samp[col_idx].permute(1, 2, 0))
            axes[row_idx, col_idx].axis('off')
            if col_idx == 0:
                axes[row_idx, col_idx].set_title(label, fontsize=10, loc='left')
                
    plt.suptitle("Model Samples After Training", fontsize=14)
    plt.tight_layout()
    logger.save_figure(fig, "sample_comparison")

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    
    # Initialize logger
    logger = ExperimentLogger(output_dir="./experiments")
    
    # 0. Visualize dataset first
    print("\n📸 Generating dataset samples...")
    iterator = CheckerboardIterator(device='cuda')
    dataset_fig = visualize_dataset_samples(iterator, resolutions=[16, 32], samples_per_res=8)
    logger.save_figure(dataset_fig, "dataset_groundtruth")
    
    # 1. Run training
    df_naive, model_naive = train_multires('naive', steps=3000, embed_dim=256, depth=8, logger=logger)
    df_fact, model_fact = train_multires('factorized', steps=3000, embed_dim=256, depth=8, logger=logger)
    
    # 2. Plot loss curves
    print("\n📈 Plotting loss curves...")
    plot_loss_curves(df_naive, df_fact, logger)
    
    # 3. Generate samples
    print("\n🎨 Generating model samples...")
    n_s16 = sample_viz(model_naive, 16)
    n_s32 = sample_viz(model_naive, 32)
    f_s16 = sample_viz(model_fact, 16)
    f_s32 = sample_viz(model_fact, 32)
    
    # 4. Plot sample comparison
    plot_sample_comparison(n_s16, n_s32, f_s16, f_s32, logger)
    
    print(f"\n✅ Experiment complete! Results in: ./experiments/")
"""

**What this gives you:**
```
./experiments/
  bench_multires_run_000_dataset_groundtruth.png  ← What we're training on
  bench_multires_run_000_loss_curves.png          ← Training dynamics
  bench_multires_run_000_sample_comparison.png    ← What the model learned
  bench_multires_run_001_dataset_groundtruth.png  ← Next run
  ...
"""