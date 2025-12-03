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
# NEW: Import CompositeIterator
from dataset import CompositeIterator
from model import HybridGemmaDiT

class ExperimentLogger:
    def __init__(self, output_dir="."):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        script_path = Path(sys.argv[0])
        self.script_name = script_path.stem
        existing = list(self.output_dir.glob(f"{self.script_name}_run_*"))
        if existing:
            run_nums = [int(p.stem.split('_run_')[1].split('_')[0]) for p in existing]
            self.run_id = max(run_nums) + 1
        else:
            self.run_id = 0
        self.figure_count = 0
        self.run_dir = self.output_dir / f"{self.script_name}_run_{self.run_id:03d}"
        self.run_dir.mkdir(exist_ok=True)
        print(f"📊 Experiment: {self.script_name} | Run: {self.run_id} | Dir: {self.run_dir}")
        
    def save_figure(self, fig, name=None):
        if name is None: name = f"fig{self.figure_count}"
        filename = f"{name}.png"
        filepath = self.run_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.figure_count += 1
        return filepath

def visualize_dataset_samples(iterator, resolutions, samples_per_res=8):
    """
    Generate samples from the composite iterator and label them.
    """
    fig, axes = plt.subplots(len(resolutions), samples_per_res, 
                            figsize=(samples_per_res * 1.5, len(resolutions) * 1.8))
    
    if len(resolutions) == 1: axes = axes.reshape(1, -1)
    
    for row_idx, res in enumerate(resolutions):
        # Generate batch
        samples = iterator.generate_batch(samples_per_res, res, num_tiles=4.0)
        labels = iterator.last_labels.cpu().numpy()
        samples = samples.cpu()
        
        for col_idx in range(samples_per_res):
            ax = axes[row_idx, col_idx]
            ax.imshow(samples[col_idx].permute(1, 2, 0).clamp(0, 1))
            ax.axis('off')
            
            # Get label name
            lbl_idx = labels[col_idx]
            lbl_name = iterator.label_map.get(lbl_idx, "Unknown")
            
            if col_idx == 0:
                ax.set_title(f"{res}px\n{lbl_name}", fontsize=8, loc='left')
            else:
                ax.set_title(f"{lbl_name}", fontsize=7)
                
    plt.suptitle("Composite Dataset Samples", fontsize=14)
    plt.tight_layout()
    return fig

def train_multires(mode, steps=1000, embed_dim=256, depth=12, logger=None):
    device = torch.device('cuda')
    print(f"\n--- Training: {mode.upper()} ---")
    
    model = HybridGemmaDiT(mode, embed_dim=embed_dim, depth=depth).to(device)
    model = torch.compile(model)
        
    # Standard Params
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    
    # NEW: Configure Composite Dataset
    mix_config = {
        'checkerboard': 0.5,
        'torus': 0.5
    }
    iterator = CompositeIterator(device, config=mix_config)
    print(f"    Dataset Mix: {mix_config}")
    
    buckets = [(16, 256), (32, 64)]
    manager = BucketManager(buckets)
    
    history = []
    
    pbar = tqdm(range(steps), desc=f"{mode}")
    for i in pbar:
        opt.zero_grad()
        res, bs = manager.next_bucket()
        
        # Generate mixed batch
        x0 = iterator.generate_batch(bs, res, num_tiles=4.0)
        labels = iterator.last_labels # [B]
        
        t = torch.rand(bs, device=device).clamp(0.001, 0.999)
        logsnr = get_schedule(t)
        alpha, sigma = get_alpha_sigma(logsnr)
        
        eps = torch.randn_like(x0)
        z_t = x0 * alpha.view(-1,1,1,1) + eps * sigma.view(-1,1,1,1)
        v_true = alpha.view(-1,1,1,1) * eps - sigma.view(-1,1,1,1) * x0
        
        # Forward
        raw, l_pred, aux_loss = model(z_t, logsnr)
        
        if mode == 'factorized':
            sigma_p = torch.sqrt(torch.sigmoid(-l_pred)).view(-1, 1, 1, 1)
            v_pred = raw * sigma_p
        else:
            v_pred = raw
            
        # Detailed Loss Calculation
        # Compute elementwise to separate by class
        loss_elem = F.mse_loss(v_pred, v_true, reduction='none').mean(dim=[1,2,3])
        total_loss = loss_elem.mean()
        
        total_loss.backward()
        opt.step()
        
        # --- LOGGING ---
        step_stats = {'step': i, 'res': res, 'loss_total': total_loss.item()}
        
        # Breakdown by dataset type
        for idx, name in iterator.label_map.items():
            mask = (labels == idx)
            if mask.any():
                step_stats[f'loss_{name}'] = loss_elem[mask].mean().item()
            else:
                step_stats[f'loss_{name}'] = None # Should handle NaN in plots
                
        history.append(step_stats)
        
        if i % 100 == 0:
            pbar.set_postfix({'loss': f'{total_loss.item():.4f}', 'res': res})
            
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
        x0 = alpha.view(-1,1,1,1)*z - sigma.view(-1,1,1,1)*v_pred
        eps = sigma.view(-1,1,1,1)*z + alpha.view(-1,1,1,1)*v_pred
        alpha_n, sigma_n = get_alpha_sigma(logsnr_n)
        z = alpha_n.view(-1,1,1,1)*x0 + sigma_n.view(-1,1,1,1)*eps
    return z.cpu().clamp(0, 1)

def plot_detailed_loss(df_naive, df_fact, logger):
    """
    Generates a 2x2 grid.
    Rows: Resolution (16, 32)
    Cols: Dataset Type (Checkerboard, Torus)
    """
    # Clean data (interpolating NaNs if batches were pure one type)
    df_naive = df_naive.interpolate()
    df_fact = df_fact.interpolate()
    
    resolutions = sorted(df_naive['res'].unique())
    datasets = ['checkerboard', 'torus']
    
    fig, axes = plt.subplots(len(resolutions), len(datasets), figsize=(12, 8))
    
    for r_idx, res in enumerate(resolutions):
        n_res = df_naive[df_naive['res'] == res]
        f_res = df_fact[df_fact['res'] == res]
        
        for d_idx, dtype in enumerate(datasets):
            ax = axes[r_idx, d_idx]
            col_name = f'loss_{dtype}'
            
            # Smoothing for cleaner plots
            roll_win = 20
            
            if col_name in n_res.columns:
                line_n = n_res[col_name].rolling(roll_win).mean()
                ax.plot(n_res['step'], line_n, label='Naive', color='tab:blue', alpha=0.8)
                
            if col_name in f_res.columns:
                line_f = f_res[col_name].rolling(roll_win).mean()
                ax.plot(f_res['step'], line_f, label='Factorized', color='tab:orange', alpha=0.8)
            
            ax.set_title(f"{dtype.capitalize()} @ {res}px")
            ax.set_yscale('log')
            ax.grid(True, which='both', alpha=0.2)
            if r_idx == 0 and d_idx == 0:
                ax.legend()
    
    plt.tight_layout()
    logger.save_figure(fig, "loss_breakdown_res_vs_type")

def plot_sample_grid(n16, n32, f16, f32, logger):
    fig, axes = plt.subplots(4, 8, figsize=(16, 9))
    rows = [("Naive 16px", n16), ("Fact 16px", f16), 
            ("Naive 32px", n32), ("Fact 32px", f32)]
    
    for r, (name, batch) in enumerate(rows):
        for c in range(8):
            axes[r, c].imshow(batch[c].permute(1,2,0))
            axes[r, c].axis('off')
            if c == 0: axes[r, c].set_title(name, fontsize=10, loc='left')
            
    plt.suptitle("Unconditional Generation (Mixed Distribution)", fontsize=16)
    plt.tight_layout()
    logger.save_figure(fig, "final_samples")

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    logger = ExperimentLogger(output_dir="./experiments_mix")
    
    # 1. Verify Data Mix
    print("\n📸 Verifying Composite Data...")
    iterator = CompositeIterator(device='cuda', config={'checkerboard': 0.5, 'torus': 0.5})
    fig_data = visualize_dataset_samples(iterator, [16, 32])
    logger.save_figure(fig_data, "dataset_mix_verification")
    
    # 2. Train
    # Reduced steps for demo, increase for real bench
    steps = 4000 
    df_n, mod_n = train_multires('naive', steps=steps, depth=4, logger=logger)
    df_f, mod_f = train_multires('factorized', steps=steps, depth=4, logger=logger)
    
    # 3. Analyze
    print("\n📈 Plotting breakdown...")
    plot_detailed_loss(df_n, df_f, logger)
    
    print("\n🎨 Sampling...")
    n16 = sample_viz(mod_n, 16); n32 = sample_viz(mod_n, 32)
    f16 = sample_viz(mod_f, 16); f32 = sample_viz(mod_f, 32)
    
    plot_sample_grid(n16, n32, f16, f32, logger)
    print(f"\n✅ Done. Check {logger.run_dir}")