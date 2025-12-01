# bench_backbone_hit_that_griddy.py
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
import gc
from pathlib import Path

# Reuse existing infrastructure
from diffusion_utils import get_schedule, get_alpha_sigma, BucketManager
from dataset import CheckerboardIterator
from model import HybridGemmaDiT

# --- Strict Duplication of Logging/Viz Code ---
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
        
        # Sub-folder for this specific run to keep things tidy
        self.run_dir = self.output_dir / f"{self.script_name}_run_{self.run_id:03d}"
        self.run_dir.mkdir(exist_ok=True)
        
        print(f"📊 Experiment: {self.script_name} | Run: {self.run_id}")
        print(f"📂 Output: {self.run_dir}")
        
    def save_figure(self, fig, name=None):
        if name is None: name = f"fig{self.figure_count}"
        filename = f"{name}.png"
        filepath = self.run_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.figure_count += 1
        return filepath
    
    def save_csv(self, df, name):
        filepath = self.run_dir / f"{name}.csv"
        df.to_csv(filepath, index=False)
        return filepath

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
    model.train() # Reset to train mode
    return z.cpu().clamp(0, 1)

def plot_single_config_samples(s16, s32, config_name, step, logger):
    """Plots samples for a single config immediately."""
    fig, axes = plt.subplots(2, 8, figsize=(16, 5))
    
    # 16px
    for i in range(8):
        axes[0, i].imshow(s16[i].permute(1, 2, 0))
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title(f"16px @ {step}", fontsize=10)
        
    # 32px
    for i in range(8):
        axes[1, i].imshow(s32[i].permute(1, 2, 0))
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title(f"32px @ {step}", fontsize=10)
        
    plt.suptitle(f"{config_name} - Step {step}")
    plt.tight_layout()
    logger.save_figure(fig, f"{config_name}_step_{step:05d}")

def plot_trajectory_grid(samples_dict, step, logger):
    """Plots samples for all active configs at a specific step."""
    configs = list(samples_dict.keys())
    resolutions = [16, 32]
    
    rows = len(configs)
    cols = len(resolutions) * 8 # 8 samples per res
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows * 1.5))
    if rows == 1: axes = axes.reshape(1, -1)
    
    for r, config in enumerate(configs):
        # 16px samples
        s16 = samples_dict[config][16]
        for i in range(8):
            ax = axes[r, i]
            ax.imshow(s16[i].permute(1, 2, 0))
            ax.axis('off')
            if i == 0: ax.set_title(f"{config} 16px @ {step}", fontsize=8, loc='left')
            
        # 32px samples
        s32 = samples_dict[config][32]
        for i in range(8):
            ax = axes[r, i + 8]
            ax.imshow(s32[i].permute(1, 2, 0))
            ax.axis('off')
            if i == 0: ax.set_title(f"{config} 32px @ {step}", fontsize=8, loc='left')
            
    plt.tight_layout()
    logger.save_figure(fig, f"combined_trajectory_step_{step:05d}")

# --- Training Logic ---

# Define the "Safe Base" that fits in 22.5GB
BASE_BUCKETS = [(16, 256), (32, 64)]

def train_config(config_name, model_kwargs, train_steps, batch_scale=1.0, sample_interval=1000, logger=None):
    device = torch.device('cuda')
    print(f"\n--- Training: {config_name} | Steps: {train_steps} | Batch Scale: {batch_scale:.2f} ---")
    print(f"    Params: {model_kwargs}")
    
    # 1. Scale Batches to avoid OOM
    # int(bs * scale) ensures we drop down from 64 to 32 if scale is 0.5
    scaled_buckets = [(res, max(1, int(bs * batch_scale))) for res, bs in BASE_BUCKETS]
    print(f"    Buckets: {scaled_buckets}")
    
    model = HybridGemmaDiT(mode='factorized', **model_kwargs).to(device)
    model = torch.compile(model)
    
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    
    iterator = CheckerboardIterator(device)
    manager = BucketManager(scaled_buckets)
    
    history = []
    samples_history = {} 
    
    pbar = tqdm(range(train_steps), desc=config_name)
    for i in pbar:
        opt.zero_grad()
        res, bs = manager.next_bucket()
        
        x0 = iterator.generate_batch(bs, res, num_tiles=4.0)
        t = torch.rand(bs, device=device).clamp(0.001, 0.999)
        logsnr = get_schedule(t)
        alpha, sigma = get_alpha_sigma(logsnr)
        
        eps = torch.randn_like(x0)
        z_t = x0 * alpha.view(-1,1,1,1) + eps * sigma.view(-1,1,1,1)
        v_true = alpha.view(-1,1,1,1) * eps - sigma.view(-1,1,1,1) * x0
        
        # Forward
        raw, l_pred, aux_loss = model(z_t, logsnr)
        
        sigma_p = torch.sqrt(torch.sigmoid(-l_pred)).view(-1, 1, 1, 1)
        v_pred = raw * sigma_p
        
        loss_diff = F.mse_loss(v_pred, v_true)
        # Scale aux loss by 0.0 to just log it effectively, or 0.01 if we actually want regularization
        loss = loss_diff + 0.0 * aux_loss 
        
        loss.backward()
        opt.step()
        
        history.append({'step': i, 'res': res, 'loss': loss.item()})
        
        if (i + 1) % sample_interval == 0:
            # Clean up VRAM before sampling
            torch.cuda.empty_cache() 
            # Force cleanup of matplotlib backends
            plt.close('all')
            
            print(f"\n[Sampling] Step {i+1}...")
            s16 = sample_viz(model, 16)
            s32 = sample_viz(model, 32)
            samples_history[i+1] = {16: s16, 32: s32}
            
            # --- Safety Save 1: Samples ---
            plot_single_config_samples(s16, s32, config_name, i+1, logger)
            
            # --- Safety Save 2: Metrics ---
            # Overwrite the CSV each time so it grows
            df_partial = pd.DataFrame(history)
            logger.save_csv(df_partial, f"{config_name}_history")
            
    return pd.DataFrame(history), samples_history

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    logger = ExperimentLogger(output_dir="./experiments_scaling")
    
    # --- Experimental Grid ---
    
    configs = {
        "Base_Long": {
            "kwargs": {"depth": 4, "interface_depth": 1, "ffn_type": "moe"},
            "steps": 6000,
            "batch_scale": 1.0 
        },
        "Deep_Backbone": {
            "kwargs": {"depth": 8, "interface_depth": 1, "ffn_type": "moe"},
            "steps": 3000,
            "batch_scale": 0.5  # Halve batch size for Double Depth
        },
        "Deep_Interface": {
            "kwargs": {"depth": 4, "interface_depth": 4, "ffn_type": "moe"},
            "steps": 3000,
            "batch_scale": 0.7  # Conservative reduction for MLP layers
        }
    }
    
    all_history = {}
    all_samples = {}
    
    for name, cfg in configs.items():
        # Aggressive GC between runs
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            df, samples = train_config(
                name, 
                cfg['kwargs'], 
                cfg['steps'], 
                batch_scale=cfg['batch_scale'],
                sample_interval=1000, 
                logger=logger
            )
            all_history[name] = df
            all_samples[name] = samples
        except Exception as e:
            print(f"❌ CRITICAL FAILURE in {name}: {e}")
            # If one fails, try to continue to the next
            continue
        
    # --- Visualization ---
    print("\n📈 Plotting Final Comparisons...")
    
    # Compare at step 3000 (Common denominator)
    comparison_3000 = {}
    for name in configs:
        if name in all_samples and 3000 in all_samples[name]:
            comparison_3000[name] = all_samples[name][3000]
            
    if comparison_3000:
        plot_trajectory_grid(comparison_3000, 3000, logger)
    
    # Loss Curves (Aggregated)
    if all_history:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = {'Base_Long': 'tab:blue', 'Deep_Backbone': 'tab:orange', 'Deep_Interface': 'tab:green'}
        
        for name, df in all_history.items():
            # 16px Loss
            d16 = df[df['res'] == 16]
            if not d16.empty:
                axes[0].plot(d16['step'], d16['loss'].rolling(50).mean(), label=name, color=colors.get(name, 'k'), alpha=0.8)
            
            # 32px Loss
            d32 = df[df['res'] == 32]
            if not d32.empty:
                axes[1].plot(d32['step'], d32['loss'].rolling(50).mean(), label=name, color=colors.get(name, 'k'), alpha=0.8)
            
        axes[0].set_title("16px Loss (Smoothed)")
        axes[0].set_yscale('log')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_title("32px Loss (Smoothed)")
        axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        logger.save_figure(fig, "scaling_loss_curves_final")
        
    print("✅ Experiment Batch Complete.")