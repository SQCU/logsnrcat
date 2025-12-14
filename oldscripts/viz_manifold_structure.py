# viz_manifold_structure.py
import inductor_cas_client
inductor_cas_client.install_cas_client() 

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pathlib import Path
import math

from diffusion_utils import get_schedule, get_alpha_sigma, BucketManager
from dataset import CompositeIterator
from model import HybridGemmaDiT

class ExperimentLogger:
    def __init__(self, output_dir="experiments_manifold"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.run_id = len(list(self.output_dir.glob("run_*")))
        self.run_dir = self.output_dir / f"run_{self.run_id:03d}"
        self.run_dir.mkdir(exist_ok=True)
        print(f"📊 Manifold Probe | Run: {self.run_id} | Dir: {self.run_dir}")
        
    def save_figure(self, fig, name):
        filepath = self.run_dir / f"{name}.png"
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"   Saved {name}")

def slerp(val, low, high):
    """Spherical Linear Interpolation for latent walks."""
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + \
          (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res

@torch.no_grad()
def generate_walk(model, res, steps=10):
    """
    Generates a sequence of images interpolating between two random seeds.
    """
    model.eval()
    
    # Two distinct random seeds
    z1 = torch.randn(1, 3, res, res, device='cuda')
    z2 = torch.randn(1, 3, res, res, device='cuda')
    
    frames = []
    alphas = torch.linspace(0, 1, steps, device='cuda')
    
    for alpha in alphas:
        # Interpolate Latent
        # We use simple linear for batch ease, or proper slerp
        # For diffusion noise, Slerp is theoretically better to maintain variance
        # z_interp = (1 - alpha) * z1 + alpha * z2 # Linear (variance drops in middle)
        
        # Slerp implementation for 1D alpha
        omega = torch.acos(torch.sum(F.normalize(z1.flatten(), dim=0) * F.normalize(z2.flatten(), dim=0)))
        so = torch.sin(omega)
        if so == 0:
            z_t = z1
        else:
            w1 = torch.sin((1.0 - alpha) * omega) / so
            w2 = torch.sin(alpha * omega) / so
            z_t = w1 * z1 + w2 * z2
            
        # Denoise
        z = z_t.clone()
        # Use fewer steps for viz speed, but enough for structure
        ts = torch.linspace(1.0, 0.001, 50, device='cuda')
        
        for i in range(49):
            t = ts[i]; t_n = ts[i+1]
            logsnr = get_schedule(torch.full((1,), t, device='cuda'))
            raw, l_pred, _ = model(z, logsnr)
            
            if model.mode == 'factorized':
                sigma_p = torch.sqrt(torch.sigmoid(-l_pred)).view(-1,1,1,1)
                v_pred = raw * sigma_p
            else:
                v_pred = raw
                
            logsnr_n = get_schedule(torch.full((1,), t_n, device='cuda'))
            alpha_s, sigma_s = get_alpha_sigma(logsnr)
            x0 = alpha_s.view(-1,1,1,1)*z - sigma_s.view(-1,1,1,1)*v_pred
            eps = sigma_s.view(-1,1,1,1)*z + alpha_s.view(-1,1,1,1)*v_pred
            alpha_n, sigma_n = get_alpha_sigma(logsnr_n)
            z = alpha_n.view(-1,1,1,1)*x0 + sigma_n.view(-1,1,1,1)*eps
            
        frames.append(z.cpu().clamp(0, 1).squeeze(0))
        
    return frames

def train_and_probe(depth=8, steps=5000, logger=None):
    device = torch.device('cuda')
    print(f"\n--- Training Depth {depth} (Probe Mode) ---")
    
    model = HybridGemmaDiT('factorized', embed_dim=256, depth=depth).to(device)
    model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    
    # Add 128px to the mix to stress test the manifold
    buckets = [(16, 256), (32, 64), (64, 16), (128, 4)]
    manager = BucketManager(buckets)
    iterator = CompositeIterator(device, config={'checkerboard': 0.5, 'torus': 0.5})
    
    pbar = tqdm(range(steps))
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
        
        raw, l_pred, _ = model(z_t, logsnr)
        sigma_p = torch.sqrt(torch.sigmoid(-l_pred)).view(-1, 1, 1, 1)
        loss = F.mse_loss(raw * sigma_p, v_true)
        
        loss.backward()
        opt.step()
        
        if i % 100 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'res': res})
            
    return model

def plot_walks(model, resolutions=[32, 64, 128], logger=None):
    """
    Plots latent walks for multiple resolutions.
    """
    rows = len(resolutions)
    cols = 10 # Steps in walk
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    if rows == 1: axes = axes.reshape(1, -1)
    
    print("\n🚶 Generating Manifold Walks...")
    
    for r, res in enumerate(resolutions):
        # Generate 1 random walk per resolution
        frames = generate_walk(model, res, steps=cols)
        
        for c, frame in enumerate(frames):
            ax = axes[r, c]
            ax.imshow(frame.permute(1, 2, 0))
            ax.axis('off')
            if c == 0:
                ax.set_title(f"Start\n{res}px", fontsize=8, loc='left')
            if c == cols - 1:
                ax.set_title("End", fontsize=8, loc='right')
                
    plt.suptitle(f"Latent Space Interpolation (Depth {model.depth})", fontsize=16)
    plt.tight_layout()
    logger.save_figure(fig, "manifold_walk")

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    logger = ExperimentLogger()
    
    # Train the capable model
    # We push to 5000 steps to ensure 128px has a chance to converge
    model = train_and_probe(depth=8, steps=5000, logger=logger)
    
    # Visual Proof
    plot_walks(model, resolutions=[32, 64, 128], logger=logger)
    
    print(f"✅ Manifold visualization complete in {logger.run_dir}")