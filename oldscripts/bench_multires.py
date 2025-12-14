# bench_multires.py
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from diffusion_utils import get_schedule, get_alpha_sigma, BucketManager
from dataset import CheckerboardIterator
from model import HybridGemmaDiT
from sampler import sample_euler_v

def train_multires(mode, steps=1000):
    device = torch.device('cuda')
    print(f"\n--- Running Multi-Res V-Prediction: {mode.upper()} ---")
    
    # Init Model
    #embed_dim=256, depth=8 defaults
    model = HybridGemmaDiT(mode, embed_dim=256, depth=12).to(device) # hehe , dtype=torch.float16
    model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    
    # Init Data
    iterator = CheckerboardIterator(device)
    
    # Buckets: (Resolution, BatchSize)
    # 16x16: 64 tokens. Batch 1024. -> 65k tokens/step
    # 32x32: 256 tokens. Batch 256. -> 65k tokens/step (Matched compute)
    buckets = [(16, 1024), (32, 256)]
    manager = BucketManager(buckets)
    
    history = []
    
    pbar = tqdm(range(steps))
    for i in pbar:
        opt.zero_grad()
        
        # 1. Get Bucket
        res, bs = manager.next_bucket()
        
        # 2. Generate Data (Tile Scale fixed at 4.0 for consistent features)
        x0 = iterator.generate_batch(bs, res, tile_scale=4.0)
        
        t = torch.rand(bs, device=device).clamp(0.001, 0.999)
        logsnr = get_schedule(t)
        alpha, sigma = get_alpha_sigma(logsnr)
        alpha, sigma = alpha.view(-1,1,1,1), sigma.view(-1,1,1,1)
        
        eps = torch.randn_like(x0)
        z_t = x0 * alpha + eps * sigma
        v_true = alpha * eps - sigma * x0
        
        # 3. Forward
        raw, l_pred = model(z_t, logsnr)
        
        # 4. Reconstruction
        if mode == 'factorized':
            sigma_p = torch.sqrt(torch.sigmoid(-l_pred)).view(-1, 1, 1, 1)
            v_pred = raw * sigma_p
        else:
            v_pred = raw
            
        loss = F.mse_loss(v_pred, v_true)
        loss.backward()
        opt.step()
        
        # Log per resolution
        history.append({
            'step': i,
            'res': res,
            'loss': loss.item()
        })
        
        if i % 10 == 0:
            pbar.set_description(f"Res {res}: {loss.item():.4f}")
            
    return pd.DataFrame(history), model

@torch.no_grad()
def sample_viz(model, res):
    model.eval()
    z = torch.randn(8, 3, res, res, device='cuda')
    ts = torch.linspace(1.0, 0.001, 50, device='cuda')
    
    for i in range(49):
        t = ts[i]; t_n = ts[i+1]
        logsnr = get_schedule(torch.full((8,), t, device='cuda'))
        
        # Forward
        raw, l_pred = model(z, logsnr)
        
        # Reconstruction Logic
        if model.mode == 'factorized':
            sigma_p = torch.sqrt(torch.sigmoid(-l_pred)).view(-1,1,1,1)
            v_pred = raw * sigma_p
        else:
            v_pred = raw
        
        # Euler Step
        logsnr_n = get_schedule(torch.full((8,), t_n, device='cuda'))
        _, sigma_c = get_alpha_sigma(logsnr)
        _, sigma_n = get_alpha_sigma(logsnr_n)
        
        # v-pred formulation: z_{next} = alpha_{next}*x0 + sigma_{next}*eps
        # x0 = alpha*z - sigma*v
        # eps = sigma*z + alpha*v
        
        alpha, sigma = get_alpha_sigma(logsnr)
        alpha, sigma = alpha.view(-1,1,1,1), sigma.view(-1,1,1,1)
        x0 = alpha * z - sigma * v_pred
        eps = sigma * z + alpha * v_pred
        
        z = get_alpha_sigma(logsnr_n)[0].view(-1,1,1,1) * x0 + sigma_n.view(-1,1,1,1) * eps
        
    return z.cpu().clamp(0, 1)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    
    # 1. Run Benchmarks
    df_naive, model_naive = train_multires('naive', steps=3000)
    df_fact, model_fact = train_multires('factorized', steps=3000)
    
    # 2. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot Res 16
    n16 = df_naive[df_naive['res'] == 16]
    f16 = df_fact[df_fact['res'] == 16]
    axes[0].plot(n16['step'], n16['loss'], label='Naive', alpha=0.7)
    axes[0].plot(f16['step'], f16['loss'], label='Factorized', alpha=0.7)
    axes[0].set_title("Resolution 16x16 Loss")
    axes[0].set_yscale('log')
    axes[0].legend()
    
    # Plot Res 32
    n32 = df_naive[df_naive['res'] == 32]
    f32 = df_fact[df_fact['res'] == 32]
    axes[1].plot(n32['step'], n32['loss'], label='Naive', alpha=0.7)
    axes[1].plot(f32['step'], f32['loss'], label='Factorized', alpha=0.7)
    axes[1].set_title("Resolution 32x32 Loss")
    axes[1].set_yscale('log')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # 3. Visual Comparison
    print("Generating Comparative Samples...")
    
    n_s16 = sample_viz(model_naive, 16)
    n_s32 = sample_viz(model_naive, 32)
    f_s16 = sample_viz(model_fact, 16)
    f_s32 = sample_viz(model_fact, 32)
    
    # Grid: Top=Naive, Bottom=Factorized. Left=16, Right=32
    fig, axes = plt.subplots(4, 8, figsize=(16, 9))
    
    # Row 0: Naive 16
    for i in range(8):
        axes[0,i].imshow(n_s16[i].permute(1,2,0)); axes[0,i].axis('off')
        if i==0: axes[0,i].set_title("Naive (16px)")
        
    # Row 1: Factorized 16
    for i in range(8):
        axes[1,i].imshow(f_s16[i].permute(1,2,0)); axes[1,i].axis('off')
        if i==0: axes[1,i].set_title("Factorized (16px)")
        
    # Row 2: Naive 32
    for i in range(8):
        axes[2,i].imshow(n_s32[i].permute(1,2,0)); axes[2,i].axis('off')
        if i==0: axes[2,i].set_title("Naive (32px)")
        
    # Row 3: Factorized 32
    for i in range(8):
        axes[3,i].imshow(f_s32[i].permute(1,2,0)); axes[3,i].axis('off')
        if i==0: axes[3,i].set_title("Factorized (32px)")
        
    plt.tight_layout()
    plt.show()