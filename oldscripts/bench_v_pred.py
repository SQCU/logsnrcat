# bench_v_pred.py
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from diffusion_utils import get_schedule, get_alpha_sigma
from dataset import RotatedCheckerboardDataset
from model import HybridGemmaDiT
from sampler import sample_euler_v

def compute_kurtosis(tensor):
    x = tensor.detach().flatten().float()
    if x.std() < 1e-9: return 0.0
    return (((x - x.mean()) / x.std()) ** 4).mean().item()

def get_grad_stats(model):
    """Aggregates gradient stats across all trainable parameters."""
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.flatten())
    
    if not grads: return 0, 0, 0
    
    all_grads = torch.cat(grads)
    norm = all_grads.norm().item()
    var = all_grads.var().item()
    kurt = compute_kurtosis(all_grads)
    return norm, var, kurt

def train_and_probe(mode, steps=1000, batch_size=512, probe_interval=100):
    device = torch.device('cuda')
    print(f"\n--- Running V-Prediction Bench: {mode.upper()} ---")
    
    model = HybridGemmaDiT(mode, depth=4).to(device) # Slightly smaller for speed
    # Note: Skipping compile for probing to ensure graph/grad access works predictably
    model = torch.compile(model) 
    
    #opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    dataset = RotatedCheckerboardDataset(batch_size, device)
    
    householder_params = [p for n,p in model.named_parameters() if 'orthogonal.vs' in n]
    other_params = [p for n,p in model.named_parameters() if 'orthogonal.vs' not in n]
    # Option B: Don't schedule Householder
    opt_main = torch.optim.AdamW(other_params, lr=5e-4, weight_decay=0.1)
    opt_house = torch.optim.AdamW(householder_params, lr=0.1, weight_decay=0.0)
    # Learning rate schedule with warmup + decay
    from torch.optim.lr_scheduler import OneCycleLR
    scheduler_main = OneCycleLR(opt_main, max_lr=1e-3, total_steps=steps, 
                        pct_start=0.1, div_factor=10, final_div_factor=100)
    scheduler_house = OneCycleLR(opt_house, max_lr=1e-2, total_steps=steps, 
                        pct_start=0.1, div_factor=10, final_div_factor=100)
        

    history = []
    
    for i, x0 in enumerate(tqdm(dataset)):
        if i >= steps: break
        
        opt_main.zero_grad()
        opt_house.zero_grad()
        t = torch.rand(batch_size, device=device).clamp(0.001, 0.999)
        logsnr = get_schedule(t)
        alpha, sigma = get_alpha_sigma(logsnr)
        alpha, sigma = alpha.view(-1,1,1,1), sigma.view(-1,1,1,1)
        
        eps = torch.randn_like(x0)
        z_t = x0 * alpha + eps * sigma
        v_true = alpha * eps - sigma * x0
        
        # Forward
        raw, l_pred = model(z_t, logsnr)
        
        # Reconstruction & Loss
        if mode == 'factorized':
            # v = raw * sigma(lambda)
            # This tests the "Factorized V" hypothesis: v_norm = v / sigma
            sigma_p = torch.sqrt(torch.sigmoid(-l_pred)).view(-1, 1, 1, 1)
            v_pred = raw * sigma_p
        else:
            v_pred = raw
            
        loss = F.mse_loss(v_pred, v_true)
        loss.backward()
        
        # PROBE
        if i % probe_interval == 0:
            norm, var, kurt = get_grad_stats(model)
            history.append({
                'step': i,
                'loss': loss.item(),
                'grad_norm': norm,
                'grad_var': var,
                'grad_kurt': kurt
            })
            
        opt_main.step()
        opt_house.step()
        scheduler_main.step()
        scheduler_house.step()
        
    return pd.DataFrame(history), model

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    
    # 1. Run Benchmarks
    df_naive, model_naive = train_and_probe('naive')
    df_fact, model_fact = train_and_probe('factorized')
    
    # 2. Visualize Stats
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Loss
    axes[0].plot(df_naive['step'], df_naive['loss'], label='Naive')
    axes[0].plot(df_fact['step'], df_fact['loss'], label='Factorized')
    axes[0].set_title('Loss')
    axes[0].set_yscale('log')
    axes[0].legend()
    
    # Gradient Norm
    axes[1].plot(df_naive['step'], df_naive['grad_norm'], label='Naive')
    axes[1].plot(df_fact['step'], df_fact['grad_norm'], label='Factorized')
    axes[1].set_title('Gradient Norm')
    axes[1].set_yscale('log')
    
    # Gradient Variance
    axes[2].plot(df_naive['step'], df_naive['grad_var'], label='Naive')
    axes[2].plot(df_fact['step'], df_fact['grad_var'], label='Factorized')
    axes[2].set_title('Gradient Variance')
    axes[2].set_yscale('log')
    
    # Gradient Kurtosis
    axes[3].plot(df_naive['step'], df_naive['grad_kurt'], label='Naive')
    axes[3].plot(df_fact['step'], df_fact['grad_kurt'], label='Factorized')
    axes[3].set_title('Gradient Kurtosis')
    
    plt.tight_layout()
    plt.show()
    
    # 3. Sample Comparison
    print("Generating Samples...")
    s_naive = sample_euler_v(model_naive)
    s_fact = sample_euler_v(model_fact)
    
    fig, ax = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        ax[0,i].imshow(s_naive[i].permute(1,2,0)); ax[0,i].axis('off')
        ax[1,i].imshow(s_fact[i].permute(1,2,0)); ax[1,i].axis('off')
    ax[0,0].set_title("Naive V-Pred")
    ax[1,0].set_title("Factorized V-Pred")
    plt.show()