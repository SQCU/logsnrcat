"""
diffusion_checkerboard_lab.py

Experiment: 
Training a 2-layer SwiGLU FFN on a Bimodal Checkerboard Dataset.
Comparing Naive vs. Factorized Passthrough parameterizations.
Using "Transfusion-style" Linear Patch Encoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

# ============================================================================
# 1. The Dataset (CIELAB Checkerboards)
# ============================================================================

def lab_to_rgb_approx(L, a, b):
    """
    Approximate Lab to RGB conversion for tensor batches.
    Not strictly color-science accurate, but preserves perceptual contrast.
    """
    # Simple transform to LMS then RGB (Rough approximation)
    y = (L + 16) / 116.0
    x = a / 500.0 + y
    z = y - b / 200.0
    
    r = 2.69*x - 1.28*y - 0.41*z
    g = -1.21*x + 2.18*y + 0.04*z
    b_val = 0.06*x - 0.12*y + 1.06*z
    
    rgb = torch.stack([r, g, b_val], dim=1)
    return torch.clamp(rgb, 0, 1)

def generate_checkerboard_batch(batch_size: int, device='cuda') -> torch.Tensor:
    """
    Generates 16x16 checkerboards (4x4 tiles).
    Two modes: Phase A (0 deg) and Phase B (90 deg).
    Colors: Random hue pairs with fixed high saturation/lightness.
    """
    # 1. Base Grid (16x16)
    # 4x4 tiles means each tile is 4 pixels wide.
    x = torch.arange(16, device=device)
    y = torch.arange(16, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    
    # Checker pattern: (x // 4 + y // 4) % 2
    pattern = ((grid_x // 4) + (grid_y // 4)) % 2  # [16, 16]
    
    # 2. Random Colors in Lab Space
    # L=70 (Bright), Distance in ab plane
    # We pick a random angle for color 1, and angle+pi for color 2
    theta = torch.rand(batch_size, device=device) * 2 * np.pi
    
    L = 70.0
    radius = 60.0 # Saturation
    
    a1 = radius * torch.cos(theta)
    b1 = radius * torch.sin(theta)
    
    a2 = radius * torch.cos(theta + np.pi)
    b2 = radius * torch.sin(theta + np.pi)
    
    c1 = lab_to_rgb_approx(torch.full_like(theta, L), a1, b1) # [B, 3]
    c2 = lab_to_rgb_approx(torch.full_like(theta, L), a2, b2) # [B, 3]
    
    # 3. Apply Modes
    # Half batch gets Phase A, Half gets Phase B (inverted pattern)
    mode = torch.randint(0, 2, (batch_size, 1, 1), device=device).float()
    
    # Broadcast pattern
    pat = pattern.unsqueeze(0).expand(batch_size, -1, -1) # [B, 16, 16]
    
    # Flip pattern based on mode
    # Fixed broadcasting error here: [B, 16, 16] - [B, 1, 1] works.
    final_pat = torch.abs(pat - mode) # [B, 16, 16]
    
    # Composite
    # img = c1 * (1 - pat) + c2 * pat
    final_pat = final_pat.unsqueeze(1) # [B, 1, 16, 16]
    c1 = c1.view(batch_size, 3, 1, 1)
    c2 = c2.view(batch_size, 3, 1, 1)
    
    img = c1 * (1 - final_pat) + c2 * final_pat
    return img

# ============================================================================
# 2. Physics (Schedule)
# ============================================================================

def get_logsnr_schedule(t: torch.Tensor) -> torch.Tensor:
    return 20.0 - 40.0 * t

def logsnr_to_alpha_sigma(logsnr: torch.Tensor):
    alpha = torch.sqrt(torch.sigmoid(logsnr))
    sigma = torch.sqrt(torch.sigmoid(-logsnr))
    return alpha, sigma

# ============================================================================
# 3. Architecture (SwiGLU + Patching)
# ============================================================================

class LinearPatchEncoder(nn.Module):
    def __init__(self, patch_size=2, in_chans=3, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_chans * patch_size * patch_size, embed_dim)
        self.unproj = nn.Linear(embed_dim, in_chans * patch_size * patch_size)
        
    def forward(self, x):
        # x: [B, 3, 16, 16] -> [B, 64, 128]
        B, C, H, W = x.shape
        # Patchify
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, -1, C * self.patch_size * self.patch_size)
        return self.proj(x)
        
    def decode(self, x):
        # x: [B, 64, 128] -> [B, 3, 16, 16]
        B, L, D = x.shape
        x = self.unproj(x)
        H = W = int(np.sqrt(L)) * self.patch_size
        
        x = x.view(B, int(H/self.patch_size), int(W/self.patch_size), 3, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, 3, H, W)
        return x

class SwiGLUMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
            
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(input_dim, hidden_dim)
        # Fix: w3 now projects to output_dim, not forced back to input_dim
        self.w3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class DiffusionModel(nn.Module):
    def __init__(self, mode='factorized', embed_dim=128):
        super().__init__()
        self.mode = mode
        self.patcher = LinearPatchEncoder(embed_dim=embed_dim)
        
        # 8x8 patches = 64 tokens. 
        # Flat Vector Input = 64 * 128 = 8192.
        self.flat_dim = 64 * embed_dim
        
        # Passthrough: +1 channel concatenated to vector
        input_dim = self.flat_dim + 1 # +1 for LogSNR
        
        if mode == 'factorized':
            output_dim = self.flat_dim + 1 # Predict [Latent, Lambda]
        else:
            output_dim = self.flat_dim # Predict [Latent]
            
        # 2-Layer SwiGLU
        # Layer 1: Input -> 2*Input -> Input
        # Layer 2: Input -> 2*Input -> Output
        self.net = nn.Sequential(
            SwiGLUMLP(input_dim, input_dim * 2, input_dim),
            SwiGLUMLP(input_dim, input_dim * 2, output_dim)
        )
        
        # Zero init final layer for stability
        with torch.no_grad():
            self.net[1].w3.weight.zero_()
            self.net[1].w3.bias.zero_()

    def forward(self, z_t, logsnr):
        B = z_t.shape[0]
        
        # 1. Flatten Latents
        z_flat = z_t.view(B, -1) # [B, 8192]
        
        # 2. Concat Lambda
        l_in = logsnr.view(B, 1)
        x_in = torch.cat([z_flat, l_in], dim=1)
        
        # 3. MLP
        out = self.net(x_in)
        
        if self.mode == 'factorized':
            z_pred_flat = out[:, :-1]
            l_pred = out[:, -1]
            return z_pred_flat.view_as(z_t), l_pred
        else:
            return out.view_as(z_t), None

# ============================================================================
# 4. Training Loop
# ============================================================================

def train_and_sample(mode='factorized', steps=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training {mode} model on {device}...")
    
    model = DiffusionModel(mode=mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    loss_history = []
    
    for i in range(steps):
        opt.zero_grad()
        
        # 1. Data
        x0_img = generate_checkerboard_batch(1024, device)
        
        # 2. Encode to Latent
        with torch.no_grad():
            x0_latent = model.patcher(x0_img) # [B, 64, 128]
            
        # 3. Noise
        t = torch.rand(1024, device=device)
        logsnr = get_logsnr_schedule(t)
        alpha, sigma = logsnr_to_alpha_sigma(logsnr)
        
        # Broadcast to latent shape
        alpha_lat = alpha.view(-1, 1, 1)
        sigma_lat = sigma.view(-1, 1, 1)
        
        eps = torch.randn_like(x0_latent)
        z_t = alpha_lat * x0_latent + sigma_lat * eps
        
        # 4. Forward
        pred_raw, l_pred = model(z_t, logsnr)
        
        # 5. Loss Calculation
        if mode == 'naive':
            # Target: Epsilon
            loss = F.mse_loss(pred_raw, eps)
        else:
            # Target: Factorized Epsilon (eps / sigma)
            # Passthrough Reconstruction: eps_pred = raw * sigma(l_pred)
            sigma_pred = torch.sqrt(torch.sigmoid(-l_pred)).view(-1, 1, 1)
            eps_pred = pred_raw * sigma_pred
            
            loss = F.mse_loss(eps_pred, eps)
            
        loss.backward()
        opt.step()
        loss_history.append(loss.item())
        
        if i % 100 == 0:
            print(f"Step {i}: Loss {loss.item():.6f}")

    return model, loss_history

# ============================================================================
# 5. Sampling & Viz
# ============================================================================

@torch.no_grad()
def sample_viz(model, mode):
    device = next(model.parameters()).device
    model.eval()
    
    # Start from noise
    z = torch.randn(4, 64, 128, device=device)
    
    # Simple Euler Ancestral Sampler
    timesteps = torch.linspace(1, 0, 50, device=device)
    
    for i in range(len(timesteps) - 1):
        t_cur = timesteps[i]
        t_next = timesteps[i+1]
        
        logsnr_cur = get_logsnr_schedule(torch.full((4,), t_cur, device=device))
        logsnr_next = get_logsnr_schedule(torch.full((4,), t_next, device=device))
        
        alpha_c, sigma_c = logsnr_to_alpha_sigma(logsnr_cur)
        alpha_n, sigma_n = logsnr_to_alpha_sigma(logsnr_next)
        
        # Model Prediction
        pred_raw, l_pred = model(z, logsnr_cur)
        
        # Recover Epsilon
        if mode == 'naive':
            eps_pred = pred_raw
        else:
            # Use PREDICTED sigma for decoding
            sigma_p = torch.sqrt(torch.sigmoid(-l_pred)).view(-1, 1, 1)
            eps_pred = pred_raw * sigma_p
            
        # Denoise Step (Euler)
        sigma_c = sigma_c.view(-1, 1, 1)
        sigma_n = sigma_n.view(-1, 1, 1)
        
        d = eps_pred
        z = z + (sigma_n - sigma_c) * d
        
    # Decode
    imgs = model.patcher.decode(z)
    return imgs.permute(0, 2, 3, 1).cpu().clamp(0, 1)

if __name__ == "__main__":
    # Run Experiment
    model_naive, loss_naive = train_and_sample('naive')
    model_fact, loss_fact = train_and_sample('factorized')
    
    # Viz
    imgs_naive = sample_viz(model_naive, 'naive')
    imgs_fact = sample_viz(model_fact, 'factorized')
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    for i in range(4):
        axes[0,i].imshow(imgs_naive[i])
        axes[0,i].axis('off')
        if i==0: axes[0,i].set_title("Naive")
        
        axes[1,i].imshow(imgs_fact[i])
        axes[1,i].axis('off')
        if i==0: axes[1,i].set_title("Factorized")
        
    plt.tight_layout()
    plt.show()
    
    # Plot Loss
    plt.figure()
    plt.plot(loss_naive, label='Naive')
    plt.plot(loss_fact, label='Factorized')
    plt.yscale('log')
    plt.legend()
    plt.title("Training Loss")
    plt.show()