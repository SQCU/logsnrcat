"""
diffusion_checkerboard_vis.py

High-Throughput Edition (Batch Size 4096).
Includes Sampling Visualization to compare Naive vs Factorized outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 1. Fourier Spin-Digits & Architecture
# ============================================================================

class FourierFeatures(nn.Module):
    def __init__(self, num_bands=4, max_range=40.0):
        super().__init__()
        base_freq = 2 * np.pi / max_range
        freqs = base_freq * (2.0 ** torch.arange(num_bands))
        self.register_buffer('freqs', freqs)

    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(-1)
        args = x * self.freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.g

class SwiGLUBlock(nn.Module):
    def __init__(self, dim, hidden_dim, output_dim=None):
        super().__init__()
        if output_dim is None: output_dim = dim
        self.norm = RMSNorm(dim)
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, output_dim)
        
        with torch.no_grad():
            self.w3.weight.zero_()
            self.w3.bias.zero_()

    def forward(self, x):
        n = self.norm(x)
        h = F.silu(self.w1(n)) * self.w2(n)
        out = self.w3(h)
        return x + out if x.shape == out.shape else out

class DiffusionModel(nn.Module):
    def __init__(self, mode='factorized', embed_dim=128):
        super().__init__()
        self.mode = mode
        self.spin_encoder = FourierFeatures(num_bands=4, max_range=50.0)
        
        # Input: 12 (Patch) + 8 (Fourier) = 20
        self.embed = nn.Linear(20, embed_dim)
        self.patch_out = nn.Linear(embed_dim, 12) # 2x2x3
        
        self.start_norm = RMSNorm(embed_dim)
        self.block1 = SwiGLUBlock(embed_dim, embed_dim * 2)
        self.block2 = SwiGLUBlock(embed_dim, embed_dim * 2)
        
        if mode == 'factorized':
            self.head = nn.Linear(embed_dim, embed_dim + 1)
        else:
            self.head = nn.Linear(embed_dim, embed_dim)

    def forward(self, z_t, logsnr):
        B = z_t.shape[0]
        # Patchify [B, 3, 16, 16] -> [B, 64, 12]
        patches = z_t.unfold(2, 2, 2).unfold(3, 2, 2).permute(0, 2, 3, 1, 4, 5).reshape(B, 64, 12)
        
        # Fourier Features [B, 8] -> [B, 64, 8]
        spins = self.spin_encoder(logsnr).unsqueeze(1).expand(-1, 64, -1)
        
        # Concat & Process
        h = self.embed(torch.cat([patches, spins], dim=2))
        h = self.start_norm(h)
        h = self.block1(h)
        h = self.block2(h)
        out = self.head(h)
        
        # Decode
        if self.mode == 'factorized':
            z_pred = self.patch_out(out[:, :, :-1])
            l_pred = out[:, :, -1].mean(dim=1)
        else:
            z_pred = self.patch_out(out)
            l_pred = None
            
        # Unpatchify [B, 64, 12] -> [B, 3, 16, 16]
        z_pred = z_pred.view(B, 8, 8, 3, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, 3, 16, 16)
        return z_pred, l_pred

# ============================================================================
# 2. Helpers
# ============================================================================

def get_checkerboard_data(B, device='cuda'):
    x = torch.arange(16, device=device)
    y = torch.arange(16, device=device)
    gx, gy = torch.meshgrid(x, y, indexing='ij')
    pat = ((gx // 4) + (gy // 4)) % 2
    mode = torch.randint(0, 2, (B, 1, 1), device=device).float()
    pat = pat.unsqueeze(0).expand(B, -1, -1).float()
    img = torch.abs(pat - mode)
    return img.unsqueeze(1).repeat(1, 3, 1, 1)

def get_schedule(t): return 20.0 - 40.0 * t
def get_alpha_sigma(logsnr):
    return torch.sqrt(torch.sigmoid(logsnr)), torch.sqrt(torch.sigmoid(-logsnr))

# ============================================================================
# 3. Training & Sampling
# ============================================================================

def train_model(mode, batch_size=4096, steps=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training {mode.upper()} | Batch Size: {batch_size}...")
    
    model = DiffusionModel(mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    
    for i in range(steps):
        opt.zero_grad()
        
        x0 = get_checkerboard_data(batch_size, device)
        t = torch.rand(batch_size, device=device).clamp(0.001, 0.999)
        logsnr = get_schedule(t)
        alpha, sigma = get_alpha_sigma(logsnr)
        
        eps = torch.randn_like(x0)
        z_t = x0 * alpha.view(-1,1,1,1) + eps * sigma.view(-1,1,1,1)
        
        raw, l_pred = model(z_t, logsnr)
        
        if mode == 'naive':
            loss = F.mse_loss(raw, eps)
        else:
            # Reconstruct eps = raw * sigma(lambda)
            sigma_p = torch.sqrt(torch.sigmoid(-l_pred)).view(-1, 1, 1, 1)
            loss = F.mse_loss(raw * sigma_p, eps)
            
        loss.backward()
        opt.step()
        losses.append(loss.item())
        
        if i % 200 == 0:
            print(f"Step {i}: {loss.item():.5f}")
            
    return model, losses

@torch.no_grad()
def sample(model, mode, steps=50):
    device = next(model.parameters()).device
    model.eval()
    
    z = torch.randn(8, 3, 16, 16, device=device) # 8 Samples
    
    # Euler Schedule
    ts = torch.linspace(1.0, 0.001, steps, device=device)
    
    for i in range(len(ts) - 1):
        t_cur = ts[i]
        t_next = ts[i+1]
        
        # Current LogSNR
        logsnr = get_schedule(torch.full((8,), t_cur, device=device))
        alpha_c, sigma_c = get_alpha_sigma(logsnr)
        
        # Next LogSNR
        logsnr_next = get_schedule(torch.full((8,), t_next, device=device))
        _, sigma_n = get_alpha_sigma(logsnr_next)
        
        # Predict Epsilon
        raw, l_pred = model(z, logsnr)
        
        if mode == 'naive':
            eps = raw
        else:
            # Use predicted lambda for decoding
            sigma_p = torch.sqrt(torch.sigmoid(-l_pred)).view(-1, 1, 1, 1)
            eps = raw * sigma_p
            
        # Euler Step: z_{i+1} = z_i + (sigma_n - sigma_c) * eps
        # (d_i = eps)
        sigma_c = sigma_c.view(-1,1,1,1)
        sigma_n = sigma_n.view(-1,1,1,1)
        z = z + (sigma_n - sigma_c) * eps
        
    # Final cleanup (optional, naive decode)
    return z.clamp(0, 1).cpu()

if __name__ == "__main__":
    # Train
    model_n, loss_n = train_model('naive', batch_size=4096)
    model_f, loss_f = train_model('factorized', batch_size=4096)
    
    # Plot Loss
    plt.figure(figsize=(10, 4))
    plt.plot(loss_n, label='Naive', alpha=0.6)
    plt.plot(loss_f, label='Factorized', alpha=0.6)
    plt.yscale('log')
    plt.title("Training Loss (Batch Size 4096)")
    plt.legend()
    plt.show()
    
    # Generate Samples
    print("Sampling...")
    samples_n = sample(model_n, 'naive')
    samples_f = sample(model_f, 'factorized')
    
    # Visualize Samples
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        axes[0, i].imshow(samples_n[i].permute(1, 2, 0))
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Naive")
        
        axes[1, i].imshow(samples_f[i].permute(1, 2, 0))
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title("Factorized")
        
    plt.tight_layout()
    plt.show()