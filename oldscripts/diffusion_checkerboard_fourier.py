"""
diffusion_checkerboard_fourier.py

Robust Edition with "Spin-Digit" (Fourier) Conditioning.
- Replaces raw scalar inputs with concatenated Fourier Features.
- Solves input scale divergence naturally (inputs are always [-1, 1]).
- Uses RMSNorm + ResNet structure for stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 1. Fourier "Spin-Digit" Encoder
# ============================================================================

class FourierFeatures(nn.Module):
    def __init__(self, num_bands=4, max_range=40.0):
        super().__init__()
        # We need the lowest freq to not wrap within the range [-20, 20].
        # Wavelength = max_range -> w = 2*pi / max_range
        base_freq = 2 * np.pi / max_range
        
        # Create frequencies: [w, 2w, 4w, 8w]
        freqs = base_freq * (2.0 ** torch.arange(num_bands))
        self.register_buffer('freqs', freqs) # [num_bands]

    def forward(self, x):
        # x: [B, 1] or [B]
        if x.dim() == 1: x = x.unsqueeze(-1)
        
        # x * freqs -> [B, num_bands]
        args = x * self.freqs
        
        # [sin, cos] -> [B, num_bands, 2] -> Flatten to [B, 2*num_bands]
        features = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return features

# ============================================================================
# 2. Architecture
# ============================================================================

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
        
        # 1. Fourier Encoder for Lambda (The Spin Digits)
        self.fourier_dim = 8 # 4 bands * 2 (sin/cos)
        self.spin_encoder = FourierFeatures(num_bands=4, max_range=50.0) # slightly > 40 margin
        
        # 2. Patch Input (2x2x3 = 12 floats)
        patch_dim = 12
        
        # 3. Concatenated Input Dim
        # We concat [Patch, Fourier] -> [12 + 8] = 20
        input_raw_dim = patch_dim + self.fourier_dim
        
        # 4. Embedding Layer
        self.embed = nn.Linear(input_raw_dim, embed_dim)
        self.patch_out = nn.Linear(embed_dim, patch_dim)
        
        # 5. Backbone
        self.start_norm = RMSNorm(embed_dim)
        self.block1 = SwiGLUBlock(embed_dim, embed_dim * 2)
        self.block2 = SwiGLUBlock(embed_dim, embed_dim * 2)
        
        # 6. Heads
        if mode == 'factorized':
            # Output: Latent(embed_dim) + Lambda_Scalar(1)
            self.head = nn.Linear(embed_dim, embed_dim + 1)
        else:
            self.head = nn.Linear(embed_dim, embed_dim)

    def forward(self, z_t, logsnr):
        B = z_t.shape[0]
        
        # 1. Patchify: [B, 3, 16, 16] -> [B, 64, 12]
        patches = z_t.unfold(2, 2, 2).unfold(3, 2, 2).permute(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(B, 64, 12)
        
        # 2. Spin Digits for Lambda: [B, 8]
        spins = self.spin_encoder(logsnr)
        
        # Broadcast spins to every patch: [B, 64, 8]
        spins = spins.unsqueeze(1).expand(-1, 64, -1)
        
        # 3. Concatenate: [B, 64, 20]
        raw_in = torch.cat([patches, spins], dim=2)
        
        # 4. Embed & Process
        h = self.embed(raw_in) # -> [B, 64, 128]
        h = self.start_norm(h)
        h = self.block1(h)
        h = self.block2(h)
        
        out = self.head(h)
        
        if self.mode == 'factorized':
            z_pred = out[:, :, :-1]
            l_pred = out[:, :, -1].mean(dim=1) # Mean pooling for scalar lambda
            
            # Unpatchify
            z_pred = self.patch_out(z_pred)
            z_pred = z_pred.view(B, 8, 8, 3, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, 3, 16, 16)
            return z_pred, l_pred
        else:
            z_pred = self.patch_out(out)
            z_pred = z_pred.view(B, 8, 8, 3, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, 3, 16, 16)
            return z_pred, None

# ============================================================================
# 3. Helpers (Data & Physics)
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

def get_schedule(t):
    return 20.0 - 40.0 * t

def get_alpha_sigma(logsnr):
    alpha = torch.sqrt(torch.sigmoid(logsnr))
    sigma = torch.sqrt(torch.sigmoid(-logsnr))
    return alpha, sigma

# ============================================================================
# 4. Training
# ============================================================================

def train(mode='factorized'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Training {mode} with Fourier Inputs ---")
    
    model = DiffusionModel(mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3) # Back to spicy LR!
    
    # History
    losses = []
    
    for i in range(1001):
        opt.zero_grad()
        
        # Data
        x0 = get_checkerboard_data(256, device)
        t = torch.rand(256, device=device)
        t = torch.clamp(t, 0.001, 0.999) # Avoid infinite targets
        
        logsnr = get_schedule(t)
        alpha, sigma = get_alpha_sigma(logsnr)
        
        eps = torch.randn_like(x0)
        z_t = x0 * alpha.view(-1,1,1,1) + eps * sigma.view(-1,1,1,1)
        
        # Forward
        raw_pred, l_pred = model(z_t, logsnr)
        
        # Loss
        if mode == 'naive':
            loss = F.mse_loss(raw_pred, eps)
        else:
            # Factorized Epsilon Recosntruction
            # eps = raw * sigma(lambda)
            sigma_p = torch.sqrt(torch.sigmoid(-l_pred)).view(-1, 1, 1, 1)
            eps_pred = raw_pred * sigma_p
            loss = F.mse_loss(eps_pred, eps)
            
        if torch.isnan(loss):
            print("NaN detected!")
            break
            
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        losses.append(loss.item())
        if i % 200 == 0:
            print(f"Step {i}: Loss {loss.item():.5f}")
            
    return losses

if __name__ == "__main__":
    l_naive = train('naive')
    l_fact = train('factorized')
    
    plt.figure(figsize=(10, 5))
    plt.plot(l_naive, label='Naive', alpha=0.7)
    plt.plot(l_fact, label='Factorized', alpha=0.7)
    plt.yscale('log')
    plt.title("Training Loss: Naive vs Factorized (Fourier Inputs)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()