"""
check_dataset.py
Verifying the ground truth geometry of the rotated checkerboards.
"""
import torch
import matplotlib.pyplot as plt
import math

def generate_rotated_checkerboards_debug(B, device='cpu'):
    # Grid [-8, 8]
    linspace = torch.linspace(-8, 8, 16, device=device)
    y, x = torch.meshgrid(linspace, linspace, indexing='ij')
    
    x_flat = x.flatten().unsqueeze(0).expand(B, -1)
    y_flat = y.flatten().unsqueeze(0).expand(B, -1)
    
    # 0, 30, 45, 60, etc degrees
    theta = torch.linspace(0, math.pi, B, device=device).unsqueeze(1)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    
    x_rot = x_flat * cos_t + y_flat * sin_t
    y_rot = -x_flat * sin_t + y_flat * cos_t
    
    # Scale 4.0 means 4 pixels wide per tile
    scale = 4.0
    x_idx = torch.floor(x_rot / scale + 0.01)
    y_idx = torch.floor(y_rot / scale + 0.01)
    
    pat = ((x_idx + y_idx) % 2).view(B, 16, 16)
    return pat

if __name__ == "__main__":
    # Generate 8 angles
    patterns = generate_rotated_checkerboards_debug(8)
    
    fig, axes = plt.subplots(1, 8, figsize=(16, 2))
    for i in range(8):
        axes[i].imshow(patterns[i], cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f"{i * 180/8:.1f}°")
        axes[i].axis('off')
    
    plt.suptitle("Ground Truth Dataset: 16x16 Grid, Scale=4.0 (4px Tiles)")
    plt.tight_layout()
    plt.show()