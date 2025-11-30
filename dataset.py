#dataset.py

import torch
import math

class RotatedCheckerboardDataset:
    def __init__(self, batch_size, device='cuda'):
        self.batch_size = batch_size
        self.device = device
        
        # Pre-compute grid for speed
        linspace = torch.linspace(-8, 8, 16, device=device)
        self.y, self.x = torch.meshgrid(linspace, linspace, indexing='ij')
        self.x_flat = self.x.flatten().unsqueeze(0).expand(batch_size, -1)
        self.y_flat = self.y.flatten().unsqueeze(0).expand(batch_size, -1)

    def __iter__(self):
        return self

    def __next__(self):
        # Random Rotation
        theta = torch.rand(self.batch_size, 1, device=self.device) * 2 * math.pi
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        # Rotate Coordinates
        x_rot = self.x_flat * cos_t + self.y_flat * sin_t
        y_rot = -self.x_flat * sin_t + self.y_flat * cos_t
        
        # Hard Aliased Checkerboard Logic (Scale 4.0)
        scale = 4.0
        # Adding epsilon to avoid boundary flickering
        x_idx = torch.floor(x_rot / scale + 0.01)
        y_idx = torch.floor(y_rot / scale + 0.01)
        
        pat = ((x_idx + y_idx) % 2).view(self.batch_size, 16, 16)
        
        # Random Colors
        c1 = torch.rand(self.batch_size, 3, 1, 1, device=self.device)
        c2 = torch.rand(self.batch_size, 3, 1, 1, device=self.device)
        
        mask = pat.unsqueeze(1)
        img = c1 * (1 - mask) + c2 * mask
        return img