#diffusion_utils.py

import torch
import math

def get_schedule(t):
    """Linear LogSNR Schedule: +20 to -20."""
    return 20.0 - 40.0 * t

def get_alpha_sigma(logsnr):
    """VP Schedule derived from LogSNR."""
    alpha = torch.sqrt(torch.sigmoid(logsnr))
    sigma = torch.sqrt(torch.sigmoid(-logsnr))
    return alpha, sigma

class FourierFeatures(torch.nn.Module):
    def __init__(self, num_bands=4, max_range=40.0):
        super().__init__()
        base_freq = 2 * math.pi / max_range
        freqs = base_freq * (2.0 ** torch.arange(num_bands))
        self.register_buffer('freqs', freqs)

    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(-1)
        args = x * self.freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)