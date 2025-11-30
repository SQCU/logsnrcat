import torch
import math

class CheckerboardIterator:
    def __init__(self, device='cuda'):
        self.device = device
        
    def generate_batch(self, batch_size, resolution, num_tiles=4):
        """
        Args:
            num_tiles: Number of tiles across (fixed at 4 for 4×4 checkerboard)
        """
        # Scale tile size with resolution to maintain constant tile count
        tile_scale = resolution / num_tiles  # 16/4=4.0, 32/4=8.0
        
        # Map to [-num_tiles/2, num_tiles/2] in tile-space
        half_tiles = num_tiles / 2.0
        linspace = torch.linspace(-half_tiles, half_tiles, resolution, device=self.device)
        y, x = torch.meshgrid(linspace, linspace, indexing='ij')
        
        x_flat = x.flatten().unsqueeze(0).expand(batch_size, -1)
        y_flat = y.flatten().unsqueeze(0).expand(batch_size, -1)
        
        theta = torch.rand(batch_size, 1, device=self.device) * 2 * math.pi
        cos_t = torch.cos(theta); sin_t = torch.sin(theta)
        
        x_rot = x_flat * cos_t + y_flat * sin_t
        y_rot = -x_flat * sin_t + y_flat * cos_t
        
        # Now tile_scale is in tile-space units (1.0 = one tile)
        x_idx = torch.floor(x_rot + 0.01)
        y_idx = torch.floor(y_rot + 0.01)
        
        pat = ((x_idx + y_idx) % 2).view(batch_size, resolution, resolution)
        
        c1 = torch.rand(batch_size, 3, 1, 1, device=self.device)
        c2 = torch.rand(batch_size, 3, 1, 1, device=self.device)
        mask = pat.unsqueeze(1)
        
        return c1 * (1 - mask) + c2 * mask