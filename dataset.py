# dataset.py
import torch
import math
import numpy as np

# Try importing the native torus, otherwise skip (allows standalone checkerboard usage)
try:
    from dataset_torus_native import TorusIterator
except ImportError:
    TorusIterator = None

class CheckerboardIterator:
    def __init__(self, device='cuda'):
        self.device = device
        
    def generate_batch(self, batch_size, resolution, num_tiles=4.0, **kwargs):
        """
        Args:
            num_tiles: Number of tiles across (fixed at 4 for 4x4 checkerboard usually)
        """
        # Scale tile size with resolution
        tile_scale = resolution / num_tiles 
        
        # Map to [-num_tiles/2, num_tiles/2]
        half_tiles = num_tiles / 2.0
        linspace = torch.linspace(-half_tiles, half_tiles, resolution, device=self.device)
        y, x = torch.meshgrid(linspace, linspace, indexing='ij')
        
        x_flat = x.flatten().unsqueeze(0).expand(batch_size, -1)
        y_flat = y.flatten().unsqueeze(0).expand(batch_size, -1)
        
        # Random Rotation
        theta = torch.rand(batch_size, 1, device=self.device) * 2 * math.pi
        cos_t = torch.cos(theta); sin_t = torch.sin(theta)
        
        x_rot = x_flat * cos_t + y_flat * sin_t
        y_rot = -x_flat * sin_t + y_flat * cos_t
        
        # Checker logic
        x_idx = torch.floor(x_rot + 0.01)
        y_idx = torch.floor(y_rot + 0.01)
        
        pat = ((x_idx + y_idx) % 2).view(batch_size, resolution, resolution)
        
        # Random Colors
        c1 = torch.rand(batch_size, 3, 1, 1, device=self.device)
        c2 = torch.rand(batch_size, 3, 1, 1, device=self.device)
        mask = pat.unsqueeze(1)
        
        return c1 * (1 - mask) + c2 * mask

class CompositeIterator:
    def __init__(self, device='cuda', config=None):
        """
        A unified iterator that mixes samples from multiple sources.
        
        config example:
        {
            'checkerboard': 0.5, 
            'torus': 0.5
        }
        OR with params:
        {
            'checkerboard': {'ratio': 0.5, 'params': {'num_tiles': 4.0}},
            'torus': {'ratio': 0.5}
        }
        """
        self.device = device
        
        # Default to legacy behavior (100% Checkerboard)
        if config is None:
            config = {'checkerboard': 1.0}
            
        self.sources = []
        self.ratios = []
        self.params = []
        self.iterators = {}
        
        # Initialize sources
        if 'checkerboard' in config:
            self.iterators['checkerboard'] = CheckerboardIterator(device)
            self._parse_config('checkerboard', config['checkerboard'])
            
        if 'torus' in config:
            if TorusIterator is None:
                raise ImportError("TorusIterator not found. Check dataset_torus_native.py")
            self.iterators['torus'] = TorusIterator(device)
            self._parse_config('torus', config['torus'])
            
        # Normalize ratios to sum to 1.0
        total_ratio = sum(self.ratios)
        self.ratios = [r / total_ratio for r in self.ratios]
        
        # Metadata storage (Stateful trick for drop-in compatibility)
        self.last_labels = None
        self.label_map = {i: name for i, name in enumerate(self.sources)}

    def _parse_config(self, name, cfg):
        self.sources.append(name)
        if isinstance(cfg, (float, int)):
            self.ratios.append(float(cfg))
            self.params.append({})
        elif isinstance(cfg, dict):
            self.ratios.append(cfg.get('ratio', 1.0))
            self.params.append(cfg.get('params', {}))
        else:
            raise ValueError(f"Invalid config for {name}")

    def generate_batch(self, batch_size, resolution, **kwargs):
        """
        Generates a mixed batch. 
        Metadata regarding which sample came from where is stored in self.last_labels.
        """
        # 1. Determine split
        counts = [int(batch_size * r) for r in self.ratios]
        # Fix rounding errors by dumping remainder into the first source
        counts[0] += batch_size - sum(counts)
        
        batch_parts = []
        labels_parts = []
        
        # 2. Generate sub-batches
        for idx, (name, count) in enumerate(zip(self.sources, counts)):
            if count == 0: continue
            
            iterator = self.iterators[name]
            specific_params = self.params[idx]
            
            # Merge defaults (kwargs) with specific params, specific overrides default
            # e.g. kwargs has num_tiles=4.0, specific might have num_tiles=8.0
            call_kwargs = {**kwargs, **specific_params}
            
            # Generate
            imgs = iterator.generate_batch(count, resolution, **call_kwargs)
            batch_parts.append(imgs)
            
            # Create labels
            labels_parts.append(torch.full((count,), idx, device=self.device, dtype=torch.long))
            
        # 3. Concatenate
        full_batch = torch.cat(batch_parts, dim=0)
        full_labels = torch.cat(labels_parts, dim=0)
        
        # 4. Shuffle (Important for batch statistics and diverse minibatches)
        perm = torch.randperm(batch_size, device=self.device)
        full_batch = full_batch[perm]
        
        # Store metadata statefully
        self.last_labels = full_labels[perm]
        
        return full_batch

# Helper to easily visualize/debug the new iterator
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    
    # Test Config
    mix_config = {
        'checkerboard': 0.5,
        'torus': 0.5
    }
    
    print("Testing CompositeIterator...")
    iterator = CompositeIterator(device='cuda', config=mix_config)
    
    # Generate
    batch_res = 64
    batch_bs = 8
    images = iterator.generate_batch(batch_bs, batch_res, num_tiles=4.0)
    labels = iterator.last_labels
    
    print(f"Generated {images.shape}")
    print(f"Labels: {labels.cpu().numpy()}")
    print(f"Map: {iterator.label_map}")
    
    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    imgs_np = images.permute(0, 2, 3, 1).cpu().numpy()
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs_np[i])
        lbl_idx = labels[i].item()
        name = iterator.label_map[lbl_idx]
        ax.set_title(name)
        ax.axis('off')
        
    os.makedirs("test_mix", exist_ok=True)
    plt.savefig("test_mix/composite_debug.png")
    print("Saved debug image.")