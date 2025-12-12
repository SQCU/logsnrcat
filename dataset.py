import torch
import torch.nn.functional as F
import math
import numpy as np

# --- 1. Noise Topology Generators ---

def generate_split_logsnr(B, H, W, device, min_snr=-4.0, max_snr=1.0, angle_range_deg=30.0, jitter_pct=0.05):
    """
    Generates a batch of logSNR maps with a randomized split-screen topology.
    Equation: x*cos(theta) + y*sin(theta) > offset
    """
    # 1. Randomized Hyperplanes
    angle_rad = angle_range_deg * math.pi / 180.0
    theta = (torch.rand(B, device=device) * 2 - 1) * angle_rad 
    
    # Offset: +/- jitter_pct of min dimension
    offset_scale = jitter_pct * min(H, W)
    offset = (torch.rand(B, device=device) * 2 - 1) * offset_scale
    
    # Create coordinate grid [B, H, W]
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device), 
        torch.arange(W, device=device), 
        indexing='ij'
    )
    # Center coordinates
    y_c = y_grid - H / 2.0
    x_c = x_grid - W / 2.0
    
    # Broadcast params to [B, H, W]
    theta = theta.view(B, 1, 1)
    offset = offset.view(B, 1, 1)
    
    # 2. Generate Masks
    projection = x_c * torch.cos(theta) + y_c * torch.sin(theta)
    mask_high = (projection > offset).float().unsqueeze(1) # [B, 1, H, W]
    
    # 3. Sample Noise Levels
    # Side A: Standard training range (Noisy Side)
    snr_low = torch.rand(B, device=device) * (max_snr - min_snr) + min_snr
    
    # Side B: Halfway between Side A and Max (Clean-ish Side)
    snr_high_target = max_snr 
    snr_high = snr_low + 0.5 * (snr_high_target - snr_low)
    
    # Expand to maps
    snr_low_map = snr_low.view(B, 1, 1, 1).expand(B, 1, H, W)
    snr_high_map = snr_high.view(B, 1, 1, 1).expand(B, 1, H, W)
    
    # 4. Composite
    logsnr_map = mask_high * snr_high_map + (1.0 - mask_high) * snr_low_map
    
    return logsnr_map

def generate_uniform_logsnr(B, H, W, device, min_snr=-4.0, max_snr=1.0):
    """
    Standard scalar noise level broadcast to spatial map.
    """
    snr = torch.rand(B, device=device) * (max_snr - min_snr) + min_snr
    return snr.view(B, 1, 1, 1).expand(B, 1, H, W)

def get_logsnr_batch(mode, B, H, W, device, params):
    """Dispatcher for noise topology."""
    if mode == 'split':
        return generate_split_logsnr(B, H, W, device, **params)
    else:
        return generate_uniform_logsnr(B, H, W, device, **params)


# --- 2. Geometric Datasets ---

class TorusRaymarcher:
    def __init__(self, device='cuda'):
        self.device = device
        self.R_major = 1.0
        self.r_minor = 0.4

    def get_camera_rays(self, batch_size, resolution):
        i, j = torch.meshgrid(
            torch.linspace(-1, 1, resolution, device=self.device),
            torch.linspace(-1, 1, resolution, device=self.device),
            indexing='ij'
        ) 
        dirs_cam = torch.stack([j, -i, torch.ones_like(i)], dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)
        dirs_cam = F.normalize(dirs_cam, dim=-1)
        
        in_hole = torch.rand(batch_size, device=self.device) < 0.3
        rho_hole = torch.rand(batch_size, device=self.device) * 0.5 
        rho_ext = torch.rand(batch_size, device=self.device) * 2.0 + 1.5
        rho = torch.where(in_hole, rho_hole, rho_ext)
        y_cam = (torch.rand(batch_size, device=self.device) * 4.0 - 2.0)
        phi_cam = torch.rand(batch_size, device=self.device) * 2 * math.pi
        
        cx = rho * torch.cos(phi_cam)
        cy = y_cam
        cz = rho * torch.sin(phi_cam)
        origin = torch.stack([cx, cy, cz], dim=1) 
        
        target_jitter = (torch.rand(batch_size, device=self.device) - 0.5) * 0.5
        theta_tgt = phi_cam + target_jitter
        tx = self.R_major * torch.cos(theta_tgt)
        ty = torch.zeros_like(tx) + (torch.rand(batch_size, device=self.device) - 0.5) * 0.2
        tz = self.R_major * torch.sin(theta_tgt)
        target = torch.stack([tx, ty, tz], dim=1)
        
        forward = F.normalize(target - origin, dim=1)
        world_up = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(batch_size, -1)
        right = F.normalize(torch.cross(forward, world_up, dim=1), dim=1)
        up = torch.cross(right, forward, dim=1)
        R = torch.stack([right, up, forward], dim=2) 
        
        rays_world = torch.bmm(dirs_cam.view(batch_size, -1, 3), R).view(batch_size, resolution, resolution, 3)
        return origin, rays_world

    def sdf_torus(self, p):
        q_xy = torch.norm(p[..., [0, 2]], dim=-1) - self.R_major
        q_z = p[..., 1]
        d = torch.sqrt(q_xy**2 + q_z**2) - self.r_minor
        return d

    def intersect(self, origin, rays, max_steps=64):
        B, H, W, _ = rays.shape
        o = origin.view(B, 1, 3).expand(-1, H*W, -1).reshape(-1, 3)
        d = rays.view(-1, 3)
        
        t = torch.zeros(o.shape[0], device=self.device)
        active_mask = torch.ones_like(t, dtype=torch.bool)
        
        for _ in range(max_steps):
            if not active_mask.any(): break
            p = o + d * t.unsqueeze(-1)
            dist = self.sdf_torus(p)
            t = torch.where(active_mask, t + dist, t)
            hit = dist < 0.001
            miss = t > 6.0 
            active_mask = active_mask & (~hit) & (~miss)
        
        p_final = o + d * t.unsqueeze(-1)
        final_dist = self.sdf_torus(p_final)
        valid_mask = final_dist < 0.01
        return p_final, valid_mask.view(B, H, W)

    def get_uv(self, p):
        u = torch.atan2(p[..., 2], p[..., 0]) / (2 * math.pi) + 0.5
        q_xy = torch.norm(p[..., [0, 2]], dim=-1) - self.R_major
        q_z = p[..., 1]
        v = torch.atan2(q_z, q_xy) / (2 * math.pi) + 0.5
        return u, v

    def lab_to_rgb(self, L, a, b):
        y = (L + 16.) / 116.
        x = a / 500. + y
        z = y - b / 200.
        func = lambda t: torch.where(t > 0.2068966, t ** 3, (t - 16. / 116.) / 7.787)
        X, Y, Z = 0.95047 * func(x), 1.00000 * func(y), 1.08883 * func(z)
        R = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
        G = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
        B = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
        rgb = torch.stack([R, G, B], dim=-1)
        rgb = torch.where(rgb > 0.0031308, 1.055 * (torch.abs(rgb) ** (1/2.4)) - 0.055, 12.92 * rgb)
        return torch.clamp(rgb, 0, 1)

    def shade_batch(self, p, valid_mask, rays, resolution):
        B = valid_mask.shape[0]
        H = W = resolution
        
        L_bright = torch.rand(B, 1, device=self.device) * 20.0 + 70.0 
        c_bright = self.lab_to_rgb(L_bright, torch.zeros_like(L_bright), torch.zeros_like(L_bright))
        L_dark = torch.rand(B, 1, device=self.device) * 30.0 + 20.0
        hue = torch.rand(B, 1, device=self.device) * 2 * math.pi
        chroma = torch.rand(B, 1, device=self.device) * 50.0 + 60.0
        a_dark = chroma * torch.cos(hue)
        b_dark = chroma * torch.sin(hue)
        c_dark = self.lab_to_rgb(L_dark, a_dark, b_dark)
        
        u, v = self.get_uv(p)
        freq_u = torch.randint(8, 16, (B, 1, 1), device=self.device).float()
        freq_v = torch.randint(4, 8, (B, 1, 1), device=self.device).float()
        u_scaled = u * freq_u
        v_scaled = v * freq_v
        col_idx = torch.floor(u_scaled)
        stripe_freq = 2.0 * math.pi
        pat_even = torch.cos((u_scaled + v_scaled) * stripe_freq)
        pat_odd  = torch.cos((u_scaled - v_scaled) * stripe_freq)
        is_even = (col_idx % 2 == 0)
        pattern = torch.where(is_even, pat_even, pat_odd)
        is_dark = pattern > 0
        
        c_bright_exp = c_bright.view(B, 1, 1, 3).expand(-1, H, W, -1)
        c_dark_exp = c_dark.view(B, 1, 1, 3).expand(-1, H, W, -1)
        surface_color = torch.where(is_dark.unsqueeze(-1), c_dark_exp, c_bright_exp)
        
        cp = p.clone()
        p_xz_norm = F.normalize(p[..., [0, 2]], dim=-1)
        cp[..., 0] = p_xz_norm[..., 0] * self.R_major
        cp[..., 1] = 0
        cp[..., 2] = p_xz_norm[..., 1] * self.R_major
        normal = F.normalize(p - cp, dim=-1)
        light_dir = F.normalize(torch.tensor([0.5, 1.0, 0.5], device=self.device), dim=0).view(1, 1, 1, 3)
        diffuse = torch.sum(normal * light_dir, dim=-1, keepdim=True).clamp(0.2, 1.0)
        surface_lit = surface_color * diffuse

        bg_base = torch.rand(B, 1, 1, 1, device=self.device) * 0.03 + 0.01
        bg_freq = torch.rand(B, 1, 1, 1, device=self.device) * 2.0 + 3.0
        d = rays
        bg_pat_val = torch.sin(d[..., 0] * bg_freq.squeeze(-1) * math.pi) * \
                     torch.sin(d[..., 1] * bg_freq.squeeze(-1) * math.pi)
        bg_mod = 1.0 + (bg_pat_val.unsqueeze(-1) * 0.3)
        bg_color = (bg_base * bg_mod).clamp(0.0, 1.0)
        
        final_img = torch.where(valid_mask.unsqueeze(-1), surface_lit, bg_color)
        return final_img

class TorusIterator:
    def __init__(self, device='cuda'):
        self.marcher = TorusRaymarcher(device)
        
    def generate_batch(self, batch_size, resolution, **kwargs):
        origins, rays = self.marcher.get_camera_rays(batch_size, resolution)
        p, mask = self.marcher.intersect(origins, rays)
        p_reshaped = p.view(batch_size, resolution, resolution, 3)
        images = self.marcher.shade_batch(p_reshaped, mask, rays, resolution)
        return images.permute(0, 3, 1, 2)

class CheckerboardIterator:
    def __init__(self, device='cuda'):
        self.device = device
        
    def generate_batch(self, batch_size, resolution, num_tiles=4.0, **kwargs):
        tile_scale = resolution / num_tiles 
        half_tiles = num_tiles / 2.0
        linspace = torch.linspace(-half_tiles, half_tiles, resolution, device=self.device)
        y, x = torch.meshgrid(linspace, linspace, indexing='ij')
        
        x_flat = x.flatten().unsqueeze(0).expand(batch_size, -1)
        y_flat = y.flatten().unsqueeze(0).expand(batch_size, -1)
        
        theta = torch.rand(batch_size, 1, device=self.device) * 2 * math.pi
        cos_t = torch.cos(theta); sin_t = torch.sin(theta)
        x_rot = x_flat * cos_t + y_flat * sin_t
        y_rot = -x_flat * sin_t + y_flat * cos_t
        
        x_idx = torch.floor(x_rot + 0.01)
        y_idx = torch.floor(y_rot + 0.01)
        
        pat = ((x_idx + y_idx) % 2).view(batch_size, resolution, resolution)
        
        c1 = torch.rand(batch_size, 3, 1, 1, device=self.device)
        c2 = torch.rand(batch_size, 3, 1, 1, device=self.device)
        mask = pat.unsqueeze(1)
        
        return c1 * (1 - mask) + c2 * mask

# --- 3. Composite Iterator (Refactored) ---

class CompositeIterator:
    _ITERATOR_MAP = {
        'checkerboard': CheckerboardIterator,
        'torus': TorusIterator
    }

    def __init__(self, device='cuda', config=None):
        """
        Advanced config structure:
        {
            'my_split_name_1': {
                'type': 'checkerboard',  # Underlying generator
                'ratio': 0.5,
                'params': {'num_tiles': 4.0},
                'noise_mode': 'uniform',
                'noise_params': {'max_snr': 5.0}
            },
            'my_split_name_2': {
                'type': 'torus',
                'ratio': 0.5,
                ...
            }
        }
        Legacy (shorthand) support:
        { 'checkerboard': 0.5, 'torus': 0.5 }
        """
        self.device = device
        if config is None: config = {'checkerboard': 1.0}
        
        self.splits = [] # List of dicts: {name, iterator, ratio, d_params, n_mode, n_params}
        
        # 1. Parse Config
        for split_key, cfg in config.items():
            
            # Handle shorthand: 'checkerboard': 0.5
            if isinstance(cfg, (float, int)):
                cfg = {'ratio': float(cfg), 'type': split_key}
            
            # Identify generator type
            # If 'type' is missing, assume the split key is the type (legacy behavior)
            gen_type = cfg.get('type', split_key)
            
            if gen_type not in self._ITERATOR_MAP:
                raise ValueError(f"Unknown generator type '{gen_type}' in split '{split_key}'. Available: {list(self._ITERATOR_MAP.keys())}")
            
            # Create Iterator Instance
            # We instantiate fresh for every split to keep parameter injection clean
            iterator_cls = self._ITERATOR_MAP[gen_type]
            iterator_instance = iterator_cls(device)
            
            self.splits.append({
                'name': split_key,
                'iterator': iterator_instance,
                'ratio': cfg.get('ratio', 1.0),
                'd_params': cfg.get('params', {}),
                'n_mode': cfg.get('noise_mode', 'uniform'),
                'n_params': cfg.get('noise_params', {})
            })
            
        # 2. Normalize Ratios
        total_ratio = sum(s['ratio'] for s in self.splits)
        if total_ratio <= 0: raise ValueError("Total ratio must be positive.")
        
        # Normalize in place
        for s in self.splits:
            s['ratio'] /= total_ratio
            
        self.last_labels = None
        # Map label index -> split name
        self.label_map = {i: s['name'] for i, s in enumerate(self.splits)}

    def generate_batch(self, batch_size, resolution, **kwargs):
        """
        Returns:
            images: [B, 3, H, W] (Clean x0)
            logsnr: [B, 1, H, W] (Topology-aware noise map)
        """
        counts = [int(batch_size * s['ratio']) for s in self.splits]
        
        # Dump remainder into first non-zero split
        remainder = batch_size - sum(counts)
        for i in range(len(counts)):
            if counts[i] > 0 or (i == len(counts)-1): # Ensure we dump somewhere
                counts[i] += remainder
                break
        
        batch_imgs = []
        batch_snr = []
        labels_parts = []
        
        for idx, split in enumerate(self.splits):
            count = counts[idx]
            if count == 0: continue
            
            # 1. Generate Images
            # Merge global defaults (kwargs) with split-specific params
            # Split params override global kwargs
            d_params = {**kwargs, **split['d_params']}
            
            imgs = split['iterator'].generate_batch(count, resolution, **d_params)
            batch_imgs.append(imgs)
            
            # 2. Generate Noise Map
            n_mode = split['n_mode']
            n_params = split['n_params']
            snr_map = get_logsnr_batch(n_mode, count, resolution, resolution, self.device, n_params)
            batch_snr.append(snr_map)
            
            # 3. Labels
            labels_parts.append(torch.full((count,), idx, device=self.device, dtype=torch.long))
            
        # 3. Concatenate & Shuffle
        full_imgs = torch.cat(batch_imgs, dim=0)
        full_snr = torch.cat(batch_snr, dim=0)
        full_labels = torch.cat(labels_parts, dim=0)
        
        perm = torch.randperm(batch_size, device=self.device)
        full_imgs = full_imgs[perm]
        full_snr = full_snr[perm]
        self.last_labels = full_labels[perm]
        
        return full_imgs, full_snr

# --- 4. Debug/Visualization ---

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    
    # Advanced Config Demonstration: Multi-modal splits!
    mix_config = {
        'checkerboard_easy': {
            'type': 'checkerboard',
            'ratio': 0.3,
            'noise_mode': 'uniform',
            'noise_params': {'min_snr': -2.0, 'max_snr': 2.0} # Clean-ish
        },
        'checkerboard_hplane': {
            'type': 'checkerboard',
            'ratio': 0.2,
            'noise_mode': 'split',
            'noise_params': {
                'min_snr': -6.0,   
                'max_snr': 6.0,
                'angle_range_deg': 15.0 # Mild angles
            }
        },
        'torus_hard': {
            'type': 'torus',
            'ratio': 0.5,
            'noise_mode': 'split', 
            'noise_params': {
                'min_snr': -4.0, 
                'max_snr': 1.0,
                'angle_range_deg': 45.0 # Wild angles
            }
        }
    }
    
    print("Testing CompositeIterator with Split-Based Architecture...")
    try:
        iterator = CompositeIterator(device='cuda', config=mix_config)
        
        batch_res = 64
        batch_bs = 10
        
        # Iterator now returns (x0, logsnr)
        images, logsnrs = iterator.generate_batch(batch_bs, batch_res, num_tiles=4.0)
        labels = iterator.last_labels
        
        print(f"Generated Images: {images.shape}, LogSNR: {logsnrs.shape}")
        
        # Visualization
        fig, axes = plt.subplots(batch_bs, 2, figsize=(6, 3 * batch_bs))
        if batch_bs == 1: axes = axes.reshape(1, -1)
        
        imgs_np = images.permute(0, 2, 3, 1).cpu().numpy()
        snr_np = logsnrs.squeeze(1).cpu().numpy()
        
        for i in range(batch_bs):
            # Col 0: Image
            lbl_idx = labels[i].item()
            split_name = iterator.label_map[lbl_idx]
            
            axes[i, 0].imshow(imgs_np[i])
            axes[i, 0].set_title(f"{split_name}\n(x0)")
            axes[i, 0].axis('off')
            
            # Col 1: LogSNR Map
            s_min, s_max = snr_np[i].min(), snr_np[i].max()
            axes[i, 1].imshow(snr_np[i], cmap='viridis')
            axes[i, 1].set_title(f"LogSNR\n[{s_min:.2f}, {s_max:.2f}]")
            axes[i, 1].axis('off')
            
        os.makedirs("test_mix", exist_ok=True)
        plt.tight_layout()
        plt.savefig("test_mix/composite_split_debug.png")
        print("Saved debug visualization to test_mix/composite_split_debug.png")
        
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()