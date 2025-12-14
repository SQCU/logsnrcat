# src/data.py - Fields, noise, ContextBlock, iterator
import torch
import torch.nn.functional as F
import math
import glob
import os
import random
from pathlib import Path
from .model import ContextBlock  # Import the canonical definition

# Try importing torchvision for video handling
# --- Dependency Check: Swap torchvision for torchcodec ---
try:
    from torchcodec.decoders import VideoDecoder
    HAS_TORCHCODEC = True
except ImportError:
    HAS_TORCHCODEC = False
    print("⚠️ torchcodec not found. Video iterator will fail if used.")

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
    if mode == "split":
        return generate_split_logsnr(B, H, W, device, **params)
    else:
        # Filter to only params that uniform accepts
        uniform_params = {k: v for k, v in params.items() if k in ("min_snr", "max_snr")}
        return generate_uniform_logsnr(B, H, W, device, **uniform_params)

# --- 1. Metaprogrammatic Wrapper Classes ---

class FrameBatchProxy:
    """
    Wraps a torchcodec FrameBatch. 
    Intercepts access to .data and moves it to the target device on the fly.
    """
    def __init__(self, real_batch, target_device):
        self._real_batch = real_batch
        self._target_device = target_device
        # Eagerly move the heavy data. 
        # Usually uint8 -> GPU transfer is very fast.
        self._gpu_data = real_batch.data.to(target_device)

    @property
    def data(self):
        return self._gpu_data

    def __getattr__(self, name):
        # Delegate everything else (pts_seconds, duration_seconds) to original
        return getattr(self._real_batch, name)

class DeviceCorrectingDecoder:
    """
    Wraps a VideoDecoder. 
    If a method returns something with a .data tensor (like FrameBatch),
    it wraps that result in a FrameBatchProxy to enforce device placement.
    """
    def __init__(self, real_decoder, target_device):
        self._decoder = real_decoder
        self._target_device = target_device

    def __getattr__(self, name):
        attr = getattr(self._decoder, name)
        
        # If it's just a property (like .metadata), return it directly
        if not callable(attr):
            return attr

        # If it's a method, wrap it
        def wrapper(*args, **kwargs):
            result = attr(*args, **kwargs)
            
            # Check if we got a FrameBatch-like object (has .data tensor)
            if hasattr(result, 'data') and isinstance(result.data, torch.Tensor):
                # The magic happens here: correct the device mismatch
                return FrameBatchProxy(result, self._target_device)
            
            return result
        return wrapper

    def __getitem__(self, key):
        # Handle decoder[idx] indexing if used
        result = self._decoder[key]
        if isinstance(result, torch.Tensor):
            return result.to(self._target_device)
        return result

# --- 2. The Smart Factory Function ---

def safe_create_decoder(video_path, device):
    """
    Creates a VideoDecoder. 
    If hardware decoding fails, falls back to CPU but returns a wrapper
    that auto-moves outputs to the originally requested device.
    """
    # Normalize device string/object
    if isinstance(device, str):
        device = torch.device(device)
        
    try:
        # 1. Happy Path: Try creating exactly what was asked for
        return VideoDecoder(str(video_path), device=device)
        
    except (ValueError, RuntimeError) as e:
        msg = str(e).lower()
        # Check specifically for the "unsupported device" error from torchcodec
        if "unsupported device" in msg and device.type == 'cuda':
            print(f"⚠️  NVDEC unavailable. Decoding {Path(video_path).name} on CPU -> Auto-streaming to {device}")
            
            # 2. Fallback: Create on CPU
            cpu_decoder = VideoDecoder(str(video_path), device='cpu')
            
            # 3. Metaprogramming: Wrap it so it acts like a CUDA decoder
            return DeviceCorrectingDecoder(cpu_decoder, device)
            
        raise e


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
        
    def generate_batch_list(self, batch_size, resolution, num_tiles=4.0, **kwargs):
        origins, rays = self.marcher.get_camera_rays(batch_size, resolution)
        p, mask = self.marcher.intersect(origins, rays)
        p_reshaped = p.view(batch_size, resolution, resolution, 3)
        images = self.marcher.shade_batch(p_reshaped, mask, rays, resolution).permute(0, 3, 1, 2)
        
        # Generate standard metadata
        blocks = []
        for i in range(batch_size):
            blocks.append(ContextBlock(
                content=images[i],
                type='latent',
                causal=True,
                group_id=i, # Independent
                id=f"torus_{i}"
            ))
        return blocks

class VideoFolderIterator:
    def __init__(self, folder_path, device='cuda'):
        if not HAS_TORCHCODEC: raise ImportError("torchcodec required")
        self.device = device
        self.folder = Path(folder_path)
        self.files = sorted(list(self.folder.glob("**/*.mp4")))
        if not self.files: raise ValueError(f"No videos in {folder_path}")

    def _sample_indices(self, total_frames, seq_len, sampler_config):
        min_pct = sampler_config.get('min_pct', 0.0)
        max_pct = sampler_config.get('max_pct', 1.0)
        stride = sampler_config.get('stride', None)
        if total_frames < seq_len: return [i % total_frames for i in range(seq_len)]
        
        min_span = max(seq_len, int(total_frames * min_pct))
        max_span = max(min_span, int(total_frames * max_pct))
        max_span = min(max_span, total_frames); min_span = min(min_span, max_span)
        actual_span = random.randint(min_span, max_span)
        max_start = total_frames - actual_span
        start_idx = random.randint(0, max_start)
        window_range = range(start_idx, start_idx + actual_span)
        
        if stride is not None:
            indices = [start_idx + i * stride for i in range(seq_len)]
            indices = [i % total_frames for i in indices]
        else:
            indices = sorted(random.sample(window_range, seq_len))
        return indices

    def generate_batch_list(self, batch_size, sequence_config, start_group_id=0):
        from concurrent.futures import ThreadPoolExecutor
        
        seq_len = len(sequence_config)
        chosen_files = random.choices(self.files, k=batch_size)
        sampler_config = sequence_config[0].get('time_sampler', {})

        # Define the worker function for a single video sequence
        def fetch_sequence(args):
            fpath, group_id = args
            blocks = []
            try:
                # 1. Open
                decoder = safe_create_decoder(fpath, self.device)
                total_frames = decoder.metadata.num_frames
                if total_frames is None: return []
                
                # 2. Plan
                indices = self._sample_indices(total_frames, seq_len, sampler_config)
                
                # 3. Decode (Heavy Lifting - happens in parallel now)
                # Note: safe_create_decoder ensures data lands on self.device (GPU)
                # Moving to GPU inside a thread is generally safe in PyTorch
                raw_batch = decoder.get_frames_at(indices).data.float() / 255.0
                
                # 4. Resize & Pack
                for t, (frame_raw, spec) in enumerate(zip(raw_batch, sequence_config)):
                    target_res = spec.get('res', 32)
                    if frame_raw.shape[-1] != target_res:
                        frame = F.interpolate(
                            frame_raw.unsqueeze(0), 
                            size=(target_res, target_res), 
                            mode='area'
                        ).squeeze(0)
                    else: 
                        frame = frame_raw
                    
                    n_mode = spec.get('noise_mode', 'uniform')
                    n_params = spec.get('noise_params', {})
                    # Optimization: Generate logsnr on device, no CPU trip needed
                    lsnr = get_logsnr_batch(n_mode, 1, target_res, target_res, self.device, n_params).squeeze(0)
                    
                    blocks.append(ContextBlock(
                        content=frame,
                        type='latent',
                        causal=True,
                        logsnr=lsnr,
                        group_id=group_id,
                        id=f"vid_{group_id}_{t}"
                    ))
                return blocks
            except Exception as e:
                # print(f"Fetch fail {fpath.name}: {e}") # Optional noise
                return []

        # Prepare arguments for parallel execution
        # We pre-assign group_ids to keep them distinct
        tasks = [(f, start_group_id + i) for i, f in enumerate(chosen_files)]
        
        all_blocks = []
        
        # ThreadPoolExecutor efficiently distributes the CPU-bound decoding tasks
        # max_workers=batch_size ensures we try to decode the whole batch at once
        with ThreadPoolExecutor(max_workers=min(batch_size, 16)) as executor:
            results = executor.map(fetch_sequence, tasks)
            
        for res in results:
            if res:
                all_blocks.extend(res)
                
        return all_blocks

class CheckerboardIterator:
    def __init__(self, device='cuda'): self.device = device
    def generate_batch_list(self, batch_size, resolution, num_tiles=4.0, **kwargs):
        # Generate batch of images
        # Logic copied from original CheckerboardIterator.generate_batch
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
        x_idx = torch.floor(x_rot + 0.01); y_idx = torch.floor(y_rot + 0.01)
        pat = ((x_idx + y_idx) % 2).view(batch_size, resolution, resolution)
        c1 = torch.rand(batch_size, 3, 1, 1, device=self.device)
        c2 = torch.rand(batch_size, 3, 1, 1, device=self.device)
        mask = pat.unsqueeze(1)
        imgs = c1 * (1 - mask) + c2 * mask # [B, 3, H, W]

        # Generate standard metadata
        blocks = []
        for i in range(batch_size):
            blocks.append(ContextBlock(
                content=imgs[i],
                type='latent',
                causal=True,
                group_id=i, # Independent
                id=f"checker_{i}"
            ))
        return blocks



# --- 3. Composite Iterator (Refactored) ---

class CompositeIterator:
    _ITERATOR_MAP = {
        'checkerboard': CheckerboardIterator,
        'torus': TorusIterator,
        'video': VideoFolderIterator # Register new type
    }

    def __init__(self, device='cuda', config=None):
        self.device = device
        if config is None: config = {'checkerboard': 1.0}
        
        self.splits = []
        
        for split_key, cfg in config.items():
            if isinstance(cfg, (float, int)): 
                cfg = {'ratio': float(cfg), 'type': split_key}
            
            gen_type = cfg.get('type', split_key)
            
            # Fallback only if type is genuinely unknown (prevents crash, but warns)
            if gen_type not in self._ITERATOR_MAP:
                print(f"⚠️ Unknown iterator type '{gen_type}', defaulting to Checkerboard.")
                iterator_cls = CheckerboardIterator
            else:
                iterator_cls = self._ITERATOR_MAP[gen_type]
            
            if gen_type == 'video':
                 path = cfg.get('params', {}).get('path', None)
                 iterator_instance = iterator_cls(path, device=device)
            else: 
                 iterator_instance = iterator_cls(device)
            
            self.splits.append({
                'name': split_key, 
                'type': gen_type, # <--- FIX: Store the type!
                'iterator': iterator_instance, 
                'ratio': cfg.get('ratio', 1.0), 
                'd_params': cfg.get('params', {}),
                'n_mode': cfg.get('noise_mode', 'uniform'), 
                'n_params': cfg.get('noise_params', {})
            })
            
        total = sum(s['ratio'] for s in self.splits)
        for s in self.splits: 
            s['ratio'] /= total
        self.label_map = {i: s['name'] for i, s in enumerate(self.splits)}

    def generate_batch_list(self, batch_size, **kwargs):
        counts = [int(batch_size * s['ratio']) for s in self.splits]
        remainder = batch_size - sum(counts)
        if counts: counts[0] += remainder
        
        all_blocks = []
        global_group_id = 0
        
        for idx, split in enumerate(self.splits):
            count = counts[idx]
            if count == 0: continue
            split_name = split['name'] # Capture the source name
            gen_type = split['type']
            if gen_type == 'video':
                seq_conf = split['d_params'].get('sequence_structure', [{'res':32}])
                blocks = split['iterator'].generate_batch_list(count, seq_conf, start_group_id=global_group_id)
                for b in blocks: b.source = split_name
                # Video iterator increments group_id internally, need to resync global
                if blocks: global_group_id = max(b.group_id for b in blocks) + 1
            else:
                d_params = {**kwargs, **split['d_params']}
                res = d_params.get('resolution', 32)
                blocks = split['iterator'].generate_batch_list(count, res, **d_params)
                
                # Apply LogSNR metadata (Video handles this internally, geometric does not)
                n_mode = split['n_mode']; n_params = split['n_params']
                raw_snrs = get_logsnr_batch(n_mode, len(blocks), res, res, self.device, n_params)
                
                for i, b in enumerate(blocks):
                    b.source = split_name
                    b.logsnr = raw_snrs[i]
                    b.group_id = global_group_id
                    global_group_id += 1
            
            all_blocks.extend(blocks)
        return all_blocks

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