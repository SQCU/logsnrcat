# src/data.py - Fields, noise, ContextBlock, iterator
import torch
import torch.nn.functional as F
import math
import glob
import os
import random
from pathlib import Path
from .model import ContextBlock  # Import the canonical definition

# lookahead iterator
import time
import random
import threading
import queue
from collections import defaultdict

# Try importing torchvision for video handling
# --- Dependency Check: Swap torchvision for torchcodec ---
try:
    from torchcodec.decoders import VideoDecoder
    HAS_TORCHCODEC = True
except ImportError:
    HAS_TORCHCODEC = False
    print("⚠️ torchcodec not found. Video iterator will fail if used.")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️ psutil not found. RAM safety checks disabled.")
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
                shape_meta=(resolution, resolution),  # Explicit: (H, W)
                group_id=i,
                id=f"torus_{i}"
            ))
        return blocks


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
                shape_meta=(resolution, resolution),  # Explicit: (H, W)
                group_id=i,
                id=f"checker_{i}"
            ))
        return blocks



class RawSequenceBatch:
    """Buffer object. Frames are now small (resized) uint8 tensors."""
    def __init__(self, small_frames, configs, group_id, req_idx):
        self.small_frames = small_frames # [L, C, Res, Res] uint8, CPU
        self.configs = configs
        self.group_id = group_id
        self.req_idx = req_idx

class VideoFolderIterator:
    def __init__(self, folder_path, device='cuda', horizon=256, result_queue_depth=1024, num_workers=None, max_ram_pct=95.0, target_dtype=torch.float32, caching_resolution=128):
        if not HAS_TORCHCODEC: raise ImportError("torchcodec required")
        self.device = device
        self.target_dtype = target_dtype
        self.folder = Path(folder_path)
        self.files = sorted(list(self.folder.glob("**/*.mp4")))
        if not self.files: raise ValueError(f"No videos in {folder_path}")
        
        self.num_workers = num_workers if num_workers else max(4, (os.cpu_count()//2) - 1)
        self.max_ram_pct = max_ram_pct
        
        # NEW: Fixed resolution for the worker queue.
        # All frames in the queue will be this size.
        # Downstream batches will interpolate FROM this size TO their target size.
        self.caching_resolution = caching_resolution
        
        print(f"🚀 Video Iterator: {self.num_workers} threads, RAM Limit: {max_ram_pct}%, Cache Res: {caching_resolution}px")
        
        self.horizon = horizon
        self.stop_event = threading.Event()
        
        # Queues
        self.job_queue = queue.Queue(maxsize=horizon * 2)
        # Note: We can increase result queue depth now because items are small!
        self.result_queue = queue.Queue(maxsize=result_queue_depth)
        
        self.lock = threading.Lock()
        self.global_group_id = 0
        self.current_seq_config = None
        
        # Start Threads
        self.planner_thread = threading.Thread(target=self._planner_loop, daemon=True)
        self.planner_thread.start()
        
        self.workers = []
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            t.start()
            self.workers.append(t)

    def _check_memory_pressure(self):
        """Returns True if we should pause due to RAM usage."""
        if HAS_PSUTIL:
            return psutil.virtual_memory().percent > self.max_ram_pct
        return False

    def _sample_indices(self, total_frames, seq_len, sampler_config):
        min_pct = sampler_config['min_pct']
        max_pct = sampler_config['max_pct']
        stride = sampler_config['stride']
        
        if total_frames <= seq_len: return [i % total_frames for i in range(seq_len)]
        start_min = int(total_frames * min_pct)
        start_max = int(total_frames * max_pct) - seq_len
        if start_max <= start_min: start_max = start_min + 1
        
        start_idx = random.randint(start_min, max(start_min, start_max))
        
        if stride:
            indices = [(start_idx + i * stride) % total_frames for i in range(seq_len)]
        else:
            indices = [(start_idx + i) % total_frames for i in range(seq_len)]
        return indices

    def _planner_loop(self):
        # Planner logic remains mostly same, but we don't need to stress about 
        # the specific resolution in the config, just the sequence length/sampler params.
        while not self.stop_event.is_set():
            if self._check_memory_pressure():
                time.sleep(1.0)
                continue

            if self.result_queue.qsize() > (self.result_queue.maxsize - self.horizon):
                time.sleep(0.05)
                continue
            
            if self.current_seq_config is None:
                time.sleep(0.1)
                continue

            if self.job_queue.full():
                time.sleep(0.01)
                continue

            # We use the current config mainly for the Time Sampler settings
            seq_config = self.current_seq_config
            sampler = seq_config[0]['time_sampler']
            seq_len = len(seq_config)
            
            batch_files = random.choices(self.files, k=self.horizon)
            
            with self.lock:
                start_gid = self.global_group_id
                self.global_group_id += self.horizon
            
            jobs_by_file = defaultdict(list)
            for i, fpath in enumerate(batch_files):
                req_idx = i 
                gid = start_gid + i
                jobs_by_file[fpath].append((req_idx, gid))
            
            for fpath, requests in jobs_by_file.items():
                try:
                    # Pass the sequence length, not the full config, to keep workers simple
                    self.job_queue.put((fpath, requests, seq_len, sampler), timeout=1.0)
                except queue.Full:
                    pass

    def _worker_loop(self, worker_id):
        while not self.stop_event.is_set():
            if self._check_memory_pressure():
                time.sleep(1.0)
                continue

            try:
                job = self.job_queue.get(timeout=1.0)
                fpath, requests, seq_len, sampler = job # unpacked
            except queue.Empty:
                continue

            try:
                decoder = VideoDecoder(str(fpath), device='cpu')
                total_frames = decoder.metadata.num_frames
                if not total_frames: 
                    self.job_queue.task_done()
                    continue

                req_to_idxs = {}
                all_idxs = set()
                
                for req_idx, gid in requests:
                    idxs = self._sample_indices(total_frames, seq_len, sampler)
                    req_to_idxs[req_idx] = idxs
                    all_idxs.update(idxs)
                
                if not all_idxs:
                    self.job_queue.task_done()
                    continue

                sorted_idxs = sorted(list(all_idxs))
                heavy_frames = decoder.get_frames_at(sorted_idxs).data 
                
                # --- DECOUPLED RESIZING ---
                # Always resize to self.caching_resolution
                # This ensures the queue is uniform and buckets don't invalidate work.
                
                # 1. Float Cast
                hf_float = heavy_frames.float()
                
                # 2. Resize to Caching Resolution
                sf_float = F.interpolate(
                    hf_float, 
                    size=(self.caching_resolution, self.caching_resolution), 
                    mode='area' 
                )
                
                # 3. Cast back to uint8
                small_frames_batch = sf_float.to(torch.uint8)
                
                del heavy_frames, hf_float, sf_float

                # Map back to indices
                idx_to_small_tensor = {}
                for i, idx in enumerate(sorted_idxs):
                    idx_to_small_tensor[idx] = small_frames_batch[i]
                
                # Assemble Requests
                for req_idx, gid in requests:
                    needed = req_to_idxs[req_idx]
                    seq_stack = torch.stack([idx_to_small_tensor[x] for x in needed])
                    
                    # Note: We pass None for configs here, main thread handles that
                    result = RawSequenceBatch(seq_stack, None, gid, req_idx)
                    self.result_queue.put(result)
                    
            except Exception as e:
                pass
            finally:
                self.job_queue.task_done()

    def generate_batch_list(self, batch_size, sequence_config, start_group_id=0):
        # Update current config so planner knows what to schedule (sequence length)
        self.current_seq_config = sequence_config
            
        out_blocks = []
        fetched = 0
        
        while fetched < batch_size:
            try:
                # Wait longer if queue is empty to avoid crashing loop
                batch_raw = self.result_queue.get(timeout=15.0)
            except queue.Empty:
                print("⚠️ Video Queue Empty! Workers lagging.")
                break
                
            # 1. GPU Transfer (Cached Res -> GPU)
            # frames_gpu is [Seq_Len, 3, Cache_Res, Cache_Res]
            frames_gpu = batch_raw.small_frames.to(device=self.device, dtype=self.target_dtype, non_blocking=True) / 255.0
            
            seq_blocks = []
            
            # 2. Sequence Assembly & Final Resize
            for t, (frame, spec) in enumerate(zip(frames_gpu, sequence_config)):
                target_res = spec['res']

                # Resize if the cached resolution != target resolution
                if frame.shape[-1] != target_res:
                    # Note: We use bilinear here as it's faster on GPU and we are likely
                    # going from 128 -> 32 or 64. If going 128 -> 256, bilinear is also fine.
                    frame = F.interpolate(
                        frame.unsqueeze(0),
                        size=(target_res, target_res),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)

                n_mode = spec['noise_mode']
                n_params = spec['noise_params']
                lsnr = get_logsnr_batch(n_mode, 1, target_res, target_res, self.device, n_params).squeeze(0)
                
                seq_blocks.append(ContextBlock(
                    content=frame,
                    type='latent',
                    causal=True,
                    shape_meta=(target_res, target_res),  # Explicit: (H, W) for broadcast
                    logsnr=lsnr,
                    group_id=batch_raw.group_id,
                    id=f"vid_{batch_raw.group_id}_{t}"
                ))
            
            out_blocks.extend(seq_blocks)
            fetched += 1
            
        return out_blocks

# --- 3. Composite Iterator (Refactored) ---

class CompositeIterator:
    _ITERATOR_MAP = {
        'checkerboard': CheckerboardIterator,
        'torus': TorusIterator,
        'video': VideoFolderIterator 
    }

    def __init__(self, device='cuda', config=None, target_dtype=torch.float32, caching_resolution=256):
        self.device = device
        if config is None: config = {'checkerboard': 1.0}
        
        self.splits = []
        
        for split_key, cfg in config.items():
            if isinstance(cfg, (float, int)):
                # Normalize shorthand to full config with all required fields
                cfg = {
                    'ratio': float(cfg),
                    'type': split_key,
                    'params': {},
                    'noise_mode': 'uniform',
                    'noise_params': {}
                }

            gen_type = cfg['type']

            if gen_type not in self._ITERATOR_MAP:
                print(f"⚠️ Unknown iterator type '{gen_type}', defaulting to Checkerboard.")
                iterator_cls = CheckerboardIterator
            else:
                iterator_cls = self._ITERATOR_MAP[gen_type]

            raw_params = cfg['params']

            if gen_type == 'video':
                 path = raw_params['path']  # Required for video
                 iterator_instance = iterator_cls(
                     path,
                     device=device,
                     target_dtype=target_dtype,
                     caching_resolution=caching_resolution
                 )
            else:
                 iterator_instance = iterator_cls(device)

            self.splits.append({
                'name': split_key,
                'type': gen_type,
                'iterator': iterator_instance,
                'ratio': cfg['ratio'],
                'd_params': raw_params,
                'n_mode': cfg['noise_mode'],
                'n_params': cfg['noise_params']
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
            
            split_name = split['name']
            gen_type = split['type']
            
            # Extract params strictly
            d_params = split['d_params']

            if gen_type == 'video':
                # strict access assumes sequence_structure is set in config if type is video
                # We can keep a .get here if the default is logic-dependent, or enforce in schema
                seq_conf = d_params['sequence_structure']
                
                # Check for resolution overrides (from Bucketing)
                resolution = kwargs.get('resolution')  # kwargs is external interface, .get() OK
                if resolution is not None:
                    # Apply Relative Scaling Logic
                    overridden_seq = []
                    for frame_cfg in seq_conf:
                        new_cfg = frame_cfg.copy()
                        rel = new_cfg['relative_res']
                        # Abs = Bucket * Relative
                        abs_res = int(resolution * rel)
                        if abs_res % 2 != 0: abs_res += 1
                        new_cfg['res'] = abs_res
                        overridden_seq.append(new_cfg)
                    seq_conf = overridden_seq

                blocks = split['iterator'].generate_batch_list(count, seq_conf, start_group_id=global_group_id)
                for b in blocks: b.source = split_name
                if blocks: global_group_id = max(b.group_id for b in blocks) + 1
            
            else:
                # Geometric Logic
                # Merge kwargs (like resolution) with dataset params
                merged_params = {**kwargs, **d_params}
                
                # Extract resolution to pass positionally
                res = merged_params.pop('resolution', 32) # <--- FIX: pop() removes it from dict
                
                # Now passing **merged_params won't collide with 'res'
                blocks = split['iterator'].generate_batch_list(count, res, **merged_params)
                
                n_mode = split['n_mode']
                n_params = split['n_params']
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