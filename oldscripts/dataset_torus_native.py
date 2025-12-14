import torch
import torch.nn.functional as F
import math
import os

class TorusRaymarcher:
    def __init__(self, device='cuda'):
        self.device = device
        # Torus geometry constants
        self.R_major = 1.0
        self.r_minor = 0.4

    def get_camera_rays(self, batch_size, resolution):
        """
        Generates camera rays using a topology-aware cylindrical sampler.
        """
        # 1. Camera Intrinsics
        i, j = torch.meshgrid(
            torch.linspace(-1, 1, resolution, device=self.device),
            torch.linspace(-1, 1, resolution, device=self.device),
            indexing='ij'
        ) 
        
        # Rays in camera space (Z-forward)
        dirs_cam = torch.stack([j, -i, torch.ones_like(i)], dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)
        dirs_cam = F.normalize(dirs_cam, dim=-1)
        
        # 2. Camera Extrinsics: Topological Sampling
        in_hole = torch.rand(batch_size, device=self.device) < 0.3
        
        # Radius (rho)
        rho_hole = torch.rand(batch_size, device=self.device) * 0.5 
        rho_ext = torch.rand(batch_size, device=self.device) * 2.0 + 1.5
        rho = torch.where(in_hole, rho_hole, rho_ext)
        
        # Height (y) & Angle (phi)
        y_cam = (torch.rand(batch_size, device=self.device) * 4.0 - 2.0)
        phi_cam = torch.rand(batch_size, device=self.device) * 2 * math.pi
        
        # Camera Position
        cx = rho * torch.cos(phi_cam)
        cy = y_cam
        cz = rho * torch.sin(phi_cam)
        origin = torch.stack([cx, cy, cz], dim=1) 
        
        # 3. Targeting Strategy
        # Look at the spine point roughly closest to the camera
        target_jitter = (torch.rand(batch_size, device=self.device) - 0.5) * 0.5
        theta_tgt = phi_cam + target_jitter
        
        tx = self.R_major * torch.cos(theta_tgt)
        ty = torch.zeros_like(tx) + (torch.rand(batch_size, device=self.device) - 0.5) * 0.2
        tz = self.R_major * torch.sin(theta_tgt)
        target = torch.stack([tx, ty, tz], dim=1)
        
        # LookAt Matrix
        forward = F.normalize(target - origin, dim=1)
        world_up = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(batch_size, -1)
        
        right = F.normalize(torch.cross(forward, world_up, dim=1), dim=1)
        up = torch.cross(right, forward, dim=1)
        R = torch.stack([right, up, forward], dim=2) 
        
        # Transform rays
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
        """
        Shades the torus and generates a procedural background for misses.
        """
        B = valid_mask.shape[0]
        H = W = resolution
        
        # --- 1. Torus Surface Shading ---
        
        # Randomize Colors
        L_bright = torch.rand(B, 1, device=self.device) * 20.0 + 70.0 
        c_bright = self.lab_to_rgb(L_bright, torch.zeros_like(L_bright), torch.zeros_like(L_bright))
        
        L_dark = torch.rand(B, 1, device=self.device) * 30.0 + 20.0
        hue = torch.rand(B, 1, device=self.device) * 2 * math.pi
        chroma = torch.rand(B, 1, device=self.device) * 50.0 + 60.0
        a_dark = chroma * torch.cos(hue)
        b_dark = chroma * torch.sin(hue)
        c_dark = self.lab_to_rgb(L_dark, a_dark, b_dark)
        
        # Texture Sampling
        u, v = self.get_uv(p)
        
        # Random Frequency (B, 1, 1)
        freq_u = torch.randint(8, 16, (B, 1, 1), device=self.device).float()
        freq_v = torch.randint(4, 8, (B, 1, 1), device=self.device).float()
        
        u_scaled = u * freq_u
        v_scaled = v * freq_v
        
        # Herringbone Pattern
        col_idx = torch.floor(u_scaled)
        stripe_freq = 2.0 * math.pi
        pat_even = torch.cos((u_scaled + v_scaled) * stripe_freq)
        pat_odd  = torch.cos((u_scaled - v_scaled) * stripe_freq)
        is_even = (col_idx % 2 == 0)
        pattern = torch.where(is_even, pat_even, pat_odd)
        is_dark = pattern > 0
        
        # Composite Surface
        c_bright_exp = c_bright.view(B, 1, 1, 3).expand(-1, H, W, -1)
        c_dark_exp = c_dark.view(B, 1, 1, 3).expand(-1, H, W, -1)
        surface_color = torch.where(is_dark.unsqueeze(-1), c_dark_exp, c_bright_exp)
        
        # Diffuse Lighting
        cp = p.clone()
        p_xz_norm = F.normalize(p[..., [0, 2]], dim=-1)
        cp[..., 0] = p_xz_norm[..., 0] * self.R_major
        cp[..., 1] = 0
        cp[..., 2] = p_xz_norm[..., 1] * self.R_major
        normal = F.normalize(p - cp, dim=-1)
        light_dir = F.normalize(torch.tensor([0.5, 1.0, 0.5], device=self.device), dim=0).view(1, 1, 1, 3)
        diffuse = torch.sum(normal * light_dir, dim=-1, keepdim=True).clamp(0.2, 1.0)
        surface_lit = surface_color * diffuse

        # --- 2. Background Shading ---
        
        # Base Grey Level: 1% to 4% (0.01 to 0.04)
        bg_base = torch.rand(B, 1, 1, 1, device=self.device) * 0.03 + 0.01
        
        # Frequency: Period of 3 to 5 across the sphere
        bg_freq = torch.rand(B, 1, 1, 1, device=self.device) * 2.0 + 3.0
        
        # Use ray direction (normalized) as coordinates for the "Sky Sphere"
        # rays: [B, H, W, 3]
        d = rays
        
        # Pattern: sin(x * freq) * sin(y * freq)
        # This creates a soft grid/blob pattern
        bg_pat_val = torch.sin(d[..., 0] * bg_freq.squeeze(-1) * math.pi) * \
                     torch.sin(d[..., 1] * bg_freq.squeeze(-1) * math.pi)
        
        # Modulation: Keep it subtle (e.g. +/- 20% of the base grey)
        # Add 1.0 to center it around the base color
        bg_mod = 1.0 + (bg_pat_val.unsqueeze(-1) * 0.3)
        
        bg_color = bg_base * bg_mod
        bg_color = bg_color.clamp(0.0, 1.0)
        
        # --- 3. Final Composite ---
        final_img = torch.where(valid_mask.unsqueeze(-1), surface_lit, bg_color)
        
        return final_img

class TorusIterator:
    def __init__(self, device='cuda'):
        self.marcher = TorusRaymarcher(device)
        
    def generate_batch(self, batch_size, resolution, **kwargs):
        origins, rays = self.marcher.get_camera_rays(batch_size, resolution)
        p, mask = self.marcher.intersect(origins, rays)
        
        p_reshaped = p.view(batch_size, resolution, resolution, 3)
        
        # Pass 'rays' to shade_batch for background generation
        images = self.marcher.shade_batch(p_reshaped, mask, rays, resolution)
        
        return images.permute(0, 3, 1, 2)