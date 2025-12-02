# dataset_torus_native.py
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
        Ensures camera is never inside the object and always looks at geometry.
        """
        # 1. Camera Intrinsics
        # Screen coordinates [H, W]
        i, j = torch.meshgrid(
            torch.linspace(-1, 1, resolution, device=self.device),
            torch.linspace(-1, 1, resolution, device=self.device),
            indexing='ij'
        ) 
        
        # Rays in camera space (Z-forward)
        dirs_cam = torch.stack([j, -i, torch.ones_like(i)], dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)
        dirs_cam = F.normalize(dirs_cam, dim=-1)
        
        # 2. Camera Extrinsics: Topological Sampling
        
        # We sample in Cylindrical Coordinates (rho, phi, y) relative to the central Y-axis.
        # To avoid the solid torus, we sample two safe zones:
        # Zone A (Hole): Inside the ring (rho < R - r)
        # Zone B (Exterior): Outside the ring (rho > R + r)
        
        # Probability of being in the hole (Macro texture shots) vs Outside (Shape shots)
        in_hole = torch.rand(batch_size, device=self.device) < 0.3
        
        # -- Radius (rho) --
        # Hole: 0.0 to (R - r - margin). R=1, r=0.4 => max safe rho ~0.5
        rho_hole = torch.rand(batch_size, device=self.device) * 0.5 
        
        # Exterior: (R + r + margin) to 3.5. min safe rho ~1.5
        # We bias towards closer shots for detail
        rho_ext = torch.rand(batch_size, device=self.device) * 2.0 + 1.5
        
        rho = torch.where(in_hole, rho_hole, rho_ext)
        
        # -- Height (y) --
        # "Displaced up and down the central axis"
        # We want diversity: some planar shots (y~0), some high-angle (y~2)
        y_cam = (torch.rand(batch_size, device=self.device) * 4.0 - 2.0) # [-2, 2]
        
        # -- Angle (phi) --
        # "Displaced along the arc"
        phi_cam = torch.rand(batch_size, device=self.device) * 2 * math.pi
        
        # Convert to Cartesian Camera Position
        cx = rho * torch.cos(phi_cam)
        cy = y_cam
        cz = rho * torch.sin(phi_cam)
        origin = torch.stack([cx, cy, cz], dim=1) # [B, 3]
        
        # 3. Targeting Strategy (LookAt)
        # We must look at the spine (the central ring of the tube).
        # Spine points are at (cos(theta), 0, sin(theta)).
        
        # Heuristic: Look at the spine point roughly closest to the camera.
        # If we look at the far side, we get occlusion or view-through-hole (which is cool but harder to control).
        # Looking at the nearest spine point guarantees pixel coverage.
        
        # For hole shots (rho < 1), looking at phi_cam means looking radially OUTWARD at the wall. (Correct)
        # For ext shots (rho > 1), looking at phi_cam means looking radially INWARD at the wall. (Correct)
        
        # Add slight jitter to the target angle so we don't always look perfectly center-mass
        target_jitter = (torch.rand(batch_size, device=self.device) - 0.5) * 0.5 # +/- 0.25 rad
        theta_tgt = phi_cam + target_jitter
        
        tx = self.R_major * torch.cos(theta_tgt)
        ty = torch.zeros_like(tx) + (torch.rand(batch_size, device=self.device) - 0.5) * 0.2 # Slight vertical jitter
        tz = self.R_major * torch.sin(theta_tgt)
        
        target = torch.stack([tx, ty, tz], dim=1)
        
        # -- LookAt Matrix --
        forward = F.normalize(target - origin, dim=1)
        
        # Standard Up is (0,1,0), but if looking straight down/up, this is unstable.
        # We use a robust up vector calculation.
        world_up = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(batch_size, -1)
        
        # If forward is too close to world_up, perturb world_up
        # (Dot product check omitted for brevity in simple renderer, usually rare with random sampling)
        
        # Fix: Explicit dim=1 for torch.cross to suppress warnings
        right = F.normalize(torch.cross(forward, world_up, dim=1), dim=1)
        up = torch.cross(right, forward, dim=1)
        
        R = torch.stack([right, up, forward], dim=2) 
        
        # Transform rays
        rays_world = torch.bmm(dirs_cam.view(batch_size, -1, 3), R).view(batch_size, resolution, resolution, 3)
        
        return origin, rays_world

    def sdf_torus(self, p):
        # Distance to the ring in XZ plane
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
        
        # Optimization: Bounding Sphere for early exit? 
        # For simple batch, simple marching is often faster than python-loop overhead of complex checks.
        
        for _ in range(max_steps):
            if not active_mask.any(): break
            
            p = o + d * t.unsqueeze(-1)
            dist = self.sdf_torus(p)
            
            t = torch.where(active_mask, t + dist, t)
            
            hit = dist < 0.001
            miss = t > 6.0 # Max render distance
            active_mask = active_mask & (~hit) & (~miss)
        
        p_final = o + d * t.unsqueeze(-1)
        final_dist = self.sdf_torus(p_final)
        valid_mask = final_dist < 0.01
        
        return p_final, valid_mask.view(B, H, W)

    def get_uv(self, p):
        # U: Major Angle
        u = torch.atan2(p[..., 2], p[..., 0]) / (2 * math.pi) + 0.5
        # V: Minor Angle
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

    def shade_batch(self, p, valid_mask, resolution):
        B = valid_mask.shape[0]
        H = W = resolution
        
        # Randomize Colors
        L_bright = torch.rand(B, 1, device=self.device) * 20.0 + 70.0 # Varying brightness
        c_bright = self.lab_to_rgb(L_bright, torch.zeros_like(L_bright), torch.zeros_like(L_bright))
        
        # Random saturated colors
        L_dark = torch.rand(B, 1, device=self.device) * 30.0 + 20.0
        # Random hue rotation in Lab
        hue = torch.rand(B, 1, device=self.device) * 2 * math.pi
        chroma = torch.rand(B, 1, device=self.device) * 50.0 + 60.0 # High saturation
        a_dark = chroma * torch.cos(hue)
        b_dark = chroma * torch.sin(hue)
        c_dark = self.lab_to_rgb(L_dark, a_dark, b_dark)
        
        # Texture
        u, v = self.get_uv(p)
        
        # Randomized Frequency per batch item
        # FIX: Ensure shape is (B, 1, 1) for broadcasting against (B, H, W)
        freq_u = torch.randint(8, 16, (B, 1, 1), device=self.device).float()
        freq_v = torch.randint(4, 8, (B, 1, 1), device=self.device).float()
        
        u_scaled = u * freq_u
        v_scaled = v * freq_v
        
        # Herringbone-ish pattern
        col_idx = torch.floor(u_scaled)
        stripe_freq = 2.0 * math.pi
        pat_even = torch.cos((u_scaled + v_scaled) * stripe_freq)
        pat_odd  = torch.cos((u_scaled - v_scaled) * stripe_freq)
        is_even = (col_idx % 2 == 0)
        pattern = torch.where(is_even, pat_even, pat_odd)
        is_dark = pattern > 0
        
        # Composite
        c_bright_exp = c_bright.view(B, 1, 1, 3).expand(-1, H, W, -1)
        c_dark_exp = c_dark.view(B, 1, 1, 3).expand(-1, H, W, -1)
        
        surface_color = torch.where(is_dark.unsqueeze(-1), c_dark_exp, c_bright_exp)
        
        # Simple diffuse lighting for depth cue
        # Normal approx? p is on torus surface.
        # Normal of torus at p:
        # Center of tube section for p is:
        # cp_xz = normalize(p_xz) * R_major
        cp = p.clone()
        p_xz_norm = F.normalize(p[..., [0, 2]], dim=-1)
        cp[..., 0] = p_xz_norm[..., 0] * self.R_major
        cp[..., 1] = 0
        cp[..., 2] = p_xz_norm[..., 1] * self.R_major
        
        normal = F.normalize(p - cp, dim=-1)
        
        # Light dir: from camera (headlamp)
        # Actually fixed light is better for structure training
        light_dir = F.normalize(torch.tensor([0.5, 1.0, 0.5], device=self.device), dim=0).view(1, 1, 1, 3)
        diffuse = torch.sum(normal * light_dir, dim=-1, keepdim=True).clamp(0.2, 1.0)
        
        final_color = surface_color * diffuse
        
        return torch.where(valid_mask.unsqueeze(-1), final_color, torch.zeros_like(final_color))

class TorusIterator:
    def __init__(self, device='cuda'):
        self.marcher = TorusRaymarcher(device)
        
    def generate_batch(self, batch_size, resolution, **kwargs):
        origins, rays = self.marcher.get_camera_rays(batch_size, resolution)
        p, mask = self.marcher.intersect(origins, rays)
        p_reshaped = p.view(batch_size, resolution, resolution, 3)
        images = self.marcher.shade_batch(p_reshaped, mask, resolution)
        return images.permute(0, 3, 1, 2)