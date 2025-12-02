# dataset_torus_native.py
import torch
import torch.nn.functional as F
import math
import os

class TorusRaymarcher:
    def __init__(self, device='cuda'):
        self.device = device

    def get_camera_rays(self, batch_size, resolution):
        """
        Generates camera rays with smart targeting.
        Mixes 'Full Object' shots and 'Macro Surface' shots.
        """
        # 1. Camera Intrinsics (Assume FOV ~60-90 deg)
        # Screen coordinates
        i, j = torch.meshgrid(
            torch.linspace(-1, 1, resolution, device=self.device),
            torch.linspace(-1, 1, resolution, device=self.device),
            indexing='ij'
        ) # [H, W]
        
        # Rays in camera space (Z-forward)
        # We use a slightly wider implicit FOV to ensure we catch things close up
        dirs_cam = torch.stack([j, -i, torch.ones_like(i)], dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)
        dirs_cam = F.normalize(dirs_cam, dim=-1)
        
        # 2. Camera Extrinsics (Targeting Logic)
        
        # Mode Selection: 60% Macro (Close-up), 40% Full View
        is_macro = torch.rand(batch_size, device=self.device) < 0.6
        
        # -- Target Point Calculation --
        # Full View: Look at (0,0,0)
        # Macro View: Look at a random point on the major ring (Radius=1.0)
        target_angle = torch.rand(batch_size, device=self.device) * 2 * math.pi
        
        tx = torch.cos(target_angle) # Radius 1.0
        tz = torch.sin(target_angle)
        
        target = torch.zeros(batch_size, 3, device=self.device)
        # Apply macro targets
        target[is_macro, 0] = tx[is_macro]
        target[is_macro, 2] = tz[is_macro]
        # Jitter Y target slightly for macro to look at top/bottom of tube surface
        target[is_macro, 1] = (torch.rand(is_macro.sum(), device=self.device) - 0.5) * 0.2
        
        # -- Camera Position Calculation --
        # Distances relative to the TARGET
        # Full: Far enough to see whole ring (2.5 to 4.0)
        # Macro: Close enough to see texture, safe enough not to clip (0.6 to 1.2)
        # Note: Tube radius is 0.4. Dist 0.6 leaves 0.2 clearance.
        dist_full = torch.rand(batch_size, device=self.device) * 1.5 + 2.5
        dist_macro = torch.rand(batch_size, device=self.device) * 0.6 + 0.6
        dist = torch.where(is_macro, dist_macro, dist_full)
        
        # Orbit around the TARGET
        theta = torch.rand(batch_size, device=self.device) * 2 * math.pi
        # Clamp phi to avoid degenerate LookAt (poles)
        phi = torch.rand(batch_size, device=self.device) * 2.8 + 0.17 # ~10 to 170 degrees
        
        cx = dist * torch.sin(phi) * torch.cos(theta)
        cy = dist * torch.cos(phi)
        cz = dist * torch.sin(phi) * torch.sin(theta)
        offset = torch.stack([cx, cy, cz], dim=1)
        
        origin = target + offset # [B, 3]
        
        # -- LookAt Matrix --
        # Forward = Direction from Camera to Target
        forward = F.normalize(target - origin, dim=1)
        
        # Up vector (World Up = 0,1,0)
        world_up = torch.tensor([0.0, 1.0, 0.0], device=self.device).view(1, 3).expand(batch_size, -1)
        
        # Standard Gram-Schmidt
        right = F.normalize(torch.cross(forward, world_up), dim=1)
        up = torch.cross(right, forward)
        
        # Transform rays to World Space
        R = torch.stack([right, up, forward], dim=2) # [B, 3, 3]
        
        # Apply rotation: [B, H*W, 3] @ [B, 3, 3]
        rays_world = torch.bmm(dirs_cam.view(batch_size, -1, 3), R).view(batch_size, resolution, resolution, 3)
        
        return origin, rays_world

    def sdf_torus(self, p, R_major=1.0, r_minor=0.4):
        """
        Signed Distance Function for a Torus lying on XZ plane.
        """
        # Distance to the ring in XZ plane
        q_xy = torch.norm(p[..., [0, 2]], dim=-1) - R_major
        q_z = p[..., 1]
        
        # Distance to the tube surface
        d = torch.sqrt(q_xy**2 + q_z**2) - r_minor
        return d

    def intersect(self, origin, rays, max_steps=64):
        """
        Sphere tracing.
        origin: [B, 3]
        rays: [B, H, W, 3]
        """
        B, H, W, _ = rays.shape
        
        # Flatten for marching
        o = origin.view(B, 1, 3).expand(-1, H*W, -1).reshape(-1, 3)
        d = rays.view(-1, 3)
        
        t = torch.zeros(o.shape[0], device=self.device)
        active_mask = torch.ones_like(t, dtype=torch.bool)
        
        # Optimization: Don't march rays that look away from bounding sphere (approx)
        # But for torus, just march everything, it's cheap enough.
        
        for _ in range(max_steps):
            if not active_mask.any(): break
            
            p = o + d * t.unsqueeze(-1)
            dist = self.sdf_torus(p)
            
            # Update t only for active rays
            # We use `where` to avoid inplace ops on masked tensors which can be slow/tricky
            t_next = t + dist
            t = torch.where(active_mask, t_next, t)
            
            # Check convergence
            hit = dist < 0.001
            miss = t > 8.0 # Tighter bound than 10.0 for speed
            
            # Update mask
            active_mask = active_mask & (~hit) & (~miss)
        
        p_final = o + d * t.unsqueeze(-1)
        
        # Final validity check
        final_dist = self.sdf_torus(p_final)
        valid_mask = final_dist < 0.01
        
        return p_final, valid_mask.view(B, H, W)

    def get_uv(self, p, R_major=1.0):
        """
        Map 3D points on torus surface to UVs.
        """
        # U: Angle around Y axis (Major)
        u = torch.atan2(p[..., 2], p[..., 0]) / (2 * math.pi) + 0.5
        
        # V: Angle around the tube center (Minor)
        q_xy = torch.norm(p[..., [0, 2]], dim=-1) - R_major
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
        
        # 1. Colors
        L_bright = torch.rand(B, 1, device=self.device) * 10.0 + 85.0
        c_bright = self.lab_to_rgb(L_bright, torch.zeros_like(L_bright), torch.zeros_like(L_bright))
        
        L_dark = torch.rand(B, 1, device=self.device) * 30.0 + 20.0
        a_dark = (torch.rand(B, 1, device=self.device) - 0.5) * 160.0
        b_dark = (torch.rand(B, 1, device=self.device) - 0.5) * 160.0
        c_dark = self.lab_to_rgb(L_dark, a_dark, b_dark)
        
        # 2. UVs & Pattern
        u, v = self.get_uv(p)
        
        # Slightly higher frequency for macro details
        u_freq = 12.0
        v_freq = 6.0 
        
        u_scaled = u * u_freq
        v_scaled = v * v_freq
        
        col_idx = torch.floor(u_scaled)
        
        stripe_freq = 2.0 * math.pi
        pat_even = torch.cos((u_scaled + v_scaled) * stripe_freq)
        pat_odd  = torch.cos((u_scaled - v_scaled) * stripe_freq)
        
        is_even = (col_idx % 2 == 0)
        pattern = torch.where(is_even, pat_even, pat_odd)
        is_dark = pattern > 0
        
        # 3. Composite
        c_bright_expanded = c_bright.view(B, 1, 1, 3).expand(-1, H, W, -1)
        c_dark_expanded = c_dark.view(B, 1, 1, 3).expand(-1, H, W, -1)
        
        surface_color = torch.where(is_dark.unsqueeze(-1), c_dark_expanded, c_bright_expanded)
        
        final_img = torch.where(valid_mask.unsqueeze(-1), surface_color, torch.zeros_like(surface_color))
        return final_img

class TorusIterator:
    def __init__(self, device='cuda'):
        self.marcher = TorusRaymarcher(device)
        
    def generate_batch(self, batch_size, resolution, **kwargs):
        origins, rays = self.marcher.get_camera_rays(batch_size, resolution)
        p, mask = self.marcher.intersect(origins, rays)
        p_reshaped = p.view(batch_size, resolution, resolution, 3)
        images = self.marcher.shade_batch(p_reshaped, mask, resolution)
        return images.permute(0, 3, 1, 2)

if __name__ == "__main__":
    # Self-test if run directly
    import matplotlib.pyplot as plt
    iterator = TorusIterator(device='cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs("torus_test", exist_ok=True)
    img = iterator.generate_batch(16, 64)
    # Save a grid
    grid = img.permute(0, 2, 3, 1).cpu().numpy()
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(grid[i])
        ax.axis('off')
    plt.savefig("torus_test/test_grid.png")
    print("Test grid saved to torus_test/test_grid.png")