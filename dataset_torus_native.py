# dataset_torus_native.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
import os

class TorusRaymarcher:
    def __init__(self, device='cuda'):
        self.device = device

    def get_camera_rays(self, batch_size, resolution, dist_range=(2.0, 4.0)):
        """
        Generates camera rays looking at the origin.
        Random orbit around the object.
        """
        # 1. Camera Intrinsics (Assume FOV ~60 deg)
        # Screen coordinates
        i, j = torch.meshgrid(
            torch.linspace(-1, 1, resolution, device=self.device),
            torch.linspace(-1, 1, resolution, device=self.device),
            indexing='ij'
        ) # [H, W]
        
        # Rays in camera space (Z-forward)
        # [B, H, W, 3]
        dirs_cam = torch.stack([j, -i, torch.ones_like(i)], dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)
        dirs_cam = F.normalize(dirs_cam, dim=-1)
        
        # 2. Camera Extrinsics (Orbit)
        # Random spherical coords
        theta = torch.rand(batch_size, device=self.device) * 2 * math.pi
        phi = torch.rand(batch_size, device=self.device) * math.pi * 0.8 + 0.1 # Avoid pure poles
        radius = torch.rand(batch_size, device=self.device) * (dist_range[1] - dist_range[0]) + dist_range[0]
        
        # Camera Position (Eye)
        cx = radius * torch.sin(phi) * torch.cos(theta)
        cy = radius * torch.cos(phi)
        cz = radius * torch.sin(phi) * torch.sin(theta)
        origin = torch.stack([cx, cy, cz], dim=1) # [B, 3]
        
        # LookAt Matrix (Target = 0,0,0)
        # Forward = -Position (normalized)
        forward = -F.normalize(origin, dim=1)
        # Up vector (World Up = 0,1,0)
        world_up = torch.tensor([0.0, 1.0, 0.0], device=self.device).view(1, 3).expand(batch_size, -1)
        right = F.normalize(torch.cross(forward, world_up), dim=1)
        up = torch.cross(right, forward)
        
        # Transform rays to World Space
        # [B, 3, 3] Rotation Matrix
        R = torch.stack([right, up, forward], dim=2) 
        
        # Apply rotation to rays: [B, H, W, 3] @ [B, 3, 3] 
        # We flatten rays for matmul, then reshape
        rays_world = torch.bmm(dirs_cam.view(batch_size, -1, 3), R).view(batch_size, resolution, resolution, 3)
        
        return origin, rays_world

    def sdf_torus(self, p, R_major=1.0, r_minor=0.4):
        """
        Signed Distance Function for a Torus lying on XZ plane.
        p: [N, 3]
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
        
        # Current position along ray
        t = torch.zeros(o.shape[0], device=self.device)
        
        # Mask of active rays (haven't hit, haven't escaped)
        active_mask = torch.ones_like(t, dtype=torch.bool)
        
        for _ in range(max_steps):
            if not active_mask.any(): break
            
            p = o + d * t.unsqueeze(-1)
            dist = self.sdf_torus(p)
            
            # Update t
            t[active_mask] += dist[active_mask]
            
            # Convergence check
            hit = dist < 0.001
            miss = t > 10.0
            
            # We don't stop marching hit rays immediately in simple vectorized, 
            # we just stop updating them effectively or let them jitter.
            # Ideally we mask.
            active_mask = active_mask & (~hit) & (~miss)
        
        # Compute final points
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
        # Vector from center of major ring to point
        center_to_p_xz = F.normalize(p[..., [0, 2]], dim=-1) * R_major
        center_to_p = p.clone()
        center_to_p[..., 0] -= center_to_p_xz[..., 0]
        center_to_p[..., 2] -= center_to_p_xz[..., 1]
        
        # Now center_to_p is the vector from the tube spine to the surface
        # Project onto the local frame of the tube section
        # Local Y is world Y (p[..., 1])
        # Local X is the radial direction outwards
        
        # Simplified: V is angle of (q_xy, q_z) from SDF logic
        q_xy = torch.norm(p[..., [0, 2]], dim=-1) - R_major
        q_z = p[..., 1]
        v = torch.atan2(q_z, q_xy) / (2 * math.pi) + 0.5
        
        return u, v

    def lab_to_rgb(self, L, a, b):
        # ... [Same color conversion as before] ...
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
        
        # 1. Get Colors per batch item
        # Bright: Grey
        L_bright = torch.rand(B, 1, device=self.device) * 10.0 + 80.0
        c_bright = self.lab_to_rgb(L_bright, torch.zeros_like(L_bright), torch.zeros_like(L_bright))
        
        # Dark: Random Color
        L_dark = torch.rand(B, 1, device=self.device) * 30.0 + 20.0
        a_dark = (torch.rand(B, 1, device=self.device) - 0.5) * 160.0
        b_dark = (torch.rand(B, 1, device=self.device) - 0.5) * 160.0
        c_dark = self.lab_to_rgb(L_dark, a_dark, b_dark)
        
        # 2. Compute UVs
        u, v = self.get_uv(p)
        
        # 3. Herringbone Logic (Vectorized)
        # Tile density
        u_freq = 12.0
        v_freq = 4.0
        
        u_scaled = u * u_freq
        v_scaled = v * v_freq
        
        # Column Index
        col_idx = torch.floor(u_scaled)
        
        # Local pattern (x +/- y)
        # Use simple stripes
        stripe_freq = 2.0 * math.pi
        
        pat_even = torch.cos((u_scaled + v_scaled) * stripe_freq)
        pat_odd  = torch.cos((u_scaled - v_scaled) * stripe_freq)
        
        # Mix based on column parity
        is_even = (col_idx % 2 == 0)
        pattern = torch.where(is_even, pat_even, pat_odd)
        
        # Threshold
        is_dark = pattern > 0
        
        # 4. Composite
        # Flatten valid_mask to match p [N_total, 3]? No, p is [N_total, 3]
        # We need to map back to B,H,W
        
        # Colors: [B, 3] -> broadcast to [B, H, W, 3]
        img = torch.zeros(B, H, W, 3, device=self.device)
        
        # We need to construct the texture per pixel
        # This is tricky because `p` is flattened or masked.
        # Let's assume `p` passed in is [B, H, W, 3] but contains garbage where !valid_mask
        
        c_bright_expanded = c_bright.view(B, 1, 1, 3).expand(-1, H, W, -1)
        c_dark_expanded = c_dark.view(B, 1, 1, 3).expand(-1, H, W, -1)
        
        surface_color = torch.where(is_dark.unsqueeze(-1), c_dark_expanded, c_bright_expanded)
        
        # Apply Background (Black or Noise? Let's do Black)
        final_img = torch.where(valid_mask.unsqueeze(-1), surface_color, torch.zeros_like(surface_color))
        
        return final_img

# --- The Iterator Replacement ---

class TorusIterator:
    def __init__(self, device='cuda'):
        self.marcher = TorusRaymarcher(device)
        
    def generate_batch(self, batch_size, resolution, **kwargs):
        # 1. Rays
        origins, rays = self.marcher.get_camera_rays(batch_size, resolution)
        
        # 2. March
        p, mask = self.marcher.intersect(origins, rays)
        
        # 3. Shade
        # We reshape p to [B, H, W, 3] for shading logic
        p_reshaped = p.view(batch_size, resolution, resolution, 3)
        images = self.marcher.shade_batch(p_reshaped, mask, resolution)
        
        # Permute to [B, C, H, W] for the model
        return images.permute(0, 3, 1, 2)

# --- Test Dump ---
if __name__ == "__main__":
    iterator = TorusIterator(device='cuda')
    os.makedirs("torus_native_samples", exist_ok=True)
    
    configs = [(16, 16), (32, 16), (256, 4)]
    
    for res, bs in configs:
        print(f"Rendering {res}x{res}...")
        batch = iterator.generate_batch(bs, res)
        
        # Save Grid
        batch_np = batch.permute(0, 2, 3, 1).cpu().numpy()
        rows = int(math.sqrt(bs)); cols = math.ceil(bs/rows)
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        for i, ax in enumerate(axes.flat):
            if i < bs:
                ax.imshow(batch_np[i])
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"torus_native_samples/native_{res}px.png", dpi=150)
        plt.close()