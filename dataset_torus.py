#dataset_torus.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
import os

# PyTorch3D Imports
# Assuming a standard environment where these are available.
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer, 
    SoftPhongShader,
    TexturesUV,
    AmbientLights
)

# --- Color Space Utilities ---

def lab_to_rgb(L, a, b):
    """
    Manual CIELAB to RGB conversion to avoid extra dependencies.
    L: [0, 100], a: [-128, 127], b: [-128, 127]
    Returns tensor [B, 3] in range [0, 1]
    """
    # 1. Lab -> XYZ
    y = (L + 16.) / 116.
    x = a / 500. + y
    z = y - b / 200.

    def func(t):
        return torch.where(t > 0.2068966, t ** 3, (t - 16. / 116.) / 7.787)

    X = 0.95047 * func(x)
    Y = 1.00000 * func(y)
    Z = 1.08883 * func(z)

    # 2. XYZ -> RGB (sRGB D65)
    R = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    G = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    B = 0.0557 * X - 0.2040 * Y + 1.0570 * Z

    rgb = torch.stack([R, G, B], dim=-1)
    
    # 3. Gamma Correction
    rgb = torch.where(rgb > 0.0031308, 1.055 * (torch.abs(rgb) ** (1/2.4)) - 0.055, 12.92 * rgb)
    return torch.clamp(rgb, 0, 1)

def generate_random_materials(batch_size, device):
    """
    Generates colors:
    - 'Bright': Neutral greys (L=80-90)
    - 'Dark': Random Hue/Sat (L=20-50)
    """
    # Bright Color (The Grout/Background)
    L_bright = torch.rand(batch_size, device=device) * 10.0 + 80.0
    a_bright = torch.zeros(batch_size, device=device) # Neutral
    b_bright = torch.zeros(batch_size, device=device)
    c_bright = lab_to_rgb(L_bright, a_bright, b_bright)

    # Dark Color (The Herringbone)
    L_dark = torch.rand(batch_size, device=device) * 30.0 + 20.0
    # Random a, b in generic range, filtered by magnitude to ensure some color
    a_dark = (torch.rand(batch_size, device=device) - 0.5) * 160.0
    b_dark = (torch.rand(batch_size, device=device) - 0.5) * 160.0
    c_dark = lab_to_rgb(L_dark, a_dark, b_dark)
    
    return c_bright, c_dark

# --- Texture Generation ---

def generate_herringbone_texture(batch_size, height=512, width=512, num_stripes=16, device='cuda'):
    """
    Procedurally generates a herringbone pattern batch.
    Returns: [B, H, W, 3]
    """
    c_bright, c_dark = generate_random_materials(batch_size, device)
    
    # Coordinate Grid
    y = torch.linspace(0, num_stripes, height, device=device)
    x = torch.linspace(0, num_stripes, width, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    # Herringbone Logic:
    # 1. Divide into columns (zigzag strips)
    # 2. Alternating columns have diagonal lines +45 and -45 deg
    
    # Column index
    col_idx = torch.floor(grid_x).long()
    
    # Local coords within tile
    x_fract = grid_x - torch.floor(grid_x)
    y_fract = grid_y - torch.floor(grid_y)
    
    # Pattern Logic
    # Even columns: / (x + y)
    # Odd columns:  \ (x - y)
    
    # We create diagonal stripes by taking (x +/- y) modulo periodicity
    # The 'frequency' of stripes within a column determines the density
    
    stripe_density = 4.0 
    
    pattern_even = torch.cos((grid_y + grid_x) * stripe_density * math.pi)
    pattern_odd  = torch.cos((grid_y - grid_x) * stripe_density * math.pi)
    
    mask_even = (col_idx % 2 == 0).float()
    
    pattern = mask_even * pattern_even + (1 - mask_even) * pattern_odd
    
    # Binarize for sharp texture
    # > 0 -> Dark, < 0 -> Bright
    mask = (pattern > 0).float().unsqueeze(0).unsqueeze(-1) # [1, H, W, 1]
    
    # Broadcast colors
    # c_bright: [B, 3] -> [B, 1, 1, 3]
    img = c_dark.view(batch_size, 1, 1, 3) * mask + c_bright.view(batch_size, 1, 1, 3) * (1 - mask)
    
    return img

# --- Mesh Generation ---

def create_torus_mesh(R=1.0, r=0.4, rings=64, segments=32, device='cuda'):
    """
    Creates a Torus mesh with UV coordinates.
    R: Major radius (center to tube center)
    r: Minor radius (tube radius)
    """
    u = torch.linspace(0, 1, rings + 1, device=device)[:-1] # periodic
    v = torch.linspace(0, 1, segments + 1, device=device)[:-1]
    
    grid_u, grid_v = torch.meshgrid(u, v, indexing='ij')
    grid_u = grid_u.flatten()
    grid_v = grid_v.flatten()
    
    theta = 2 * math.pi * grid_u
    phi = 2 * math.pi * grid_v
    
    # Parametric Torus
    x = (R + r * torch.cos(theta)) * torch.cos(phi)
    y = (R + r * torch.cos(theta)) * torch.sin(phi)
    z = r * torch.sin(theta)
    
    verts = torch.stack([x, y, z], dim=1)
    
    # Faces topology (Triangulating the grid)
    faces = []
    for i in range(rings):
        for j in range(segments):
            # Indices in the flattened grid
            p0 = i * segments + j
            p1 = i * segments + (j + 1) % segments
            p2 = ((i + 1) % rings) * segments + (j + 1) % segments
            p3 = ((i + 1) % rings) * segments + j
            
            # Two triangles per quad
            faces.append([p0, p1, p2])
            faces.append([p0, p2, p3])
            
    faces = torch.tensor(faces, device=device)
    
    # UV Map
    # Map u, v directly to texture coordinates
    # We repeat the texture 4 times around the major axis and 2 times around minor
    # to maintain aspect ratio of the herringbone
    verts_uvs = torch.stack([grid_u * 4.0, grid_v * 1.0], dim=1) 
    
    return verts, faces, verts_uvs

# --- The Iterator Class ---

class TorusIterator:
    def __init__(self, device='cuda', texture_size=512):
        self.device = device
        self.texture_size = texture_size
        
        # 1. Init Renderer Components
        # We assume square aspect ratio
        self.lights = AmbientLights(device=device) # Flat lighting mostly, relying on texture
        
        # Create base geometry once (cheap)
        self.verts, self.faces, self.verts_uvs = create_torus_mesh(device=device)
        
    def generate_batch(self, batch_size, resolution, zoom_range=(1.2, 2.5)):
        """
        Generates a batch of rendered tori.
        zoom_range: Controls camera distance. 
                    Lower = Farther (Full torus). 
                    Higher = Closer (Texture macro).
        """
        # 1. Texture Generation
        # [B, H, W, 3]
        textures_map = generate_herringbone_texture(batch_size, 
                                                    height=self.texture_size, 
                                                    width=self.texture_size, 
                                                    device=self.device)
        
        # Create Textures object
        # Note: PyTorch3D expects specific UV formatting.
        # Ideally we construct a Meshes object per batch to handle the batch texture.
        
        # Expand geometry for batch
        # verts: [V, 3] -> [B, V, 3]
        b_verts = self.verts.unsqueeze(0).expand(batch_size, -1, -1)
        b_faces = self.faces.unsqueeze(0).expand(batch_size, -1, -1)
        b_uvs = self.verts_uvs.unsqueeze(0).expand(batch_size, -1, -1)
        
        textures = TexturesUV(maps=textures_map, faces_uvs=b_faces, verts_uvs=b_uvs)
        
        meshes = Meshes(verts=b_verts, faces=b_faces, textures=textures)
        
        # 2. Camera Sampling (The "Zoom" Logic)
        # We sample points on a sphere around the torus
        # Radius varies to control zoom/crop
        min_dist, max_dist = 2.0, 3.5 # Base distance units
        
        # Adjust distance based on resolution to ensure "1.3px width" heuristic?
        # Actually, simpler to just vary distance randomly and let the model see multiscale.
        # But specifically for 16px, we need to be CLOSE.
        
        dist = torch.rand(batch_size, device=self.device) * (max_dist - min_dist) + min_dist
        
        # Random spherical angles
        elev = torch.rand(batch_size, device=self.device) * 180.0 - 90.0 # -90 to 90
        azim = torch.rand(batch_size, device=self.device) * 360.0
        
        # Random Rotation of the Torus itself (or camera roll)
        # PyTorch3D cameras usually look at (0,0,0) with up=(0,1,0). 
        # Rotating the object is mathematically equivalent.
        # Let's just orbit the camera.
        
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        
        # 3. Renderer Setup (Per batch resolution)
        raster_settings = RasterizationSettings(
            image_size=resolution,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device, 
                cameras=cameras,
                lights=self.lights
            )
        )
        
        # 4. Render
        images = renderer(meshes) # [B, H, W, 4] (RGBA)
        
        # Drop Alpha, Rearrange to [B, C, H, W]
        images = images[..., :3].permute(0, 3, 1, 2)
        
        return images

# --- Test Script ---

if __name__ == "__main__":
    print("Initialize Generator...")
    iterator = TorusIterator(device='cuda')
    
    # Output dir
    os.makedirs("torus_samples", exist_ok=True)
    
    print("Generating 16x16 Batch...")
    batch_16 = iterator.generate_batch(16, 16)
    
    print("Generating 32x32 Batch...")
    batch_32 = iterator.generate_batch(16, 32)
    
    print("Generating 256x256 High-Res Reference Batch...")
    batch_256 = iterator.generate_batch(8, 256)
    
    def save_grid(batch, name):
        # batch: [B, 3, H, W]
        B = batch.shape[0]
        rows = int(math.sqrt(B))
        cols = math.ceil(B / rows)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        for i, ax in enumerate(axes.flat):
            if i < B:
                img = batch[i].permute(1, 2, 0).cpu().numpy()
                ax.imshow(img)
                ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"torus_samples/{name}.png", dpi=150)
        plt.close()
        print(f"Saved {name}.png")

    save_grid(batch_16, "torus_16px")
    save_grid(batch_32, "torus_32px")
    save_grid(batch_256, "torus_256px_ref")