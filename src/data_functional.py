# src/data_functional.py
import torch
import torch.nn.functional as F
import math
import json
import random

# ==============================================================================
# 1. Parameter Generation (Query Logic)
# ==============================================================================

def generate_checkerboard_query(seed: int, config: dict) -> dict:
    rng = random.Random(seed)
    num_tiles = rng.uniform(2.0, 8.0)
    angle = rng.uniform(0, 2 * math.pi)
    
    def rand_color(): return [rng.random(), rng.random(), rng.random()]
    c1 = rand_color()
    c2 = [1.0 - x for x in c1] if config.get('force_high_contrast', False) else rand_color()

    return {
        "type": "checkerboard",
        "tiles": round(num_tiles, 2),
        "angle": round(angle, 3),
        "c1": [round(x, 3) for x in c1],
        "c2": [round(x, 3) for x in c2]
    }

def generate_torus_query(seed: int, config: dict) -> dict:
    rng = random.Random(seed)
    
    # Lab-ish color generator helper
    def rand_rgb(): return [rng.random(), rng.random(), rng.random()]

    return {
        "type": "torus",
        "R_major": 1.0,
        "r_minor": round(rng.uniform(0.2, 0.6), 2),
        # Camera Params
        "cam_yaw": round(rng.uniform(0, 6.28), 3),
        "cam_pitch": round(rng.uniform(-0.5, 0.5), 3),
        "cam_dist": round(rng.uniform(2.5, 4.0), 2),
        # Style Params
        "c_surf_1": rand_rgb(),
        "c_surf_2": rand_rgb(),
        "stripe_freq": round(rng.uniform(4.0, 16.0), 1)
    }

# ==============================================================================
# 2. Rendering (The "Compiler")
# ==============================================================================

def serialize_query(query: dict) -> torch.Tensor:
    """Canonical JSON -> ASCII Bytes -> Tensor"""
    json_str = json.dumps(query, sort_keys=True, separators=(',', ':'))
    tokens = [ord(c) for c in json_str]
    return torch.tensor(tokens, dtype=torch.long)

def render_checkerboard(query: dict, resolution: int, device: torch.device) -> torch.Tensor:
    num_tiles = query['tiles']
    angle = query['angle']
    c1 = torch.tensor(query['c1'], device=device).view(3, 1, 1)
    c2 = torch.tensor(query['c2'], device=device).view(3, 1, 1)
    
    linspace = torch.linspace(-num_tiles/2, num_tiles/2, resolution, device=device)
    y, x = torch.meshgrid(linspace, linspace, indexing='ij')
    
    cos_t = math.cos(angle); sin_t = math.sin(angle)
    x_rot = x * cos_t + y * sin_t
    y_rot = -x * sin_t + y * cos_t
    
    pat = ((torch.floor(x_rot) + torch.floor(y_rot)) % 2).unsqueeze(0)
    return c1 * (1 - pat) + c2 * pat

# --- Torus Logic ---

def _sdf_torus(p, R, r):
    q_xy = torch.norm(p[..., [0, 2]], dim=-1) - R
    q_z = p[..., 1]
    return torch.sqrt(q_xy**2 + q_z**2) - r

def _get_uv(p, R):
    u = torch.atan2(p[..., 2], p[..., 0]) / (2 * math.pi) + 0.5
    q_xy = torch.norm(p[..., [0, 2]], dim=-1) - R
    q_z = p[..., 1]
    v = torch.atan2(q_z, q_xy) / (2 * math.pi) + 0.5
    return u, v

def render_torus(query: dict, resolution: int, device: torch.device) -> torch.Tensor:
    # 1. Extract Params
    R = query['R_major']
    r = query['r_minor']
    yaw, pitch, dist = query['cam_yaw'], query['cam_pitch'], query['cam_dist']
    c1 = torch.tensor(query['c_surf_1'], device=device).view(1, 1, 3)
    c2 = torch.tensor(query['c_surf_2'], device=device).view(1, 1, 3)
    freq = query['stripe_freq']

    # 2. Camera Rays
    cx, cy, cz = dist * math.cos(yaw), dist * math.sin(pitch), dist * math.sin(yaw)
    origin = torch.tensor([cx, cy, cz], device=device).view(1, 1, 3)
    target = torch.zeros(1, 1, 3, device=device) # Look at center
    
    # Basis
    forward = F.normalize(target - origin, dim=-1)
    world_up = torch.tensor([0.0, 1.0, 0.0], device=device).view(1, 1, 3)
    right = F.normalize(torch.cross(forward, world_up, dim=-1), dim=-1)
    up = torch.cross(right, forward, dim=-1)
    
    # Image Plane
    i, j = torch.meshgrid(
        torch.linspace(-1, 1, resolution, device=device),
        torch.linspace(-1, 1, resolution, device=device),
        indexing='ij'
    )
    # [H, W, 3]
    dirs_cam = torch.stack([j, -i, torch.ones_like(i)], dim=-1)
    dirs_cam = F.normalize(dirs_cam, dim=-1)
    
    # Rotate Rays to World
    # R_mat: [3, 3]
    R_mat = torch.cat([right.view(3, 1), up.view(3, 1), forward.view(3, 1)], dim=1)
    rays_world = torch.matmul(dirs_cam, R_mat.t()) # [H, W, 3]

    # 3. Raymarch
    # Flatten for batching: [H*W, 3]
    o_flat = origin.expand(resolution * resolution, -1)
    d_flat = rays_world.view(-1, 3)
    
    t = torch.zeros(resolution * resolution, device=device)
    active = torch.ones_like(t, dtype=torch.bool)
    
    for _ in range(32): # 32 steps usually enough for simple torus
        if not active.any(): break
        p = o_flat + d_flat * t.unsqueeze(-1)
        dist_val = _sdf_torus(p, R, r)
        t = torch.where(active, t + dist_val, t)
        active = active & (dist_val > 0.001) & (t < 8.0)

    # 4. Shade
    p_final = o_flat + d_flat * t.unsqueeze(-1)
    valid_mask = (_sdf_torus(p_final, R, r) < 0.01).view(resolution, resolution)
    p_final = p_final.view(resolution, resolution, 3)
    
    # UV Mapping
    u, v = _get_uv(p_final, R)
    
    # Pattern
    pat_val = torch.cos((u * freq + v * freq) * math.pi)
    is_stripe = pat_val > 0
    
    # Color Mix
    surf_color = torch.where(is_stripe.unsqueeze(-1), c1, c2)
    
    # Simple Diffuse Light
    light_dir = F.normalize(torch.tensor([0.5, 1.0, 0.5], device=device), dim=0)
    # Normal estimation (analytical for torus)
    # n = normalize(p - closest_point_on_major_ring)
    p_xz_norm = F.normalize(p_final[..., [0, 2]], dim=-1)
    center_ring = torch.stack([p_xz_norm[..., 0] * R, torch.zeros_like(u), p_xz_norm[..., 1] * R], dim=-1)
    normal = F.normalize(p_final - center_ring, dim=-1)
    diffuse = (normal @ light_dir).clamp(0.2, 1.0).unsqueeze(-1)
    
    final_rgb = surf_color * diffuse
    bg_color = torch.tensor([0.05, 0.05, 0.08], device=device)
    
    img = torch.where(valid_mask.unsqueeze(-1), final_rgb, bg_color)
    return img.permute(2, 0, 1) # [3, H, W]

# ==============================================================================
# 3. Serialization (Text Tokenization)
# ==============================================================================


class TokenizerWrapper:
    """
    Wraps a tokenizer. If transformers is missing, falls back to ASCII.
    Target Vocab: ~152k (Qwen 2.5)
    """
    def __init__(self):
        try:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
            self.mode = "qwen"
            print("Loaded Qwen 3 Tokenizer.")
        except Exception as e:
            print(f"Tokenizer fallback (ASCII): {e}")
            self.mode = "fallback"

    def encode(self, text: str) -> torch.Tensor:
        if self.mode == "qwen":
            # Return Long Tensor
            return torch.tensor(self.tok.encode(text), dtype=torch.long)
        else:
            # Fallback: Ordinal mapping, shifted to avoid 0/1/2 specials if needed
            return torch.tensor([ord(c) for c in text], dtype=torch.long)

# Global singleton (lazy init)
_TOKENIZER = None

def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = TokenizerWrapper()
    return _TOKENIZER

def serialize_query(query: dict) -> torch.Tensor:
    # Deterministic JSON
    text = json.dumps(query, sort_keys=True, separators=(',', ':'))
    return get_tokenizer().encode(text)