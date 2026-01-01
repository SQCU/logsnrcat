"""Train FSQ autoencoder - SwiGLU neighbor decoder with 2D RoPE.
Center patch gates the neighbors via SwiGLU.
2D Rotary Position Embeddings in transformer attention.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
import time
import yaml
from losses import get_loss_fn


class ImageDataset(Dataset):
    def __init__(self, paths, size):
        self.paths = paths
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
            w, h = img.size
            m = min(w, h)
            left, top = (w - m) // 2, (h - m) // 2
            img = img.crop((left, top, left + m, top + m))
            img = img.resize((self.size, self.size), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)
        except:
            return torch.rand(3, self.size, self.size)


def get_2d_rope_freqs(grid_size, head_dim, device, base=10000.0):
    """Generate 2D rotary position embedding frequencies.

    Split head_dim in half: first half for x, second half for y.
    """
    half_dim = head_dim // 2
    # Frequencies for each half
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2, device=device).float() / half_dim))

    # Grid positions
    pos_x = torch.arange(grid_size, device=device).float()
    pos_y = torch.arange(grid_size, device=device).float()

    # Outer product: [grid_size, half_dim//2]
    freqs_x = torch.outer(pos_x, freqs)  # [grid_size, half_dim//2]
    freqs_y = torch.outer(pos_y, freqs)  # [grid_size, half_dim//2]

    # Expand to 2D grid: [grid_size, grid_size, half_dim//2]
    freqs_x = freqs_x.unsqueeze(1).expand(-1, grid_size, -1)  # [H, W, d]
    freqs_y = freqs_y.unsqueeze(0).expand(grid_size, -1, -1)  # [H, W, d]

    # Flatten to sequence: [grid_size*grid_size, half_dim//2]
    freqs_x = freqs_x.reshape(-1, freqs_x.shape[-1])
    freqs_y = freqs_y.reshape(-1, freqs_y.shape[-1])

    # Interleave for sin/cos pattern: [seq_len, half_dim]
    freqs_x = torch.stack([freqs_x, freqs_x], dim=-1).flatten(-2)
    freqs_y = torch.stack([freqs_y, freqs_y], dim=-1).flatten(-2)

    # Concat x and y: [seq_len, head_dim]
    freqs = torch.cat([freqs_x, freqs_y], dim=-1)

    return freqs


def apply_rotary_emb(x, freqs):
    """Apply rotary embeddings to x.

    x: [B, n_heads, seq_len, head_dim]
    freqs: [seq_len, head_dim]
    """
    # Reshape x for rotation: treat pairs
    x_reshape = x.float().reshape(*x.shape[:-1], -1, 2)

    # Get sin/cos
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)

    cos = cos.reshape(*cos.shape[:-1], -1, 2)  # [1, 1, seq_len, head_dim//2, 2]
    sin = sin.reshape(*sin.shape[:-1], -1, 2)

    # Rotate: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
    x_out = torch.stack([
        x_reshape[..., 0] * cos[..., 0] - x_reshape[..., 1] * sin[..., 0],
        x_reshape[..., 0] * sin[..., 1] + x_reshape[..., 1] * cos[..., 1],
    ], dim=-1)

    return x_out.flatten(-2).type_as(x)


class GQATransformerLayerRoPE(nn.Module):
    def __init__(self, dim, n_query_heads, n_kv_heads):
        super().__init__()
        self.dim = dim
        self.n_query_heads = n_query_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_query_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, self.head_dim * n_kv_heads)
        self.v_proj = nn.Linear(dim, self.head_dim * n_kv_heads)
        self.out_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Post-SDPA sigmoid gating (per-head, query-dependent)
        # Each head gets its own gate projection
        self.attn_gate = nn.Parameter(torch.zeros(n_query_heads, self.head_dim))
        self.attn_gate_bias = nn.Parameter(torch.zeros(n_query_heads))

        # SwiGLU FFN
        self.w1 = nn.Linear(dim, dim * 4)  # gate
        self.w2 = nn.Linear(dim, dim * 4)  # value
        self.w3 = nn.Linear(dim * 4, dim)  # out

    def forward(self, x, rope_freqs=None):
        B, N, D = x.shape
        normed = self.norm1(x)
        q = self.q_proj(normed).view(B, N, self.n_query_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to q and k
        if rope_freqs is not None:
            q = apply_rotary_emb(q, rope_freqs)
            k = apply_rotary_emb(k, rope_freqs)

        heads_per_kv = self.n_query_heads // self.n_kv_heads
        k = k.repeat_interleave(heads_per_kv, dim=1)
        v = v.repeat_interleave(heads_per_kv, dim=1)
        attn_out = F.scaled_dot_product_attention(q, k, v)

        # Post-SDPA sigmoid gating: query-dependent per-head gate
        # q: [B, n_heads, N, head_dim], attn_gate: [n_heads, head_dim]
        # einsum output: [B, n_heads, N], bias: [n_heads] -> need [1, n_heads, 1]
        gate_logits = torch.einsum('bhnd,hd->bhn', q, self.attn_gate) + self.attn_gate_bias.view(1, -1, 1)
        gate = torch.sigmoid(gate_logits).unsqueeze(-1)  # [B, n_heads, N, 1]
        attn_out = attn_out * gate

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, D)
        x = x + self.out_proj(attn_out)

        # SwiGLU FFN
        normed2 = self.norm2(x)
        x = x + self.w3(F.silu(self.w1(normed2)) * self.w2(normed2))
        return x


class TransformerBlockRoPE(nn.Module):
    def __init__(self, dim, n_layers, n_query_heads, n_kv_heads):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_query_heads
        self.layers = nn.ModuleList([GQATransformerLayerRoPE(dim, n_query_heads, n_kv_heads) for _ in range(n_layers)])
        self._rope_cache = {}

    def get_rope_freqs(self, grid_size, device):
        key = (grid_size, device)
        if key not in self._rope_cache:
            self._rope_cache[key] = get_2d_rope_freqs(grid_size, self.head_dim, device)
        return self._rope_cache[key]

    def forward(self, x):
        B, N, D = x.shape
        grid_size = int(math.sqrt(N))
        rope_freqs = self.get_rope_freqs(grid_size, x.device)

        for layer in self.layers:
            x = layer(x, rope_freqs)
        return x


class SwiGLUNeighborHead(nn.Module):
    """Just the neighbor gathering + SwiGLU gating (no projections)."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_proj = nn.Linear(9 * hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(9 * hidden_dim, hidden_dim)

    def forward(self, h):
        # h: [B, N, hidden_dim] after transformer
        B, N, _ = h.shape
        grid_size = int(math.sqrt(N))

        h_grid = h.view(B, grid_size, grid_size, self.hidden_dim)
        h_padded = F.pad(h_grid.permute(0, 3, 1, 2), (1, 1, 1, 1), mode='reflect')
        h_padded = h_padded.permute(0, 2, 3, 1)

        neighbors_list = []
        for i in range(grid_size):
            for j in range(grid_size):
                neighborhood = h_padded[:, i:i+3, j:j+3, :].reshape(B, 9 * self.hidden_dim)
                neighbors_list.append(neighborhood)

        neighbors = torch.stack(neighbors_list, dim=1)
        gate = F.silu(self.gate_proj(neighbors))
        value = self.value_proj(neighbors)
        return gate * value  # [B, n_patches, hidden_dim]


class SwiGLUNeighborDecoder(nn.Module):
    """Decoder with SwiGLU: center patch gates the neighbors."""
    def __init__(self, code_dim, hidden_dim, output_dim, n_layers, n_query_heads, n_kv_heads):
        super().__init__()
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        # First: code -> hidden
        self.to_hidden = nn.Linear(code_dim, hidden_dim)
        # Transformer for global context (with RoPE)
        self.transformer = TransformerBlockRoPE(hidden_dim, n_layers, n_query_heads, n_kv_heads)

        # SwiGLU projections
        # Gate: all 9 neighbors -> gate values (not just center)
        self.gate_proj = nn.Linear(9 * hidden_dim, hidden_dim)
        # Value: all 9 neighbors -> values
        self.value_proj = nn.Linear(9 * hidden_dim, hidden_dim)
        # Output
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, codes):
        # codes: [B, n_patches, code_dim]
        B, N, _ = codes.shape
        grid_size = int(math.sqrt(N))

        # To hidden and transformer
        h = self.to_hidden(codes)  # [B, N, hidden_dim]
        h = self.transformer(h)    # [B, N, hidden_dim]

        # Reshape to grid
        h_grid = h.view(B, grid_size, grid_size, self.hidden_dim)

        # Pad for neighbor gathering (reflect padding)
        h_padded = F.pad(h_grid.permute(0, 3, 1, 2), (1, 1, 1, 1), mode='reflect')
        h_padded = h_padded.permute(0, 2, 3, 1)  # [B, H+2, W+2, hidden_dim]

        # Gather 3x3 neighborhoods and centers
        neighbors_list = []
        centers_list = []
        for i in range(grid_size):
            for j in range(grid_size):
                neighborhood = h_padded[:, i:i+3, j:j+3, :].reshape(B, 9 * self.hidden_dim)
                center = h_grid[:, i, j, :]  # [B, hidden_dim]
                neighbors_list.append(neighborhood)
                centers_list.append(center)

        neighbors = torch.stack(neighbors_list, dim=1)  # [B, n_patches, 9*hidden_dim]
        centers = torch.stack(centers_list, dim=1)      # [B, n_patches, hidden_dim]

        # SwiGLU: gate from all 9 neighbors, value from all 9 neighbors
        gate = F.silu(self.gate_proj(neighbors))  # [B, n_patches, hidden_dim]
        value = self.value_proj(neighbors)         # [B, n_patches, hidden_dim]

        # Gated output
        gated = gate * value  # [B, n_patches, hidden_dim]

        # Project to pixels
        pixels = self.out_proj(gated)  # [B, n_patches, output_dim]
        return pixels


class MultiScaleRouter(nn.Module):
    """Fixed learned routing per level - same for all images."""
    def __init__(self, code_dim, n_levels, k_dims):
        super().__init__()
        self.code_dim = code_dim
        self.n_levels = n_levels
        # k_dims can be int (same for all) or list (per-level)
        self.k = k_dims if isinstance(k_dims, list) else [k_dims] * n_levels
        # Which dims are active per level (for topk selection)
        self.dim_logits = nn.Parameter(torch.randn(n_levels, code_dim))
        # Fixed learned values per level (not image-dependent)
        self.level_values = nn.Parameter(torch.randn(n_levels, code_dim) * 0.1)

    def forward(self, batch_size, device, k_override=None):
        # k_override can be int (same for all) or list (per-level)
        if k_override is not None:
            ks = k_override if isinstance(k_override, list) else [k_override] * self.n_levels
        else:
            ks = self.k

        # Per-level topk mask
        global_mask = torch.zeros(self.n_levels, self.code_dim, device=device)
        all_indices = []
        for lvl in range(self.n_levels):
            _, idx = self.dim_logits[lvl].topk(ks[lvl])
            global_mask[lvl].scatter_(0, idx, 1.0)
            all_indices.append(idx)

        # Fixed values per level, quantized to 2-bit
        values = torch.sigmoid(self.level_values)  # [n_levels, code_dim]

        # 2-bit quantization: 4 levels -> -1.5, -0.5, 0.5, 1.5
        n_levels_fsq = 4
        scaled = values * (n_levels_fsq - 1)  # 0 to 3
        quantized = torch.round(scaled)
        quantized_ste = scaled + (quantized - scaled).detach()
        # Map 0,1,2,3 to -1.5, -0.5, 0.5, 1.5
        soft_out = (quantized_ste - 1.5) / 1.5  # normalized to [-1, 1]

        # Apply mask and expand for batch
        soft_masked = (soft_out * global_mask).unsqueeze(0).expand(batch_size, -1, -1)

        # Stack indices per level
        max_k = max(ks)
        indices_padded = torch.zeros(self.n_levels, max_k, dtype=torch.long, device=device)
        for lvl, idx in enumerate(all_indices):
            indices_padded[lvl, :len(idx)] = idx
        indices_batch = indices_padded.unsqueeze(0).expand(batch_size, -1, -1)
        return soft_masked, indices_batch, ks


class MultiScaleFSQAutoencoder(nn.Module):
    def __init__(self, image_size, patches_per_level, dec_canvas_sizes, n_levels,
                 hidden_dim, code_dim, k_dims, n_layers, residual_scale,
                 n_query_heads, n_kv_heads, share_weights):
        super().__init__()
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.code_dim = code_dim
        self.k_dims = k_dims
        self.residual_scale = residual_scale
        self.n_levels = n_levels
        self.n_patches_per_level = patches_per_level

        # Compute patch sizes from config (per-level)
        grid_sizes = [int(math.sqrt(p)) for p in patches_per_level]
        self.enc_patch_sizes = [image_size // gs for gs in grid_sizes]

        # Decode canvas sizes from config
        self.dec_canvas_sizes = dec_canvas_sizes
        self.dec_patch_sizes = [dec_canvas_sizes[i] // grid_sizes[i] for i in range(n_levels)]

        self.enc_patch_dims = [ps * ps * 3 for ps in self.enc_patch_sizes]
        self.dec_patch_dims = [ps * ps * 3 for ps in self.dec_patch_sizes]
        self.share_weights = share_weights

        self.router = MultiScaleRouter(code_dim, self.n_levels, k_dims)

        if share_weights:
            # Shared transformer, per-level projections
            self.enc_in_projs = nn.ModuleList([nn.Linear(epd, hidden_dim) for epd in self.enc_patch_dims])
            self.enc_transformer = TransformerBlockRoPE(hidden_dim, n_layers, n_query_heads, n_kv_heads)
            self.enc_out_proj = nn.Linear(hidden_dim, code_dim)

            self.dec_in_proj = nn.Linear(code_dim, hidden_dim)
            self.dec_transformer = TransformerBlockRoPE(hidden_dim, n_layers, n_query_heads, n_kv_heads)
            self.dec_neighbor = SwiGLUNeighborHead(hidden_dim)
            self.dec_out_projs = nn.ModuleList([nn.Linear(hidden_dim, dpd) for dpd in self.dec_patch_dims])
        else:
            # Separate encoder/decoder per level
            self.encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(epd, hidden_dim),
                    TransformerBlockRoPE(hidden_dim, n_layers, n_query_heads, n_kv_heads),
                    nn.Linear(hidden_dim, code_dim)
                )
                for epd in self.enc_patch_dims
            ])

            self.decoders = nn.ModuleList([
                SwiGLUNeighborDecoder(code_dim, hidden_dim, dpd, n_layers, n_query_heads, n_kv_heads)
                for dpd in self.dec_patch_dims
            ])

    def patchify(self, images, patch_size):
        B, C, H, W = images.shape
        p = patch_size
        n_h, n_w = H // p, W // p
        patches = images.view(B, C, n_h, p, n_w, p)
        patches = patches.permute(0, 2, 4, 3, 5, 1).contiguous()
        return patches.view(B, n_h * n_w, p * p * C)

    def unpatchify_to_canvas(self, patches, patch_size, canvas_size):
        B = patches.shape[0]
        p = patch_size
        n_h = n_w = canvas_size // p
        patches = patches.view(B, n_h, n_w, p, p, 3)
        patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
        return patches.view(B, 3, canvas_size, canvas_size)

    def forward(self, images, k_override=None):
        B = images.shape[0]
        device = images.device

        masks, indices, ks = self.router(B, device, k_override)

        all_soft = []
        all_hard = []
        level_recons = []
        cumulative_recon = torch.zeros_like(images)

        for level in range(self.n_levels):
            enc_ps = self.enc_patch_sizes[level]
            dec_ps = self.dec_patch_sizes[level]
            dec_canvas = self.dec_canvas_sizes[level]
            level_mask = masks[:, level, :]

            if level > 0:
                residual = (images - cumulative_recon.detach()) * self.residual_scale
            else:
                residual = images

            patches = self.patchify(residual, enc_ps)

            # Encode
            if self.share_weights:
                h = self.enc_in_projs[level](patches)
                h = self.enc_transformer(h)
                logits = self.enc_out_proj(h)
            else:
                logits = self.encoders[level](patches)

            # Binary quantization
            soft = torch.sigmoid(logits)
            hard = (soft > 0.5).float()
            hard = soft + (hard - soft).detach()  # STE
            hard = hard * 2 - 1  # normalize to [-1, 1]

            masked_soft = soft * level_mask.unsqueeze(1)
            masked_hard = hard * level_mask.unsqueeze(1)

            all_soft.append(masked_soft)
            all_hard.append(masked_hard)

            # Decode
            if self.share_weights:
                h = self.dec_in_proj(masked_hard)
                h = self.dec_transformer(h)
                h = self.dec_neighbor(h)
                decoded_patches = self.dec_out_projs[level](h)
            else:
                decoded_patches = self.decoders[level](masked_hard)
            decoded_small = self.unpatchify_to_canvas(decoded_patches, dec_ps, dec_canvas)

            if dec_canvas != self.image_size:
                decoded_img = F.interpolate(decoded_small, size=(self.image_size, self.image_size),
                                           mode='bilinear', align_corners=False)
            else:
                decoded_img = decoded_small

            if level > 0:
                decoded_img = decoded_img / self.residual_scale

            cumulative_recon = cumulative_recon + decoded_img
            level_recons.append(cumulative_recon.clone())

        return {
            'recon': level_recons[-1],
            'level_recons': level_recons,
            'soft': all_soft,
            'hard': all_hard,
            'indices': indices,
        }


def save_comparison_image(originals, level_recons, path, n_samples=4):
    originals = originals[:n_samples].detach().cpu().clamp(0, 1).numpy().transpose(0, 2, 3, 1)
    recons = [r[:n_samples].detach().cpu().clamp(0, 1).numpy().transpose(0, 2, 3, 1) for r in level_recons]
    rows = []
    for i in range(n_samples):
        row = [originals[i]] + [r[i] for r in recons]
        rows.append(np.concatenate(row, axis=1))
    grid = np.concatenate(rows, axis=0)
    Image.fromarray((grid * 255).astype(np.uint8)).save(path)


def get_image_paths(folder_path):
    folder = Path(folder_path)
    paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
        paths.extend(folder.glob(ext))
    return paths


def get_k_for_step(step, k_starts, k_ends, anneal_steps=2000):
    """Exponential decay per-level from k_starts to k_ends over anneal_steps."""
    t = min(step / anneal_steps, 1.0)
    ks = []
    for k_start, k_end in zip(k_starts, k_ends):
        k = k_start * ((k_end / k_start) ** t)
        ks.append(max(int(round(k)), k_end))
    return ks


def main():
    device = 'cuda:0'
    script_dir = Path(__file__).parent
    with open(script_dir / "config.yaml") as f:
        config = yaml.safe_load(f)

    data_folder = config["data_folder"]
    batch_size = config["batch_size"]
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    max_steps = config["max_steps"]
    image_size = config["image_size"]

    # Support both explicit patches_per_level OR derived from grid_size
    has_patches = "patches_per_level" in config
    has_grid = "grid_size" in config
    assert not (has_patches and has_grid), "Specify patches_per_level OR grid_size, not both"
    assert has_patches or has_grid, "Must specify patches_per_level or grid_size"

    if has_patches:
        patches_per_level = config["patches_per_level"]
        n_levels = len(patches_per_level)
    else:
        grid_size = config["grid_size"]
        n_levels = config["n_levels"]
        patches_per_level = [grid_size * grid_size] * n_levels

    # Support both explicit dec_canvas_sizes OR derived
    if "dec_canvas_sizes" in config:
        dec_canvas_sizes = config["dec_canvas_sizes"]
    else:
        # Exponential spacing from small to image_size
        dec_canvas_sizes = [image_size // (2 ** (n_levels - 1 - i)) for i in range(n_levels)]
    hidden_dim = config["hidden_dim"]
    code_dim = config["code_dim"]
    n_layers = config["n_layers"]
    residual_scale = float(config["residual_scale"])
    share_weights = config["share_weights"]
    n_query_heads = config["n_query_heads"]
    n_kv_heads = config["n_kv_heads"]
    output_dir = script_dir / config["output_dir"]
    save_interval_seconds = config["save_interval_seconds"]
    k_starts = config["k_starts"]
    k_ends = config["k_ends"]
    anneal_steps = config["anneal_steps"]
    clear_samples_on_launch = config["clear_samples_on_launch"]
    loss_fn = get_loss_fn(config["loss_fn"])
    output_dir.mkdir(exist_ok=True)

    if clear_samples_on_launch:
        for png in output_dir.glob("*.png"):
            png.unlink()

    model = MultiScaleFSQAutoencoder(
        image_size=image_size,
        patches_per_level=patches_per_level,
        dec_canvas_sizes=dec_canvas_sizes,
        n_levels=n_levels,
        hidden_dim=hidden_dim,
        code_dim=code_dim,
        k_dims=k_starts,
        n_layers=n_layers,
        residual_scale=residual_scale,
        n_query_heads=n_query_heads,
        n_kv_heads=n_kv_heads,
        share_weights=share_weights
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"SwiGLU 9-way + RoPE + SwiGLU FFN + post-SDPA gate | patches: {'+'.join(map(str, patches_per_level))}")
    print(f"Binary FSQ (2 levels: -1, 1)")
    print(f"Per-level k annealing: {k_starts} -> {k_ends} over {anneal_steps} steps")
    print(f"Batch size: {batch_size}, Total steps: {max_steps}")
    print("=" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_steps)

    all_paths = get_image_paths(data_folder)
    print(f"Found {len(all_paths)} images")

    dataset = ImageDataset(all_paths, size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    model.train()
    step = 0
    start_time = time.time()
    last_image_time = start_time

    print("Starting training...")

    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break

            batch = batch.to(device, non_blocking=True)

            # Get current per-level k
            ks = get_k_for_step(step, k_starts, k_ends, anneal_steps)

            optimizer.zero_grad()
            output = model(batch, k_override=ks)

            loss, loss_info = loss_fn(output, batch)
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1
            current_time = time.time()

            if step % 1 == 0:
                elapsed = current_time - start_time
                patches_str = '+'.join([str(h.shape[1]) for h in output['hard']])
                ks_str = '/'.join([str(k) for k in ks])
                loss_str = ' | '.join(f"{k}: {v:.5f}" if isinstance(v, float) else f"{k}: {v}"
                                       for k, v in loss_info.items() if k != 'per_level')
                print(f"Step {step:5d} | k=[{ks_str}] | {loss_str} | "
                      f"patches: {patches_str} | {elapsed:.1f}s", flush=True)

            if current_time - last_image_time >= save_interval_seconds:
                img_path = output_dir / f"comparison_step{step:06d}.png"
                save_comparison_image(batch, output['level_recons'], str(img_path))
                print(f"  -> Saved {img_path}")
                last_image_time = current_time

            if step % 1000 == 0:
                torch.save({
                    'step': step,
                    'ks': ks,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, output_dir / f"checkpoint_step{step:05d}.pt")
                print(f"  Saved checkpoint at step {step}")

    torch.save({
        'step': step,
        'ks': k_ends,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, output_dir / "checkpoint_final.pt")
    print(f"Training complete! Final checkpoint saved.")


if __name__ == "__main__":
    main()
