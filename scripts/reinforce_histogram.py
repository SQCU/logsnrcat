#!/usr/bin/env python3
"""
REINFORCE trainer for histogram fidelity via eval server API.

Uses histogram divergence as reward signal to fine-tune AE via low-rank
adapters. Goal: improve foreground (Vaporeon) color capture without
large optimizer state overhead.

Architecture:
    - LoRA-style adapters on encoder/decoder projection layers
    - Histogram divergence reward (lower = better)
    - REINFORCE policy gradient on adapter weights only
    - Interactive training via eval server API

Usage:
    python scripts/reinforce_histogram.py --vaporeon --steps 100
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests

DEFAULT_HOST = "172.26.160.1"
DEFAULT_PORT = 8421
DEFAULT_OUTPUT = "experiments_swiglu_ae/main_run_096"


def eval_code(code: str, host: str, port: int, timeout: int = 180) -> dict:
    """Execute Python code on eval server."""
    url = f"http://{host}:{port}/eval"
    resp = requests.post(url, json={"code": code}, timeout=timeout)
    return resp.json()


def format_error(error) -> str:
    """Format error from eval server (handles both string and dict formats)."""
    if isinstance(error, str):
        return error
    if isinstance(error, dict):
        msg = f"{error.get('type', 'Error')}: {error.get('message', 'Unknown error')}"
        if error.get('traceback'):
            msg += f"\n{error['traceback']}"
        return msg
    return str(error)


def save_histogram_grid(host: str, port: int, step: int, n_samples: int,
                        output_dir: Path, run_id: str, vaporeon: bool, n_bins: int = 32):
    """Generate and save histogram comparison grid at current policy state."""
    viz_code = f'''
import torch
import numpy as np

# Get fresh test batch
blocks = ctx._rl_iterator.generate_batch_list(batch_size={n_samples * 4}, resolution=64)
matching = [b.content for b in blocks if b.content.shape[-1] == 64][:{n_samples}]
images = torch.stack(matching).to(ctx.device)

ae = model.sparse_ae
p = ae.patch_size
grid_shape = (64 // p, 64 // p)
encoder_masks, decoder_masks = ae.build_masks(grid_shape, images.device)

with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    codes_list = ae.encode(images, grid_shape=grid_shape,
                           encoder_masks=encoder_masks, decoder_masks=decoder_masks)

    # Apply LoRA adapters if available
    if hasattr(ctx, '_lora_enc') and hasattr(ctx, '_lora_dec'):
        codes_adapted = []
        for codes in codes_list:
            adapted = codes + ctx._lora_enc(codes)
            adapted = adapted + ctx._lora_dec(adapted)
            codes_adapted.append(adapted)
        recon = ae.decode(codes_adapted, grid_shape, decoder_masks)
    else:
        recon = ae.decode(codes_list, grid_shape, decoder_masks)

# Convert to numpy (detach to be safe)
images_np = images.detach().float().cpu().numpy()
recon_np = recon.detach().float().cpu().numpy()

# Compute histograms
def compute_histogram(img_batch, n_bins):
    B, C, H, W = img_batch.shape
    hists = []
    for c in range(C):
        channel_data = img_batch[:, c, :, :].flatten()
        hist, _ = np.histogram(channel_data, bins=n_bins, range=(0, 1), density=True)
        hists.append((hist / (hist.sum() + 1e-10)).tolist())
    return hists

hist_input = compute_histogram(images_np, {n_bins})
hist_recon = compute_histogram(recon_np, {n_bins})

# JS divergence
def js_div(p, q):
    p, q = np.array(p), np.array(q)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + 1e-10) / (m + 1e-10)))
    kl_qm = np.sum(q * np.log((q + 1e-10) / (m + 1e-10)))
    return 0.5 * kl_pm + 0.5 * kl_qm

js_per_channel = [js_div(hist_input[c], hist_recon[c]) for c in range(3)]

ctx._hist_viz = {{
    "images": [images_np[i].transpose(1,2,0).clip(0,1).tolist() for i in range({n_samples})],
    "recons": [recon_np[i].transpose(1,2,0).clip(0,1).tolist() for i in range({n_samples})],
    "hist_input": hist_input,
    "hist_recon": hist_recon,
    "js_per_channel": js_per_channel,
    "js_mean": float(np.mean(js_per_channel)),
    "mse": float(F.mse_loss(recon.float(), images.float()).item())
}}
'''
    result = eval_code(viz_code, host, port)
    if not result['success']:
        print(f"    Warning: Could not generate viz at step {step}: {format_error(result['error'])}")
        return

    fetch = eval_code("ctx._hist_viz", host, port)
    if not fetch['success']:
        return

    viz_data = fetch['result']

    # Create figure: top row = histograms, bottom row = sample images
    fig = plt.figure(figsize=(14, 6))

    # Histograms
    colors = ['red', 'green', 'blue']
    channel_names = ['R', 'G', 'B']
    bins = np.linspace(0, 1, n_bins)

    for c in range(3):
        ax = fig.add_subplot(2, 4, c + 1)
        ax.bar(bins, viz_data['hist_input'][c], width=1/n_bins, alpha=0.5, label='Input', color=colors[c])
        ax.bar(bins, viz_data['hist_recon'][c], width=1/n_bins, alpha=0.5, label='Recon', color='gray')
        ax.set_title(f'{channel_names[c]} JS={viz_data["js_per_channel"][c]:.4f}')
        ax.legend(fontsize=7)

    # Summary
    ax = fig.add_subplot(2, 4, 4)
    ax.axis('off')
    ax.text(0.1, 0.5, f"Step {step}\n\nJS Mean: {viz_data['js_mean']:.4f}\nMSE: {viz_data['mse']:.6f}",
            fontsize=11, family='monospace', verticalalignment='center', transform=ax.transAxes)

    # Sample images (input | recon pairs)
    for i in range(min(4, n_samples)):
        ax = fig.add_subplot(2, 4, 5 + i)
        img_in = np.array(viz_data['images'][i])
        img_re = np.array(viz_data['recons'][i])
        combined = np.concatenate([img_in, img_re], axis=1)
        ax.imshow(np.clip(combined, 0, 1))
        ax.set_title(f'Sample {i+1}: In | Recon')
        ax.axis('off')

    suffix = "_vaporeon" if vaporeon else ""
    plt.suptitle(f'Histogram Fidelity @ Step {step} ({run_id})', fontsize=12)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"histogram_step{step:04d}_{run_id}{suffix}.png"
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description="REINFORCE histogram trainer")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--steps", type=int, default=50, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA adapter rank")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adapter learning rate")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--vaporeon", action="store_true")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Save histogram image grid every N steps (0 to disable)")
    parser.add_argument("--n-viz-samples", type=int, default=4,
                        help="Number of samples in periodic visualization grids")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    if args.run_id is None:
        import datetime
        args.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Connecting to eval server at http://{args.host}:{args.port}...")
    health = requests.get(f"http://{args.host}:{args.port}/health").json()
    print(f"  Status: {health['status']}, Weights: {health['weights_loaded']}")

    # Setup: create LoRA adapters and Vaporeon iterator
    setup_code = f'''
import torch
import torch.nn as nn
import numpy as np

# LoRA adapter module
class LoRAAdapter(nn.Module):
    """Low-rank adapter: out = x + scale * B @ A @ x"""
    def __init__(self, in_dim, out_dim, rank={args.lora_rank}, scale=0.001):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_dim, rank))
        self.scale = scale

    def forward(self, x):
        # x: [..., in_dim] -> [..., out_dim]
        # Adapter adds: scale * (x @ A.T @ B.T)
        return self.scale * (x @ self.A.T @ self.B.T)

# Find encoder/decoder projection layers to patch
ae = model.sparse_ae
code_dim = ae.code_dim  # 128

# Create adapters for code space (code_dim -> code_dim)
# Applied after encode (refine codes) and before decode (adjust for decode)
# Use fp32 for training stability (will convert codes to fp32 during adaptation)
ctx._lora_enc = LoRAAdapter(code_dim, code_dim, rank={args.lora_rank}).to(ctx.device)
ctx._lora_dec = LoRAAdapter(code_dim, code_dim, rank={args.lora_rank}).to(ctx.device)

# Optimizer for adapter weights only
ctx._lora_optimizer = torch.optim.Adam(
    list(ctx._lora_enc.parameters()) + list(ctx._lora_dec.parameters()),
    lr={args.lr}
)

# Setup iterator
'''

    # Build iterator setup code conditionally (outside f-string)
    if args.vaporeon:
        iterator_setup = '''
from src.sprite_atlas import SpriteAtlasIterator
vaporeon_config = {
    "data_dir": "data/infinite_fusion",
    "sampling_config": {
        "split": "all", "mode": "uniform_sprites",
        "adjustment_mode": "additive", "temperature": 1.0, "seed": 42,
        "adjustments": {"134": 10.0, "*.134": 10.0}
    },
    "render_config": {"res_scaling": "do_not", "background_mode": "solid_random", "jitter": True}
}
ctx._rl_iterator = SpriteAtlasIterator(ctx.device, vaporeon_config)
'''
    else:
        iterator_setup = '''
ctx._rl_iterator = ctx.iterator
'''

    setup_code += iterator_setup + f'''

# Training history
ctx._rl_history = {{"step": [], "reward": [], "mse": [], "js_div": [], "loss": []}}

n_adapter_params = sum(p.numel() for p in ctx._lora_enc.parameters()) + sum(p.numel() for p in ctx._lora_dec.parameters())
f"LoRA adapters ready: {{n_adapter_params}} params (rank={args.lora_rank})"
'''

    print(f"\nSetting up LoRA adapters (rank={args.lora_rank})...")
    result = eval_code(setup_code, args.host, args.port)
    if not result['success']:
        print(f"ERROR: {format_error(result['error'])}")
        return
    print(f"  {result['result']}")

    # Training loop
    train_step_code = f'''
import torch
import torch.nn.functional as F
import numpy as np

batch_size = {args.batch_size}
resolution = {args.resolution}

# Get batch
blocks = ctx._rl_iterator.generate_batch_list(batch_size=batch_size * 4, resolution=resolution)
matching = [b.content for b in blocks if b.content.shape[-1] == resolution][:batch_size]
images = torch.stack(matching).to(ctx.device)

ae = model.sparse_ae
p = ae.patch_size
H, W = images.shape[2], images.shape[3]
grid_shape = (H // p, W // p)
encoder_masks, decoder_masks = ae.build_masks(grid_shape, images.device)

# Forward with LoRA adapters
# Note: encode returns list of tensors (one per level)
with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    # Encode - returns list of [B, n_patches, code_dim] per level
    codes_list = ae.encode(images, grid_shape=grid_shape,
                           encoder_masks=encoder_masks, decoder_masks=decoder_masks)

    # Apply LoRA adapters to each level's codes (adapters already in bf16)
    codes_adapted = []
    for codes in codes_list:
        adapted = codes + ctx._lora_enc(codes)
        adapted = adapted + ctx._lora_dec(adapted)
        codes_adapted.append(adapted)

    # Decode with adapted codes
    recon = ae.decode(codes_adapted, grid_shape, decoder_masks)

# Compute histogram divergence (reward signal)
def compute_js_divergence(imgs, recons, n_bins=32):
    imgs_np = imgs.float().cpu().numpy()
    recons_np = recons.float().cpu().numpy()

    js_total = 0.0
    for c in range(3):
        hist_in, _ = np.histogram(imgs_np[:, c].flatten(), bins=n_bins, range=(0, 1), density=True)
        hist_re, _ = np.histogram(recons_np[:, c].flatten(), bins=n_bins, range=(0, 1), density=True)
        hist_in = hist_in / (hist_in.sum() + 1e-10)
        hist_re = hist_re / (hist_re.sum() + 1e-10)

        m = 0.5 * (hist_in + hist_re)
        kl_pm = np.sum(hist_in * np.log((hist_in + 1e-10) / (m + 1e-10)))
        kl_qm = np.sum(hist_re * np.log((hist_re + 1e-10) / (m + 1e-10)))
        js_total += 0.5 * kl_pm + 0.5 * kl_qm

    return js_total / 3.0

with torch.no_grad():
    js_div = compute_js_divergence(images, recon)
    mse = F.mse_loss(recon.float(), images.float()).item()

# REINFORCE: reward = -js_div (lower divergence = higher reward)
# Baseline: running mean of rewards
if not hasattr(ctx, '_reward_baseline'):
    ctx._reward_baseline = -js_div
else:
    ctx._reward_baseline = 0.9 * ctx._reward_baseline + 0.1 * (-js_div)

reward = -js_div
advantage = reward - ctx._reward_baseline

ctx._lora_optimizer.zero_grad()

# Recompute with gradients in fp32 for training stability
# Encode with frozen AE (no grad through AE, only through adapters)
with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    codes_list = ae.encode(images, grid_shape=grid_shape,
                           encoder_masks=encoder_masks, decoder_masks=decoder_masks)
    # Detach and convert to fp32
    codes_list = [c.detach().float() for c in codes_list]

# Apply adapters in fp32 (these have gradients)
codes_adapted = []
for codes in codes_list:
    adapted = codes + ctx._lora_enc(codes)
    adapted = adapted + ctx._lora_dec(adapted)
    codes_adapted.append(adapted)

# Decode in fp32
with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
    recon = ae.decode([c.to(torch.bfloat16) for c in codes_adapted], grid_shape, decoder_masks)
    recon = recon.float()

# Differentiable soft histogram loss (aligns training objective with reward)
def soft_histogram_loss(img, ref, n_bins=32, sigma=0.05):
    \"\"\"Soft histogram matching loss - differentiable approximation of JS divergence.\"\"\"
    B, C, H, W = img.shape
    bins = torch.linspace(0, 1, n_bins, device=img.device, dtype=img.dtype)

    total_loss = 0.0
    for c in range(C):
        img_flat = img[:, c].reshape(-1)  # [B*H*W]
        ref_flat = ref[:, c].reshape(-1)

        # Soft histogram: Gaussian kernel around each bin
        # [n_pixels, n_bins] distance matrix
        img_dists = (img_flat.unsqueeze(1) - bins.unsqueeze(0)) / sigma
        ref_dists = (ref_flat.unsqueeze(1) - bins.unsqueeze(0)) / sigma

        # Soft bin counts via Gaussian
        img_hist = torch.exp(-0.5 * img_dists ** 2).sum(0)
        ref_hist = torch.exp(-0.5 * ref_dists ** 2).sum(0)

        # Normalize to probability
        img_hist = img_hist / (img_hist.sum() + 1e-8)
        ref_hist = ref_hist / (ref_hist.sum() + 1e-8)

        # JS divergence (differentiable)
        m = 0.5 * (img_hist + ref_hist)
        kl_pm = (ref_hist * torch.log((ref_hist + 1e-8) / (m + 1e-8))).sum()
        kl_qm = (img_hist * torch.log((img_hist + 1e-8) / (m + 1e-8))).sum()
        total_loss += 0.5 * kl_pm + 0.5 * kl_qm

    return total_loss / C

mse_loss = F.mse_loss(recon, images.float())
hist_loss = soft_histogram_loss(recon, images.float())

# L2 regularization on adapter weights to prevent drift
l2_reg = 0.0
for p in list(ctx._lora_enc.parameters()) + list(ctx._lora_dec.parameters()):
    l2_reg += (p ** 2).sum()

# Combined loss: MSE + histogram (weighted by advantage) + regularization
# When advantage > 0 (doing better), emphasize histogram; when < 0, emphasize MSE stability
hist_weight = 1.0 + max(0, advantage * 5.0)  # 1.0 to ~1.35
loss = mse_loss + hist_weight * hist_loss + 0.01 * l2_reg

loss.backward()
ctx._lora_optimizer.step()

# Log
step_num = len(ctx._rl_history["step"])
ctx._rl_history["step"].append(step_num)
ctx._rl_history["reward"].append(reward)
ctx._rl_history["mse"].append(mse)
ctx._rl_history["js_div"].append(js_div)
ctx._rl_history["loss"].append(loss.item())
if "hist_loss" not in ctx._rl_history:
    ctx._rl_history["hist_loss"] = []
ctx._rl_history["hist_loss"].append(hist_loss.item())

# Store step result for retrieval
ctx._last_step_result = {{"step": step_num, "reward": reward, "mse": mse, "js_div": js_div, "loss": loss.item(), "hist_loss": hist_loss.item(), "advantage": advantage}}
'''

    print(f"\nTraining for {args.steps} steps...")
    print("-" * 70)
    print(f"{'Step':>6} {'Reward':>10} {'JS Div':>10} {'MSE':>10} {'Hist Loss':>10} {'Adv':>8}")
    print("-" * 70)

    for step in range(args.steps):
        result = eval_code(train_step_code, args.host, args.port)
        if not result['success']:
            print(f"ERROR at step {step}: {format_error(result['error'])}")
            break

        # Fetch step result
        fetch = eval_code("ctx._last_step_result", args.host, args.port)
        if not fetch['success']:
            print(f"ERROR fetching step result: {format_error(fetch['error'])}")
            break

        metrics = fetch['result']
        if step % 5 == 0 or step == args.steps - 1:
            print(f"{metrics['step']:>6} {metrics['reward']:>10.4f} {metrics['js_div']:>10.4f} "
                  f"{metrics['mse']:>10.6f} {metrics['hist_loss']:>10.4f} {metrics['advantage']:>+8.4f}")

        # Periodic image saving
        if args.save_interval > 0 and (step % args.save_interval == 0 or step == args.steps - 1):
            save_histogram_grid(args.host, args.port, step, args.n_viz_samples,
                                Path(args.output_dir), args.run_id, args.vaporeon)

    # Fetch training history and plot
    print("\nFetching training history...")
    history = eval_code("ctx._rl_history", args.host, args.port)['result']

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    axes[0, 0].plot(history['step'], history['reward'])
    axes[0, 0].set_title('Reward (-JS Divergence)')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].axhline(y=np.mean(history['reward'][-10:]), color='r', linestyle='--', alpha=0.5)

    axes[0, 1].plot(history['step'], history['js_div'])
    axes[0, 1].set_title('JS Divergence (eval)')
    axes[0, 1].set_xlabel('Step')

    axes[0, 2].plot(history['step'], history.get('hist_loss', [0] * len(history['step'])))
    axes[0, 2].set_title('Soft Histogram Loss (train)')
    axes[0, 2].set_xlabel('Step')

    axes[1, 0].plot(history['step'], history['mse'])
    axes[1, 0].set_title('MSE')
    axes[1, 0].set_xlabel('Step')

    axes[1, 1].plot(history['step'], history['loss'])
    axes[1, 1].set_title('Total Loss')
    axes[1, 1].set_xlabel('Step')

    # Improvement over baseline
    if len(history['reward']) > 1:
        baseline = history['reward'][0]
        improvement = [(r - baseline) for r in history['reward']]
        axes[1, 2].plot(history['step'], improvement)
        axes[1, 2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 2].set_title('Reward vs Baseline')
        axes[1, 2].set_xlabel('Step')
    else:
        axes[1, 2].axis('off')

    plt.suptitle(f'REINFORCE Histogram Training ({"Vaporeon" if args.vaporeon else "Mixed"}, '
                 f'rank={args.lora_rank}, lr={args.lr})', fontsize=12)
    plt.tight_layout()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_vaporeon" if args.vaporeon else ""
    output_path = output_dir / f"reinforce_histogram{suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved training plot to: {output_path}")
    print(f"\nFinal metrics (last 10 steps):")
    print(f"  Reward: {np.mean(history['reward'][-10:]):.4f}")
    print(f"  JS Div: {np.mean(history['js_div'][-10:]):.4f}")
    print(f"  MSE: {np.mean(history['mse'][-10:]):.6f}")
    if 'hist_loss' in history and history['hist_loss']:
        print(f"  Hist Loss: {np.mean(history['hist_loss'][-10:]):.4f}")

    # Show improvement from start
    if len(history['reward']) > 1:
        start_reward = history['reward'][0]
        end_reward = np.mean(history['reward'][-10:])
        print(f"\n  Improvement: {end_reward - start_reward:+.4f} (start: {start_reward:.4f})")


if __name__ == "__main__":
    main()
