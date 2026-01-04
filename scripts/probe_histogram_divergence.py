#!/usr/bin/env python3
"""
Probe histogram divergence between input and reconstructed images.

Tests whether the AE captures foreground objects (Vaporeon) or "cheats"
by reconstructing backgrounds well while mangling the sprite.

Vaporeon sprites have consistent blue tones; random backgrounds don't.
If histogram divergence is high, AE is losing foreground color info.

Usage:
    python scripts/probe_histogram_divergence.py [--vaporeon]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests

DEFAULT_HOST = "172.26.160.1"
DEFAULT_PORT = 8421
DEFAULT_OUTPUT = "experiments_swiglu_ae/main_run_091"


def eval_code(code: str, host: str, port: int, timeout: int = 120) -> dict:
    """Execute Python code on eval server."""
    url = f"http://{host}:{port}/eval"
    resp = requests.post(url, json={"code": code}, timeout=timeout)
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Probe histogram divergence")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--n-bins", type=int, default=32, help="Histogram bins per channel")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--vaporeon", action="store_true")
    args = parser.parse_args()

    print(f"Connecting to eval server at http://{args.host}:{args.port}...")
    health = requests.get(f"http://{args.host}:{args.port}/health").json()
    print(f"  Status: {health['status']}, Weights: {health['weights_loaded']}")

    # Setup iterator (Vaporeon-biased or composite)
    if args.vaporeon:
        setup_code = f'''
from src.sprite_atlas import SpriteAtlasIterator

vaporeon_config = {{
    "data_dir": "data/infinite_fusion",
    "sampling_config": {{
        "split": "all",
        "mode": "uniform_sprites",
        "adjustment_mode": "additive",
        "temperature": 1.0,
        "seed": 42,
        "adjustments": {{"134": 10.0, "*.134": 10.0}}
    }},
    "render_config": {{
        "res_scaling": "do_not",
        "background_mode": "solid_random",
        "jitter": True
    }}
}}
ctx._hist_iterator = SpriteAtlasIterator(ctx.device, vaporeon_config)
"Vaporeon iterator ready"
'''
        print("\nSetting up Vaporeon iterator...")
    else:
        setup_code = '''
ctx._hist_iterator = ctx.iterator
"Using composite iterator"
'''
        print("\nUsing composite iterator...")

    result = eval_code(setup_code, args.host, args.port)
    if not result['success']:
        print(f"ERROR: {result['error']}")
        return

    # Generate images and compute histograms
    hist_code = f'''
import torch
import numpy as np

n_samples = {args.n_samples}
resolution = {args.resolution}
n_bins = {args.n_bins}

# Get images (filter by resolution for composite iterator)
blocks = ctx._hist_iterator.generate_batch_list(batch_size=n_samples * 4, resolution=resolution)
matching = [b.content for b in blocks if b.content.shape[-1] == resolution][:n_samples]
images = torch.stack(matching).to(ctx.device)

# Encode/decode
ae = model.sparse_ae
p = ae.patch_size
H, W = images.shape[2], images.shape[3]
grid_shape = (H // p, W // p)
encoder_masks, decoder_masks = ae.build_masks(grid_shape, images.device)

with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    codes = ae.encode(images, grid_shape=grid_shape,
                      encoder_masks=encoder_masks, decoder_masks=decoder_masks)
    recon = ae.decode(codes, grid_shape, decoder_masks)

# Convert to numpy for histogram computation
images_np = images.float().cpu().numpy()  # [B, C, H, W]
recon_np = recon.float().cpu().numpy()

# Compute per-channel histograms
def compute_histogram(img_batch, n_bins):
    """Compute normalized histogram for each channel, averaged over batch."""
    B, C, H, W = img_batch.shape
    hists = []
    for c in range(C):
        channel_data = img_batch[:, c, :, :].flatten()
        hist, _ = np.histogram(channel_data, bins=n_bins, range=(0, 1), density=True)
        hists.append(hist / hist.sum())  # Normalize to probability
    return np.array(hists)  # [C, n_bins]

hist_input = compute_histogram(images_np, n_bins)
hist_recon = compute_histogram(recon_np, n_bins)

# Compute divergence metrics
def kl_divergence(p, q, eps=1e-10):
    """KL(P || Q) - how much info lost encoding P as Q."""
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.sum(p * np.log(p / q))

def js_divergence(p, q):
    """Jensen-Shannon divergence - symmetric, bounded [0, ln(2)]."""
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def hellinger_distance(p, q):
    """Hellinger distance - bounded [0, 1]."""
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))

# Per-channel metrics
metrics = {{
    "kl_per_channel": [kl_divergence(hist_input[c], hist_recon[c]) for c in range(3)],
    "js_per_channel": [js_divergence(hist_input[c], hist_recon[c]) for c in range(3)],
    "hellinger_per_channel": [hellinger_distance(hist_input[c], hist_recon[c]) for c in range(3)],
}}

# Aggregate metrics
metrics["kl_mean"] = np.mean(metrics["kl_per_channel"])
metrics["js_mean"] = np.mean(metrics["js_per_channel"])
metrics["hellinger_mean"] = np.mean(metrics["hellinger_per_channel"])

# MSE for comparison
metrics["mse"] = float(F.mse_loss(torch.from_numpy(recon_np), torch.from_numpy(images_np)).item())

# Store histograms for plotting
ctx._hist_metrics = metrics
ctx._hist_input = hist_input.tolist()
ctx._hist_recon = hist_recon.tolist()
ctx._hist_images = images_np[:4].tolist()  # First 4 for viz
ctx._hist_recons = recon_np[:4].tolist()

metrics
'''

    print(f"\nComputing histograms for {args.n_samples} images...")
    result = eval_code(hist_code, args.host, args.port)
    if not result['success']:
        print(f"ERROR: {result['error']}")
        return

    # Fetch results
    fetch = eval_code("ctx._hist_metrics", args.host, args.port)
    metrics = fetch['result']

    print("\n" + "="*60)
    print("HISTOGRAM DIVERGENCE METRICS")
    print("="*60)
    print(f"\nReconstruction MSE: {metrics['mse']:.6f}")
    print(f"\nKL Divergence (input || recon):")
    print(f"  R: {metrics['kl_per_channel'][0]:.4f}")
    print(f"  G: {metrics['kl_per_channel'][1]:.4f}")
    print(f"  B: {metrics['kl_per_channel'][2]:.4f}")
    print(f"  Mean: {metrics['kl_mean']:.4f}")
    print(f"\nJensen-Shannon Divergence:")
    print(f"  R: {metrics['js_per_channel'][0]:.4f}")
    print(f"  G: {metrics['js_per_channel'][1]:.4f}")
    print(f"  B: {metrics['js_per_channel'][2]:.4f}")
    print(f"  Mean: {metrics['js_mean']:.4f}")
    print(f"\nHellinger Distance:")
    print(f"  R: {metrics['hellinger_per_channel'][0]:.4f}")
    print(f"  G: {metrics['hellinger_per_channel'][1]:.4f}")
    print(f"  B: {metrics['hellinger_per_channel'][2]:.4f}")
    print(f"  Mean: {metrics['hellinger_mean']:.4f}")

    # Fetch histogram data for plotting
    hist_input = np.array(eval_code("ctx._hist_input", args.host, args.port)['result'])
    hist_recon = np.array(eval_code("ctx._hist_recon", args.host, args.port)['result'])
    images_viz = np.array(eval_code("ctx._hist_images", args.host, args.port)['result'])
    recons_viz = np.array(eval_code("ctx._hist_recons", args.host, args.port)['result'])

    # Plot histograms and examples
    fig = plt.figure(figsize=(14, 8))

    # Top row: histogram comparison per channel
    colors = ['red', 'green', 'blue']
    channel_names = ['R', 'G', 'B']
    bins = np.linspace(0, 1, args.n_bins + 1)[:-1]

    for c in range(3):
        ax = fig.add_subplot(2, 4, c + 1)
        ax.bar(bins, hist_input[c], width=1/args.n_bins, alpha=0.5, label='Input', color=colors[c])
        ax.bar(bins, hist_recon[c], width=1/args.n_bins, alpha=0.5, label='Recon', color='gray')
        ax.set_title(f'{channel_names[c]} Channel\nJS={metrics["js_per_channel"][c]:.4f}')
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)

    # Summary metrics
    ax = fig.add_subplot(2, 4, 4)
    ax.axis('off')
    summary = f"""Histogram Divergence Summary

MSE: {metrics['mse']:.6f}

KL Divergence: {metrics['kl_mean']:.4f}
JS Divergence: {metrics['js_mean']:.4f}
Hellinger Dist: {metrics['hellinger_mean']:.4f}

Lower = better color fidelity
High divergence = losing foreground"""
    ax.text(0.1, 0.5, summary, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax.transAxes)

    # Bottom row: example images
    for i in range(4):
        ax = fig.add_subplot(2, 4, 5 + i)
        # Stack input and recon side by side
        img_in = np.transpose(images_viz[i], (1, 2, 0))
        img_re = np.transpose(recons_viz[i], (1, 2, 0))
        combined = np.concatenate([img_in, img_re], axis=1)
        ax.imshow(np.clip(combined, 0, 1))
        ax.set_title(f'Sample {i+1}: In | Recon')
        ax.axis('off')

    plt.suptitle(f'Histogram Divergence Analysis ({"Vaporeon" if args.vaporeon else "Mixed"})',
                 fontsize=12, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_vaporeon" if args.vaporeon else ""
    output_path = output_dir / f"histogram_divergence{suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved plot to: {output_path}")


if __name__ == "__main__":
    main()
