#!/usr/bin/env python3
"""
Probe subspace ablation via eval server API.

Grabs Vaporeons from sprite atlas, tests wavelet vs amplitude ablation,
reports MSE/BCE losses at different ablation rates, and saves visual grids.

Usage:
    python scripts/probe_subspace_ablation.py [--host HOST] [--port PORT] [--output-dir DIR]
"""

import argparse
import base64
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests

# Default eval server address (Windows host from WSL)
DEFAULT_HOST = "172.26.160.1"
DEFAULT_PORT = 8421
DEFAULT_OUTPUT = "experiments_swiglu_ae/main_run_096"


def eval_code(code: str, host: str, port: int) -> dict:
    """Execute Python code on eval server and return result."""
    url = f"http://{host}:{port}/eval"
    resp = requests.post(url, json={"code": code}, timeout=120)
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Probe subspace ablation via eval server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Eval server host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Eval server port")
    parser.add_argument("--n-samples", type=int, default=8, help="Number of sprites to test")
    parser.add_argument("--n-display", type=int, default=4, help="Number of sprites to display in grid")
    parser.add_argument("--resolution", type=int, default=64, help="Image resolution")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT, help="Output directory for plots")
    parser.add_argument("--vaporeon", action="store_true",
                        help="Use dedicated SpriteAtlasIterator with strong Vaporeon (#134) bias")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    # Health check
    print(f"Connecting to eval server at {base_url}...")
    health = requests.get(f"{base_url}/health").json()
    print(f"  Status: {health['status']}")
    print(f"  Weights loaded: {health['weights_loaded']}")
    print(f"  Params: {health['params']:,}")

    if not health['weights_loaded']:
        print("ERROR: No weights loaded. Run training with yeet enabled first.")
        return

    # Setup code - imports and get sprites
    # Namespace has: ctx, model, ae, config, get_batch, torch, nn, F, autocast
    if args.vaporeon:
        # Create dedicated SpriteAtlasIterator with strong Vaporeon bias
        setup_code = f'''
from src.sprite_atlas import SpriteAtlasIterator

n_samples = {args.n_samples}
resolution = {args.resolution}

# Config with heavy Vaporeon (#134) bias via logit adjustments
# +10.0 additive logit makes Vaporeon ~22000x more likely than baseline
vaporeon_config = {{
    "data_dir": "data/infinite_fusion",
    "sampling_config": {{
        "split": "all",
        "mode": "uniform_sprites",
        "adjustment_mode": "additive",
        "temperature": 1.0,
        "seed": 42,
        "adjustments": {{
            "134": 10.0,      # Head ID 134 (Vaporeon) +10 logits
            "*.134": 10.0,    # Body ID 134 +10 logits
        }}
    }},
    "render_config": {{
        "res_scaling": "do_not",
        "background_mode": "solid_random",
        "jitter": True
    }}
}}

# Create dedicated iterator for Vaporeons
vaporeon_iterator = SpriteAtlasIterator(ctx.device, vaporeon_config)

# Generate batch - should be mostly Vaporeons now
blocks = vaporeon_iterator.generate_batch_list(batch_size=n_samples, resolution=resolution)
images = torch.stack([b.content for b in blocks]).to(ctx.device)

# Store for later use
ctx._probe_images = images
ctx._vaporeon_iterator = vaporeon_iterator  # Keep alive
f"Got {{images.shape[0]}} Vaporeon sprites at {{resolution}}px"
'''
        print(f"\nSetting up Vaporeon-specific iterator (ID #134, +10 logit bias)...")
    else:
        setup_code = f'''
# Get sprite images from composite iterator (full dataset mix)
n_samples = {args.n_samples}
resolution = {args.resolution}

# Use composite iterator - samples from full dataset mix
blocks = ctx.iterator.generate_batch_list(batch_size=n_samples * 4, resolution=resolution)
matching = [b.content for b in blocks if b.content.shape[-1] == resolution][:n_samples]
images = torch.stack(matching).to(ctx.device)

# Store for later use
ctx._probe_images = images
f"Got {{images.shape[0]}} images at {{resolution}}px"
'''
        print(f"\nUsing composite iterator (full dataset mix)...")

    print(f"\nFetching {args.n_samples} sprites at {args.resolution}px...")
    result = eval_code(setup_code, args.host, args.port)
    if not result['success']:
        print(f"ERROR: {result['error']}")
        return
    print(f"  {result['result']}")

    # Run ablation sweep
    ablation_code = '''
images = ctx._probe_images
ae = model.sparse_ae
p = ae.patch_size
H, W = images.shape[2], images.shape[3]
grid_shape = (H // p, W // p)

# Build masks
encoder_masks, decoder_masks = ae.build_masks(grid_shape, images.device)

# Encode once
with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    codes = ae.encode(images, grid_shape=grid_shape,
                      encoder_masks=encoder_masks, decoder_masks=decoder_masks)
    baseline = ae.decode(codes, grid_shape, decoder_masks)

# Ablation rates to test
rates = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

results = {"rates": rates, "wavelet": {"mse": [], "bce": []}, "amplitude": {"mse": [], "bce": []}}

# Collect reconstructions under autocast
recons_wav = []
recons_amp = []
with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    for rate in rates:
        recon_wav = ae.decode_with_ablation(codes, grid_shape,
                                            ablate_wavelet=rate, ablate_amplitude=0.0,
                                            decoder_masks=decoder_masks, deterministic=True)
        recons_wav.append(recon_wav.float())  # Convert to fp32 immediately

        recon_amp = ae.decode_with_ablation(codes, grid_shape,
                                            ablate_wavelet=0.0, ablate_amplitude=rate,
                                            decoder_masks=decoder_masks, deterministic=True)
        recons_amp.append(recon_amp.float())

# Compute losses outside autocast (BCE is unsafe under autocast)
images_f = images.float()
baseline_f = baseline.float()

for i, rate in enumerate(rates):
    mse_wav = F.mse_loss(recons_wav[i], images_f).item()
    bce_wav = F.binary_cross_entropy(recons_wav[i].clamp(1e-6, 1-1e-6), images_f).item()
    results["wavelet"]["mse"].append(mse_wav)
    results["wavelet"]["bce"].append(bce_wav)

    mse_amp = F.mse_loss(recons_amp[i], images_f).item()
    bce_amp = F.binary_cross_entropy(recons_amp[i].clamp(1e-6, 1-1e-6), images_f).item()
    results["amplitude"]["mse"].append(mse_amp)
    results["amplitude"]["bce"].append(bce_amp)

# Baseline metrics
baseline_mse = F.mse_loss(baseline_f, images_f).item()
baseline_bce = F.binary_cross_entropy(baseline_f.clamp(1e-6, 1-1e-6), images_f).item()
results["baseline"] = {"mse": baseline_mse, "bce": baseline_bce}

# Subspace info (n_wavelet_dims defaults to code_dim // 2 if None)
n_wav = ae.n_wavelet_dims if ae.n_wavelet_dims is not None else ae.code_dim // 2
n_amp = ae.code_dim - n_wav
results["subspace_dims"] = {"wavelet": n_wav, "amplitude": n_amp, "total": ae.code_dim, "k": ae.k_per_patch}

# Store for retrieval (exec doesn't return values)
ctx._ablation_results = results
'''

    print("\nRunning subspace ablation sweep...")
    result = eval_code(ablation_code, args.host, args.port)

    if not result['success']:
        print(f"ERROR: {result['error']}")
        return

    # Fetch stored results (exec doesn't return values)
    fetch_result = eval_code("ctx._ablation_results", args.host, args.port)
    if not fetch_result['success']:
        print(f"ERROR fetching results: {fetch_result['error']}")
        return

    data = fetch_result['result']

    # Pretty print results
    print("\n" + "="*70)
    print("SUBSPACE ABLATION RESULTS")
    print("="*70)

    dims = data['subspace_dims']
    print(f"\nCode dimensions: {dims['total']} total, k={dims['k']} active")
    print(f"  Wavelet subspace:   {dims['wavelet']} dims")
    print(f"  Amplitude subspace: {dims['amplitude']} dims")

    print(f"\nBaseline (no ablation):")
    print(f"  MSE: {data['baseline']['mse']:.6f}")
    print(f"  BCE: {data['baseline']['bce']:.6f}")

    print("\n" + "-"*70)
    print("WAVELET ABLATION (zeroing frequency/texture info)")
    print("-"*70)
    print(f"{'Rate':>8} {'MSE':>12} {'BCE':>12} {'MSE delta':>12} {'BCE delta':>12}")
    print("-"*70)
    for i, rate in enumerate(data['rates']):
        mse = data['wavelet']['mse'][i]
        bce = data['wavelet']['bce'][i]
        mse_d = mse - data['baseline']['mse']
        bce_d = bce - data['baseline']['bce']
        print(f"{rate:>8.0%} {mse:>12.6f} {bce:>12.6f} {mse_d:>+12.6f} {bce_d:>+12.6f}")

    print("\n" + "-"*70)
    print("AMPLITUDE ABLATION (zeroing intensity/color info)")
    print("-"*70)
    print(f"{'Rate':>8} {'MSE':>12} {'BCE':>12} {'MSE delta':>12} {'BCE delta':>12}")
    print("-"*70)
    for i, rate in enumerate(data['rates']):
        mse = data['amplitude']['mse'][i]
        bce = data['amplitude']['bce'][i]
        mse_d = mse - data['baseline']['mse']
        bce_d = bce - data['baseline']['bce']
        print(f"{rate:>8.0%} {mse:>12.6f} {bce:>12.6f} {mse_d:>+12.6f} {bce_d:>+12.6f}")

    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY: Which subspace matters more?")
    print("="*70)

    # Compare 100% ablation impact
    wav_100_mse = data['wavelet']['mse'][-1] - data['baseline']['mse']
    amp_100_mse = data['amplitude']['mse'][-1] - data['baseline']['mse']
    wav_100_bce = data['wavelet']['bce'][-1] - data['baseline']['bce']
    amp_100_bce = data['amplitude']['bce'][-1] - data['baseline']['bce']

    print(f"\nFull ablation impact (100% zeroed):")
    print(f"  Wavelet:   MSE +{wav_100_mse:.6f}, BCE +{wav_100_bce:.6f}")
    print(f"  Amplitude: MSE +{amp_100_mse:.6f}, BCE +{amp_100_bce:.6f}")

    if wav_100_mse > amp_100_mse:
        ratio = wav_100_mse / (amp_100_mse + 1e-8)
        print(f"\n  -> Wavelet subspace is {ratio:.1f}x more important (by MSE)")
    else:
        ratio = amp_100_mse / (wav_100_mse + 1e-8)
        print(f"\n  -> Amplitude subspace is {ratio:.1f}x more important (by MSE)")

    # Generate visual grid of ablated images
    print("\n" + "="*70)
    print("GENERATING VISUAL GRID")
    print("="*70)

    # Visualization rates (subset for cleaner grid)
    viz_rates = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_display = min(args.n_display, args.n_samples)

    # Code to generate image grid data on server
    viz_code = f'''
import torch
import io
import base64

# Get stored images and AE
images = ctx._probe_images[:{n_display}]
ae = model.sparse_ae
p = ae.patch_size
H, W = images.shape[2], images.shape[3]
grid_shape = (H // p, W // p)

encoder_masks, decoder_masks = ae.build_masks(grid_shape, images.device)

viz_rates = {viz_rates}
grid_data = {{"originals": [], "baseline": [], "wav_ablations": {{}}, "amp_ablations": {{}}}}

with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    codes = ae.encode(images, grid_shape=grid_shape,
                      encoder_masks=encoder_masks, decoder_masks=decoder_masks)
    baseline = ae.decode(codes, grid_shape, decoder_masks)

    # Convert to numpy lists (JSON serializable)
    for i in range({n_display}):
        grid_data["originals"].append(images[i].float().permute(1,2,0).cpu().numpy().clip(0,1).tolist())
        grid_data["baseline"].append(baseline[i].float().permute(1,2,0).cpu().numpy().clip(0,1).tolist())

    for rate in viz_rates:
        grid_data["wav_ablations"][rate] = []
        grid_data["amp_ablations"][rate] = []

        recon_wav = ae.decode_with_ablation(codes, grid_shape,
                                            ablate_wavelet=rate, ablate_amplitude=0.0,
                                            decoder_masks=decoder_masks, deterministic=True)
        recon_amp = ae.decode_with_ablation(codes, grid_shape,
                                            ablate_wavelet=0.0, ablate_amplitude=rate,
                                            decoder_masks=decoder_masks, deterministic=True)

        for i in range({n_display}):
            grid_data["wav_ablations"][rate].append(recon_wav[i].float().permute(1,2,0).cpu().numpy().clip(0,1).tolist())
            grid_data["amp_ablations"][rate].append(recon_amp[i].float().permute(1,2,0).cpu().numpy().clip(0,1).tolist())

ctx._viz_grid_data = grid_data
'''

    print(f"  Generating {n_display} exemplars at {len(viz_rates)} ablation rates...")
    result = eval_code(viz_code, args.host, args.port)
    if not result['success']:
        print(f"ERROR generating viz data: {result['error']}")
        return

    # Fetch the grid data
    fetch_result = eval_code("ctx._viz_grid_data", args.host, args.port)
    if not fetch_result['success']:
        print(f"ERROR fetching viz data: {fetch_result['error']}")
        return

    grid_data = fetch_result['result']

    # Create matplotlib figure
    # Columns: Original | Baseline | Wav@0% | Wav@25% | ... | Amp@0% | Amp@25% | ...
    n_cols = 2 + 2 * len(viz_rates)
    fig, axes = plt.subplots(n_display, n_cols, figsize=(2 * n_cols, 2 * n_display))
    if n_display == 1:
        axes = axes[np.newaxis, :]

    for row in range(n_display):
        col = 0

        # Original
        axes[row, col].imshow(np.array(grid_data["originals"][row]))
        axes[row, col].set_title("Original" if row == 0 else "")
        axes[row, col].axis('off')
        col += 1

        # Baseline
        axes[row, col].imshow(np.array(grid_data["baseline"][row]))
        axes[row, col].set_title("Baseline" if row == 0 else "")
        axes[row, col].axis('off')
        col += 1

        # Wavelet ablations
        for rate in viz_rates:
            img = np.array(grid_data["wav_ablations"][str(rate)][row])
            axes[row, col].imshow(img)
            axes[row, col].set_title(f"Wav {int(rate*100)}%" if row == 0 else "")
            axes[row, col].axis('off')
            col += 1

        # Amplitude ablations
        for rate in viz_rates:
            img = np.array(grid_data["amp_ablations"][str(rate)][row])
            axes[row, col].imshow(img)
            axes[row, col].set_title(f"Amp {int(rate*100)}%" if row == 0 else "")
            axes[row, col].axis('off')
            col += 1

    plt.suptitle(f'Subspace Ablation Exemplars @ {args.resolution}px\n'
                 f'(Wavelet = frequency/texture | Amplitude = intensity/color)',
                 fontsize=12, y=1.02)
    plt.tight_layout()

    # Save to output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_vaporeon" if args.vaporeon else ""
    output_path = output_dir / f"probe_subspace_ablation_{args.resolution}px{suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved visual grid to: {output_path}")


if __name__ == "__main__":
    main()
