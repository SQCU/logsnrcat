#!/usr/bin/env python3
"""
Probe model health on eval server.

Queries architecture, runs reconstruction tests, saves diagnostics to run directory.

Usage:
    python scripts/probe_model_health.py [--host HOST] [--port PORT]
"""

import argparse
import json
from pathlib import Path

import requests

DEFAULT_HOST = "172.26.160.1"
DEFAULT_PORT = 8421


def eval_code(code: str, host: str, port: int, timeout: int = 120) -> dict:
    """Execute Python code on eval server."""
    url = f"http://{host}:{port}/eval"
    resp = requests.post(url, json={"code": code}, timeout=timeout)
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Probe model health")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    print(f"Connecting to eval server at {base_url}...")

    # Health check
    health = requests.get(f"{base_url}/health").json()
    print(f"  Status: {health['status']}, Weights: {health['weights_loaded']}")
    print(f"  Params: {health.get('params', 'N/A'):,}")

    # Provenance
    prov = requests.get(f"{base_url}/provenance").json()
    run_id = prov.get("run_id", "unknown")
    run_path = prov.get("run_path", "experiments_swiglu_ae/unknown")
    print(f"  Provenance: {run_id}")
    print(f"  Path: {run_path}")

    # Determine output directory
    output_dir = Path(run_path.replace("\\", "/"))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output dir: {output_dir}")

    # Status
    status = requests.get(f"{base_url}/status").json()
    print(f"\nServer Status:")
    print(f"  dtype: {status['dtype']}")
    print(f"  device: {status['device']}")
    print(f"  ae_present: {status['ae_present']}")

    print("\n" + "=" * 60)
    print("ARCHITECTURE")
    print("=" * 60)

    # Query architecture
    ae_type = eval_code("type(model.sparse_ae).__name__", args.host, args.port)
    print(f"AE Type: {ae_type['result']}")

    wavelet_gating = eval_code("model.sparse_ae.wavelet_gating", args.host, args.port)
    print(f"Wavelet Gating: {wavelet_gating['result']}")

    n_levels = eval_code("model.sparse_ae.n_levels", args.host, args.port)
    print(f"N Levels: {n_levels['result']}")

    # Encoder details
    n_wav = eval_code("model.sparse_ae.encoders[0].n_wavelet_dims", args.host, args.port)
    n_amp = eval_code("model.sparse_ae.encoders[0].n_amplitude_dims", args.host, args.port)
    code_dim = eval_code("model.sparse_ae.encoders[0].code_dim", args.host, args.port)
    sparsity_type = eval_code("type(model.sparse_ae.encoders[0].sparsity).__name__", args.host, args.port)

    print(f"\nEncoder[0]:")
    print(f"  code_dim: {code_dim['result']}")
    print(f"  n_wavelet_dims: {n_wav['result']}")
    print(f"  n_amplitude_dims: {n_amp['result']}")
    print(f"  sparsity: {sparsity_type['result']}")

    # Check dual pathways
    has_amp_proj = eval_code("hasattr(model.sparse_ae.encoders[0], 'amplitude_proj') and model.sparse_ae.encoders[0].amplitude_proj is not None", args.host, args.port)
    has_wav_proj = eval_code("hasattr(model.sparse_ae.encoders[0], 'wavelet_proj') and model.sparse_ae.encoders[0].wavelet_proj is not None", args.host, args.port)
    print(f"  has amplitude_proj: {has_amp_proj['result']}")
    print(f"  has wavelet_proj: {has_wav_proj['result']}")

    # Decoder details
    has_wav_embed = eval_code("hasattr(model.sparse_ae.decoders[0], 'wav_embed') and model.sparse_ae.decoders[0].wav_embed is not None", args.host, args.port)
    has_amp_embed = eval_code("hasattr(model.sparse_ae.decoders[0], 'amp_embed') and model.sparse_ae.decoders[0].amp_embed is not None", args.host, args.port)
    has_wav_head = eval_code("hasattr(model.sparse_ae.decoders[0], 'wav_head') and model.sparse_ae.decoders[0].wav_head is not None", args.host, args.port)
    has_amp_head = eval_code("hasattr(model.sparse_ae.decoders[0], 'amp_head') and model.sparse_ae.decoders[0].amp_head is not None", args.host, args.port)

    print(f"\nDecoder[0]:")
    print(f"  has wav_embed: {has_wav_embed['result']}")
    print(f"  has amp_embed: {has_amp_embed['result']}")
    print(f"  has wav_head: {has_wav_head['result']}")
    print(f"  has amp_head: {has_amp_head['result']}")

    print("\n" + "=" * 60)
    print("RECONSTRUCTION TEST")
    print("=" * 60)

    # Run reconstruction test
    recon_code = '''
import torch
import torch.nn.functional as F

# Get test batch
images = ctx.get_batch(64, 8)

ae = model.sparse_ae
p = ae.patch_size
H, W = images.shape[2], images.shape[3]
grid_shape = (H // p, W // p)
encoder_masks, decoder_masks = ae.build_masks(grid_shape, images.device)

with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    codes = ae.encode(images, grid_shape=grid_shape, encoder_masks=encoder_masks, decoder_masks=decoder_masks)
    recon = ae.decode(codes, grid_shape, decoder_masks)

mse = F.mse_loss(recon, images).item()
psnr = -10 * torch.log10(torch.tensor(mse)).item()

# Store for later retrieval
ctx._health_mse = mse
ctx._health_psnr = psnr
ctx._health_images = images
ctx._health_recon = recon
ctx._health_codes = codes

(mse, psnr, len(codes))
'''

    result = eval_code(recon_code, args.host, args.port, timeout=60)
    if not result['success']:
        print(f"ERROR: {result['error']}")
        return

    # Result is a tuple/list - handle variable length
    res = result['result']
    if isinstance(res, (list, tuple)) and len(res) >= 3:
        mse, psnr, n_codes = res[0], res[1], res[2]
    else:
        # Fallback: fetch stored values
        mse = eval_code("ctx._health_mse", args.host, args.port)['result']
        psnr = eval_code("ctx._health_psnr", args.host, args.port)['result']
        n_codes = eval_code("len(ctx._health_codes)", args.host, args.port)['result']

    print(f"MSE: {mse:.6f}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"N codes: {n_codes}")

    # Get code statistics
    code_stats_code = '[(i, c.shape[-1], (c.abs() < 1e-6).float().mean().item(), c.flatten(0, -2).unique(dim=0).shape[0]) for i, c in enumerate(ctx._health_codes)]'
    code_stats = eval_code(code_stats_code, args.host, args.port)
    if code_stats['success'] and code_stats['result']:
        print("\nCode Statistics:")
        print(f"  {'Level':<6} {'Dim':<6} {'Sparsity':<10} {'Unique':<8}")
        for item in code_stats['result']:
            if isinstance(item, (list, tuple)) and len(item) >= 4:
                level, dim, sparsity, unique = item[0], item[1], item[2], item[3]
                print(f"  {level:<6} {dim:<6} {sparsity:<10.4f} {unique:<8}")

    print("\n" + "=" * 60)
    print("SAVING VISUALIZATIONS")
    print("=" * 60)

    # Load deps and save visualization
    requests.get(f"{base_url}/load_deps")

    viz_code = f'''
import torch
import numpy as np

images = ctx._health_images
recon = ctx._health_recon

# Make grid of input vs recon
from torchvision.utils import make_grid, save_image

# Interleave input and recon for side-by-side comparison
n_show = min(8, images.shape[0])
pairs = torch.stack([images[:n_show], recon[:n_show]], dim=1).flatten(0, 1)
grid = make_grid(pairs.float().clamp(0, 1), nrow=4, padding=2)

save_image(grid, "{output_dir}/health_recon_grid.png")
"saved health_recon_grid.png"
'''
    viz_result = eval_code(viz_code.replace("\\", "/"), args.host, args.port)
    print(f"Visualization: {viz_result['result'] if viz_result['success'] else viz_result['error']}")

    # Save detailed plot with histograms
    plot_code = f'''
import matplotlib.pyplot as plt
import numpy as np
import torch

images = ctx._health_images.float().cpu().numpy()
recon = ctx._health_recon.float().cpu().numpy()

fig, axes = plt.subplots(3, 4, figsize=(14, 10))

# Row 1: Sample reconstructions
for i in range(4):
    ax = axes[0, i]
    img_in = np.transpose(images[i], (1, 2, 0))
    img_re = np.transpose(recon[i], (1, 2, 0))
    combined = np.concatenate([img_in, img_re], axis=1)
    ax.imshow(np.clip(combined, 0, 1))
    ax.set_title(f"Sample {{i+1}}: In | Recon")
    ax.axis("off")

# Row 2: Per-channel histograms
colors = ["red", "green", "blue"]
channel_names = ["R", "G", "B"]
for c in range(3):
    ax = axes[1, c]
    ax.hist(images[:, c].flatten(), bins=50, alpha=0.5, label="Input", color=colors[c], density=True)
    ax.hist(recon[:, c].flatten(), bins=50, alpha=0.5, label="Recon", color="gray", density=True)
    ax.set_title(f"{{channel_names[c]}} Channel Histogram")
    ax.legend(fontsize=8)

# Summary stats in row 2, col 4
ax = axes[1, 3]
ax.axis("off")
mse = ctx._health_mse
psnr = ctx._health_psnr
summary = f"""Reconstruction Summary

MSE: {{mse:.6f}}
PSNR: {{psnr:.2f}} dB

Input range: [{{images.min():.3f}}, {{images.max():.3f}}]
Recon range: [{{recon.min():.3f}}, {{recon.max():.3f}}]

Input std: {{images.std():.4f}}
Recon std: {{recon.std():.4f}}"""
ax.text(0.1, 0.5, summary, fontsize=10, family="monospace", verticalalignment="center", transform=ax.transAxes)

# Row 3: Error analysis
ax = axes[2, 0]
error = np.abs(recon - images).mean(axis=1)  # Mean across channels
ax.imshow(error[0], cmap="hot")
ax.set_title("Sample 1 Error Map")
ax.axis("off")

ax = axes[2, 1]
ax.hist(error.flatten(), bins=50, color="red", alpha=0.7)
ax.set_title("Error Distribution")
ax.set_xlabel("Absolute Error")

ax = axes[2, 2]
per_sample_mse = ((recon - images) ** 2).mean(axis=(1, 2, 3))
ax.bar(range(len(per_sample_mse)), per_sample_mse)
ax.set_title("Per-Sample MSE")
ax.set_xlabel("Sample")

ax = axes[2, 3]
ax.axis("off")

plt.suptitle("Model Health Check - {run_id}", fontsize=14)
plt.tight_layout()
plt.savefig("{output_dir}/health_detailed.png", dpi=150, bbox_inches="tight")
plt.close()
"saved health_detailed.png"
'''
    plot_result = eval_code(plot_code.replace("\\", "/"), args.host, args.port)
    print(f"Detailed plot: {plot_result['result'] if plot_result['success'] else plot_result['error']}")

    # Save health summary as JSON
    summary = {
        "run_id": run_id,
        "run_path": run_path,
        "architecture": {
            "ae_type": ae_type['result'],
            "wavelet_gating": wavelet_gating['result'],
            "n_levels": n_levels['result'],
            "code_dim": code_dim['result'],
            "n_wavelet_dims": n_wav['result'],
            "n_amplitude_dims": n_amp['result'],
            "sparsity_type": sparsity_type['result'],
        },
        "reconstruction": {
            "mse": mse,
            "psnr": psnr,
        },
        "code_stats": code_stats['result'] if code_stats['success'] else None,
    }

    summary_path = output_dir / "health_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON: {summary_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
