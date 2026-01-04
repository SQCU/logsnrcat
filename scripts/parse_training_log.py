#!/usr/bin/env python3
"""
Parse training log files to extract loss metrics over steps.

Usage:
    python scripts/parse_training_log.py sparse_ae_swiglu_ix.log
    python scripts/parse_training_log.py sparse_ae_swiglu_ix.log --plot
"""
import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import json


def detect_encoding(filepath: Path) -> str:
    """Detect file encoding by checking for BOM or null bytes."""
    with open(filepath, 'rb') as f:
        raw = f.read(4)

    # UTF-16 LE BOM
    if raw[:2] == b'\xff\xfe':
        return 'utf-16-le'
    # UTF-16 BE BOM
    if raw[:2] == b'\xfe\xff':
        return 'utf-16-be'
    # Check for null bytes (UTF-16 without BOM)
    if b'\x00' in raw:
        return 'utf-16'
    return 'utf-8'


def parse_ae_line(line: str) -> Dict:
    """Parse AE training progress line."""
    # Pattern: train-ae: XX% | step/total [time, recon=X.XXXX, sparse=XX.X%, k=XX]
    pattern = r'train-ae:\s+(\d+)%.*?(\d+)/(\d+).*?recon=([0-9.]+),\s*sparse=([0-9.]+)%,\s*k=(\d+)'
    match = re.search(pattern, line)
    if match:
        return {
            'phase': 'ae',
            'progress': int(match.group(1)),
            'step': int(match.group(2)),
            'total': int(match.group(3)),
            'recon_loss': float(match.group(4)),
            'sparsity': float(match.group(5)),
            'k': int(match.group(6))
        }
    return None


def parse_diffusion_line(line: str) -> Dict:
    """Parse diffusion training progress line."""
    # Pattern for latent diffusion: train-latent-diff: XX% | step/total [..., v=X.XXXX, rec=X.XXXX, v_bce=X.XX, ...]
    latent_pattern = r'train-latent-diff:\s+(\d+)%.*?(\d+)/(\d+).*?v=([0-9.]+),\s*rec=([0-9.]+),\s*v_bce=([0-9.]+)'
    match = re.search(latent_pattern, line)
    if match:
        return {
            'phase': 'diffusion',
            'progress': int(match.group(1)),
            'step': int(match.group(2)),
            'total': int(match.group(3)),
            'v_loss': float(match.group(4)),
            'rec_loss': float(match.group(5)),
            'v_bce': float(match.group(6)),
            'loss': float(match.group(4))  # Use v_loss as primary loss
        }

    # Pattern for standard denoising: train-denoise: XX% | step/total [time, loss=X.XXXX, ...]
    patterns = [
        r'train-denoise:\s+(\d+)%.*?(\d+)/(\d+).*?loss=([0-9.]+)',
        r'train-latent:\s+(\d+)%.*?(\d+)/(\d+).*?loss=([0-9.]+)',
        r'denoise:\s+(\d+)%.*?(\d+)/(\d+).*?loss=([0-9.]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, line)
        if match:
            return {
                'phase': 'diffusion',
                'progress': int(match.group(1)),
                'step': int(match.group(2)),
                'total': int(match.group(3)),
                'loss': float(match.group(4))
            }
    return None


def parse_log(filepath: Path) -> Tuple[List[Dict], List[Dict]]:
    """Parse training log file, return (ae_metrics, diffusion_metrics)."""
    encoding = detect_encoding(filepath)
    print(f"Detected encoding: {encoding}")

    ae_metrics = []
    diffusion_metrics = []

    with open(filepath, 'r', encoding=encoding, errors='replace') as f:
        for line in f:
            # Clean up any weird characters
            line = line.strip()
            if not line:
                continue

            # Try AE pattern
            ae_data = parse_ae_line(line)
            if ae_data:
                ae_metrics.append(ae_data)
                continue

            # Try diffusion pattern
            diff_data = parse_diffusion_line(line)
            if diff_data:
                diffusion_metrics.append(diff_data)

    return ae_metrics, diffusion_metrics


def dedupe_metrics(metrics: List[Dict]) -> List[Dict]:
    """Remove duplicate entries (same step logged twice)."""
    seen = set()
    deduped = []
    for m in metrics:
        key = m['step']
        if key not in seen:
            seen.add(key)
            deduped.append(m)
    return deduped


def summarize_metrics(metrics: List[Dict], phase: str):
    """Print summary statistics."""
    if not metrics:
        print(f"\n{phase.upper()}: No metrics found")
        return

    metrics = dedupe_metrics(metrics)

    print(f"\n{'='*60}")
    print(f"{phase.upper()} Training Metrics")
    print(f"{'='*60}")
    print(f"Total steps logged: {len(metrics)}")

    if phase == 'ae':
        losses = [m['recon_loss'] for m in metrics]
        k_values = [m['k'] for m in metrics]
        sparsity = [m['sparsity'] for m in metrics]

        print(f"\nReconstruction Loss:")
        print(f"  Start: {losses[0]:.4f}")
        print(f"  End:   {losses[-1]:.4f}")
        print(f"  Min:   {min(losses):.4f}")
        print(f"  Max:   {max(losses):.4f}")

        print(f"\nK (active dims):")
        print(f"  Start: {k_values[0]}")
        print(f"  End:   {k_values[-1]}")

        print(f"\nSparsity:")
        print(f"  Start: {sparsity[0]:.1f}%")
        print(f"  End:   {sparsity[-1]:.1f}%")

    elif phase == 'diffusion':
        losses = [m['loss'] for m in metrics]

        print(f"\nDiffusion Loss (v-field):")
        print(f"  Start: {losses[0]:.4f}")
        print(f"  End:   {losses[-1]:.4f}")
        print(f"  Min:   {min(losses):.4f}")
        print(f"  Max:   {max(losses):.4f}")

        # Check for latent diffusion specific metrics
        if 'rec_loss' in metrics[0]:
            rec_losses = [m['rec_loss'] for m in metrics]
            print(f"\nReconstruction Loss:")
            print(f"  Start: {rec_losses[0]:.4f}")
            print(f"  End:   {rec_losses[-1]:.4f}")
            print(f"  Min:   {min(rec_losses):.4f}")

        if 'v_bce' in metrics[0]:
            v_bce = [m['v_bce'] for m in metrics]
            print(f"\nV-field BCE:")
            print(f"  Start: {v_bce[0]:.4f}")
            print(f"  End:   {v_bce[-1]:.4f}")

    # Sample milestones
    print(f"\nMilestones (step, loss):")
    milestones = [0, len(metrics)//4, len(metrics)//2, 3*len(metrics)//4, len(metrics)-1]
    for idx in milestones:
        if idx < len(metrics):
            m = metrics[idx]
            if phase == 'ae':
                print(f"  Step {m['step']:5d}: recon={m['recon_loss']:.4f}, k={m['k']}")
            else:
                print(f"  Step {m['step']:5d}: loss={m['loss']:.4f}")


def plot_metrics(ae_metrics: List[Dict], diff_metrics: List[Dict], output_path: Path):
    """Generate plots of training metrics."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available for plotting")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # AE reconstruction loss
    if ae_metrics:
        ae_metrics = dedupe_metrics(ae_metrics)
        steps = [m['step'] for m in ae_metrics]
        recon = [m['recon_loss'] for m in ae_metrics]
        k_vals = [m['k'] for m in ae_metrics]

        ax = axes[0, 0]
        ax.semilogy(steps, recon, 'b-', linewidth=0.5, alpha=0.7)
        # Add smoothed line
        if len(recon) > 50:
            window = min(50, len(recon)//10)
            smoothed = np.convolve(recon, np.ones(window)/window, mode='valid')
            ax.semilogy(steps[window//2:window//2+len(smoothed)], smoothed, 'b-', linewidth=2, label='Smoothed')
        ax.set_xlabel('Step')
        ax.set_ylabel('Reconstruction Loss')
        ax.set_title('AE Reconstruction Loss')
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(steps, k_vals, 'g-', linewidth=1)
        ax.set_xlabel('Step')
        ax.set_ylabel('K (active dims)')
        ax.set_title('K Annealing Schedule')
        ax.grid(True, alpha=0.3)

    # Diffusion loss
    if diff_metrics:
        diff_metrics = dedupe_metrics(diff_metrics)
        steps = [m['step'] for m in diff_metrics]
        losses = [m['loss'] for m in diff_metrics]

        ax = axes[1, 0]
        ax.semilogy(steps, losses, 'r-', linewidth=0.5, alpha=0.7)
        if len(losses) > 50:
            window = min(50, len(losses)//10)
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax.semilogy(steps[window//2:window//2+len(smoothed)], smoothed, 'r-', linewidth=2, label='Smoothed')
        ax.set_xlabel('Step')
        ax.set_ylabel('Diffusion Loss')
        ax.set_title('Diffusion Training Loss')
        ax.grid(True, alpha=0.3)

    # Combined view
    ax = axes[1, 1]
    if ae_metrics:
        ae_steps = [m['step'] for m in ae_metrics]
        ae_loss = [m['recon_loss'] for m in ae_metrics]
        ax.semilogy(ae_steps, ae_loss, 'b-', alpha=0.5, label='AE Recon')
    if diff_metrics:
        # Offset diffusion steps by AE total
        ae_total = ae_metrics[-1]['step'] if ae_metrics else 0
        diff_steps = [m['step'] + ae_total for m in diff_metrics]
        diff_loss = [m['loss'] for m in diff_metrics]
        ax.semilogy(diff_steps, diff_loss, 'r-', alpha=0.5, label='Diffusion')
    ax.axvline(ae_total, color='gray', linestyle='--', alpha=0.5, label='Phase boundary')
    ax.set_xlabel('Total Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Combined Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")


def export_json(ae_metrics: List[Dict], diff_metrics: List[Dict], output_path: Path):
    """Export metrics to JSON for further analysis."""
    data = {
        'ae_metrics': dedupe_metrics(ae_metrics),
        'diffusion_metrics': dedupe_metrics(diff_metrics)
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"JSON exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Parse training log files")
    parser.add_argument("logfile", type=Path, help="Path to log file")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--json", action="store_true", help="Export to JSON")
    parser.add_argument("-o", "--output", type=Path, help="Output directory", default=Path("."))
    args = parser.parse_args()

    if not args.logfile.exists():
        print(f"Error: {args.logfile} not found")
        sys.exit(1)

    print(f"Parsing: {args.logfile}")
    ae_metrics, diff_metrics = parse_log(args.logfile)

    summarize_metrics(ae_metrics, 'ae')
    summarize_metrics(diff_metrics, 'diffusion')

    if args.plot:
        plot_path = args.output / f"{args.logfile.stem}_metrics.png"
        plot_metrics(ae_metrics, diff_metrics, plot_path)

    if args.json:
        json_path = args.output / f"{args.logfile.stem}_metrics.json"
        export_json(ae_metrics, diff_metrics, json_path)


if __name__ == "__main__":
    main()
