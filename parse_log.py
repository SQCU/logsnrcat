#!/usr/bin/env python
"""Parse training log and compute summary metrics."""
import re
import sys
from collections import defaultdict


def parse_log(filepath: str):
    """Parse training log file, handling UTF-16-LE encoding."""
    metrics = defaultdict(list)
    errors = []

    # Try UTF-16-LE first (Windows console output), fallback to UTF-8
    try:
        with open(filepath, 'r', encoding='utf-16-le', errors='replace') as f:
            content = f.read()
    except:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

    lines = content.split('\n')

    for line in lines:
        line = line.strip()

        # Parse train-ae progress lines
        # Format: train-ae: 85%|...|  213/250  [01:54<00:09,  4.05it/s, recon=0.0633, sparse=91.4%, k=11]
        ae_match = re.search(
            r'train-ae:\s*(\d+)%.*?(\d+)/(\d+).*?recon=([0-9.]+).*?sparse=([0-9.]+)%.*?k=(\d+)',
            line
        )
        if ae_match:
            pct, step, total, recon, sparse, k = ae_match.groups()
            metrics['ae_step'].append(int(step))
            metrics['ae_recon'].append(float(recon))
            metrics['ae_sparse'].append(float(sparse))
            metrics['ae_k'].append(int(k))
            continue

        # Parse train-denoise progress lines
        denoise_match = re.search(
            r'train-denoise:\s*(\d+)%.*?(\d+)/(\d+).*?loss=([0-9.]+)',
            line
        )
        if denoise_match:
            pct, step, total, loss = denoise_match.groups()
            metrics['denoise_step'].append(int(step))
            metrics['denoise_loss'].append(float(loss))
            continue

        # Collect errors/tracebacks
        if 'Error' in line or 'Traceback' in line or 'Exception' in line:
            errors.append(line.strip())

    return metrics, errors


def compute_summary(metrics):
    """Compute summary statistics."""
    summary = {}

    if metrics['ae_recon']:
        recons = metrics['ae_recon']
        summary['ae'] = {
            'steps': len(set(metrics['ae_step'])),
            'recon_start': recons[0] if recons else None,
            'recon_end': recons[-1] if recons else None,
            'recon_min': min(recons),
            'recon_max': max(recons),
            'recon_mean': sum(recons) / len(recons),
            'sparse_end': metrics['ae_sparse'][-1] if metrics['ae_sparse'] else None,
            'k_end': metrics['ae_k'][-1] if metrics['ae_k'] else None,
        }

    if metrics['denoise_loss']:
        losses = metrics['denoise_loss']
        summary['denoise'] = {
            'steps': len(set(metrics['denoise_step'])),
            'loss_start': losses[0] if losses else None,
            'loss_end': losses[-1] if losses else None,
            'loss_min': min(losses),
            'loss_max': max(losses),
            'loss_mean': sum(losses) / len(losses),
        }

    return summary


def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else 'sparse_ae_swiglu_x.log'

    print(f"Parsing: {filepath}")
    metrics, errors = parse_log(filepath)
    summary = compute_summary(metrics)

    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)

    if 'ae' in summary:
        ae = summary['ae']
        print(f"\n[AE Warmup] ({ae['steps']} unique steps)")
        print(f"  recon: {ae['recon_start']:.4f} -> {ae['recon_end']:.4f}")
        print(f"  range: [{ae['recon_min']:.4f}, {ae['recon_max']:.4f}]")
        print(f"  mean:  {ae['recon_mean']:.4f}")
        print(f"  sparse: {ae['sparse_end']:.1f}%")
        print(f"  k:     {ae['k_end']}")

    if 'denoise' in summary:
        d = summary['denoise']
        print(f"\n[Denoiser] ({d['steps']} unique steps)")
        print(f"  loss: {d['loss_start']:.4f} -> {d['loss_end']:.4f}")
        print(f"  range: [{d['loss_min']:.4f}, {d['loss_max']:.4f}]")
        print(f"  mean:  {d['loss_mean']:.4f}")

    if errors:
        print(f"\n[Errors/Warnings] ({len(errors)} found)")
        for e in errors[:10]:  # Show first 10
            print(f"  {e[:100]}")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
