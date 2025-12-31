#!/usr/bin/env python3
"""
Analyze training throughput from logs using online variance estimation.
Computes d(expected_batches_hour)/d(iterations) to measure throughput stability.
"""
import re
import sys
from pathlib import Path

class WelfordOnlineStats:
    """Welford's online algorithm for mean and variance."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squared differences from mean

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.M2 / self.n if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return self.variance ** 0.5

    @property
    def cv(self) -> float:
        """Coefficient of variation (std/mean)."""
        return self.std / self.mean if self.mean > 0 else 0.0


def parse_log(log_path: str) -> list:
    """Parse training log and extract (step, seconds_per_it) pairs."""
    results = []

    # Try different encodings - PowerShell often outputs UTF-16
    for encoding in ['utf-16', 'utf-8', 'utf-8-sig', 'latin-1']:
        try:
            with open(log_path, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read()
            # Check if we got sensible content
            if 'train' in content.lower() or 's/it' in content:
                break
        except:
            continue

    # Pattern matches: "123/1000 [time<eta, X.XXs/it" or "X.XX it/s"
    pattern_s_per_it = r'(\d+)/\d+\s*\[[\d:]+<[\d:?,]+\s*([\d.]+)\s*s/it'
    pattern_it_per_s = r'(\d+)/\d+\s*\[[\d:]+<[\d:?,]+\s*([\d.]+)\s*it/s'

    for line in content.split('\n'):
        # Try s/it format first
        match = re.search(pattern_s_per_it, line)
        if match:
            step = int(match.group(1))
            s_per_it = float(match.group(2))
            results.append((step, s_per_it))
            continue

        # Try it/s format
        match = re.search(pattern_it_per_s, line)
        if match:
            step = int(match.group(1))
            it_per_s = float(match.group(2))
            s_per_it = 1.0 / it_per_s if it_per_s > 0 else 999
            results.append((step, s_per_it))

    return results


def analyze_throughput(data: list, window: int = 10) -> dict:
    """
    Analyze throughput using online statistics.

    Returns dict with:
    - Global stats (mean, std, cv of s/it)
    - Windowed derivative of batches/hour
    - Stability metrics
    """
    if len(data) < 2:
        return {"error": "Insufficient data"}

    # Global online stats for s/it
    global_stats = WelfordOnlineStats()
    for step, s_it in data:
        global_stats.update(s_it)

    # Compute batches/hour at each point
    # batches_hour = 3600 / s_per_it
    batches_hour = [(step, 3600.0 / s_it) for step, s_it in data if s_it > 0]

    # Compute windowed derivative: d(batches_hour)/d(step)
    derivatives = []
    deriv_stats = WelfordOnlineStats()

    for i in range(window, len(batches_hour)):
        step_now, bph_now = batches_hour[i]
        step_prev, bph_prev = batches_hour[i - window]

        d_step = step_now - step_prev
        d_bph = bph_now - bph_prev

        if d_step > 0:
            deriv = d_bph / d_step  # Change in batches/hour per iteration
            derivatives.append((step_now, deriv))
            deriv_stats.update(deriv)

    # Segment analysis: early (first 25%), mid (25-75%), late (75-100%)
    n = len(data)
    early = data[:n//4]
    mid = data[n//4:3*n//4]
    late = data[3*n//4:]

    def segment_stats(segment):
        stats = WelfordOnlineStats()
        for _, s_it in segment:
            stats.update(s_it)
        return {
            "mean_s_it": stats.mean,
            "std_s_it": stats.std,
            "cv": stats.cv,
            "batches_hour": 3600 / stats.mean if stats.mean > 0 else 0
        }

    return {
        "n_samples": len(data),
        "global": {
            "mean_s_it": global_stats.mean,
            "std_s_it": global_stats.std,
            "cv": global_stats.cv,
            "batches_hour": 3600 / global_stats.mean if global_stats.mean > 0 else 0
        },
        "derivative": {
            "mean": deriv_stats.mean,
            "std": deriv_stats.std,
            "interpretation": "positive=speeding up, negative=slowing down, high_std=unstable"
        },
        "segments": {
            "early": segment_stats(early),
            "mid": segment_stats(mid),
            "late": segment_stats(late)
        }
    }


def print_report(name: str, stats: dict):
    """Print formatted analysis report."""
    print(f"\n{'='*60}")
    print(f" {name}")
    print(f"{'='*60}")

    if "error" in stats:
        print(f"  Error: {stats['error']}")
        return

    g = stats["global"]
    print(f"\n  Samples: {stats['n_samples']}")
    print(f"\n  GLOBAL THROUGHPUT:")
    print(f"    Mean:     {g['mean_s_it']:.3f} s/it")
    print(f"    Std:      {g['std_s_it']:.3f} s/it")
    print(f"    CV:       {g['cv']:.1%}  (lower=more stable)")
    print(f"    Rate:     {g['batches_hour']:.1f} batches/hour")

    d = stats["derivative"]
    print(f"\n  d(batches_hour)/d(iteration):")
    print(f"    Mean:     {d['mean']:+.2f} batches/hr per step")
    print(f"    Std:      {d['std']:.2f}")
    trend = "speeding up" if d['mean'] > 0.5 else "slowing down" if d['mean'] < -0.5 else "stable"
    stability = "stable" if d['std'] < 50 else "unstable" if d['std'] > 200 else "moderate variance"
    print(f"    Trend:    {trend}, {stability}")

    print(f"\n  SEGMENT ANALYSIS:")
    for seg_name in ["early", "mid", "late"]:
        s = stats["segments"][seg_name]
        print(f"    {seg_name:5s}: {s['mean_s_it']:.2f}s/it (cv={s['cv']:.1%}, {s['batches_hour']:.0f} batch/hr)")


def main():
    logs = sys.argv[1:] if len(sys.argv) > 1 else []

    if not logs:
        # Default to finding logs in current directory
        logs = list(Path('.').glob('*.log'))
        if not logs:
            print("Usage: python analyze_throughput.py log1.log [log2.log ...]")
            return

    for log_path in logs:
        log_path = str(log_path)
        try:
            data = parse_log(log_path)
            if data:
                stats = analyze_throughput(data)
                print_report(Path(log_path).name, stats)
            else:
                print(f"\n  {log_path}: No timing data found")
        except Exception as e:
            print(f"\n  {log_path}: Error - {e}")

    # Comparison if multiple logs
    if len(logs) >= 2:
        print(f"\n{'='*60}")
        print(" COMPARISON")
        print(f"{'='*60}")

        all_stats = []
        for log_path in logs:
            data = parse_log(str(log_path))
            if data:
                all_stats.append((Path(log_path).name, analyze_throughput(data)))

        if len(all_stats) >= 2:
            for name, stats in all_stats:
                g = stats["global"]
                d = stats["derivative"]
                print(f"  {name[:30]:30s} | {g['batches_hour']:6.1f} batch/hr | cv={g['cv']:.1%} | d={d['mean']:+.1f}±{d['std']:.0f}")


if __name__ == "__main__":
    main()
