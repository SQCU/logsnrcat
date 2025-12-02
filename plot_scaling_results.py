#plot_scaling_results.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import sys

def analyze_and_plot(target_dir, smoothing_window=50):
    target_path = Path(target_dir)
    if not target_path.exists():
        print(f"❌ Directory not found: {target_path}")
        return

    # 1. Find all history CSVs
    csv_files = list(target_path.glob("*_history.csv"))
    if not csv_files:
        print(f"❌ No *_history.csv files found in {target_path}")
        return

    print(f"Found {len(csv_files)} logs: {[f.name for f in csv_files]}")
    
    data = {}
    for f in csv_files:
        # Clean name: "Deep_Backbone_history.csv" -> "Deep_Backbone"
        config_name = f.stem.replace('_history', '')
        try:
            df = pd.read_csv(f)
            data[config_name] = df
        except Exception as e:
            print(f"⚠️ Could not read {f}: {e}")

    # 2. Setup Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Consistent colors for known configs, random for others
    colors = {
        'Base_Long': '#1f77b4',      # Blue
        'Deep_Backbone': '#ff7f0e',  # Orange
        'Deep_Interface': '#2ca02c', # Green
        'Base': '#7f7f7f'            # Gray
    }
    
    summary_stats = []

    # 3. Plotting Loop
    for name, df in data.items():
        color = colors.get(name, None) # Auto-color if unknown
        
        # --- Resolution 16 ---
        d16 = df[df['res'] == 16].sort_values('step')
        if not d16.empty:
            # Rolling mean for visual clarity
            smoothed = d16['loss'].rolling(window=smoothing_window, min_periods=1).mean()
            axes[0].plot(d16['step'], smoothed, label=name, color=color, linewidth=1.5, alpha=0.9)
            
            # Calculate final convergence (avg of last 5% of steps)
            tail_n = max(5, int(len(d16) * 0.05))
            final_loss = d16['loss'].iloc[-tail_n:].mean()
            summary_stats.append({'Config': name, 'Res': 16, 'Final_Loss': final_loss, 'Steps': d16['step'].max()})

        # --- Resolution 32 ---
        d32 = df[df['res'] == 32].sort_values('step')
        if not d32.empty:
            smoothed = d32['loss'].rolling(window=smoothing_window, min_periods=1).mean()
            axes[1].plot(d32['step'], smoothed, label=name, color=color, linewidth=1.5, alpha=0.9)
            
            tail_n = max(5, int(len(d32) * 0.05))
            final_loss = d32['loss'].iloc[-tail_n:].mean()
            summary_stats.append({'Config': name, 'Res': 32, 'Final_Loss': final_loss, 'Steps': d32['step'].max()})

    # 4. Styling
    axes[0].set_title(f"Resolution 16x16 Loss (Moving Avg {smoothing_window})")
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("MSE Loss (Log Scale)")
    axes[0].set_yscale('log')
    axes[0].grid(True, which="both", ls="-", alpha=0.2)
    axes[0].legend()

    axes[1].set_title(f"Resolution 32x32 Loss (Moving Avg {smoothing_window})")
    axes[1].set_xlabel("Training Step")
    axes[1].set_yscale('log')
    axes[1].grid(True, which="both", ls="-", alpha=0.2)
    axes[1].legend()

    plt.tight_layout()
    
    # Save plot
    output_plot = target_path / "post_hoc_comparison.png"
    plt.savefig(output_plot, dpi=150)
    print(f"\n📈 Comparison plot saved to: {output_plot}")
    plt.close()

    # 5. Print Summary Table
    print("\n=== Convergence Summary (Last 5% Average) ===")
    summary_df = pd.DataFrame(summary_stats)
    # Pivot for readability
    if not summary_df.empty:
        pivot = summary_df.pivot(index='Config', columns='Res', values='Final_Loss')
        pivot['Steps'] = summary_df.groupby('Config')['Steps'].max()
        print(pivot.sort_values(32)) # Sort by 32px performance
        
        # Save summary CSV
        pivot.to_csv(target_path / "post_hoc_summary.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse and plot training history CSVs.")
    parser.add_argument("dir", type=str, help="Path to the run directory (e.g. experiments_scaling/bench_backbone_run_005)")
    parser.add_argument("--smooth", type=int, default=50, help="Window size for rolling average")
    
    args = parser.parse_args()
    
    analyze_and_plot(args.dir, args.smooth)