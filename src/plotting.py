# plotting.py
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - prevents tkinter threading crash
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

def plot_causal_sweep(sequences, predictions, snr_values, output_path):
    """
    Visualizes the Causal Sweep (Row per sequence, Col per timestep).
    """
    N_seq = len(sequences)
    M_len = len(sequences[0])
    
    fig, axes = plt.subplots(N_seq * 2, M_len, figsize=(3 * M_len, 4 * N_seq))
    if N_seq == 1 and M_len == 1: axes = np.array([[axes]])
    elif N_seq == 1: axes = axes.reshape(2, M_len)
    
    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    for i in range(N_seq):
        prefix_snr = snr_values[i]
        
        # Row 1: Prediction
        row_top = i * 2
        # Row 2: Error
        row_bot = i * 2 + 1
        
        for t in range(M_len):
            pred_t = predictions[i][t].detach().cpu().permute(1,2,0).clamp(0,1).numpy()
            gt_t = sequences[i][t].content.detach().cpu().permute(1,2,0).clamp(0,1).numpy()
            
            # Plot Prediction
            ax_img = axes[row_top, t] if N_seq > 1 else axes[t]
            ax_img.imshow(pred_t)
            ax_img.axis('off')
            if t == 0: ax_img.set_title(f"Prefix SNR: {prefix_snr:.1f}", fontsize=9)
            
            # Plot Error
            diff = (pred_t - gt_t)**2
            diff = diff.mean(axis=2) # RMS error per pixel
            
            ax_err = axes[row_bot, t]
            ax_err.imshow(diff, cmap='inferno', vmin=0, vmax=0.1)
            ax_err.axis('off')

    plt.savefig(output_path)
    plt.close(fig)


def run_pipeline_analysis(blocks, device):
    """
    Runs embedding and topological analysis.
    Constructs the ACTUAL Block-Causal Mask for visualization using the model's internal logic.
    """
    # 1. Setup Model Components (Real)
    # Note: We use the actual class definitions to ensure we test the real logic
    from src.model import coolerLDTformerZC, SpanEmbedder, render_topology_embeddings, build_dual_masks
    from collections import namedtuple
    
    # Tiny model config, just enough to drive the embeddings
    model = coolerLDTformerZC(
        dim=64, depth=1, num_heads=4, topo_dim=3,
        vocab_size=256, context_size=4, stride=2
    ).to(device)
    span_emb = SpanEmbedder(model.text_embed, model.patch_embedder)
    
    # 2. Embed Spans (Raw Data -> Z, Metadata)
    z_flat, span_objects, _ = span_emb.embed(blocks)

    # 3. Render Topology (Metadata -> Coords) - dtype from model
    dtype = model.text_embed.weight.dtype
    topo_embeds, doc_ids = render_topology_embeddings(span_objects, max_dims=3, device=device, dtype=dtype)
    
    # 4. Setup PageTable Mocks for build_dual_masks
    # The mask builder requires these structures to resolve physical addresses, 
    # even in ZC (Zero-Copy) mode.
    L = z_flat.shape[0]
    block_size = 128
    num_blocks = (L + block_size - 1) // block_size
    
    PageTableMock = namedtuple('PageTable', ['block_size'])
    page_table = PageTableMock(block_size=block_size)
    
    # Identity mappings for training (Active matches Heap 1:1)
    flat_page_table = torch.arange(num_blocks, device=device, dtype=torch.long)
    inverse_page_table = torch.arange(num_blocks, device=device, dtype=torch.long)
    
    # 5. Retrieve the Mask Closure (The Source of Truth)
    # We call the debug version of the mask builder to get the internal closure
    _, _, debug_dict = build_dual_masks(
        spans=span_objects,
        topo_active=topo_embeds,
        topo_heap=topo_embeds, # Self-attention
        page_table=page_table,
        flat_page_table=flat_page_table,
        inverse_page_table=inverse_page_table,
        window_size=10.0, # Arbitrary large window for global check
        return_mask_closures=True
    )
    
    # This is the actual python function passed to flex_attention
    mask_mod = debug_dict['mask_mod_global']
    
    # 6. Materialize the Mask Tensor
    # Construct a full LxL grid of indices to evaluate the closure
    q_idx = torch.arange(L, device=device).unsqueeze(1).expand(L, L)
    k_idx = torch.arange(L, device=device).unsqueeze(0).expand(L, L)
    
    # Evaluate the closure.
    # We pass b=0, h=0 as the logic is currently batch/head invariant for this check.
    # The closure will lookup doc_ids, span_ids, and causal_modes internally.
    final_mask = mask_mod(0, 0, q_idx, k_idx)
    
    return {
        "spans": span_objects,
        "mask": final_mask,
        "topo": topo_embeds,
        "L": L
    }

# ==============================================================================
# Logging & Plotting
# ==============================================================================

from pathlib import Path
import sys
import json
import matplotlib.pyplot as plt
import pandas as pd

class ExperimentLogger:
    def __init__(self, output_dir="."):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        script_path = Path(sys.argv[0])
        self.script_name = script_path.stem
        existing = list(self.output_dir.glob(f"{self.script_name}_run_*"))
        if existing:
            run_nums = [int(p.stem.split("_run_")[1].split("_")[0]) for p in existing]
            self.run_id = max(run_nums) + 1
        else:
            self.run_id = 0
        self.run_dir = self.output_dir / f"{self.script_name}_run_{self.run_id:03d}"
        self.run_dir.mkdir(exist_ok=True)
        print(f"Run: {self.run_id} | Dir: {self.run_dir}")
        
    def save_figure(self, fig, name):
        filepath = self.run_dir / f"{name}.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
 
    def log_text(self, filename, text):
        """Append text to a log file in the run directory."""
        filepath = self.run_dir / filename
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def save_config(self, config: dict, filename: str = "config.json"):
        """Save experiment config to JSON file in run directory.

        Called early in training to preserve config even if run crashes.
        Handles non-serializable types (Path, torch.dtype, etc.) gracefully.
        """
        def make_serializable(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif hasattr(obj, 'item'):  # torch.Tensor scalar
                return obj.item()
            elif hasattr(obj, '__name__'):  # types like torch.float32
                return str(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(v) for v in obj]
            else:
                return obj

        filepath = self.run_dir / filename
        try:
            serializable = make_serializable(config)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, default=str)
        except Exception as e:
            # Fallback: write repr if JSON fails
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(repr(config))
            print(f"[Logger] Config serialization warning: {e}")

    def save_dataframe(self, df: pd.DataFrame, name: str, format: str = "parquet"):
        """Save dataframe to run directory. Call BEFORE plotting for crash safety.

        Args:
            df: DataFrame to save
            name: Base filename (without extension)
            format: 'parquet' (fast, compact) or 'csv' (human-readable)
        """
        if df.empty:
            return

        if format == "parquet":
            try:
                filepath = self.run_dir / f"{name}.parquet"
                df.to_parquet(filepath, index=False)
            except ImportError:
                # Fallback to CSV if pyarrow/fastparquet not installed
                filepath = self.run_dir / f"{name}.csv"
                df.to_csv(filepath, index=False)
        else:
            filepath = self.run_dir / f"{name}.csv"
            df.to_csv(filepath, index=False)

def plot_losses(df_naive, df_fact, logger, metric="loss_total", title="Training Loss"):
    if df_naive.empty and df_fact.empty: return
    fig, ax = plt.subplots(figsize=(10, 6))
    if not df_naive.empty:
        df_naive = df_naive.interpolate()
        ax.plot(df_naive["step"], df_naive[metric].rolling(50).mean(), label="Naive")
    if not df_fact.empty:
        df_fact = df_fact.interpolate()
        ax.plot(df_fact["step"], df_fact[metric].rolling(50).mean(), label="Factorized")
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xlabel("Step")
    ax.set_yscale("log")
    ax.legend()
    logger.save_figure(fig, f"plot_{metric}")



def plot_multimetric_analysis(df, logger, stringy="multimetric_analysis"):
    if df.empty: return
    
    # Ensure logsnr is present (text blocks might not have it, fill with nan)
    if "logsnr" not in df.columns: df["logsnr"] = np.nan
    
    # Binning for Latent SNR analysis
    # Filter for latents only for SNR plots
    df_latent = df[df["type"] == "latent"].copy()
    if not df_latent.empty:
        df_latent["snr_bin"] = pd.cut(df_latent["logsnr"], bins=20, duplicates='drop')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. d_loss / d_logsnr (Latent Only)
    if not df_latent.empty:
        snr_grouped = df_latent.groupby("snr_bin", observed=True)["loss"].mean()
        snr_x = [i.mid for i in snr_grouped.index]
        axes[0,0].plot(snr_x, snr_grouped.values, marker="o", linewidth=2)
        axes[0,0].set_title("d_loss / d_logsnr (Latent)")
        axes[0,0].set_xlabel("LogSNR")
        axes[0,0].set_ylabel("MSE Loss")
        axes[0,0].invert_xaxis()
        axes[0,0].grid(True, alpha=0.3)
    
    # 2. d_loss / d_resolution (Latent Only, if resolution present)
    if "resolution" in df.columns and not df_latent.empty:
        res_stats = df_latent.groupby("resolution")["loss"].agg(["mean", "std"])
        axes[0,1].errorbar(res_stats.index, res_stats["mean"], yerr=res_stats["std"], fmt="-o", capsize=5)
        axes[0,1].set_title("d_loss / d_resolution")
        axes[0,1].set_xlabel("Tokens")
        axes[0,1].set_xscale("log")
        axes[0,1].grid(True, alpha=0.3)
        
    # 3. d_loss / d_source (Split by Type)
    # Group by (Source, Type) to distinguish Latent vs Text curves
    if "type" in df.columns:
        groups = df.groupby(["source", "type"])
    else:
        groups = df.groupby("source")
        
    for name, grp in groups:
        # name is tuple (source, type) or string source
        if isinstance(name, tuple):
            label = f"{name[0]} ({name[1]})"
            style = '--' if name[1] == 'text' else '-'
        else:
            label = name
            style = '-'
            
        step_avg = grp.groupby("step")["loss"].mean()
        smooth = step_avg.rolling(window=50, min_periods=1).mean()
        axes[1,0].plot(smooth.index, smooth.values, label=label, linestyle=style)
        
    axes[1,0].legend()
    axes[1,0].set_title("Training Loss Curves")
    axes[1,0].set_xlabel("Step")
    axes[1,0].set_yscale("log")
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Variance / d_logsnr (Latent Only)
    if not df_latent.empty and "loss_var" in df_latent.columns:
        var_grouped = df_latent.groupby("snr_bin", observed=True)["loss_var"].mean()
        axes[1,1].plot([i.mid for i in var_grouped.index], var_grouped.values, color="orange", marker="s")
        axes[1,1].set_title("Variance / d_logsnr")
        axes[1,1].set_xlabel("LogSNR")
        axes[1,1].invert_xaxis()
        axes[1,1].grid(True, alpha=0.3)
        
    plt.tight_layout()
    logger.save_figure(fig, stringy)


def plot_loss_schedule_analysis(df, logger, name="loss_schedule_analysis"):
    """
    Analyze MSE vs BCE loss compatibility during scheduled loss training.

    Shows how each loss component changes as the optimization target lerps
    from pure MSE to mostly BCE, revealing whether the tasks are compatible
    (both decrease together) or conflicting (one rises as other falls).

    Args:
        df: DataFrame with columns: step, mse_loss, bce_loss, mse_weight, bce_weight, lerp_t
        logger: ExperimentLogger for saving figures
        name: Output filename prefix
    """
    if df.empty:
        return

    # Check if required columns exist
    required_cols = ['mse_loss', 'bce_loss', 'lerp_t']
    if not all(col in df.columns for col in required_cols):
        print(f"[plot_loss_schedule_analysis] Missing required columns. Have: {list(df.columns)}")
        return

    # Filter for valid data
    df_valid = df[df['mse_loss'].notna() & df['bce_loss'].notna()].copy()
    if df_valid.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Raw loss values over training (top-left)
    ax = axes[0, 0]
    steps = df_valid.groupby('step').agg({'mse_loss': 'mean', 'bce_loss': 'mean'})

    ax.plot(steps.index, steps['mse_loss'].rolling(20, min_periods=1).mean(),
            label='MSE Loss', color='blue', linewidth=2)
    ax.plot(steps.index, steps['bce_loss'].rolling(20, min_periods=1).mean(),
            label='BCE Loss', color='red', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss Value')
    ax.set_title('Raw Loss Components Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 2. Loss vs lerp_t (top-right) - shows d_loss / d_lerp
    ax = axes[0, 1]
    lerp_bins = np.linspace(0, 1, 21)
    df_valid['lerp_bin'] = pd.cut(df_valid['lerp_t'], bins=lerp_bins)

    lerp_stats = df_valid.groupby('lerp_bin', observed=True).agg({
        'mse_loss': ['mean', 'std'],
        'bce_loss': ['mean', 'std']
    })

    bin_centers = [(b.left + b.right) / 2 for b in lerp_stats.index]

    ax.errorbar(bin_centers, lerp_stats['mse_loss']['mean'],
                yerr=lerp_stats['mse_loss']['std'],
                label='MSE', color='blue', fmt='-o', capsize=3, alpha=0.8)
    ax.errorbar(bin_centers, lerp_stats['bce_loss']['mean'],
                yerr=lerp_stats['bce_loss']['std'],
                label='BCE', color='red', fmt='-s', capsize=3, alpha=0.8)
    ax.set_xlabel('Lerp Progress (t: 0=pure MSE → 1=mostly BCE)')
    ax.set_ylabel('Loss Value')
    ax.set_title('d_loss / d_lerp_t (Loss vs Schedule Progress)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Weighted contributions (bottom-left)
    ax = axes[1, 0]
    if 'mse_weight' in df_valid.columns and 'bce_weight' in df_valid.columns:
        df_valid['weighted_mse'] = df_valid['mse_loss'] * df_valid['mse_weight']
        df_valid['weighted_bce'] = df_valid['bce_loss'] * df_valid['bce_weight']

        weighted_stats = df_valid.groupby('step').agg({
            'weighted_mse': 'mean', 'weighted_bce': 'mean'
        })

        ax.stackplot(weighted_stats.index,
                     weighted_stats['weighted_mse'].rolling(20, min_periods=1).mean(),
                     weighted_stats['weighted_bce'].rolling(20, min_periods=1).mean(),
                     labels=['w_mse × MSE', 'w_bce × BCE'],
                     colors=['lightblue', 'lightcoral'], alpha=0.7)
        ax.plot(weighted_stats.index,
                (weighted_stats['weighted_mse'] + weighted_stats['weighted_bce']).rolling(20, min_periods=1).mean(),
                'k--', label='Total Loss', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Weighted Loss')
        ax.set_title('Weighted Loss Contributions (Stacked)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Weights not available', ha='center', va='center', transform=ax.transAxes)

    # 4. Compatibility scatter (bottom-right) - MSE vs BCE at each step
    ax = axes[1, 1]
    scatter = ax.scatter(df_valid['mse_loss'], df_valid['bce_loss'],
                         c=df_valid['lerp_t'], cmap='viridis',
                         alpha=0.5, s=10)
    plt.colorbar(scatter, ax=ax, label='Lerp Progress (t)')
    ax.set_xlabel('MSE Loss')
    ax.set_ylabel('BCE Loss')
    ax.set_title('MSE vs BCE Compatibility\n(Diagonal = correlated tasks)')
    ax.grid(True, alpha=0.3)

    # Add correlation coefficient
    if len(df_valid) > 2:
        corr = df_valid['mse_loss'].corr(df_valid['bce_loss'])
        ax.text(0.05, 0.95, f'ρ = {corr:.3f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    logger.save_figure(fig, name)

    # Print summary statistics
    print(f"\n[Loss Schedule Analysis]")
    print(f"  Steps: {df_valid['step'].min()} → {df_valid['step'].max()}")
    print(f"  MSE: {df_valid['mse_loss'].mean():.4f} ± {df_valid['mse_loss'].std():.4f}")
    print(f"  BCE: {df_valid['bce_loss'].mean():.4f} ± {df_valid['bce_loss'].std():.4f}")
    if len(df_valid) > 2:
        print(f"  Correlation (ρ): {corr:.3f} {'(compatible)' if corr > 0.5 else '(potentially conflicting)' if corr < 0 else '(weakly related)'}")


def plot_dset_reconstruction(result_dict, logger, name="reconstruction", show_map=False, show_error=False):
    # Expect lists of tensors, potentially of mixed resolution
    x0s = result_dict["x0"]
    noisy = result_dict["noisy_input"]
    recon = result_dict["reconstruction"]
    lmaps = result_dict.get("logsnr_map", None)
    sources = result_dict.get("source", None)

    n = len(x0s)
    if n == 0: return

    # Determine columns
    cols = 3
    if show_map and lmaps is not None:
        cols += 1
    if show_error:
        cols += 1

    # Create figure
    fig, axes = plt.subplots(n, cols, figsize=(3*cols, 2*n))
    if n == 1: axes = axes.reshape(1, -1)

    for i in range(n):
        # Helper to safely visualize a single tensor (C,H,W) -> numpy (H,W,C)
        def to_img(t):
            return t.detach().cpu().permute(1,2,0).clamp(0,1).numpy()

        col = 0

        # Col 0: Ground Truth
        axes[i, col].imshow(to_img(x0s[i]))
        axes[i, col].axis("off")
        row_label = f" [{sources[i]}]" if sources else ""
        if i==0: axes[i,col].set_title("Ground Truth")
        if sources: axes[i, col].set_ylabel(sources[i], rotation=0, labelpad=40, fontsize=8)
        col += 1

        # Col 1: Noisy Input
        axes[i, col].imshow(to_img(noisy[i]))
        axes[i, col].axis("off")
        if i==0: axes[i,col].set_title("Noisy Input")
        col += 1

        # Col 2: Reconstruction
        axes[i, col].imshow(to_img(recon[i]))
        axes[i, col].axis("off")
        if i==0: axes[i,col].set_title("Reconstruction")
        col += 1

        # Optional: Error Map (vs ground truth)
        if show_error:
            # Compute MSE error per pixel, average across channels
            err = ((recon[i] - x0s[i]) ** 2).mean(dim=0)  # [H, W]
            err_np = err.detach().cpu().numpy()
            im = axes[i, col].imshow(err_np, cmap="hot", vmin=0, vmax=max(err_np.max() * 0.5, 1e-6))
            axes[i, col].axis("off")
            if i==0: axes[i,col].set_title("Recon Error")
            col += 1

        # Optional: LogSNR Map
        if show_map and lmaps is not None:
            # Map might be (1,H,W) or (H,W)
            m = lmaps[i].detach().cpu().squeeze().numpy()
            axes[i, col].imshow(m, cmap="viridis")
            axes[i, col].axis("off")
            if i==0: axes[i,col].set_title("LogSNR Map")
            col += 1

    plt.tight_layout()
    logger.save_figure(fig, name)


def plot_ae_roundtrip(components, iterator, logger, name="ae_roundtrip", n_samples=8, resolution=64):
    """
    Visualize AE reconstruction quality with round-trip analysis.

    Shows for each sample:
    - Input image (clean)
    - AE reconstruction (1st pass)
    - MSE error field (1st pass)
    - Round-trip reconstruction (AE applied twice)
    - Round-trip MSE error field

    This helps diagnose:
    - How much information is lost in single AE pass
    - Whether errors compound on multiple passes (indicates instability)
    - Spatial distribution of reconstruction errors
    """
    import torch

    # Access sparse_ae directly from model (like train_autoembed does)
    model = components[0]

    if not hasattr(model, 'sparse_ae') or model.sparse_ae is None:
        print("    plot_ae_roundtrip: Model has no sparse_ae")
        return

    sparse_ae = model.sparse_ae
    device = next(sparse_ae.parameters()).device

    # Detect model dtype from first parameter
    model_dtype = None
    for p in sparse_ae.parameters():
        model_dtype = p.dtype
        break
    if model_dtype is None:
        model_dtype = torch.bfloat16

    # Collect samples
    samples = []
    split_names = iterator.get_split_names()
    n_per_split = max(1, n_samples // max(1, len(split_names)))

    # Use autocast like train_autoembed does - this is the key fix
    use_amp = model_dtype in (torch.bfloat16, torch.float16)

    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=model_dtype, enabled=use_amp):
        for split_name in split_names:
            try:
                blocks = iterator.generate_from_split(split_name, count=n_per_split, resolution=resolution)
            except Exception as e:
                continue

            for b in blocks:
                if b.type != 'latent':
                    continue

                x0 = b.content  # [C, H, W]
                logsnr = b.logsnr if b.logsnr is not None else torch.zeros(1, *x0.shape[-2:], device=x0.device)

                try:
                    # Add batch dimension for AE forward
                    x0_batch = x0.unsqueeze(0)  # [1, C, H, W]
                    logsnr_batch = logsnr.unsqueeze(0)  # [1, 1, H, W]

                    # Compute grid shape
                    p = sparse_ae.patch_size
                    grid_shape = (x0.shape[1] // p, x0.shape[2] // p)

                    # Build masks (like train_autoembed does)
                    encoder_masks, decoder_masks = sparse_ae.build_masks(grid_shape, device)

                    # First pass: DIRECT AE forward (no projection bottleneck)
                    output1 = sparse_ae(
                        x0_batch,
                        encoder_masks=encoder_masks,
                        decoder_masks=decoder_masks,
                        grid_shape=grid_shape
                    )
                    recon1 = output1['recon'][0]  # [C, H, W]

                    # Round-trip: encode the reconstruction, decode again
                    recon1_batch = recon1.unsqueeze(0)
                    output2 = sparse_ae(
                        recon1_batch,
                        encoder_masks=encoder_masks,
                        decoder_masks=decoder_masks,
                        grid_shape=grid_shape
                    )
                    recon2 = output2['recon'][0]  # [C, H, W]

                    samples.append({
                        'input': x0.float(),  # Keep original in float32 for plotting
                        'recon1': recon1.float(),
                        'recon2': recon2.float(),
                        'source': getattr(b, 'source', split_name)
                    })
                except Exception as e:
                    print(f"    AE forward failed: {e}")
                    continue

                if len(samples) >= n_samples:
                    break
            if len(samples) >= n_samples:
                break

    if not samples:
        print("    plot_ae_roundtrip: No samples collected")
        return

    # Create figure: 5 columns per row
    # [Input, Recon1, Error1, Recon2 (roundtrip), Error2 (roundtrip)]
    n = len(samples)
    fig, axes = plt.subplots(n, 5, figsize=(15, 2.5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    def to_img(t):
        return t.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()

    mse1_total, mse2_total = 0.0, 0.0

    for i, s in enumerate(samples):
        x0 = s['input']
        r1 = s['recon1']
        r2 = s['recon2']

        # Compute error maps
        err1 = ((r1 - x0) ** 2).mean(dim=0)  # [H, W]
        err2 = ((r2 - x0) ** 2).mean(dim=0)  # [H, W] - error vs original, not vs r1

        mse1 = err1.mean().item()
        mse2 = err2.mean().item()
        mse1_total += mse1
        mse2_total += mse2

        # Column 0: Input
        axes[i, 0].imshow(to_img(x0))
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title("Input")
        axes[i, 0].set_ylabel(s['source'], rotation=0, labelpad=40, fontsize=8)

        # Column 1: Reconstruction (1st pass)
        axes[i, 1].imshow(to_img(r1))
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title("AE Recon")
        axes[i, 1].text(0.02, 0.98, f"MSE:{mse1:.4f}", transform=axes[i, 1].transAxes,
                        fontsize=7, va='top', color='white', backgroundcolor='black')

        # Column 2: Error map (1st pass)
        err1_np = err1.detach().cpu().numpy()
        vmax1 = max(err1_np.max() * 0.7, 1e-4)
        axes[i, 2].imshow(err1_np, cmap='hot', vmin=0, vmax=vmax1)
        axes[i, 2].axis('off')
        if i == 0:
            axes[i, 2].set_title("Error (1st)")

        # Column 3: Round-trip reconstruction
        axes[i, 3].imshow(to_img(r2))
        axes[i, 3].axis('off')
        if i == 0:
            axes[i, 3].set_title("Round-trip")
        axes[i, 3].text(0.02, 0.98, f"MSE:{mse2:.4f}", transform=axes[i, 3].transAxes,
                        fontsize=7, va='top', color='white', backgroundcolor='black')

        # Column 4: Round-trip error map (vs original)
        err2_np = err2.detach().cpu().numpy()
        vmax2 = max(err2_np.max() * 0.7, 1e-4)
        axes[i, 4].imshow(err2_np, cmap='hot', vmin=0, vmax=vmax2)
        axes[i, 4].axis('off')
        if i == 0:
            axes[i, 4].set_title("Error (RT)")

    # Add summary stats
    avg_mse1 = mse1_total / n
    avg_mse2 = mse2_total / n
    fig.suptitle(f"AE Round-trip Analysis @ {resolution}px | Avg MSE: 1st={avg_mse1:.5f}, RT={avg_mse2:.5f}", fontsize=10)

    plt.tight_layout()
    logger.save_figure(fig, name)
    print(f"    AE roundtrip @ {resolution}px: MSE 1st={avg_mse1:.5f}, roundtrip={avg_mse2:.5f}")



def plot_causal_sweep_v2(results: list, output_path, split_name: str):
    """
    Visualizes Causal Sweep with explicit metadata - no tensor shape inference.
    Layout per sequence:
        Row 1: [Context frames (noisy)] [Prediction] [GT]
        Row 2: [MSE: noisy vs clean]    [MSE: pred vs GT] (GT col omitted)
 
    Args:
        results: List of dicts, each containing:
            - 'snr': float, the SNR value for this sequence
            - 'gt': Tensor [C,H,W], ground truth target
            - 'pred': Tensor [C,H,W], predicted target
            - 'ctx_latents': List[Tensor], context latent frames (noisy)
            - 'ctx_latents_gt': List[Tensor], context latent frames (clean, for MSE baseline)
            - 'shape': tuple, the expected shape
            - 'seq_idx': int, sequence index
        output_path: Path to save the figure
        split_name: Name of the dataset split (for title)
    """
    if not results:
        return
 
    n_seq = len(results)
 
    # Determine max context length for uniform grid
    max_ctx = max(len(r['ctx_latents']) for r in results)
    n_cols = max_ctx + 2  # context frames + pred + GT
 
    # Layout: 2 rows per sequence (images, error maps)
    fig, axes = plt.subplots(n_seq * 2, n_cols, figsize=(2 * n_cols, 3 * n_seq))
    if n_seq == 1:
        axes = axes.reshape(2, -1)
    axes = axes.reshape(n_seq * 2, n_cols)
 
    fig.suptitle(f"Causal Sweep: {split_name}", fontsize=12)
 
    def to_img(t):
        """Safe tensor to numpy conversion."""
        return t.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
 
    def resize_for_comparison(img, target_shape):
        """Resize image if needed for MSE comparison."""
        if img.shape[:2] != target_shape[:2]:
            import torch.nn.functional as F
            t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
            t = F.interpolate(t, size=target_shape[:2], mode='bilinear', align_corners=False)
            return t.squeeze(0).permute(1, 2, 0).numpy()
        return img
 
    for i, r in enumerate(results):
        snr = r['snr']
        gt = r['gt']
        pred = r['pred']
        ctx_latents = r['ctx_latents']
        ctx_latents_gt = r.get('ctx_latents_gt', [])  # Clean versions for MSE baseline
 
        row_img = i * 2
        row_err = i * 2 + 1
 
        gt_np = to_img(gt)
 
        # --- Plot context frames and their MSE vs clean ---
        for j, ctx_t in enumerate(ctx_latents):
            ax = axes[row_img, j]
            ctx_np = to_img(ctx_t)
            ax.imshow(ctx_np)
            ax.axis('off')
            if i == 0 and j == 0:
                ax.set_title("Context", fontsize=9)
 
            # MSE: noisy context vs clean context (shows noise baseline)
            ax_err = axes[row_err, j]
            if j < len(ctx_latents_gt):
                ctx_gt_np = to_img(ctx_latents_gt[j])
                # Resize noisy to match GT resolution if needed
                if ctx_np.shape != ctx_gt_np.shape:
                    ctx_np_resized = resize_for_comparison(ctx_np, ctx_gt_np.shape)
                    diff = (ctx_np_resized - ctx_gt_np) ** 2
                else:
                    diff = (ctx_np - ctx_gt_np) ** 2
                mse_map = diff.mean(axis=2)
                ax_err.imshow(mse_map, cmap='inferno', vmin=0, vmax=0.1)
            else:
                # Fallback if no GT available
                ax_err.imshow(np.zeros((8, 8)), cmap='inferno', vmin=0, vmax=0.1)
            ax_err.axis('off')
 
        # Blank out unused context columns
        for j in range(len(ctx_latents), max_ctx):
            axes[row_img, j].axis('off')
            axes[row_err, j].axis('off')
 
        # --- Plot prediction ---
        pred_col = max_ctx
        ax_pred = axes[row_img, pred_col]
        pred_np = to_img(pred)
        ax_pred.imshow(pred_np)
        ax_pred.axis('off')
        if i == 0:
            ax_pred.set_title("Pred", fontsize=9)
 
        # --- Plot GT ---
        gt_col = max_ctx + 1
        ax_gt = axes[row_img, gt_col]
        ax_gt.imshow(gt_np)
        ax_gt.axis('off')
        if i == 0:
            ax_gt.set_title("GT", fontsize=9)
 
        # --- SNR label ---
        axes[row_img, 0].text(
            -0.3, 0.5, f"SNR {snr:.1f}",
            transform=axes[row_img, 0].transAxes,
            va='center', ha='right', fontsize=8, rotation=90
        )
 
        # --- Error map: pred vs GT (with resizing for resolution mismatch) ---
        ax_err_pred = axes[row_err, pred_col]
        if pred_np.shape != gt_np.shape:
            pred_np_resized = resize_for_comparison(pred_np, gt_np.shape)
            diff = (pred_np_resized - gt_np) ** 2
        else:
            diff = (pred_np - gt_np) ** 2
 
        mse_map = diff.mean(axis=2)
        ax_err_pred.imshow(mse_map, cmap='inferno', vmin=0, vmax=0.1)
        ax_err_pred.axis('off')
        if i == 0:
            ax_err_pred.set_title("MSE", fontsize=9)
 
        # GT column: no MSE (would be zeros, meaningless)
        ax_err_gt = axes[row_err, gt_col]
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {output_path}")


def plot_ae_diagnostic(diagnostics: list, logger, name: str = "ae_diagnostic"):
    """
    Visualizes AE vs Diffusion reconstruction comparison.

    Layout per sample:
        Row: [Ground Truth] [AE Recon] [Noisy Input] [Diffusion Recon] [Error Maps]

    Args:
        diagnostics: List of dicts from diagnostic_ae_vs_diffusion, each containing:
            - 'x0_clean': Ground truth tensor [C,H,W]
            - 'ae_recon': AE reconstruction [C,H,W] or None
            - 'diff_recon': Diffusion reconstruction [C,H,W]
            - 'z_noisy': Noisy input [C,H,W]
            - 'ae_mse': float
            - 'diff_mse': float
            - 'split': str, data source name
        logger: ExperimentLogger
        name: Output filename
    """
    if not diagnostics:
        return

    n = len(diagnostics)
    n_cols = 6  # GT, AE, Noisy, Diff, AE Error, Diff Error

    fig, axes = plt.subplots(n, n_cols, figsize=(2.5 * n_cols, 2.5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("AE vs Diffusion Diagnostic", fontsize=12)

    def to_img(t):
        if t is None:
            return np.zeros((32, 32, 3))
        return t.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()

    for i, d in enumerate(diagnostics):
        gt_np = to_img(d['x0_clean'])
        ae_np = to_img(d['ae_recon'])
        noisy_np = to_img(d['z_noisy'])
        diff_np = to_img(d['diff_recon'])

        # Column 0: Ground Truth
        axes[i, 0].imshow(gt_np)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title("GT", fontsize=9)

        # Column 1: AE Reconstruction
        axes[i, 1].imshow(ae_np)
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title("AE Recon", fontsize=9)

        # Column 2: Noisy Input
        axes[i, 2].imshow(noisy_np)
        axes[i, 2].axis('off')
        if i == 0:
            axes[i, 2].set_title("Noisy", fontsize=9)

        # Column 3: Diffusion Reconstruction
        axes[i, 3].imshow(diff_np)
        axes[i, 3].axis('off')
        if i == 0:
            axes[i, 3].set_title("Diff Recon", fontsize=9)

        # Column 4: AE Error Map
        if d['ae_recon'] is not None:
            ae_err = ((ae_np - gt_np) ** 2).mean(axis=2)
            axes[i, 4].imshow(ae_err, cmap='inferno', vmin=0, vmax=0.1)
        else:
            axes[i, 4].imshow(np.zeros_like(gt_np[:, :, 0]), cmap='inferno', vmin=0, vmax=0.1)
        axes[i, 4].axis('off')
        if i == 0:
            axes[i, 4].set_title("AE Err", fontsize=9)

        # Column 5: Diffusion Error Map
        diff_err = ((diff_np - gt_np) ** 2).mean(axis=2)
        axes[i, 5].imshow(diff_err, cmap='inferno', vmin=0, vmax=0.1)
        axes[i, 5].axis('off')
        if i == 0:
            axes[i, 5].set_title("Diff Err", fontsize=9)

        # Row label with MSE values
        import math
        ae_mse_str = f"{d['ae_mse']:.4f}" if not math.isnan(d['ae_mse']) else "N/A"
        label = f"{d['split']}\nAE:{ae_mse_str}\nDiff:{d['diff_mse']:.4f}"
        axes[i, 0].text(
            -0.15, 0.5, label,
            transform=axes[i, 0].transAxes,
            va='center', ha='right', fontsize=7
        )

    plt.tight_layout()
    logger.save_figure(fig, name)


# =============================================================================
# Subspace Routing Debug Plots
# =============================================================================

def plot_subspace_routing_stats(df, logger, name="subspace_routing"):
    """
    Plot wavelet vs amplitude subspace activation over training.

    Shows the relative contribution/activation of wavelet vs amplitude
    subspaces as training progresses.

    Args:
        df: DataFrame with columns: step, wav_active_mean, amp_active_mean, routing_entropy_mean
        logger: ExperimentLogger for saving figures
        name: Output filename prefix
    """
    if df.empty:
        return

    # Check for required columns
    required = ['step', 'wav_active_mean', 'amp_active_mean']
    if not all(col in df.columns for col in required):
        print(f"[plot_subspace_routing_stats] Missing columns. Have: {list(df.columns)}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Subspace activations over training (top-left)
    ax = axes[0, 0]
    steps = df.groupby('step').agg({
        'wav_active_mean': 'mean',
        'amp_active_mean': 'mean'
    })
    ax.plot(steps.index, steps['wav_active_mean'].rolling(20, min_periods=1).mean(),
            label='Wavelet Active', color='blue', linewidth=2)
    ax.plot(steps.index, steps['amp_active_mean'].rolling(20, min_periods=1).mean(),
            label='Amplitude Active', color='red', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Mean Active Dims')
    ax.set_title('Subspace Activation Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Ratio of wav/(wav+amp) (top-right)
    ax = axes[0, 1]
    total = steps['wav_active_mean'] + steps['amp_active_mean']
    ratio = steps['wav_active_mean'] / (total + 1e-7)
    ax.plot(steps.index, ratio.rolling(20, min_periods=1).mean(), color='purple', linewidth=2)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Balanced (0.5)')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Wavelet Fraction')
    ax.set_title('Wavelet/(Wavelet+Amplitude) Ratio')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Routing entropy if available (bottom-left)
    ax = axes[1, 0]
    if 'routing_entropy_mean' in df.columns:
        entropy = df.groupby('step')['routing_entropy_mean'].mean()
        ax.plot(entropy.index, entropy.rolling(20, min_periods=1).mean(),
                color='green', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Routing Entropy')
        ax.set_title('Subspace Routing Entropy\n(Higher = More Balanced)')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Routing entropy not available',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Routing Entropy')

    # 4. Stacked area chart (bottom-right)
    ax = axes[1, 1]
    ax.stackplot(steps.index,
                 steps['wav_active_mean'].rolling(20, min_periods=1).mean(),
                 steps['amp_active_mean'].rolling(20, min_periods=1).mean(),
                 labels=['Wavelet', 'Amplitude'],
                 colors=['lightblue', 'lightcoral'], alpha=0.7)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Active Dims (Stacked)')
    ax.set_title('Subspace Contribution (Stacked)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    logger.save_figure(fig, name)


def plot_residual_level_reconstructions(
    original: torch.Tensor,
    level_recons: list,
    logger,
    name="residual_levels",
    n_samples: int = 4
):
    """
    Visualize reconstructions using first k residual levels.

    Shows how reconstruction quality improves as more levels are added.

    Args:
        original: [B, C, H, W] original images
        level_recons: list of [B, C, H, W] cumulative reconstructions per level
        logger: ExperimentLogger for saving figures
        name: Output filename prefix
        n_samples: Number of samples to show (will use min(n_samples, batch_size))
    """
    n_levels = len(level_recons)
    B = original.shape[0]
    n_show = min(n_samples, B)

    fig, axes = plt.subplots(n_show, n_levels + 2, figsize=(2 * (n_levels + 2), 2 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_show):
        # Original
        orig_img = original[i].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        axes[i, 0].imshow(orig_img)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Original', fontsize=9)

        # Per-level cumulative reconstruction
        for lv, recon in enumerate(level_recons):
            recon_img = recon[i].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
            axes[i, lv + 1].imshow(recon_img)
            axes[i, lv + 1].axis('off')
            if i == 0:
                axes[i, lv + 1].set_title(f'Level 0-{lv}', fontsize=9)

        # Final error map
        final_recon = level_recons[-1][i].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        error = ((orig_img - final_recon) ** 2).mean(axis=2)
        axes[i, -1].imshow(error, cmap='inferno', vmin=0, vmax=0.05)
        axes[i, -1].axis('off')
        if i == 0:
            axes[i, -1].set_title('Final Error', fontsize=9)

    plt.tight_layout()
    logger.save_figure(fig, name)


def plot_subspace_ablation(
    original: torch.Tensor,
    full_recon: torch.Tensor,
    wav_only_recon: torch.Tensor,
    amp_only_recon: torch.Tensor,
    logger,
    name="subspace_ablation",
    n_samples: int = 4
):
    """
    Visualize subspace ablation: what happens when we knock out wav/amp pathways.

    Shows:
    - Original
    - Full reconstruction (both subspaces)
    - Wavelet-only (amplitude ablated)
    - Amplitude-only (wavelet ablated)
    - Error maps for each

    Args:
        original: [B, C, H, W] original images
        full_recon: [B, C, H, W] full reconstruction
        wav_only_recon: [B, C, H, W] reconstruction with amplitude ablated
        amp_only_recon: [B, C, H, W] reconstruction with wavelet ablated
        logger: ExperimentLogger
        name: Output filename prefix
        n_samples: Number of samples to show
    """
    B = original.shape[0]
    n_show = min(n_samples, B)

    fig, axes = plt.subplots(n_show, 7, figsize=(14, 2 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_show):
        orig = original[i].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        full = full_recon[i].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        wav = wav_only_recon[i].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        amp = amp_only_recon[i].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()

        # Original
        axes[i, 0].imshow(orig)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Original', fontsize=9)

        # Full reconstruction
        axes[i, 1].imshow(full)
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title('Full Recon', fontsize=9)

        # Full error
        full_err = ((orig - full) ** 2).mean(axis=2)
        axes[i, 2].imshow(full_err, cmap='inferno', vmin=0, vmax=0.05)
        axes[i, 2].axis('off')
        if i == 0:
            axes[i, 2].set_title('Full Error', fontsize=9)

        # Wavelet-only (amp ablated)
        axes[i, 3].imshow(wav)
        axes[i, 3].axis('off')
        if i == 0:
            axes[i, 3].set_title('Wav Only\n(Amp Ablated)', fontsize=9)

        # Wav error
        wav_err = ((orig - wav) ** 2).mean(axis=2)
        axes[i, 4].imshow(wav_err, cmap='inferno', vmin=0, vmax=0.1)
        axes[i, 4].axis('off')
        if i == 0:
            axes[i, 4].set_title('Wav Error', fontsize=9)

        # Amplitude-only (wav ablated)
        axes[i, 5].imshow(amp)
        axes[i, 5].axis('off')
        if i == 0:
            axes[i, 5].set_title('Amp Only\n(Wav Ablated)', fontsize=9)

        # Amp error
        amp_err = ((orig - amp) ** 2).mean(axis=2)
        axes[i, 6].imshow(amp_err, cmap='inferno', vmin=0, vmax=0.1)
        axes[i, 6].axis('off')
        if i == 0:
            axes[i, 6].set_title('Amp Error', fontsize=9)

    plt.tight_layout()
    logger.save_figure(fig, name)


def plot_subspace_contributions(
    original: torch.Tensor,
    recon: torch.Tensor,
    wav_contribution: torch.Tensor,
    amp_contribution: torch.Tensor,
    logger,
    name="subspace_contributions",
    n_samples: int = 4
):
    """
    Visualize the individual subspace contributions to reconstruction.

    Shows:
    - Original
    - Combined reconstruction
    - Wavelet pathway contribution (may have negative values, show as heatmap)
    - Amplitude pathway contribution
    - Difference between pathways

    Args:
        original: [B, C, H, W] original images
        recon: [B, C, H, W] combined reconstruction
        wav_contribution: [B, C, H, W] wavelet pathway output (before sum)
        amp_contribution: [B, C, H, W] amplitude pathway output (before sum)
        logger: ExperimentLogger
        name: Output filename prefix
        n_samples: Number of samples to show
    """
    B = original.shape[0]
    n_show = min(n_samples, B)

    fig, axes = plt.subplots(n_show, 6, figsize=(12, 2 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_show):
        orig = original[i].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        rec = recon[i].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        wav = wav_contribution[i].detach().cpu().permute(1, 2, 0).numpy()
        amp = amp_contribution[i].detach().cpu().permute(1, 2, 0).numpy()

        # Original
        axes[i, 0].imshow(orig)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Original', fontsize=9)

        # Reconstruction
        axes[i, 1].imshow(rec)
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title('Recon', fontsize=9)

        # Wavelet contribution (normalize for visualization)
        wav_mag = np.abs(wav).mean(axis=2)
        im = axes[i, 2].imshow(wav_mag, cmap='Blues')
        axes[i, 2].axis('off')
        if i == 0:
            axes[i, 2].set_title('Wav |contrib|', fontsize=9)

        # Amplitude contribution
        amp_mag = np.abs(amp).mean(axis=2)
        axes[i, 3].imshow(amp_mag, cmap='Reds')
        axes[i, 3].axis('off')
        if i == 0:
            axes[i, 3].set_title('Amp |contrib|', fontsize=9)

        # Ratio: wav/(wav+amp)
        total = wav_mag + amp_mag + 1e-7
        ratio = wav_mag / total
        axes[i, 4].imshow(ratio, cmap='coolwarm', vmin=0, vmax=1)
        axes[i, 4].axis('off')
        if i == 0:
            axes[i, 4].set_title('Wav Ratio', fontsize=9)

        # Error
        error = ((orig - rec) ** 2).mean(axis=2)
        axes[i, 5].imshow(error, cmap='inferno', vmin=0, vmax=0.05)
        axes[i, 5].axis('off')
        if i == 0:
            axes[i, 5].set_title('Error', fontsize=9)

    plt.tight_layout()
    logger.save_figure(fig, name)


def plot_subspace_sensitivity(
    sweep_results,#: Dict[str, Any],
    logger,
    name="subspace_sensitivity"
):
    """
    Plot subspace ablation sensitivity curves (d_mse / d_ablation).

    Visualizes how reconstruction MSE degrades as we stochastically ablate
    different proportions of wavelet vs amplitude subspace dimensions.

    Args:
        sweep_results: Dict from SwiGLUFSQAutoencoder.subspace_sensitivity_sweep() with:
            - ablation_rates: list of float ablation rates [0.0, 0.1, ..., 1.0]
            - mse_baseline: float, reconstruction MSE with no ablation
            - mse_wav_ablated: list of MSE values when ablating wavelet dims at each rate
            - mse_amp_ablated: list of MSE values when ablating amplitude dims at each rate
            - mse_both_ablated: list of MSE values when ablating both at each rate
            - d_mse_d_wav: list of gradient estimates (MSE increase per ablation rate)
            - d_mse_d_amp: list of gradient estimates
        logger: ExperimentLogger for saving figures
        name: Output filename prefix
    """
    rates = np.array(sweep_results['ablation_rates'])
    mse_baseline = sweep_results['mse_baseline']
    mse_wav = np.array(sweep_results['mse_wav_ablated'])
    mse_amp = np.array(sweep_results['mse_amp_ablated'])
    mse_both = np.array(sweep_results['mse_both_ablated'])
    # Note: d_mse_d_wav/d_mse_d_amp from sweep may have wrong length; compute derivatives locally

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. MSE vs Ablation Rate (top-left)
    ax = axes[0, 0]
    ax.axhline(mse_baseline, color='gray', linestyle='--', alpha=0.7, label=f'Baseline ({mse_baseline:.4f})')
    ax.plot(rates, mse_wav, 'b-o', markersize=4, linewidth=2, label='Wavelet Ablated')
    ax.plot(rates, mse_amp, 'r-s', markersize=4, linewidth=2, label='Amplitude Ablated')
    ax.plot(rates, mse_both, 'g-^', markersize=4, linewidth=2, label='Both Ablated')
    ax.set_xlabel('Ablation Rate')
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('MSE vs Subspace Ablation Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # 2. Normalized MSE (relative to baseline) (top-right)
    ax = axes[0, 1]
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.7, label='Baseline (1.0)')
    norm_wav = mse_wav / (mse_baseline + 1e-8)
    norm_amp = mse_amp / (mse_baseline + 1e-8)
    norm_both = mse_both / (mse_baseline + 1e-8)
    ax.plot(rates, norm_wav, 'b-o', markersize=4, linewidth=2, label='Wavelet')
    ax.plot(rates, norm_amp, 'r-s', markersize=4, linewidth=2, label='Amplitude')
    ax.plot(rates, norm_both, 'g-^', markersize=4, linewidth=2, label='Both')
    ax.set_xlabel('Ablation Rate')
    ax.set_ylabel('MSE / Baseline MSE')
    ax.set_title('Relative MSE Degradation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # 3. d_MSE / d_ablation gradient curves (bottom-left)
    ax = axes[1, 0]
    # Use midpoint rates for gradient display (gradients are between points)
    mid_rates = (rates[:-1] + rates[1:]) / 2 if len(rates) > 1 else rates
    # Compute derivatives from MSE values (more reliable than pre-computed d_mse_d_*)
    # This ensures length matches mid_rates (n-1 elements from n points)
    d_rates = np.diff(rates)
    d_wav_computed = np.diff(mse_wav) / (d_rates + 1e-8)
    d_amp_computed = np.diff(mse_amp) / (d_rates + 1e-8)
    ax.plot(mid_rates, d_wav_computed, 'b-o', markersize=4, linewidth=2, label='d(MSE)/d(wav_ablation)')
    ax.plot(mid_rates, d_amp_computed, 'r-s', markersize=4, linewidth=2, label='d(MSE)/d(amp_ablation)')
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Ablation Rate')
    ax.set_ylabel('d(MSE) / d(Ablation)')
    ax.set_title('MSE Sensitivity to Ablation\n(Gradient: Higher = More Sensitive)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # 4. Summary statistics (bottom-right)
    ax = axes[1, 1]
    ax.axis('off')

    # Calculate summary metrics
    # Area under MSE curve (trapezoid integration)
    # np.trapezoid replaces deprecated np.trapz in NumPy 2.0+
    auc_wav = np.trapezoid(mse_wav - mse_baseline, rates)
    auc_amp = np.trapezoid(mse_amp - mse_baseline, rates)
    auc_both = np.trapezoid(mse_both - mse_baseline, rates)

    # Max gradient (peak sensitivity) - use computed derivatives
    max_d_wav = np.max(d_wav_computed) if len(d_wav_computed) > 0 else 0
    max_d_amp = np.max(d_amp_computed) if len(d_amp_computed) > 0 else 0

    # Mean gradient
    mean_d_wav = np.mean(d_wav_computed) if len(d_wav_computed) > 0 else 0
    mean_d_amp = np.mean(d_amp_computed) if len(d_amp_computed) > 0 else 0

    # MSE at full ablation
    full_wav = mse_wav[-1] if len(mse_wav) > 0 else 0
    full_amp = mse_amp[-1] if len(mse_amp) > 0 else 0
    full_both = mse_both[-1] if len(mse_both) > 0 else 0

    # Relative importance: ratio of AUC
    total_auc = auc_wav + auc_amp + 1e-8
    wav_importance = auc_wav / total_auc
    amp_importance = auc_amp / total_auc

    summary_text = f"""
    Subspace Sensitivity Summary
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Baseline MSE: {mse_baseline:.6f}

    MSE at Full Ablation (rate=1.0):
      Wavelet ablated:   {full_wav:.6f}  (+{(full_wav/mse_baseline - 1)*100:.1f}%)
      Amplitude ablated: {full_amp:.6f}  (+{(full_amp/mse_baseline - 1)*100:.1f}%)
      Both ablated:      {full_both:.6f}  (+{(full_both/mse_baseline - 1)*100:.1f}%)

    Area Under Curve (MSE degradation):
      Wavelet:   {auc_wav:.6f}  ({wav_importance*100:.1f}%)
      Amplitude: {auc_amp:.6f}  ({amp_importance*100:.1f}%)
      Both:      {auc_both:.6f}

    Gradient Statistics (d_MSE / d_ablation):
      Wavelet   - Mean: {mean_d_wav:.6f}, Max: {max_d_wav:.6f}
      Amplitude - Mean: {mean_d_amp:.6f}, Max: {max_d_amp:.6f}

    Interpretation:
      {"Wavelet subspace more critical" if wav_importance > 0.55 else
       "Amplitude subspace more critical" if amp_importance > 0.55 else
       "Both subspaces roughly equal importance"}
    """

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    logger.save_figure(fig, name)


def plot_subspace_sensitivity_heatmap(
    sweep_results_by_resolution,#: Dict[int, Dict[str, Any]],
    logger,
    name="subspace_sensitivity_heatmap"
):
    """
    Plot sensitivity heatmap across multiple resolutions.

    Compares how subspace importance varies with image resolution.

    Args:
        sweep_results_by_resolution: Dict mapping resolution -> sweep_results
        logger: ExperimentLogger
        name: Output filename prefix
    """
    resolutions = sorted(sweep_results_by_resolution.keys())
    n_res = len(resolutions)

    if n_res == 0:
        print("[plot_subspace_sensitivity_heatmap] No data to plot")
        return

    # Get ablation rates from first result
    first_result = sweep_results_by_resolution[resolutions[0]]
    rates = np.array(first_result['ablation_rates'])
    n_rates = len(rates)

    # Build heatmaps: [n_res, n_rates]
    wav_heatmap = np.zeros((n_res, n_rates))
    amp_heatmap = np.zeros((n_res, n_rates))

    for i, res in enumerate(resolutions):
        result = sweep_results_by_resolution[res]
        baseline = result['mse_baseline']
        # Normalize by baseline for comparability across resolutions
        wav_heatmap[i, :] = (np.array(result['mse_wav_ablated']) - baseline) / (baseline + 1e-8)
        amp_heatmap[i, :] = (np.array(result['mse_amp_ablated']) - baseline) / (baseline + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Wavelet sensitivity heatmap
    ax = axes[0]
    im = ax.imshow(wav_heatmap, aspect='auto', cmap='Blues',
                   extent=[rates[0], rates[-1], resolutions[-1], resolutions[0]])
    ax.set_xlabel('Ablation Rate')
    ax.set_ylabel('Resolution')
    ax.set_title('Wavelet Ablation Sensitivity\n(Relative MSE Increase)')
    ax.set_yticks(resolutions)
    plt.colorbar(im, ax=ax, label='(MSE - baseline) / baseline')

    # Amplitude sensitivity heatmap
    ax = axes[1]
    im = ax.imshow(amp_heatmap, aspect='auto', cmap='Reds',
                   extent=[rates[0], rates[-1], resolutions[-1], resolutions[0]])
    ax.set_xlabel('Ablation Rate')
    ax.set_ylabel('Resolution')
    ax.set_title('Amplitude Ablation Sensitivity\n(Relative MSE Increase)')
    ax.set_yticks(resolutions)
    plt.colorbar(im, ax=ax, label='(MSE - baseline) / baseline')

    plt.tight_layout()
    logger.save_figure(fig, name)


def plot_subspace_sensitivity_exemplars(
    ae,
    images: "torch.Tensor",
    logger,
    ablation_rates: list = [0.0, 0.25, 0.5, 0.75, 1.0],
    name: str = "subspace_sensitivity_exemplars",
    dtype: "torch.dtype" = None
):
    """
    Plot visual exemplars of subspace ablation effects.

    Shows actual reconstructions at different ablation rates to build
    intuition for what each subspace (wavelet vs amplitude) encodes.

    Grid layout:
        Rows: different images (n_samples)
        Columns: Original | Baseline | Wav@25% | Wav@50% | ... | Amp@25% | Amp@50% | ...

    Args:
        ae: SwiGLUFSQAutoencoder with decode_with_ablation() method
        images: [B, C, H, W] tensor of images to reconstruct
        logger: ExperimentLogger
        ablation_rates: List of ablation rates to visualize
        name: Output filename prefix
        dtype: torch dtype for autocast (bf16/fp16/fp32)
    """
    import torch

    if not getattr(ae, 'wavelet_gating', False):
        print(f"[{name}] Skipping - AE doesn't have wavelet_gating enabled")
        return

    n_samples = min(4, images.shape[0])  # Show up to 4 exemplars
    n_rates = len(ablation_rates)

    # Columns: Original, Baseline (no ablation), then wav@rates, then amp@rates
    n_cols = 2 + 2 * n_rates  # orig + baseline + wav_rates + amp_rates

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(2.5 * n_cols, 2.5 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    # Compute grid shape from images
    p = ae.patch_size
    H, W = images.shape[2], images.shape[3]
    grid_shape = (H // p, W // p)

    # Autocast context - model weights are in training dtype
    if dtype is None:
        dtype = torch.bfloat16  # Default assumption
    use_amp = dtype in (torch.bfloat16, torch.float16)

    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
        # Build masks once
        encoder_masks, decoder_masks = ae.build_masks(grid_shape, images.device)

        # Encode once
        codes = ae.encode(images[:n_samples], grid_shape=grid_shape,
                          encoder_masks=encoder_masks, decoder_masks=decoder_masks)

        # Baseline reconstruction (no ablation)
        baseline_recon = ae.decode(codes, grid_shape, decoder_masks)

        for row in range(n_samples):
            col = 0

            # Original image
            img_np = images[row].permute(1, 2, 0).float().cpu().numpy()
            img_np = np.clip(img_np, 0, 1)
            axes[row, col].imshow(img_np)
            axes[row, col].set_title('Original' if row == 0 else '')
            axes[row, col].axis('off')
            col += 1

            # Baseline (no ablation)
            recon_np = baseline_recon[row].permute(1, 2, 0).float().cpu().numpy()
            recon_np = np.clip(recon_np, 0, 1)
            axes[row, col].imshow(recon_np)
            axes[row, col].set_title('Baseline' if row == 0 else '')
            axes[row, col].axis('off')
            col += 1

            # Wavelet ablations (use deterministic=True for reproducible visuals)
            for rate in ablation_rates:
                recon = ae.decode_with_ablation(
                    codes, grid_shape, ablate_wavelet=rate, ablate_amplitude=0.0,
                    decoder_masks=decoder_masks, deterministic=True
                )
                recon_np = recon[row].permute(1, 2, 0).float().cpu().numpy()
                recon_np = np.clip(recon_np, 0, 1)
                axes[row, col].imshow(recon_np)
                axes[row, col].set_title(f'Wav {int(rate*100)}%' if row == 0 else '')
                axes[row, col].axis('off')
                col += 1

            # Amplitude ablations
            for rate in ablation_rates:
                recon = ae.decode_with_ablation(
                    codes, grid_shape, ablate_wavelet=0.0, ablate_amplitude=rate,
                    decoder_masks=decoder_masks, deterministic=True
                )
                recon_np = recon[row].permute(1, 2, 0).float().cpu().numpy()
                recon_np = np.clip(recon_np, 0, 1)
                axes[row, col].imshow(recon_np)
                axes[row, col].set_title(f'Amp {int(rate*100)}%' if row == 0 else '')
                axes[row, col].axis('off')
                col += 1

    plt.suptitle('Subspace Ablation Exemplars\n(Wavelet = frequency/texture | Amplitude = intensity/color)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    logger.save_figure(fig, name)