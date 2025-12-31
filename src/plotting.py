# plotting.py
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

    span_emb = components[1]
    span_unemb = components[2]

    # Get the actual patch embedder/unembedder
    if hasattr(span_emb, 'patch_emb'):
        patch_emb = span_emb.patch_emb
    elif hasattr(span_emb, 'patch_embedder'):
        patch_emb = span_emb.patch_embedder
    else:
        print("    plot_ae_roundtrip: No patch embedder found")
        return

    if hasattr(span_unemb, 'patch_unembed'):
        patch_unemb = span_unemb.patch_unembed
    elif hasattr(span_unemb, 'patch_unembedder'):
        patch_unemb = span_unemb.patch_unembedder
    else:
        print("    plot_ae_roundtrip: No patch unembedder found")
        return

    # Detect model dtype from first parameter
    model_dtype = None
    for p in patch_emb.parameters():
        model_dtype = p.dtype
        break
    if model_dtype is None:
        model_dtype = torch.float32

    # Collect samples
    samples = []
    split_names = iterator.get_split_names()
    n_per_split = max(1, n_samples // max(1, len(split_names)))

    with torch.no_grad():
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

                # Convert to model dtype
                x0_cast = x0.to(model_dtype)
                logsnr_cast = logsnr.to(model_dtype)

                try:
                    # First pass: encode then decode
                    z1, grid_shape = patch_emb(x0_cast, logsnr_cast)
                    recon1_full = patch_unemb(z1, grid_shape)
                    recon1 = recon1_full[:3]  # RGB only

                    # Round-trip: encode the reconstruction, decode again
                    # Use same logsnr (or could use predicted logsnr from recon1_full[-1:])
                    z2, grid_shape2 = patch_emb(recon1, logsnr_cast)
                    recon2_full = patch_unemb(z2, grid_shape2)
                    recon2 = recon2_full[:3]

                    samples.append({
                        'input': x0.float(),  # Keep original in float32
                        'recon1': recon1.float(),  # Convert to float32 for plotting
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