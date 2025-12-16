# plotting.py
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

def plot_dset_reconstruction(x0s, noisy, recon, lmaps, output_path, show_map=False):
    """
    Standard 3-4 column reconstruction plot.
    """
    n = len(x0s)
    cols = 4 if (show_map and lmaps is not None) else 3
    
    fig, axes = plt.subplots(n, cols, figsize=(3*cols, 2*n))
    if n == 1: axes = axes.reshape(1, -1)
    
    def to_img(t):
        return t.detach().cpu().permute(1,2,0).clamp(0,1).numpy()

    for i in range(n):
        # GT
        axes[i, 0].imshow(to_img(x0s[i]))
        axes[i, 0].set_title("Ground Truth" if i==0 else "")
        axes[i, 0].axis("off")

        # Noisy
        axes[i, 1].imshow(to_img(noisy[i]))
        axes[i, 1].set_title("Noisy Input" if i==0 else "")
        axes[i, 1].axis("off")

        # Recon
        axes[i, 2].imshow(to_img(recon[i]))
        axes[i, 2].set_title("Reconstruction" if i==0 else "")
        axes[i, 2].axis("off")
        
        # Map
        if show_map and lmaps is not None:
            m = lmaps[i].detach().cpu().squeeze().numpy()
            axes[i, 3].imshow(m, cmap="viridis")
            axes[i, 3].set_title("LogSNR Map" if i==0 else "")
            axes[i, 3].axis("off")
            
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

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
    
    # 3. Render Topology (Metadata -> Coords)
    topo_embeds, doc_ids = render_topology_embeddings(span_objects, max_dims=3, device=device)
    
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
    df["snr_bin"] = pd.cut(df["logsnr"], bins=20)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    snr_grouped = df.groupby("snr_bin", observed=True)["loss"].mean()
    snr_x = [i.mid for i in snr_grouped.index]
    axes[0,0].plot(snr_x, snr_grouped.values, marker="o", linewidth=2)
    axes[0,0].set_title("d_loss / d_logsnr")
    axes[0,0].set_xlabel("LogSNR")
    axes[0,0].set_ylabel("MSE Loss")
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].invert_xaxis()
    if "resolution" in df.columns:
        res_stats = df.groupby("resolution")["loss"].agg(["mean", "std"])
        axes[0,1].errorbar(res_stats.index, res_stats["mean"], yerr=res_stats["std"], fmt="-o", capsize=5)
        axes[0,1].set_title("d_loss / d_resolution")
        axes[0,1].set_xlabel("Tokens")
        axes[0,1].set_xscale("log")
        axes[0,1].grid(True, alpha=0.3)
    for source, grp in df.groupby("source"):
        step_avg = grp.groupby("step")["loss"].mean()
        smooth = step_avg.rolling(window=50, min_periods=1).mean()
        axes[1,0].plot(smooth.index, smooth.values, label=source)
    axes[1,0].legend()
    axes[1,0].set_title("d_loss / d_source")
    axes[1,0].set_xlabel("Step")
    axes[1,0].set_yscale("log")
    axes[1,0].grid(True, alpha=0.3)
    var_grouped = df.groupby("snr_bin", observed=True)["loss_var"].mean()
    axes[1,1].plot([i.mid for i in var_grouped.index], var_grouped.values, color="orange", marker="s")
    axes[1,1].set_title("Variance / d_logsnr")
    axes[1,1].set_xlabel("LogSNR")
    axes[1,1].invert_xaxis()
    axes[1,1].grid(True, alpha=0.3)
    plt.tight_layout()
    logger.save_figure(fig, stringy)



def plot_dset_reconstruction(result_dict, logger, name="reconstruction", show_map=False):
    # Expect lists of tensors, potentially of mixed resolution
    x0s = result_dict["x0"]
    noisy = result_dict["noisy_input"]
    recon = result_dict["reconstruction"]
    lmaps = result_dict.get("logsnr_map", None)
    
    n = len(x0s)
    if n == 0: return

    cols = 4 if (show_map and lmaps is not None) else 3
    
    # Create figure
    fig, axes = plt.subplots(n, cols, figsize=(3*cols, 2*n))
    if n == 1: axes = axes.reshape(1, -1)
    
    for i in range(n):
        # Helper to safely visualize a single tensor (C,H,W) -> numpy (H,W,C)
        def to_img(t):
            return t.detach().cpu().permute(1,2,0).clamp(0,1).numpy()

        # Col 0: Ground Truth
        axes[i, 0].imshow(to_img(x0s[i]))
        axes[i, 0].axis("off")
        if i==0: axes[i,0].set_title("Ground Truth")

        # Col 1: Noisy Input
        axes[i, 1].imshow(to_img(noisy[i]))
        axes[i, 1].axis("off")
        if i==0: axes[i,1].set_title("Noisy Input")

        # Col 2: Reconstruction
        axes[i, 2].imshow(to_img(recon[i]))
        axes[i, 2].axis("off")
        if i==0: axes[i,2].set_title("Reconstruction")
        
        # Col 3: LogSNR Map
        if show_map and lmaps is not None:
            # Map might be (1,H,W) or (H,W)
            m = lmaps[i].detach().cpu().squeeze().numpy()
            axes[i, 3].imshow(m, cmap="viridis")
            axes[i, 3].axis("off")
            if i==0: axes[i,3].set_title("Split Map")
            
    plt.tight_layout()
    logger.save_figure(fig, name)