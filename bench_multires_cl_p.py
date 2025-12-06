# bench_multires_cl_p.py
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path

from diffusion_utils import get_schedule, get_alpha_sigma, BucketManager
from dataset import CompositeIterator
#from model import HybridGemmaDiT

from ld_tformer import coolerLDTformer, SpanEmbedder, SpanUnembedder, build_composed_mask
from ld_tformer_embedding_functional import render_topology_embeddings
from memory_manager import KVTManager, PageTable

# Hook the ZMQ compiler backend immediately if available
try:
    import inductor_cas_client
    inductor_cas_client.install_cas_client()
except ImportError:
    print("we tried to import inductor_cas_client but mom won't let us")
    pass

class ExperimentLogger:
    def __init__(self, output_dir="."):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        script_path = Path(sys.argv[0])
        self.script_name = script_path.stem
        existing = list(self.output_dir.glob(f"{self.script_name}_run_*"))
        if existing:
            run_nums = [int(p.stem.split('_run_')[1].split('_')[0]) for p in existing]
            self.run_id = max(run_nums) + 1
        else:
            self.run_id = 0
        self.figure_count = 0
        self.run_dir = self.output_dir / f"{self.script_name}_run_{self.run_id:03d}"
        self.run_dir.mkdir(exist_ok=True)
        print(f"📊 Experiment: {self.script_name} | Run: {self.run_id} | Dir: {self.run_dir}")
        
    def save_figure(self, fig, name=None):
        if name is None: name = f"fig{self.figure_count}"
        filename = f"{name}.png"
        filepath = self.run_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.figure_count += 1
        return filepath

### wowie this is where we put diffusion helpers!!!!

def predict_velocity_field(components, z, logsnr, spans, mode):
    """
    Unified prediction logic.
    Returns:
        v_final: [B, C, H, W] - Spatially corrected velocity field
        aux_loss: Scalar - Internal model auxiliary loss (e.g. MoE balancing)
    """
    model, _, span_unemb, _, _ = components
    
    # 1. Forward Pass
    # Naive mode passes zeroed logsnr input, but the architecture is invariant
    suppress = (mode == 'naive')
    z_flat, aux_loss, objs = run_forward_step(
        components, z, logsnr, spans, 
        suppress_logsnr_input=suppress
    )
    
    # 2. Unembed
    decoded = span_unemb.decode(z_flat, objs)
    
    # 3. Stack & Parse
    # [B, C, H, W]
    v_raw = torch.stack([d['image_vpreds'] for d in decoded])
    
    if mode == 'factorized':
        # [B, 1, H, W] - Preserve spatial dimensions!
        l_maps = torch.stack([d['image_logsnrs'] for d in decoded])
        
        # Spatial Correction: Apply sigmoid scale element-wise per pixel
        # sigma_p(x,y) = sqrt(sigmoid(-logsnr(x,y)))
        sigma_p = torch.sqrt(torch.sigmoid(-l_maps))
        
        # [B, C, H, W] * [B, 1, H, W] -> Broadcasts spatially
        v_final = v_raw * sigma_p
    else:
        v_final = v_raw
        
    return v_final, aux_loss

def run_forward_step(
    components, 
    z, 
    logsnr, 
    base_spans,
    suppress_logsnr_input: bool = False
):
    """
    Step logic. 
    If suppress_logsnr_input is True, we pass zeros to the embedder (Naive Mode),
    but the graph execution path remains identical.
    """
    model, span_embedder, _, kvt_manager, page_table = components
    B, C, H, W = z.shape
    device = z.device
    num_layers = len(model.layers)
    
    # 1. Prepare Metadata & Inputs
    batch_spans_meta = []
    images = []
    logsnr_maps = []
    
    zero_map = torch.zeros((1, H, W), device=device)
    
    for i in range(B):
        # Create unique IDs
        item_spans = [s.copy() for s in base_spans]
        for s in item_spans: s['id'] = f"req_{i}"
        batch_spans_meta.extend(item_spans)
        
        images.append(z[i])
        
        # CONTROL POINT: Naive mode sees zeroed logsnr map
        if suppress_logsnr_input:
            logsnr_maps.append(zero_map)
        else:
            logsnr_maps.append(logsnr[i].view(1, 1, 1).expand(1, H, W))

    # 2. Embed
    z_flat, span_objects, content_hashes = span_embedder.embed(
        batch_spans_meta,
        text_tokens=[None] * B,
        images=images,
        logsnr_maps=logsnr_maps
    )
    
    # 3. Topo & Alloc
    topo_embeds, _ = render_topology_embeddings(batch_spans_meta, 3, device)
    
    req_ids = list(range(B))
    cursor = 0
    for rid in req_ids:
        span_len = batch_spans_meta[rid]['len'] 
        kvt_manager.allocate_and_write_sequence(
            rid, 
            content_hashes[cursor:cursor+span_len], 
            topo_embeds[cursor:cursor+span_len]
        )
        cursor += span_len

    # 4. Inputs & Mask
    attn_inputs = kvt_manager.get_attention_inputs(req_ids, layer_idx=0)
    pt_table, phys_to_log = kvt_manager.get_batch_mappings(req_ids)
    
    page_table.page_table[:B, :pt_table.shape[1]] = pt_table
    page_table.physical_to_logical[:B, :phys_to_log.shape[1]] = phys_to_log
    
    block_mask = build_composed_mask(
        span_objects,
        # FIX: Pass the flattened tensor 'topo_embeds' instead of the list 'attn_inputs['topo_active']'
        topo_active=topo_embeds,
        topo_heap=attn_inputs['topo_heap'],
        page_table=page_table,
        batch_idx=attn_inputs['batch_idx']
    )
    
    # 5. Forward (Compiled Graph)
    z_out, aux_loss = model(
        z_flat.unsqueeze(0),
        topo_embeds.unsqueeze(0),
        k_caches=[kvt_manager.k_cache[i] for i in range(num_layers)], # Use k_cache
        v_caches=[kvt_manager.v_cache[i] for i in range(num_layers)], # Use v_cache
        slot_mapping=attn_inputs['slot_mapping'],
        block_mask=block_mask
    )
    
    # 6. Cleanup
    for rid in req_ids:
        kvt_manager.free_sequence(kvt_manager.req_tables[rid])
        
    return z_out.squeeze(0), aux_loss, span_objects


def logsnr_to_alpha_sigma(logsnr):
    """Convert log(SNR) to (alpha, sigma) for noise schedule."""
    # logsnr = log(alpha^2 / sigma^2)
    # SNR = alpha^2 / sigma^2
    snr = torch.exp(logsnr)
    # alpha^2 = SNR / (1 + SNR), sigma^2 = 1 / (1 + SNR)
    alpha_sq = snr / (1.0 + snr)
    sigma_sq = 1.0 / (1.0 + snr)
    return torch.sqrt(alpha_sq), torch.sqrt(sigma_sq)

def sample_logsnr_triplet(batch_size, device, min_logsnr=-10.0, max_logsnr=0.0, min_gap=1.0):
    """
    Sample three noise levels at least min_gap apart.
    Returns: (logsnr_low, logsnr_mid, logsnr_high) where low > mid > high
    """
    # Sample lowest (least noisy, highest logsnr)
    logsnr_low = torch.rand(batch_size, device=device) * (max_logsnr - min_logsnr) + min_logsnr
    
    # Sample mid: 1-4 logsnr units below low
    gap_mid = torch.rand(batch_size, device=device) * 3.0 + min_gap
    logsnr_mid = (logsnr_low - gap_mid).clamp(min=min_logsnr)
    
    # Sample high: 1-4 logsnr units below mid
    gap_high = torch.rand(batch_size, device=device) * 3.0 + min_gap
    logsnr_high = (logsnr_mid - gap_high).clamp(min=min_logsnr)
    
    return logsnr_low, logsnr_mid, logsnr_high

def euler_reverse_step(z_t, v_pred, logsnr_from, logsnr_to):
    """
    Take one deterministic reverse diffusion step using v-prediction.
    
    Args:
        z_t: Current noisy latent [B, C, H, W]
        v_pred: Model's v-prediction [B, C, H, W]
        logsnr_from: Current log(SNR) [B]
        logsnr_to: Target log(SNR) [B]
    
    Returns:
        z_to: Denoised latent at target noise level [B, C, H, W]
    """
    alpha_from, sigma_from = logsnr_to_alpha_sigma(logsnr_from)
    alpha_to, sigma_to = logsnr_to_alpha_sigma(logsnr_to)
    
    # v-prediction: v = alpha * eps - sigma * x0
    # Solve for x0: x0 = (alpha * z_t - sigma * v) / (alpha^2 + sigma^2)
    # But alpha^2 + sigma^2 = 1, so:
    x0_pred = alpha_from.view(-1,1,1,1) * z_t - sigma_from.view(-1,1,1,1) * v_pred
    
    # eps prediction: eps = (sigma * z_t + alpha * v) / (alpha^2 + sigma^2) = sigma * z_t + alpha * v
    eps_pred = sigma_from.view(-1,1,1,1) * z_t + alpha_from.view(-1,1,1,1) * v_pred
    
    # DDIM step (deterministic): z_to = alpha_to * x0_pred + sigma_to * eps_pred
    z_to = alpha_to.view(-1,1,1,1) * x0_pred + sigma_to.view(-1,1,1,1) * eps_pred
    
    return z_to

### metadata helpers!!!

def get_image_spans(resolution):
    """
    Helper to generate span metadata for a single 2D continuous latent (image).
    """
    # PATCH FIX: Adjust for model stride
    latent_res = resolution // 2
    length = latent_res * latent_res
    # Standard format: 1 span, not causal (bidirectional), with 2D shape
    # e.g. 72px^2 image of 12x6px-> 6x3 flat embedding of len 18.
    # [{'len': 18, 'shape': (6, 3), 'causal': False}]
    # e.g. 3 separate 480px^2 images of 20x24px, 16x30px, 30x16px
    # -> 10x12, 8x15, 15x8 embeddings of len 120, 120, 120.
    # [{'len': 120, 'shape': (10, 12), 'causal': False},
    #{'len': 120, 'shape': (8, 15), 'causal': False},
    #{'len': 120, 'shape': (15, 8), 'causal': False}]
    return [{'type': 'latent', 'len': length, 'shape': (latent_res, latent_res), 'causal': False}]

### plotting and historiography helpers!!

def visualize_dataset_samples(iterator, resolutions, samples_per_res=8):
    """
    Generate samples from the composite iterator and label them.
    """
    fig, axes = plt.subplots(len(resolutions), samples_per_res, 
                            figsize=(samples_per_res * 1.5, len(resolutions) * 1.8))
    
    # Handle single resolution case (axes is 1D)
    if len(resolutions) == 1: 
        axes = axes.reshape(1, -1)
    
    for row_idx, res in enumerate(resolutions):
        # Generate batch
        samples = iterator.generate_batch(samples_per_res, res, num_tiles=4.0)
        labels = iterator.last_labels.cpu().numpy()
        samples = samples.cpu()
        
        for col_idx in range(samples_per_res):
            ax = axes[row_idx, col_idx]
            ax.imshow(samples[col_idx].permute(1, 2, 0).clamp(0, 1))
            ax.axis('off')
            
            # Get label name
            lbl_idx = labels[col_idx]
            lbl_name = iterator.label_map.get(lbl_idx, "Unknown")
            
            if col_idx == 0:
                ax.set_title(f"{res}px\n{lbl_name}", fontsize=8, loc='left')
            else:
                ax.set_title(f"{lbl_name}", fontsize=7)
                
    plt.suptitle("Composite Dataset Samples", fontsize=14)
    plt.tight_layout()
    return fig

def plot_detailed_loss(df_naive, df_fact, logger):
    """
    Generates a grid dynamically based on resolutions found in history.
    Rows: Resolution
    Cols: Dataset Type
    """
    df_naive = df_naive.interpolate()
    df_fact = df_fact.interpolate()
    
    resolutions = sorted(df_naive['res'].unique())
    datasets = ['checkerboard', 'torus']
    
    fig, axes = plt.subplots(len(resolutions), len(datasets), 
                            figsize=(12, 4 * len(resolutions)))
    
    # Handle single resolution case (axes is 1D)
    if len(resolutions) == 1: 
        axes = axes.reshape(1, -1)
        
    for r_idx, res in enumerate(resolutions):
        n_res = df_naive[df_naive['res'] == res]
        f_res = df_fact[df_fact['res'] == res]
        
        for d_idx, dtype in enumerate(datasets):
            ax = axes[r_idx, d_idx]
            col_name = f'loss_{dtype}'
            
            roll_win = 20
            
            if col_name in n_res.columns:
                line_n = n_res[col_name].rolling(roll_win).mean()
                ax.plot(n_res['step'], line_n, label='Naive', color='tab:blue', alpha=0.8)
                
            if col_name in f_res.columns:
                line_f = f_res[col_name].rolling(roll_win).mean()
                ax.plot(f_res['step'], line_f, label='Factorized', color='tab:orange', alpha=0.8)
            
            ax.set_title(f"{dtype.capitalize()} @ {res}px")
            ax.set_yscale('log')
            ax.grid(True, which='both', alpha=0.2)
            if r_idx == 0 and d_idx == 0:
                ax.legend()
    
    plt.tight_layout()
    logger.save_figure(fig, "loss_breakdown_res_vs_type")

def plot_sample_grid(samples_list, logger, string="final_samples"):
    """
    Dynamically plots a list of sample batches.
    Args:
        samples_list: List of tuples (Title, TensorBatch)
    """
    num_rows = len(samples_list)
    cols = 8 # Fixed sample count per batch
    
    fig, axes = plt.subplots(num_rows, cols, figsize=(cols * 2, num_rows * 2))
    
    # Handle single row case
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    for r, (name, batch) in enumerate(samples_list):
        for c in range(cols):
            if c < batch.shape[0]:
                axes[r, c].imshow(batch[c].permute(1,2,0).cpu().numpy())
            axes[r, c].axis('off')
            if c == 0: 
                axes[r, c].set_title(name, fontsize=10, loc='left')
            
    plt.suptitle("Unconditional Generation (Mixed Distribution)", fontsize=16)
    plt.tight_layout()
    logger.save_figure(fig, string)

def plot_distillation_loss(df_naive, df_fact, logger):
    """Plot consistency loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Consistency loss
    ax = axes[0]
    ax.plot(df_naive['step'], df_naive['loss_consistency'].rolling(20).mean(), 
            label='Naive', color='tab:blue')
    ax.plot(df_fact['step'], df_fact['loss_consistency'].rolling(20).mean(), 
            label='Factorized', color='tab:orange')
    ax.set_title("Trajectory Consistency Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Denoising loss (for reference)
    ax = axes[1]
    ax.plot(df_naive['step'], df_naive['loss_denoise'].rolling(20).mean(), 
            label='Naive', color='tab:blue', alpha=0.7)
    ax.plot(df_fact['step'], df_fact['loss_denoise'].rolling(20).mean(), 
            label='Factorized', color='tab:orange', alpha=0.7)
    ax.set_title("Denoising Loss (Auxiliary)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    logger.save_figure(fig, "distillation_loss")

def plot_comparison_grid(samples_before, samples_after, resolutions):
    """
    Create a before/after comparison grid.
    
    Layout:
    - Rows: [Naive 16px Before, Naive 16px After, Fact 16px Before, Fact 16px After, ...]
    - Cols: 8 samples
    """
    num_rows = len(samples_before) + len(samples_after)  # Interleaved
    cols = 8
    
    fig, axes = plt.subplots(num_rows, cols, figsize=(cols * 2, num_rows * 1.5))
    
    row_idx = 0
    for i, (name_before, batch_before) in enumerate(samples_before):
        name_after, batch_after = samples_after[i]
        
        # Plot "Before" row
        for c in range(cols):
            if c < batch_before.shape[0]:
                axes[row_idx, c].imshow(batch_before[c].permute(1,2,0).cpu().numpy())
            axes[row_idx, c].axis('off')
            if c == 0:
                axes[row_idx, c].set_title(f"{name_before}\n(Before)", 
                                          fontsize=9, loc='left')
        row_idx += 1
        
        # Plot "After" row
        for c in range(cols):
            if c < batch_after.shape[0]:
                axes[row_idx, c].imshow(batch_after[c].permute(1,2,0).cpu().numpy())
            axes[row_idx, c].axis('off')
            if c == 0:
                axes[row_idx, c].set_title(f"{name_after}\n(After)", 
                                          fontsize=9, loc='left', color='green')
        row_idx += 1
    
    plt.suptitle("Trajectory Consistency Distillation: Before vs After", fontsize=16)
    plt.tight_layout()
    return fig

### crunchy stuff!!!

@torch.no_grad()
def sample_viz(components, res, num_samples=8):
    model, span_emb, span_unemb, kvt, pt = components
    model.eval()
    
    z = torch.randn(num_samples, 3, res, res, device='cuda')
    ts = torch.linspace(1.0, 0.001, 50, device='cuda')
    base_spans = get_image_spans(res)

    for i in range(49):
        t = ts[i]; t_n = ts[i+1]
        logsnr = get_schedule(torch.full((num_samples,), t, device='cuda'))
        
        # Forward
        z_out_flat, _, span_objs = run_forward_step(
            model, span_emb, kvt, pt, z, logsnr, base_spans
        )
        
        outputs = span_unemb.decode(z_out_flat, span_objs)
        v_pred_raw = torch.stack(outputs['image_vpreds'])
        l_pred = torch.stack(outputs['image_logsnrs']).mean(dim=[2,3])
        
        # Sampler Logic
        # (Assuming 'factorized' check can be inferred or passed. Defaulting to raw for viz)
        # Hack: Check if model wrapper has mode? We passed components tuple.
        # Let's assume naive for viz unless we stored mode.
        v_pred = v_pred_raw # Simplify for viz
        
        logsnr_n = get_schedule(torch.full((num_samples,), t_n, device='cuda'))
        alpha, sigma = get_alpha_sigma(logsnr)
        alpha_n, sigma_n = get_alpha_sigma(logsnr_n)
        
        x0 = alpha.view(-1,1,1,1)*z - sigma.view(-1,1,1,1)*v_pred
        eps = sigma.view(-1,1,1,1)*z + alpha.view(-1,1,1,1)*v_pred
        z = alpha_n.view(-1,1,1,1)*x0 + sigma_n.view(-1,1,1,1)*eps
        
    model.train()
    return z.cpu().clamp(0, 1)

def warmup_model(model, buckets):
    print("🔥 Warming up compilation cache...")
    # 1. Warmup Training Graph
    model.train()
    for res, bs in buckets:
        print(f"   ...compiling train graph for {res}px")
        # Create dummy inputs
        z = torch.randn(bs, 3, res, res, device='cuda')
        t = torch.rand(bs, device='cuda')
        logsnr = get_schedule(t)
        spans = get_image_spans(res)
        
        # Run one step (Forward + Backward)
        opt = torch.optim.AdamW(model.parameters())
        opt.zero_grad()
        out, _, _ = model(z, logsnr, spans)
        loss = out.mean()
        loss.backward()
        opt.step()
        opt.zero_grad() # cleanup

    # 2. Warmup Inference Graph
    model.eval()
    with torch.no_grad():
        for res, _ in buckets: # BS doesn't strictly matter for shape generalization usually
            print(f"   ...compiling inference graph for {res}px")
            z = torch.randn(2, 3, res, res, device='cuda') # Small batch
            logsnr = get_schedule(torch.rand(2, device='cuda'))
            spans = get_image_spans(res)
            model(z, logsnr, spans)
    model.train()
    print("✅ Warmup complete. No more stalls expected.")

def compute_consistency_loss(components, x0, spans, mode='factorized', min_logsnr=-5.0, max_logsnr=5.0):
    """
    Consolidated Consistency Loss.
    Optimizes 2 objectives:
    1. Trajectory Consistency: 1-Step (Coarse) should match 2-Step (Fine).
    2. Drift Correction: Generated states should yield predictions matching Ground Truth states.
    """
    B = x0.shape[0]
    device = x0.device

    # 1. Setup Trajectory Points
    l_start, l_mid, l_end = sample_logsnr_triplet(B, device, min_logsnr, max_logsnr)
    
    # Create Starting State (Noisy)
    a_start, s_start = logsnr_to_alpha_sigma(l_start)
    z_start = x0 * a_start.view(-1,1,1,1) + torch.randn_like(x0) * s_start.view(-1,1,1,1)
    
    # === A. FINE TRAJECTORY (The Target) ===
    # Step 1: Start -> Mid
    # We differentiate through v_start to optimize the start of the chain
    v_start, aux_start = predict_velocity_field(components, z_start, l_start, spans, mode)
    z_mid_gen = euler_reverse_step(z_start, v_start, l_start, l_mid)
    
    # Step 2: Mid -> End
    # We detach z_mid_gen for the *prediction* calculation to prevent 
    # the target from moving to meet the student.
    v_mid_gen, aux_mid = predict_velocity_field(components, z_mid_gen.detach(), l_mid, spans, mode)
    z_end_fine = euler_reverse_step(z_mid_gen, v_mid_gen, l_mid, l_end)
    
    # === B. COARSE TRAJECTORY (The Student) ===
    # Start -> End (Directly using v_start)
    z_end_coarse = euler_reverse_step(z_start, v_start, l_start, l_end)
    
    # LOSS 1: Trajectory Consistency
    # "Fine" is the ground truth approximation. "Coarse" tries to match it.
    loss_traj = F.mse_loss(z_end_coarse, z_end_fine.detach())
    
    # === C. DRIFT CORRECTION ===
    # Ensure the model's prediction at the *generated* mid-point (z_mid_gen)
    # is consistent with what it would predict at a *true* mid-point (z_mid_real).
    
    # Create Ground Truth Mid State
    a_mid, s_mid = logsnr_to_alpha_sigma(l_mid)
    z_mid_real = x0 * a_mid.view(-1,1,1,1) + torch.randn_like(x0) * s_mid.view(-1,1,1,1)
    
    # Get Teacher Prediction (at real state)
    with torch.no_grad():
        v_mid_real, _ = predict_velocity_field(components, z_mid_real, l_mid, spans, mode)
    
    # LOSS 2: Drift / Denoising Consistency
    # v_mid_gen (from A) should match v_mid_real
    loss_drift = F.mse_loss(v_mid_gen, v_mid_real)
    
    return loss_traj + loss_drift, aux_start + aux_mid


def distill_multires(components, mode, buckets, steps=1000, logger=None):
    print(f"\n--- Distilling: {mode.upper()} ---")
    model = components[0]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    
    # Halve batch size for memory safety (student + teacher forwards)
    buckets_distill = [(res, max(1, bs // 2)) for res, bs in buckets]
    
    iterator = CompositeIterator(model.text_embed.weight.device, config={'checkerboard': 0.5, 'torus': 0.5})
    manager = BucketManager(buckets_distill)
    history = []
    scaler = torch.amp.GradScaler('cuda')
    
    pbar = tqdm(range(steps), desc=f"distill-{mode}")
    for i in pbar:
        opt.zero_grad()
        res, bs = manager.next_bucket()
        
        x0 = iterator.generate_batch(bs, res, num_tiles=4.0)
        spans = get_image_spans(res)
        
        with torch.amp.autocast('cuda'):
            # 1. Consistency Loss
            loss_c = compute_consistency_loss(
                components, x0, spans, mode=mode,
                min_logsnr=-4.0, max_logsnr=4.0
            )
            
            # 2. Denoising Regularization (Prevent drift)
            t = torch.rand(bs, device=x0.device).clamp(0.001, 0.999)
            l_den = get_schedule(t)
            a, s = logsnr_to_alpha_sigma(l_den)
            eps = torch.randn_like(x0)
            z_t = x0 * a.view(-1,1,1,1) + eps * s.view(-1,1,1,1)
            v_t = a.view(-1,1,1,1)*eps - s.view(-1,1,1,1)*x0
            
            v_pred, aux_loss = predict_velocity_field(components, z_t, l_den, spans, mode)
            
            loss_d = F.mse_loss(v_pred, v_t)
            loss = loss_c + 0.1 * loss_d + aux_loss
            
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        
        history.append({'step': i, 'res': res, 'loss_total': loss.item(), 'loss_cons': loss_c.item()})
        if i % 50 == 0:
            pbar.set_postfix({'cons': f'{loss_c.item():.4f}', 'den': f'{loss_d.item():.4f}'})
            
    return pd.DataFrame(history)

def train_multires(components, mode, buckets, steps=1000, logger=None):
    print(f"\n--- Training: {mode.upper()} ---")
    model = components[0]
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    
    iterator = CompositeIterator(model.text_embed.weight.device, config={'checkerboard': 0.5, 'torus': 0.5})
    manager = BucketManager(buckets)
    history = []
    
    pbar = tqdm(range(steps), desc=f"{mode}")
    for i in pbar:
        opt.zero_grad()
        res, bs = manager.next_bucket()
        
        # Data
        x0 = iterator.generate_batch(bs, res, num_tiles=4.0)
        t = torch.rand(bs, device=x0.device).clamp(0.001, 0.999)
        logsnr = get_schedule(t)
        alpha, sigma = get_alpha_sigma(logsnr)
        
        eps = torch.randn_like(x0)
        z_t = x0 * alpha.view(-1,1,1,1) + eps * sigma.view(-1,1,1,1)
        v_true = alpha.view(-1,1,1,1) * eps - sigma.view(-1,1,1,1) * x0

        base_spans = get_image_spans(res)
        
        # === Unified Prediction Call ===
        v_pred, aux_loss = predict_velocity_field(components, z_t, logsnr, base_spans, mode)
            
        loss_elem = F.mse_loss(v_pred, v_true, reduction='none').mean(dim=[1,2,3])
        total_loss = loss_elem.mean() + aux_loss
        
        total_loss.backward()
        opt.step()
        
        history.append({'step': i, 'res': res, 'loss_total': total_loss.item()})
        if i % 100 == 0:
            pbar.set_postfix({'loss': f'{total_loss.item():.4f}', 'res': res})
            
    return pd.DataFrame(history)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    logger = ExperimentLogger(output_dir="./experiments_mix")
    device = torch.device('cuda')

    BUCKETS = [(16, 128), (32, 64)]
    STEPS = 500
    
    # 1. Singleton Stack
    print("🔧 Initializing Singleton Model Stack...")
    embed_dim = 256; depth = 4
    
    model = coolerLDTformer(dim=embed_dim, depth=depth, num_heads=8, topo_dim=3).to(device)
    span_emb = SpanEmbedder(model.text_embed, model.patch_embedder)
    span_unemb = SpanUnembedder(model.text_head, model.patch_unembedder)
    
    kvt_manager = KVTManager(max_blocks=2000, block_size=128, kv_dim=embed_dim, layers=depth, heads=8, topo_dim=3, device=device)
    page_table = PageTable(num_blocks=2000, block_size=128, max_batch_size=256, max_logical_blocks=256, device=device)
    
    components = (model, span_emb, span_unemb, kvt_manager, page_table)
    
    # 2. Run A (Naive)
    print("🚀 Run A: Naive")
    model.param_init() # Fresh Init
    df_n = train_multires(components, 'naive', BUCKETS, STEPS, logger)
    params_naive = model.dump() 
    
    # 3. Run B (Factorized)
    print("🚀 Run B: Factorized")
    model.flush() # Zero
    model.param_init() # Re-Init
    df_f = train_multires(components, 'factorized', BUCKETS, STEPS, logger)
    params_fact = model.dump()
    
    print("✅ Done.")