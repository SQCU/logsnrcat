# bench_multires_cl_zc.py
import os
import sys
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from torch.optim.lr_scheduler import OneCycleLR

#get_schedule,
from diffusion_utils import get_alpha_sigma, BucketManager
from dataset import CompositeIterator

# NOTE: Importing the ZC (Zero-Cache) variant for training
from ld_tformer import coolerLDTformerZC as coolerLDTformer
from ld_tformer import SpanEmbedder, SpanUnembedder, build_dual_masks
from ld_tformer_embedding_functional import render_topology_embeddings
from memory_manager import KVTManager, PageTable
from kv_cache_allocator import allocate_kv_cache_safely
# KVTManager and Allocator are no longer needed for ZC training

# Hook the ZMQ compiler backend immediately if available
try:
    import inductor_cas_client
    inductor_cas_client.install_cas_client()
except ImportError:
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

### Diffusion Helpers

def get_schedule(t, schedule_bounds: tuple = (5,-1)):
    return schedule_bounds[0]-t*(schedule_bounds[1]-schedule_bounds[0])
    #return 20.0 - 40.0 * t

def predict_velocity_field(components, z, logsnr, spans, mode):
    """
    Returns:
        v_final: [B, C, H, W] - Velocity field
        aux_loss: Scalar
        cleanup: Callable - No-op for ZC mode
    """
    model, _, span_unemb, _, _ = components
    
    # 1. Forward Pass
    suppress = (mode == 'naive')
    z_flat, aux_loss, objs, _ = run_forward_step(
        components, z, logsnr, spans, 
        suppress_logsnr_input=suppress
    )
    
    # 2. Unembed
    decoded = span_unemb.decode(z_flat, objs)
    
    # 3. Stack & Parse
    v_raw = torch.stack([d['image_vpreds'] for d in decoded])
    
    if mode == 'factorized':
        l_maps = torch.stack([d['image_logsnrs'] for d in decoded])
        sigma_p = torch.sqrt(torch.sigmoid(-l_maps))
        v_final = v_raw * sigma_p
    else:
        v_final = v_raw
    
    # 4. Cleanup (No-op for ZC/Cacheless)
    return v_final, aux_loss, lambda: None

def compute_nll_loss(v_pred_mean, v_pred_logvar, v_target):
    """
    Computes Heteroscedastic Gaussian NLL.
    
    Args:
        v_pred_mean: The predicted velocity vector.
        v_pred_logvar: Predicted log-variance (uncertainty).
        v_target: The ground truth velocity.
    """
    # 1. The Precision Term (MSE weighted by uncertainty)
    # If logvar is high (uncertain), this term shrinks.
    # We use exp(-logvar) for numerical stability.
    precision = torch.exp(-v_pred_logvar)
    mse = (v_pred_mean - v_target) ** 2
    loss_precision = 0.5 * precision * mse
    
    # 2. The Entropy Term (Penalty for being uncertain)
    # This prevents the model from predicting infinite variance to cheat.
    loss_uncertainty = 0.5 * v_pred_logvar
    
    # Total Loss per pixel
    loss = loss_precision + loss_uncertainty
    
    return loss.mean()

def predict_probabilistic_field(components, z, logsnr, spans):
    model, _, span_unemb, _, _ = components
    
    # Run Forward
    z_flat, aux_loss, objs, _ = run_forward_step(components, z, logsnr, spans)
    decoded = span_unemb.decode(z_flat, objs)
    
    # Unpack Prediction
    v_mean = torch.stack([d['image_vpreds'] for d in decoded])
    
    # Interpret the 'logsnr' head as the Model's Uncertainty regarding v
    # High predicted LogSNR = Low Variance (Confident)
    # Low predicted LogSNR = High Variance (Uncertain)
    pred_logsnr = torch.stack([d['image_logsnrs'] for d in decoded])
    
    # Variance = 1 / SNR, so LogVar = -LogSNR
    # We allow the model to learn a shift/scale on this if needed
    v_logvar = -pred_logsnr 
    
    return v_mean, v_logvar, aux_loss

def run_forward_step_kvc(
    components, 
    z, 
    logsnr, 
    base_spans,
    suppress_logsnr_input: bool = False
):
    """
    Returns
        z_out: [L_total, D]
        aux_loss: Scalar
        span_objects: List[Span]
        req_ids: List[int] - Allocated request IDs (caller must free!)
    """
    model, span_embedder, _, kvt_manager, page_table = components
    B, C, H, W = z.shape
    device = z.device
    num_layers = len(model.layers)
    
    # 1. Prepare Metadata & Inputs
    batch_spans_meta = []
    images = [z[i] for i in range(B)]
    
    if suppress_logsnr_input:
        zero_map = torch.zeros((1, H, W), device=device)
        logsnr_maps = [zero_map] * B
    else:
        # ✅ FIX: Handle both scalar and spatial logsnr
        if logsnr.dim() == 1:  # Scalar per sample: [B]
            # Broadcast to spatial map: [B] -> [B, 1, H, W]
            logsnr_spatial = logsnr.view(B, 1, 1, 1).expand(B, 1, H, W)
            logsnr_maps = [logsnr_spatial[i] for i in range(B)]
        elif logsnr.dim() == 4:  # Already spatial: [B, 1, H, W]
            logsnr_maps = [logsnr[i] for i in range(B)]
        else:
            raise ValueError(f"Invalid logsnr shape: {logsnr.shape}")

    for i in range(B):
        item_spans = [s.copy() for s in base_spans]
        for s in item_spans: s['id'] = f"req_{i}" 
        batch_spans_meta.extend(item_spans)

    # 2. Embed
    z_flat, span_objects, content_hashes = span_embedder.embed(
        batch_spans_meta,
        text_tokens=[None] * B,
        images=images,
        logsnr_maps=logsnr_maps
    )
    
    # 3. Render Topology
    from ld_tformer_embedding_functional import render_topology_embeddings
    topo_embeds, _ = render_topology_embeddings(batch_spans_meta, 3, device)
    
    # 4. Allocate KV Cache
    req_ids = list(range(B))
    tokens_per_req = sum(s['len'] for s in base_spans)
    
    cursor = 0
    for rid in req_ids:
        chunk_hashes = content_hashes[cursor : cursor + tokens_per_req]
        chunk_topo = topo_embeds[cursor : cursor + tokens_per_req]
        kvt_manager.allocate_and_write_sequence(rid, chunk_hashes, chunk_topo)
        cursor += tokens_per_req

    # 5. Get Physical Mappings
    flat_page_table, inverse_page_table = kvt_manager.get_flat_page_mapping(req_ids)
    
    # 6. Get Slot Mapping
    block_tables = [kvt_manager.req_tables[rid] for rid in req_ids]
    seq_lengths = [kvt_manager.req_lengths[rid] for rid in req_ids]
    slot_mapping = kvt_manager.get_slot_mapping(block_tables, seq_lengths)

    # 7. Build Dual Masks (Local + Global)
    # Returns tuple (local_mask, global_mask)
    # Note: Using topo_embeds as topo_heap for ZC mode (Identity Heap)
    block_masks = build_dual_masks(
        span_objects,
        topo_active=topo_embeds,
        topo_heap=topo_embeds,
        page_table=page_table,
        flat_page_table=flat_page_table,
        inverse_page_table=None
    )
    
    # 8. Forward Pass
    # CHANGED: Retrieve flattened views [1, H, Capacity, D] from manager
    k_views = []
    v_views = []
    for i in range(num_layers):
        k, v = kvt_manager.get_flat_kv_view(i)
        k_views.append(k)
        v_views.append(v)

    base_ref_len = 64.0 
    L_total = z_flat.shape[0]
    rope_scale = max(1.0, L_total / base_ref_len)

    z_out, aux_loss = model(
        z_flat.unsqueeze(0),
        topo_embeds.unsqueeze(0),
        k_caches=k_views,
        v_caches=v_views,
        slot_mapping=slot_mapping,
        block_mask=block_masks,
        scale=rope_scale
    )
    
    return z_out.squeeze(0), aux_loss, span_objects, req_ids

def run_forward_step(
    components, 
    z, 
    logsnr, 
    base_spans,
    suppress_logsnr_input: bool = False
):
    """
    Zero-Cache Forward Step (ZC Mode).
    
    CRITICAL CHANGE: This generates a mask for the EPHEMERAL batch tensor,
    not the persistent KVT heap.
    
    1. Embeds Spans -> Flat Tensor (Contiguous)
    2. Generates Identity Topology (Logical Block i -> Physical Block i)
    3. Runs Model with mask sized to L_active, not Capacity.
    """
    model, span_embedder, _, kvt_manager, page_table = components
    B, C, H, W = z.shape
    device = z.device
    
    # 1. Prepare Metadata & Inputs
    batch_spans_meta = []
    images = [z[i] for i in range(B)]
    
    if suppress_logsnr_input:
        zero_map = torch.zeros((1, H, W), device=device)
        logsnr_maps = [zero_map] * B
    else:
        if logsnr.dim() == 1:
            logsnr_spatial = logsnr.view(B, 1, 1, 1).expand(B, 1, H, W)
            logsnr_maps = [logsnr_spatial[i] for i in range(B)]
        elif logsnr.dim() == 4:
            logsnr_maps = [logsnr[i] for i in range(B)]
        else:
            raise ValueError(f"Invalid logsnr shape: {logsnr.shape}")

    for i in range(B):
        item_spans = [s.copy() for s in base_spans]
        for s in item_spans: s['id'] = f"req_{i}" 
        batch_spans_meta.extend(item_spans)

    # 2. Embed -> z_flat [Total_L, D]
    z_flat, span_objects, _ = span_embedder.embed(
        batch_spans_meta,
        text_tokens=[None] * B,
        images=images,
        logsnr_maps=logsnr_maps
    )
    
    # 3. Render Topology
    # topo_embeds: [Total_L, Topo_Dim]
    topo_embeds, _ = render_topology_embeddings(batch_spans_meta, 3, device)
    
    # 4. Identity Mapping (The "Virtual" Page Table)
    # The tensor is contiguous. Logical Block i is Physical Block i.
    L_total = z_flat.shape[0]
    block_size = page_table.block_size
    num_blocks = (L_total + block_size - 1) // block_size
    
    # Identity mapping: [0, 1, 2, ... num_blocks-1]
    flat_page_table = torch.arange(num_blocks, device=device, dtype=torch.long)
    
    # 5. Build Dual Masks (Local + Global)
    # Returns tuple (local_mask, global_mask)
    # Note: Using topo_embeds as topo_heap for ZC mode (Identity Heap)
    block_masks = build_dual_masks(
        span_objects,
        topo_active=topo_embeds,
        topo_heap=topo_embeds,
        page_table=page_table,
        flat_page_table=flat_page_table,
        inverse_page_table=None
    )
    
    base_ref_len = 64.0 
    L_total = z_flat.shape[0]
    rope_scale = max(1.0, L_total / base_ref_len)

    # 6. Forward Pass
    # coolerLDTformerZC now expects 'block_masks' (tuple), not 'block_mask'
    z_out, aux_loss = model(
        z_flat.unsqueeze(0),
        topo_embeds.unsqueeze(0),
        slot_mapping=None,
        block_masks=block_masks,  # <--- CHANGED ARGUMENT
        scale=rope_scale
    )
    
    # No request IDs to cleanup in ZC mode
    return z_out.squeeze(0), aux_loss, span_objects, []

def logsnr_to_alpha_sigma(logsnr):
    alpha = torch.sqrt(torch.sigmoid(logsnr))
    sigma = torch.sqrt(torch.sigmoid(-logsnr))
    return alpha, sigma

def sample_logsnr_triplet(batch_size, device, min_logsnr=-10.0, max_logsnr=0.0, min_gap=1.0):
    logsnr_low = torch.rand(batch_size, device=device) * (max_logsnr - min_logsnr) + min_logsnr
    gap_mid = torch.rand(batch_size, device=device) * 3.0 + min_gap
    logsnr_mid = (logsnr_low - gap_mid).clamp(min=min_logsnr)
    gap_high = torch.rand(batch_size, device=device) * 3.0 + min_gap
    logsnr_high = (logsnr_mid - gap_high).clamp(min=min_logsnr)
    return logsnr_low, logsnr_mid, logsnr_high

def euler_reverse_step(z_t, v_pred, logsnr_from, logsnr_to):
    alpha_from, sigma_from = logsnr_to_alpha_sigma(logsnr_from)
    alpha_to, sigma_to = logsnr_to_alpha_sigma(logsnr_to)
    x0_pred = alpha_from.view(-1,1,1,1) * z_t - sigma_from.view(-1,1,1,1) * v_pred
    eps_pred = sigma_from.view(-1,1,1,1) * z_t + alpha_from.view(-1,1,1,1) * v_pred
    z_to = alpha_to.view(-1,1,1,1) * x0_pred + sigma_to.view(-1,1,1,1) * eps_pred
    return z_to

def get_image_spans(resolution):
    latent_res = resolution // 2
    length = latent_res * latent_res
    return [{'type': 'latent', 'len': length, 'shape': (latent_res, latent_res), 'causal': False}]

### Plotting Helpers

def plot_detailed_loss(df_naive, df_fact, logger):
    df_naive = df_naive.interpolate()
    df_fact = df_fact.interpolate()
    resolutions = sorted(df_naive['res'].unique())
    datasets = ['checkerboard', 'torus']
    
    fig, axes = plt.subplots(len(resolutions), len(datasets), 
                            figsize=(12, 4 * len(resolutions)))
    if len(resolutions) == 1: axes = axes.reshape(1, -1)
        
    for r_idx, res in enumerate(resolutions):
        n_res = df_naive[df_naive['res'] == res]
        f_res = df_fact[df_fact['res'] == res]
        for d_idx, dtype in enumerate(datasets):
            ax = axes[r_idx, d_idx]
            col_name = f'loss_{dtype}'
            if col_name in n_res.columns:
                ax.plot(n_res['step'], n_res[col_name].rolling(20).mean(), label='Naive')
            if col_name in f_res.columns:
                ax.plot(f_res['step'], f_res[col_name].rolling(20).mean(), label='Factorized')
            ax.set_title(f"{dtype.capitalize()} @ {res}px")
            ax.set_yscale('log')
            if r_idx == 0 and d_idx == 0: ax.legend()
    plt.tight_layout()
    logger.save_figure(fig, "loss_breakdown_res_vs_type")

def plot_sample_grid(samples_list, logger, string="final_samples"):
    num_rows = len(samples_list); cols = 8
    fig, axes = plt.subplots(num_rows, cols, figsize=(cols * 2, num_rows * 2))
    if num_rows == 1: axes = axes.reshape(1, -1)
    
    for r, (name, batch) in enumerate(samples_list):
        for c in range(cols):
            if c < batch.shape[0]:
                axes[r, c].imshow(batch[c].permute(1,2,0).cpu().numpy())
            axes[r, c].axis('off')
            if c == 0: axes[r, c].set_title(name, fontsize=10, loc='left')
    plt.tight_layout()
    logger.save_figure(fig, string)

def plot_distillation_loss(df_naive, df_fact, logger):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.plot(df_naive['step'], df_naive['loss_consistency'].rolling(20).mean(), label='Naive')
    ax.plot(df_fact['step'], df_fact['loss_consistency'].rolling(20).mean(), label='Factorized')
    ax.set_title("Trajectory Consistency Loss"); ax.set_yscale('log'); ax.legend()
    
    ax = axes[1]
    ax.plot(df_naive['step'], df_naive['loss_denoise'].rolling(20).mean(), label='Naive')
    ax.plot(df_fact['step'], df_fact['loss_denoise'].rolling(20).mean(), label='Factorized')
    ax.set_title("Denoising Loss (Auxiliary)"); ax.set_yscale('log'); ax.legend()
    plt.tight_layout()
    logger.save_figure(fig, "distillation_loss")

def plot_comparison_grid(samples_before, samples_after, resolutions):
    num_rows = len(samples_before) + len(samples_after)
    cols = 8
    fig, axes = plt.subplots(num_rows, cols, figsize=(cols * 2, num_rows * 1.5))
    row_idx = 0
    for i, (name_before, batch_before) in enumerate(samples_before):
        name_after, batch_after = samples_after[i]
        for c in range(cols):
            if c < batch_before.shape[0]:
                axes[row_idx, c].imshow(batch_before[c].permute(1,2,0).cpu().numpy())
            axes[row_idx, c].axis('off')
            if c == 0: axes[row_idx, c].set_title(f"{name_before}\n(Before)", fontsize=9, loc='left')
        row_idx += 1
        for c in range(cols):
            if c < batch_after.shape[0]:
                axes[row_idx, c].imshow(batch_after[c].permute(1,2,0).cpu().numpy())
            axes[row_idx, c].axis('off')
            if c == 0: axes[row_idx, c].set_title(f"{name_after}\n(After)", fontsize=9, loc='left', color='green')
        row_idx += 1
    plt.tight_layout()
    return fig


def plot_three_way_loss(df_naive, df_fact, df_nll, logger, string="three_way_loss"):
    """
    Plots Naive vs Factorized vs NLL training curves.
    Note: NLL loss scale is different (can be negative), so we plot it on secondary axis or separate subplot.
    """
    df_naive = df_naive.interpolate()
    df_fact = df_fact.interpolate()
    df_nll = df_nll.interpolate()
    
    resolutions = sorted(df_naive['res'].unique())
    
    fig, axes = plt.subplots(len(resolutions), 1, figsize=(10, 4 * len(resolutions)))
    if len(resolutions) == 1: axes = [axes]
    
    for r_idx, res in enumerate(resolutions):
        ax1 = axes[r_idx]
        ax2 = ax1.twinx() # Second axis for NLL
        
        n_res = df_naive[df_naive['res'] == res]
        f_res = df_fact[df_fact['res'] == res]
        nll_res = df_nll[df_nll['res'] == res]
        
        # Plot MSE models on Left Axis
        if not n_res.empty:
            ax1.plot(n_res['step'], n_res['loss_total'].rolling(20).mean(), label='Naive (MSE)', color='tab:blue')
        if not f_res.empty:
            ax1.plot(f_res['step'], f_res['loss_total'].rolling(20).mean(), label='Fact (MSE)', color='tab:orange')
            
        # Plot NLL model on Right Axis
        if not nll_res.empty:
            ax2.plot(nll_res['step'], nll_res['loss_total'].rolling(20).mean(), label='NLL (LogProb)', color='tab:green', linestyle='--')
            
        ax1.set_title(f"Training Loss @ {res}px")
        ax1.set_ylabel("MSE Loss")
        ax2.set_ylabel("NLL Loss")
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
    plt.tight_layout()
    logger.save_figure(fig, string)

### Sampling & Training

@torch.no_grad()
def sample_viz(components, res, num_samples=8, mode='naive'):
    model, span_emb, span_unemb, _, _ = components
    model.eval()
    
    z = torch.randn(num_samples, 3, res, res, device='cuda')
    ts = torch.linspace(1.0, 0.001, 50, device='cuda')
    base_spans = get_image_spans(res)

    for i in range(49):
        t = ts[i]; t_n = ts[i+1]
        logsnr = get_schedule(torch.full((num_samples,), t, device='cuda'))
        v_pred, _, _ = predict_velocity_field(components, z, logsnr, base_spans, mode)
        
        logsnr_n = get_schedule(torch.full((num_samples,), t_n, device='cuda'))
        alpha, sigma = get_alpha_sigma(logsnr)
        alpha_n, sigma_n = get_alpha_sigma(logsnr_n)
        
        x0 = alpha.view(-1,1,1,1)*z - sigma.view(-1,1,1,1)*v_pred
        eps = sigma.view(-1,1,1,1)*z + alpha.view(-1,1,1,1)*v_pred
        z = alpha_n.view(-1,1,1,1)*x0 + sigma_n.view(-1,1,1,1)*eps
        
    model.train()
    return z.cpu().clamp(0, 1)

### Improved Sampling & Visualization Logic

@torch.no_grad()
def sample_viz_dset(components, iterator, config):
    """
    Higher power sampler:
    1. Fetches real dataset items (GT).
    2. Noises them to random levels between min_logsnr and max_logsnr.
    3. Reconstructs them using the model.
    4. Computes MSE against GT.
    """
    model, _, _, _, _ = components
    model.eval()
    
    # 1. Configuration
    res = config.get('res', 32)
    n_samples = config.get('num_samples', 8)
    mode = config.get('mode', 'naive')
    
    # User requested uniform sampling between -1 and -14
    # -4 is very noisy, +1 is somewhat noisy (partial data)
    min_snr = config.get('min_logsnr', -4.0) 
    max_snr = config.get('max_logsnr', 1.0)
    target_clean_snr = 10.0 # High value for "clean" final state
    
    device = model.text_embed.weight.device
    
    # 2. Get Ground Truth Data
    # We generate a batch. Since CompositeIterator mixes shapes, we get a mix.
    x0 = iterator.generate_batch(n_samples, res, num_tiles=4.0).to(device)
    
    # 3. Determine Start Times (Stratified)
    # We assign a random starting noise level to each item in the batch
    start_logsnrs = torch.rand(n_samples, device=device) * (max_snr - min_snr) + min_snr
    # Sort for cleaner visualization (Noisiest -> Cleanest)
    start_logsnrs, sort_idx = torch.sort(start_logsnrs)
    x0 = x0[sort_idx]
    
    # 4. Prepare Solver
    # We create a global time grid from the absolute noise floor (-15) to clean (+10)
    # We will "pick up" each sample when the solver passes its specific start_logsnr
    grid_steps = 64
    schedule = torch.linspace(-15.0, target_clean_snr, grid_steps, device=device)
    
    z = torch.zeros_like(x0)
    active_mask = torch.zeros(n_samples, dtype=torch.bool, device=device)
    vis_noisy_inputs = torch.zeros_like(x0) # For visualization
    
    base_spans = get_image_spans(res)
    
    # 5. Sampling Loop (Low SNR -> High SNR)
    for i in range(grid_steps - 1):
        t_curr = schedule[i]
        t_next = schedule[i+1]
        
        # A. Injection Phase:
        # Check which samples should "enter" the simulation now.
        # Condition: Sample wants to start at S, and t_curr >= S.
        should_start = (t_curr >= start_logsnrs) & (~active_mask)
        
        if should_start.any():
            # Generate the specific partial noise for these items
            # We use t_curr as the noise level so they align with the solver grid
            a, s = logsnr_to_alpha_sigma(torch.full((n_samples,), t_curr, device=device))
            eps = torch.randn_like(x0)
            z_injected = x0 * a.view(-1,1,1,1) + eps * s.view(-1,1,1,1)
            
            # Write into state
            z = torch.where(should_start.view(-1,1,1,1), z_injected, z)
            vis_noisy_inputs = torch.where(should_start.view(-1,1,1,1), z_injected, vis_noisy_inputs)
            active_mask = active_mask | should_start
            
        if not active_mask.any():
            continue

        # B. Prediction Phase
        # We pass t_curr for all, but masking handles the logic validity
        logsnr_map = torch.full((n_samples,), t_curr, device=device)
        
        v_pred, _, _ = predict_velocity_field(components, z, logsnr_map, base_spans, mode)
        
        # C. Step Phase
        z_next = euler_reverse_step(z, v_pred, t_curr, t_next)
        
        # Only update items that have actually started
        z = torch.where(active_mask.view(-1,1,1,1), z_next, z)
        
    model.train()
    
    # 6. Calc Metrics
    # Clamp for valid image range before MSE
    z_final = z.clamp(0, 1)
    mse_per_sample = F.mse_loss(z_final, x0, reduction='none').mean(dim=[1,2,3])
    
    return {
        'x0': x0,
        'noisy_input': vis_noisy_inputs,
        'reconstruction': z_final,
        'mse': mse_per_sample,
        'start_snr': start_logsnrs
    }

def plot_dset_reconstruction(result_dict, logger, name="reconstruction"):
    """
    Plots a grid: [Ground Truth] | [Noisy Start] | [Reconstruction]
    """
    x0 = result_dict['x0'].cpu()
    noisy = result_dict['noisy_input'].cpu()
    recon = result_dict['reconstruction'].cpu()
    mses = result_dict['mse'].cpu()
    snrs = result_dict['start_snr'].cpu()
    
    n = x0.shape[0]
    fig, axes = plt.subplots(n, 3, figsize=(10, 2 * n))
    
    # Handle single sample case
    if n == 1: axes = axes.reshape(1, -1)
    
    for i in range(n):
        # 1. GT
        axes[i, 0].imshow(x0[i].permute(1,2,0).numpy())
        axes[i, 0].axis('off')
        if i == 0: axes[i, 0].set_title("Ground Truth (x0)", fontsize=10)
        
        # 2. Noisy Input
        # Remap noisy range for visualization if needed, but raw is usually fine
        axes[i, 1].imshow(noisy[i].permute(1,2,0).clamp(0,1).numpy())
        axes[i, 1].axis('off')
        if i == 0: axes[i, 1].set_title("Input (Noised)", fontsize=10)
        
        # 3. Output
        axes[i, 2].imshow(recon[i].permute(1,2,0).numpy())
        axes[i, 2].axis('off')
        if i == 0: axes[i, 2].set_title("Reconstruction (Output)", fontsize=10)
        
        # Annotations
        snr_val = snrs[i].item()
        mse_val = mses[i].item()
        axes[i, 1].text(0, -2, f"LogSNR: {snr_val:.1f}", fontsize=8, color='blue')
        axes[i, 2].text(0, -2, f"MSE: {mse_val:.5f}", fontsize=8, color='red')
        
    plt.tight_layout()
    logger.save_figure(fig, name)

#why does this distillation loss always get written backwards then also called teacher-student distillation???
def compute_consistency_loss(components, x0, spans, mode='factorized', min_logsnr=-5.0, max_logsnr=5.0):
    B = x0.shape[0]
    device = x0.device
    l_start, l_mid, l_end = sample_logsnr_triplet(B, device, min_logsnr, max_logsnr)
    a_start, s_start = logsnr_to_alpha_sigma(l_start)
    z_start = x0 * a_start.view(-1,1,1,1) + torch.randn_like(x0) * s_start.view(-1,1,1,1)
    
    # 1. coarse (Start -> End)
    v_start_coarse, aux1, _ = predict_velocity_field(components, z_start, l_start, spans, mode)
    z_end_coarse = euler_reverse_step(z_start, v_start_coarse, l_start, l_end)
    
    # 2. fine (Start -> Mid -> End)
    z_mid_fine = euler_reverse_step(z_start, v_start_coarse, l_start, l_mid)
    v_mid_fine, aux2, _ = predict_velocity_field(components, z_mid_fine, l_mid, spans, mode)
    z_end_fine = euler_reverse_step(z_mid_fine, v_mid_fine, l_mid, l_end)
    
    loss = F.mse_loss(z_end_coarse, z_end_fine.detach())
    return loss, aux1 + aux2, lambda: None

def distill_multires(components, mode, buckets, steps=1000, logger=None):
    print(f"\n--- Distilling: {mode.upper()} ---")
    model = components[0]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)  #why were you a different value?
    scheduler_main = OneCycleLR(opt, max_lr=1e-4, total_steps=steps, 
                        pct_start=0.1, div_factor=10, final_div_factor=100)
    # Scaled down buckets for distillation memory
    buckets_distill = [(res, max(1, bs // 2)) for res, bs in buckets]
    
    iterator = CompositeIterator(model.text_embed.weight.device, config={'checkerboard': 0.5, 'torus': 0.5})
    manager = BucketManager(buckets_distill)
    history = []
    
    pbar = tqdm(range(steps), desc=f"distill-{mode}")
    for i in pbar:
        opt.zero_grad()
        res, bs = manager.next_bucket()
        x0 = iterator.generate_batch(bs, res, num_tiles=4.0)
        spans = get_image_spans(res)
        
        # 1. Consistency Loss
        loss_c, aux_loss_con, _ = compute_consistency_loss(components, x0, spans, mode=mode)
        
        # 2. Denoising Reg
        t = torch.rand(bs, device=x0.device).clamp(0.001, 0.999)
        l_den = get_schedule(t)
        a, s = logsnr_to_alpha_sigma(l_den)
        eps = torch.randn_like(x0)
        z_t = x0 * a.view(-1,1,1,1) + eps * s.view(-1,1,1,1)
        v_t = a.view(-1,1,1,1)*eps - s.view(-1,1,1,1)*x0
        
        v_pred, aux_loss, _ = predict_velocity_field(components, z_t, l_den, spans, mode)
        loss_d = F.mse_loss(v_pred, v_t)
        
        loss_t = (1.0+loss_c) * 1.0 * loss_d + aux_loss + aux_loss_con
        loss_t.backward()
        opt.step()
        scheduler_main.step()
        
        step_stats = {'step': i, 'res': res, 'loss_consistency': loss_c.item(), 
                      'loss_denoise': loss_d.item(), 'loss_total': loss_t.item()}
        history.append(step_stats)
        if i % 50 == 0:
            pbar.set_postfix({'cons': f'{loss_c.item():.4f}', 'den': f'{loss_d.item():.4f}'})
            
    return pd.DataFrame(history)

def distill_nll(components, mode, buckets, steps=1000, logger=None):
    """
    Distillation where the 'Dataset' signal is the NLL Loss.
    This combines Generation Gradients (Consistency) with Probabilistic Data Gradients (NLL).
    """
    print(f"\n--- Distilling: Probabilistic (NLL + Consistency) ---")
    model = components[0]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scheduler_main = OneCycleLR(opt, max_lr=1e-4, total_steps=steps, 
                        pct_start=0.1, div_factor=10, final_div_factor=100)
    
    # Use reduced batch size for distillation memory overhead
    buckets_distill = [(res, max(1, bs // 2)) for res, bs in buckets]
    
    iterator = CompositeIterator(model.text_embed.weight.device, config={'checkerboard': 0.5, 'torus': 0.5})
    manager = BucketManager(buckets_distill)
    history = []
    
    pbar = tqdm(range(steps), desc="distill-prob")
    for i in pbar:
        opt.zero_grad()
        res, bs = manager.next_bucket()
        
        # Data
        x0 = iterator.generate_batch(bs, res, num_tiles=4.0)
        spans = get_image_spans(res)
        
        # 1. Consistency Loss (Generation Gradient)
        # We stick to MSE for the trajectory itself to force a deterministic path
        loss_c, aux_c, _ = compute_consistency_loss(components, x0, spans, mode=mode)
        
        # 2. Denoising Loss (Dataset Gradient via NLL)
        # We need a fresh noise sample for the "Denoising" objective to ensure independence
        t = torch.rand(bs, device=x0.device).clamp(0.001, 0.999)
        l_den = get_schedule(t)
        a, s = logsnr_to_alpha_sigma(l_den)
        eps = torch.randn_like(x0)
        z_t = x0 * a.view(-1,1,1,1) + eps * s.view(-1,1,1,1)
        v_true = a.view(-1,1,1,1)*eps - s.view(-1,1,1,1)*x0
        
        # Get Mean and Variance estimates
        v_mean, v_logvar, aux_d = predict_probabilistic_field(components, z_t, l_den, spans)
        
        # Compute NLL
        loss_nll = compute_nll_loss(v_mean, v_logvar, v_true)
        
        # 3. The "Agreement Gate"
        # If Consistency is bad, force high Likelihood on the data.
        # If Consistency is good, just maintain Likelihood.
        loss_t = (1.0 + loss_c) * loss_nll + aux_c+aux_d
        
        loss_t.backward()
        opt.step()
        scheduler_main.step()
        
        step_stats = {
            'step': i, 'res': res, 
            'loss_consistency': loss_c.item(),
            'loss_nll': loss_nll.item(),
            'loss_total': loss_t.item()
        }
        history.append(step_stats)
        if i % 50 == 0:
            pbar.set_postfix({'cons': f'{loss_c.item():.4f}', 'nll': f'{loss_nll.item():.4f}'})
            
    return pd.DataFrame(history)

def train_multires(components, mode, buckets, steps=1000, logger=None):
    print(f"\n--- Training: {mode.upper()} ---")
    model = components[0]
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    scheduler_main = OneCycleLR(opt, max_lr=1e-4, total_steps=steps, 
                        pct_start=0.1, div_factor=10, final_div_factor=100)
    
    iterator = CompositeIterator(model.text_embed.weight.device, config={'checkerboard': 0.5, 'torus': 0.5})
    manager = BucketManager(buckets)
    history = []
    
    pbar = tqdm(range(steps), desc=f"{mode}")
    for i in pbar:
        opt.zero_grad()
        res, bs = manager.next_bucket()
        
        x0 = iterator.generate_batch(bs, res, num_tiles=4.0)
        t = torch.rand(bs, device=x0.device).clamp(0.001, 0.999)
        logsnr = get_schedule(t)
        alpha, sigma = get_alpha_sigma(logsnr)
        
        eps = torch.randn_like(x0)
        z_t = x0 * alpha.view(-1,1,1,1) + eps * sigma.view(-1,1,1,1)
        v_true = alpha.view(-1,1,1,1) * eps - sigma.view(-1,1,1,1) * x0

        base_spans = get_image_spans(res)
        
        v_pred, aux_loss, _ = predict_velocity_field(components, z_t, logsnr, base_spans, mode)
            
        loss_elem = F.mse_loss(v_pred, v_true, reduction='none').mean(dim=[1,2,3])
        total_loss = loss_elem.mean() + aux_loss
        
        total_loss.backward()
        opt.step()
        scheduler_main.step()

        step_stats = {'step': i, 'res': res, 'loss_total': total_loss.item()}
        if hasattr(iterator, 'last_labels') and hasattr(iterator, 'label_map'):
            labels = iterator.last_labels
            for lbl_idx, lbl_name in iterator.label_map.items():
                mask = (labels == lbl_idx)
                if mask.any():
                    step_stats[f'loss_{lbl_name}'] = loss_elem[mask].mean().item()
        
        history.append(step_stats)
        if i % 100 == 0:
            pbar.set_postfix({'loss': f'{total_loss.item():.4f}', 'res': res})
            
    return pd.DataFrame(history)

def train_nll(components, mode, buckets, steps=1000, logger=None):
    """
    Trains using Heteroscedastic Gaussian NLL.
    Ignores 'mode' parameter logic for output scaling, uses probabilistic field directly.
    """
    print(f"\n--- Training: NLL (Base) ---")
    model = components[0]
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    scheduler_main = OneCycleLR(opt, max_lr=1e-4, total_steps=steps, 
                    pct_start=0.1, div_factor=10, final_div_factor=100)
    
    iterator = CompositeIterator(model.text_embed.weight.device, config={'checkerboard': 0.5, 'torus': 0.5})
    manager = BucketManager(buckets)
    history = []
    
    pbar = tqdm(range(steps), desc="nll-train")
    for i in pbar:
        opt.zero_grad()
        res, bs = manager.next_bucket()
        
        x0 = iterator.generate_batch(bs, res, num_tiles=4.0)
        t = torch.rand(bs, device=x0.device).clamp(0.001, 0.999)
        logsnr = get_schedule(t)
        alpha, sigma = get_alpha_sigma(logsnr)
        
        eps = torch.randn_like(x0)
        z_t = x0 * alpha.view(-1,1,1,1) + eps * sigma.view(-1,1,1,1)
        v_true = alpha.view(-1,1,1,1) * eps - sigma.view(-1,1,1,1) * x0

        base_spans = get_image_spans(res)
        
        # 1. Get Probabilistic Prediction (Mean, LogVar)
        # Note: We don't use predict_velocity_field here because we need the raw logsnr head
        v_mean, v_logvar, aux_loss = predict_probabilistic_field(components, z_t, logsnr, base_spans)
        
        # 2. Compute NLL Loss
        # This replaces MSE. The model learns to output high v_logvar when error is high.
        loss_nll = compute_nll_loss(v_mean, v_logvar, v_true)
        total_loss = loss_nll + aux_loss
        total_loss.backward()
        opt.step()
        scheduler_main.step()

        # Logging (NLL is not directly comparable to MSE, but we log it)
        step_stats = {'step': i, 'res': res, 'loss_total': total_loss.item()}
        history.append(step_stats)
        if i % 100 == 0:
            pbar.set_postfix({'nll': f'{loss_nll.item():.4f}', 'res': res})
            
    return pd.DataFrame(history)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    logger = ExperimentLogger(output_dir="./experiments_mix")
    device = torch.device('cuda')

    """
    BUCKETS = [(16, 128), (32, 64), (64, 16)]
    STEPS = 2000
    DISTILL_STEPS = 2000
    RESOLUTIONS = [16, 32, 64]
    """
    
    BUCKETS = [(16, 64), (32, 32)]
    STEPS = 1000
    DISTILL_STEPS = 1000
    RESOLUTIONS = [16, 32]
    
    """
    BUCKETS = [(128,4)]
    STEPS = 2000
    DISTILL_STEPS = 2000
    RESOLUTIONS = [16, 32, 64, 128]
    """

    
    print("🔧 Initializing ZC Model Stack...")
    embed_dim = 256; depth = 8; num_heads=8; topo_dim = 3 
    
    # 1. Initialize ZC Model (Cacheless)
    model = coolerLDTformer(dim=embed_dim, depth=depth, num_heads=num_heads, topo_dim=topo_dim).to(device)
    span_emb = SpanEmbedder(model.text_embed, model.patch_embedder)
    span_unemb = SpanUnembedder(model.text_head, model.patch_unembedder)
    
    # Compute safe cache AND page table sizes
    max_blocks, max_batch_size, max_logical_blocks = allocate_kv_cache_safely(
        device=device,
        block_size=128,
        embed_dim=embed_dim,
        num_layers=depth,
        num_heads=8,
        expected_batch_size=128,
        expected_seq_len=8*8,  # 16px image -> 8×8 after 2×2 pooling
        kv_cache_memory_fraction=0.85,
        safety_margin_gb=1.5,
        concurrent_requests_multiplier=1.0,  # Allow 2× batch size headroom
        verbose=True
    )

    # 2. Dummy PageTable for constant retrieval (block_size)
    page_table = PageTable(
        num_blocks=max_blocks,
        block_size=128,
        max_batch_size=max_batch_size,
        max_logical_blocks=max_logical_blocks,
        device=device
    )

    print("🔥 Compiling Model (dynamic=True)...")
    model = torch.compile(model, dynamic=True)
    
    # Components now excludes KVTManager
    components = (model, span_emb, span_unemb, None, page_table)
    
    # 2. Run A (Naive)
    print("🚀 Run A: Naive")
    model.param_init()
    df_n = train_multires(components, 'naive', BUCKETS, STEPS, logger)
    params_naive = model.dump() 
    
    # 3. Run B (Factorized)
    print("🚀 Run B: Factorized")
    model.flush()
    model.param_init()
    df_f = train_multires(components, 'factorized', BUCKETS, STEPS, logger)
    params_fact = model.dump()
    
    # 3.5 Run C: NLL
    print("🚀 Run C: NLL")
    model.flush()
    model.param_init()
    # NLL training doesn't use the 'mode' param for loss calculation (it uses the custom loop)
    # but we pass 'factorized' to prediction helpers later if we want to test that decoding path.
    df_nll = train_nll(components, 'factorized', BUCKETS, STEPS, logger)
    params_nll = model.dump()

    print("\n📈 Plotting 3-way training losses...")
    plot_three_way_loss(df_n, df_f, df_nll, logger, string="three_way_denoising_loss")

    #print("\n📈 Plotting training losses...")
    plot_detailed_loss(df_n, df_f, logger)

    eval_iterator = CompositeIterator(device, config={'checkerboard': 0.5, 'torus': 0.5})
    # 4. Sample BEFORE distillation
    for res in RESOLUTIONS:
        eval_config = {
            'res': 32,
            'num_samples': 8,
            'min_logsnr': -14.0, # Very noisy
            'max_logsnr': -1.0,  # Partially clean
        }
    
        model.param_load(params_naive)
        eval_config['mode'] = 'naive'
        res_naive = sample_viz_dset(components, eval_iterator, eval_config)
        plot_dset_reconstruction(res_naive, logger, f"eval_dset_naive_{res}px")
        
        model.param_load(params_fact)
        eval_config['mode'] = 'factorized'
        res_fact = sample_viz_dset(components, eval_iterator, eval_config)
        plot_dset_reconstruction(res_fact, logger, f"eval_dset_factorized_{res}px")
        
        model.param_load(params_nll)
        s_nll = sample_viz(components, res, mode='naive')
        eval_config['mode'] = 'naive'
        res_nll_naive = sample_viz_dset(components, eval_iterator, eval_config)
        plot_dset_reconstruction(res_nll_naive, logger, f"eval_dset_nll_naive_{res}px")
        
        model.param_load(params_nll)
        s_nll = sample_viz(components, res, mode='factorized')
        eval_config['mode'] = 'factorized'
        res_nll_fact = sample_viz_dset(components, eval_iterator, eval_config)
        plot_dset_reconstruction(res_nll_fact, logger, f"eval_dset_nll_factorized_{res}px")
    
    #plot_sample_grid(samples_before, logger, "before_distillation_3way")
    #plot_sample_grid(samples_before, logger, "before_distillation")
    
    # 5. Distillation phase
    print("\n🔮 Phase 2: Distillation")
    model.param_load(params_naive)
    df_n_dist = distill_multires(components, 'naive', BUCKETS, DISTILL_STEPS, logger)
    params_naive_dist = model.dump()
    
    model.param_load(params_fact)
    df_f_dist = distill_multires(components, 'factorized', BUCKETS, DISTILL_STEPS, logger)
    params_fact_dist = model.dump()
    
    # 5.5 Distill NLL
    print("...Distilling NLL...")
    model.param_load(params_nll)
    df_nll_dist = distill_nll(components, 'naive', BUCKETS, DISTILL_STEPS, logger)
    params_nll_dist = model.dump()

    print("\n📈 Plotting distillation losses...")
    plot_distillation_loss(df_n_dist, df_f_dist, logger)
    plot_three_way_loss(df_n_dist, df_f_dist, df_nll_dist , logger, string="three_way_distillation_loss")

    # 6. Sample AFTER distillation
    print("\n🎨 Sampling (After Distillation)...")
    samples_after = []
    for res in RESOLUTIONS:
        eval_config = {
            'res': 32,
            'num_samples': 8,
            'min_logsnr': -4.0, # Very noisy
            'max_logsnr': 1.0,  # Partially clean
        }
    

        model.param_load(params_naive_dist)
        eval_config['mode'] = 'naive'
        res_naive = sample_viz_dset(components, eval_iterator, eval_config)
        plot_dset_reconstruction(res_naive, logger, f"eval_distill_dset_naive_{res}px")
        
        model.param_load(params_fact_dist)
        eval_config['mode'] = 'factorized'
        res_fact = sample_viz_dset(components, eval_iterator, eval_config)
        plot_dset_reconstruction(res_fact, logger, f"eval_distill_dset_factorized_{res}px")
        
        model.param_load(params_nll_dist)
        eval_config['mode'] = 'naive'
        res_nll_naive = sample_viz_dset(components, eval_iterator, eval_config)
        plot_dset_reconstruction(res_nll_naive, logger, f"eval_distill_dset_nll_naive_{res}px")
        
        model.param_load(params_nll_dist)
        eval_config['mode'] = 'factorized'
        res_nll_fact = sample_viz_dset(components, eval_iterator, eval_config)
        plot_dset_reconstruction(res_nll_fact, logger, f"eval_distill_dset_nll_factorized_{res}px")
        #samples_after.append((f"Naive {res}px", s_n))
        #samples_after.append((f"Fact {res}px", s_f))
        #samples_after.append((f"NLL {res}px", s_nll))
    
    #plot_sample_grid(samples_after, logger, "after_distillation_3way")
    #fig_compare = plot_comparison_grid(samples_before, samples_after, RESOLUTIONS)
    #logger.save_figure(fig_compare, "before_after_comparison")
    
    print(f"\n✅ Done. Check {logger.run_dir}")