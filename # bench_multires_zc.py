# bench_multires_cl_p.py
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

from diffusion_utils import get_schedule, get_alpha_sigma, BucketManager
from dataset import CompositeIterator

# NOTE: Importing the ZC (Zero-Cache) variant for training
from ld_tformer import coolerLDTformerZC as coolerLDTformer
from ld_tformer import SpanEmbedder, SpanUnembedder, build_composed_mask
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

def run_forward_step(
    components, 
    z, 
    logsnr, 
    base_spans,
    suppress_logsnr_input: bool = False
):
    """
    Zero-Cache Forward Step:
    1. Embeds Spans -> Flat Tensor
    2. Generates Identity Topology
    3. Runs Model
    """
    model, span_embedder, _, _, page_table = components
    B, C, H, W = z.shape
    device = z.device
    
    # 1. Prepare Metadata & Inputs
    batch_spans_meta = []
    images = [z[i] for i in range(B)]
    
    if suppress_logsnr_input:
        zero_map = torch.zeros((1, H, W), device=device)
        logsnr_maps = [zero_map] * B
    else:
        if logsnr.dim() == 1:  # Scalar per sample: [B]
            logsnr_spatial = logsnr.view(B, 1, 1, 1).expand(B, 1, H, W)
            logsnr_maps = [logsnr_spatial[i] for i in range(B)]
        elif logsnr.dim() == 4:  # Already spatial
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
    # Because z_flat is contiguous, Logical Block i maps to Physical Block i.
    L_total = z_flat.shape[0]
    block_size = page_table.block_size
    num_blocks = (L_total + block_size - 1) // block_size
    
    # Identity mapping: [0, 1, 2, ... num_blocks-1]
    flat_page_table = torch.arange(num_blocks, device=device, dtype=torch.long)
    
    # 5. Build Mask
    # Note: topo_heap is just topo_embeds because Heap == Active in ZC mode
    block_mask = build_composed_mask(
        span_objects,
        topo_active=topo_embeds,
        topo_heap=topo_embeds,
        page_table=page_table,      # Used for block_size param
        flat_page_table=flat_page_table,
        inverse_page_table=None     # Not needed for ZC
    )
    
    # 6. Forward Pass (No K/V Caches needed)
    # Unsqueeze to [1, L, D] because model expects flattened batch dim
    z_out, aux_loss = model(
        z_flat.unsqueeze(0),
        topo_embeds.unsqueeze(0),
        slot_mapping=None,  # ZC ignores this
        block_mask=block_mask
    )
    
    return z_out.squeeze(0), aux_loss, span_objects, []

def logsnr_to_alpha_sigma(logsnr):
    snr = torch.exp(logsnr)
    alpha_sq = snr / (1.0 + snr)
    sigma_sq = 1.0 / (1.0 + snr)
    return torch.sqrt(alpha_sq), torch.sqrt(sigma_sq)

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

def compute_consistency_loss(components, x0, spans, mode='factorized', min_logsnr=-5.0, max_logsnr=5.0):
    B = x0.shape[0]
    device = x0.device
    l_start, l_mid, l_end = sample_logsnr_triplet(B, device, min_logsnr, max_logsnr)
    a_start, s_start = logsnr_to_alpha_sigma(l_start)
    z_start = x0 * a_start.view(-1,1,1,1) + torch.randn_like(x0) * s_start.view(-1,1,1,1)
    
    # 1. Teacher (Start -> End)
    v_start, aux1, _ = predict_velocity_field(components, z_start, l_start, spans, mode)
    with torch.no_grad():
        z_end_teacher = euler_reverse_step(z_start, v_start, l_start, l_end)
    
    # 2. Student (Start -> Mid -> End)
    z_mid_student = euler_reverse_step(z_start, v_start, l_start, l_mid)
    v_mid, aux2, _ = predict_velocity_field(components, z_mid_student, l_mid, spans, mode)
    z_end_student = euler_reverse_step(z_mid_student, v_mid, l_mid, l_end)
    
    loss = F.mse_loss(z_end_student, z_end_teacher.detach())
    return loss, aux1 + aux2, lambda: None

def distill_multires(components, mode, buckets, steps=1000, logger=None):
    print(f"\n--- Distilling: {mode.upper()} ---")
    model = components[0]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
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
        
        loss_t = loss_c + 0.1 * loss_d + aux_loss + aux_loss_con
        loss_t.backward()
        opt.step()
        
        step_stats = {'step': i, 'res': res, 'loss_consistency': loss_c.item(), 
                      'loss_denoise': loss_d.item(), 'loss_total': loss_t.item()}
        history.append(step_stats)
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

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    logger = ExperimentLogger(output_dir="./experiments_mix")
    device = torch.device('cuda')

    BUCKETS = [(16, 128), (32, 64)]
    STEPS = 500
    DISTILL_STEPS = 500
    RESOLUTIONS = [16, 32]
    
    print("🔧 Initializing ZC Model Stack...")
    embed_dim = 256; depth = 4
    
    # 1. Initialize ZC Model (Cacheless)
    model = coolerLDTformer(dim=embed_dim, depth=depth, num_heads=8, topo_dim=3).to(device)
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
        concurrent_requests_multiplier=2.0,  # Allow 2× batch size headroom
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
    
    print("\n📈 Plotting training losses...")
    plot_detailed_loss(df_n, df_f, logger)

    # 4. Sample BEFORE distillation
    print("\n🎨 Sampling (Before Distillation)...")
    samples_before = []
    for res in RESOLUTIONS:
        model.param_load(params_naive)
        s_n = sample_viz(components, res, mode='naive')
        
        model.param_load(params_fact)
        s_f = sample_viz(components, res, mode='factorized')
        
        samples_before.append((f"Naive {res}px", s_n))
        samples_before.append((f"Fact {res}px", s_f))
    
    plot_sample_grid(samples_before, logger, "before_distillation")
    
    # 5. Distillation phase
    print("\n🔮 Phase 2: Distillation")
    model.param_load(params_naive)
    df_n_dist = distill_multires(components, 'naive', BUCKETS, DISTILL_STEPS, logger)
    params_naive_dist = model.dump()
    
    model.param_load(params_fact)
    df_f_dist = distill_multires(components, 'factorized', BUCKETS, DISTILL_STEPS, logger)
    params_fact_dist = model.dump()
    
    print("\n📈 Plotting distillation losses...")
    plot_distillation_loss(df_n_dist, df_f_dist, logger)

    # 6. Sample AFTER distillation
    print("\n🎨 Sampling (After Distillation)...")
    samples_after = []
    for res in RESOLUTIONS:
        model.param_load(params_naive_dist)
        s_n = sample_viz(components, res, mode='naive')
        
        model.param_load(params_fact_dist)
        s_f = sample_viz(components, res, mode='factorized')
        
        samples_after.append((f"Naive {res}px", s_n))
        samples_after.append((f"Fact {res}px", s_f))
    
    plot_sample_grid(samples_after, logger, "after_distillation")
    fig_compare = plot_comparison_grid(samples_before, samples_after, RESOLUTIONS)
    logger.save_figure(fig_compare, "before_after_comparison")
    
    print(f"\n✅ Done. Check {logger.run_dir}")