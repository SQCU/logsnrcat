# bench_multisnr_zc.py
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

from diffusion_utils import BucketManager
from dataset import CompositeIterator
from ld_tformer import coolerLDTformerZC as coolerLDTformer
from ld_tformer import SpanEmbedder, SpanUnembedder, build_dual_masks, ContextBlock
from ld_tformer_embedding_functional import render_topology_embeddings
from memory_manager import PageTable
from kv_cache_allocator import allocate_kv_cache_safely

# Hook ZMQ if available
try:
    import inductor_cas_client
    inductor_cas_client.install_cas_client()
except ImportError:
    pass

# ==============================================================================
# 1. Math & Physics Helpers (Deduplicated)
# ==============================================================================

def get_schedule(t, schedule_bounds: tuple = (5, -4)):
    """Linear LogSNR schedule."""
    return schedule_bounds[0] - t * (schedule_bounds[1] - schedule_bounds[0])

def logsnr_to_alpha_sigma(logsnr):
    """
    Returns alpha, sigma for a given logsnr.
    Handles broadcasting if logsnr is [B, 1, H, W].
    """
    # Ensure numerical stability
    sigmoid_lsnr = torch.sigmoid(logsnr)
    sigmoid_neg_lsnr = torch.sigmoid(-logsnr)
    alpha = torch.sqrt(sigmoid_lsnr)
    sigma = torch.sqrt(sigmoid_neg_lsnr)
    return alpha, sigma

def euler_forward_step(x0, logsnr, noise=None):
    """
    Diffuses x0 -> z_t. Returns z_t and the target velocity v_true.
    """
    if noise is None:
        noise = torch.randn_like(x0)
    
    alpha, sigma = logsnr_to_alpha_sigma(logsnr)
    
    # Broadcast check: logsnr might be [B, 1, H, W] or [B]
    if alpha.ndim == 1:
        alpha = alpha.view(-1, 1, 1, 1)
        sigma = sigma.view(-1, 1, 1, 1)
        
    z_t = x0 * alpha + noise * sigma
    v_true = alpha * noise - sigma * x0
    return z_t, v_true, noise

def euler_reverse_step(z_t, v_pred, logsnr_from, logsnr_to):
    """
    Denoises z_t -> z_{t-1}.
    """
    alpha_from, sigma_from = logsnr_to_alpha_sigma(logsnr_from)
    alpha_to, sigma_to = logsnr_to_alpha_sigma(logsnr_to)
    
    if alpha_from.ndim == 1:
        alpha_from = alpha_from.view(-1, 1, 1, 1)
        sigma_from = sigma_from.view(-1, 1, 1, 1)
        alpha_to = alpha_to.view(-1, 1, 1, 1)
        sigma_to = sigma_to.view(-1, 1, 1, 1)

    # Reconstruct x0 (prediction)
    x0_pred = alpha_from * z_t - sigma_from * v_pred
    # Reconstruct eps (prediction)
    eps_pred = sigma_from * z_t + alpha_from * v_pred
    
    # Step to next level
    z_next = alpha_to * x0_pred + sigma_to * eps_pred
    return z_next

def get_image_spans(resolution):
    latent_res = resolution // 2
    length = latent_res * latent_res
    return [{'type': 'latent', 'len': length, 'shape': (latent_res, latent_res), 'causal': False}]

# ==============================================================================
# 2. Model Wrappers & Sampling
# ==============================================================================

def run_model_forward(components, blocks: List[ContextBlock]):
    """
    Unified forward pass. 
    The 'blocks' contain the Tensors (z_t) and the Metadata (logsnr, shape, id).
    """
    model, span_embedder, span_unembedder, _, page_table = components
    device = model.text_embed.weight.device 
    
    # 1. Embed (Pass blocks directly)
    z_flat, span_objects, _ = span_embedder.embed(blocks)
    
    # 2. Topology
    topo_embeds, _ = render_topology_embeddings(span_objects, 3, device)
    
    # 3. Masking
    L_total = z_flat.shape[0]
    block_size = page_table.block_size
    num_blocks = (L_total + block_size - 1) // block_size
    flat_page_table = torch.arange(num_blocks, device=device, dtype=torch.long)
    
    block_masks = build_dual_masks(
        span_objects, topo_embeds, topo_embeds,
        page_table, flat_page_table, None
    )
    
    # 4. Transformer
    rope_scale = max(1.0, L_total / 64.0)
    z_out, aux_loss = model(
        z_flat.unsqueeze(0),
        topo_embeds.unsqueeze(0),
        slot_mapping=None,
        block_masks=block_masks,
        scale=rope_scale
    )
    
    # 5. Unembed
    decoded = span_unembedder.decode(z_out.squeeze(0), span_objects)
    return decoded, aux_loss


def predict_velocity_from_blocks(components, blocks: List[ContextBlock], mode='naive'):
    """
    Wrapper that calls model and processes outputs (factorization, etc).
    """
    decoded, aux_loss = run_model_forward(components, blocks)
    
    v_final_list = []
    pred_logsnr_list = []
    
    for i, d in enumerate(decoded):
        if 'image_vpreds' in d:
            v_raw = d['image_vpreds']
            pred_l = d['image_logsnrs']
            
            if mode == 'factorized':
                sigma_p = torch.sqrt(torch.sigmoid(-pred_l))
                v_final = v_raw * sigma_p
            else:
                v_final = v_raw
            
            v_final_list.append(v_final)
            pred_logsnr_list.append(pred_l)
        else:
            # For text-only blocks, we might not have vpreds relevant to diffusion loss
            # Just append None or dummy
            v_final_list.append(None)
            pred_logsnr_list.append(None)
        
    return v_final_list, pred_logsnr_list, aux_loss

@torch.no_grad()
def autoregressive_sample_loop(components, x0_shape, config):
    """
    Unified sampler for visual validation.
    """
    model, _, _, _, _ = components
    model.eval()
    
    res = config.get('res', 32)
    n_samples = config.get('num_samples', 8)
    mode = config.get('mode', 'naive')
    device = model.text_embed.weight.device
    
    # Initialize Noise
    z = torch.randn(n_samples, 3, res, res, device=device)
    base_spans = get_image_spans(res)
    
    # Scheduler
    steps = 50
    ts = torch.linspace(1.0, 0.001, steps, device=device)
    
    for i in range(steps - 1):
        t_curr = ts[i]
        t_next = ts[i+1]
        
        # 1. Get LogSNR (Global scalar broadcast to map)
        lsnr_val = get_schedule(t_curr)
        logsnr_map = torch.full((n_samples, 1, res, res), lsnr_val, device=device)
        
        # 2. Predict
        v_pred, _, _ = predict_velocity(components, z, logsnr_map, base_spans, mode)
        
        # 3. Step
        lsnr_curr = get_schedule(t_curr)
        lsnr_next = get_schedule(t_next)
        z = euler_reverse_step(z, v_pred, lsnr_curr, lsnr_next)
        
    model.train()
    return z.clamp(0, 1)

# ==============================================================================
# 3. Training Loops (Config Driven)
# ==============================================================================


def train_autoembed(components, config, logger=None):
    mode = config['mode']; steps = config['ae_steps']
    # We ignore buckets/manager for list mode simplicity or use a fixed one
    bs = 8 
    
    print(f"\n--- Training: Auto-Encoder ({mode.upper()}) ---")
    model = components[0]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    
    iterator = CompositeIterator(model.text_embed.weight.device, config=config['dataset_mix'])
    history = []
    
    pbar = tqdm(range(steps), desc="train-ae")
    for i in pbar:
        opt.zero_grad()
        
        # 1. Get Clean Blocks
        clean_blocks = iterator.generate_batch_list(bs)
        
        # 2. Noise Injection (Forward Step)
        noisy_blocks = []
        target_imgs = []
        target_lsnrs = []
        
        for b in clean_blocks:
            if b.type == 'latent':
                # AE training usually reconstructs from z_t. 
                # For pure AE (identity), we can use z_t = x0 (clean).
                # But let's stick to the script: x0 -> noise -> predict.
                z_t, _, _ = euler_forward_step(b.content, b.logsnr)
                
                # Construct input block
                noisy_blocks.append(ContextBlock(
                    content=z_t, logsnr=b.logsnr, type='latent', causal=b.causal,
                    shape_meta=b.shape_meta, group_id=b.group_id, id=b.id
                ))
                target_imgs.append(z_t) # AE target is input (identity)
                target_lsnrs.append(b.logsnr)
            else:
                noisy_blocks.append(b) # Pass text through
        
        # 3. Forward
        # Use run_model_forward directly to get raw outputs
        decoded, aux = run_model_forward(components, noisy_blocks)
        
        loss_img = 0.0; loss_meta = 0.0; count = 0
        for j, res in enumerate(decoded):
            if 'image_vpreds' in res:
                loss_img += F.mse_loss(res['image_vpreds'], target_imgs[j])
                loss_meta += F.l1_loss(res['image_logsnrs'], target_lsnrs[j])
                count += 1
        
        if count > 0:
            loss_img /= count; loss_meta /= count
            
        total_loss = loss_img + 0.1 * loss_meta
        total_loss.backward()
        opt.step(); scheduler.step()
        
        history.append({'step': i, 'loss_ae': loss_img.item() if count else 0})
        if i % 100 == 0: pbar.set_postfix({'ae': f'{loss_img:.4f}'})
            
    return pd.DataFrame(history)

def train_denoise(components, config, logger=None):
    mode = config['mode']; steps = config['steps']; lambda_coeff = config.get('lambda_coeff', 0.2)
    model = components[0]
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    scheduler = OneCycleLR(opt, max_lr=5e-4, total_steps=steps, pct_start=0.1)
    iterator = CompositeIterator(model.text_embed.weight.device, config=config['dataset_mix'])
    bs = 8
    
    history = []
    pbar = tqdm(range(steps), desc=f"train-{mode}")
    
    for i in pbar:
        opt.zero_grad()
        
        # 1. Clean Data
        clean_blocks = iterator.generate_batch_list(bs)
        
        # 2. Noise
        noisy_blocks = []
        targets_v = []
        targets_l = []
        
        for b in clean_blocks:
            if b.type == 'latent':
                z, v, _ = euler_forward_step(b.content, b.logsnr)
                noisy_blocks.append(ContextBlock(
                    content=z, logsnr=b.logsnr, type='latent', causal=b.causal,
                    shape_meta=b.shape_meta, group_id=b.group_id, id=b.id
                ))
                targets_v.append(v)
                targets_l.append(b.logsnr)
            else:
                noisy_blocks.append(b) # Text
                
        # 3. Predict
        v_preds, l_preds, aux = predict_velocity_from_blocks(components, noisy_blocks, mode)
        
        # 4. Loss
        loss_v = 0.0; loss_lam = 0.0; count = 0
        for j, (vp, lp) in enumerate(zip(v_preds, l_preds)):
            if vp is not None:
                loss_v += F.mse_loss(vp, targets_v[count])
                loss_lam += F.l1_loss(lp, targets_l[count])
                count += 1 # targets indices match only latent blocks
        
        if count > 0:
            loss_v /= count; loss_lam /= count
        
        total_loss = loss_v + lambda_coeff * loss_lam + aux
        total_loss.backward()
        opt.step(); scheduler.step()
        
        history.append({'step': i, 'loss_v': loss_v, 'loss_lam': loss_lam})
        if i % 100 == 0: pbar.set_postfix({'v': f'{loss_v:.4f}'})
            
    return pd.DataFrame(history)
# ==============================================================================
# 4. Logger
# ==============================================================================

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
        self.run_dir = self.output_dir / f"{self.script_name}_run_{self.run_id:03d}"
        self.run_dir.mkdir(exist_ok=True)
        print(f"📊 Run: {self.run_id} | Dir: {self.run_dir}")
        
    def save_figure(self, fig, name):
        filepath = self.run_dir / f"{name}.png"
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

def plot_losses(df_naive, df_fact, logger, metric='loss_total', title='Training Loss'):
    if df_naive.empty and df_fact.empty: return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if not df_naive.empty:
        df_naive = df_naive.interpolate()
        ax.plot(df_naive['step'], df_naive[metric].rolling(50).mean(), label='Naive')
        
    if not df_fact.empty:
        df_fact = df_fact.interpolate()
        ax.plot(df_fact['step'], df_fact[metric].rolling(50).mean(), label='Factorized')
        
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xlabel("Step")
    ax.set_yscale('log')
    ax.legend()
    logger.save_figure(fig, f"plot_{metric}")

# ==============================================================================
# 6. Sampling & Visualization Tools
# ==============================================================================

@torch.no_grad()
def spatial_euler_solver(components, start_blocks: List[ContextBlock], target_logsnr, steps, mode, config, fixed_data=None):
    """
    Evolves a list of ContextBlocks from their initial state/logsnr to target_logsnr.
    """
    device = start_blocks[0].content.device
    taus = torch.linspace(0.0, 1.0, steps + 1, device=device)
    
    # Extract mutable state
    z_list = [b.content for b in start_blocks]
    lsnr_start_list = [b.logsnr for b in start_blocks]
    
    def get_target_map(start_map):
        if isinstance(target_logsnr, (float, int)): return torch.full_like(start_map, target_logsnr)
        return target_logsnr

    target_maps = [get_target_map(m) for m in lsnr_start_list]

    for i in range(steps):
        tau_curr = taus[i]
        tau_next = taus[i+1]

        # 1. Interpolate Schedule & Rebuild Blocks
        lsnr_curr_list = []
        lsnr_next_list = []
        current_blocks = []
        
        for idx, (b, start, end) in enumerate(zip(start_blocks, lsnr_start_list, target_maps)):
            l_curr = (1 - tau_curr) * start + tau_curr * end
            l_next = (1 - tau_next) * start + tau_next * end
            lsnr_curr_list.append(l_curr)
            lsnr_next_list.append(l_next)
            
            # Reconstruct block with current state
            current_blocks.append(ContextBlock(
                content=z_list[idx], logsnr=l_curr, # Current State
                type=b.type, causal=b.causal, shape_meta=b.shape_meta,
                group_id=b.group_id, id=b.id
            ))

        # 2. Predict
        v_pred_list, _, _ = predict_velocity_from_blocks(components, current_blocks, mode)

        # 3. Step
        z_next_list = []
        for idx, (z, v, l_curr, l_next) in enumerate(zip(z_list, v_pred_list, lsnr_curr_list, lsnr_next_list)):
            if v is None: # Non-latent block
                z_next_list.append(z)
                continue
                
            if fixed_data is not None and fixed_data[idx] is not None:
                z_next_list.append(fixed_data[idx])
            else:
                z_new = euler_reverse_step(z, v, l_curr, l_next)
                z_next_list.append(z_new)
        
        z_list = z_next_list
        
    return [z.clamp(0, 1) for z in z_list]



@torch.no_grad()
def sample_viz_dset(components, iterator, config):
    model, _, _, _, _ = components
    model.eval()
    
    res = config.get('res', 32)
    n = config.get('num_samples', 8)
    
    # 1. Get Clean Data (as Blocks)
    # We force specific resolution via params if needed, but here we trust iterator default/config
    # Or create a custom config for this fetch
    clean_blocks = iterator.generate_batch_list(n)
    # Filter to only latents for this viz
    clean_blocks = [b for b in clean_blocks if b.type == 'latent'][:n]
    
    # 2. Stratify Noise
    min_snr = config.get('min_logsnr', -4.0); max_snr = config.get('max_logsnr', 1.0)
    device = clean_blocks[0].content.device
    start_vals = torch.rand(len(clean_blocks), device=device) * (max_snr - min_snr) + min_snr
    start_vals, _ = torch.sort(start_vals)
    
    # 3. Create Start Blocks
    start_blocks = []
    noisy_inputs = [] # for viz
    x0s = []
    
    for i, b in enumerate(clean_blocks):
        x0s.append(b.content)
        l_map = torch.full_like(b.logsnr, start_vals[i])
        alpha, sigma = logsnr_to_alpha_sigma(l_map)
        eps = torch.randn_like(b.content)
        z_start = b.content * alpha + eps * sigma
        
        start_blocks.append(ContextBlock(
            content=z_start, logsnr=l_map,
            type=b.type, causal=b.causal, shape_meta=b.shape_meta, group_id=b.group_id, id=b.id
        ))
        noisy_inputs.append(z_start)
        
    # 4. Solve
    z_final = spatial_euler_solver(components, start_blocks, 10.0, 50, config.get('mode', 'naive'), config)
    model.train()
    
    # Pack results for plotter
    # The plotter expects tensors, we have lists. Stack them.
    return {
        'x0': torch.stack(x0s),
        'noisy_input': torch.stack(noisy_inputs),
        'reconstruction': torch.stack(z_final),
        'logsnr_map': torch.stack([b.logsnr for b in clean_blocks]) # Original maps
    }

@torch.no_grad()
def sample_viz_split_topology(components, iterator, config):
    # Similar to above, but uses the iterator's logsnr directly
    model, _, _, _, _ = components
    model.eval()
    
    res = config.get('res', 32)
    n = config.get('num_samples', 8)
    clean_blocks = iterator.generate_batch_list(n)
    clean_blocks = [b for b in clean_blocks if b.type == 'latent'][:n]
    
    start_blocks = []
    noisy_inputs = []
    x0s = []
    
    for b in clean_blocks:
        x0s.append(b.content)
        # Use block's OWN logsnr (which is split topology)
        alpha, sigma = logsnr_to_alpha_sigma(b.logsnr)
        eps = torch.randn_like(b.content)
        z_start = b.content * alpha + eps * sigma
        
        start_blocks.append(ContextBlock(
            content=z_start, logsnr=b.logsnr,
            type=b.type, causal=b.causal, shape_meta=b.shape_meta, group_id=b.group_id, id=b.id
        ))
        noisy_inputs.append(z_start)
        
    z_final = spatial_euler_solver(components, start_blocks, 10.0, 50, config.get('mode', 'naive'), config)
    model.train()
    
    return {
        'x0': torch.stack(x0s),
        'noisy_input': torch.stack(noisy_inputs),
        'reconstruction': torch.stack(z_final),
        'logsnr_map': torch.stack([b.logsnr for b in clean_blocks])
    }

@torch.no_grad()
def sample_viz_causal_prefix_fig(components, iterator, config):
    model, _, _, _, _ = components
    model.eval()
    
    # 1. Get a Sequence Group
    # Fetch enough blocks to find a sequence
    # Note: iterator returns flat list, we need to inspect group_ids to reconstruct sequences
    # Assuming video iterator is dominant or we requested video
    
    blocks = iterator.generate_batch_list(32)
    from collections import defaultdict
    groups = defaultdict(list)
    for b in blocks:
        if b.type == 'latent': groups[b.group_id].append(b)
    
    sequences = [sorted(g, key=lambda x: x.id) for g in groups.values() if len(g) >= 4]
    if not sequences: return None
    
    results = []
    suffix_idx = 3
    
    for seq in sequences[:2]: # Do 2 sequences
        start_blocks = []
        fixed_data = []
        
        for t, b in enumerate(seq[:4]):
            if t < suffix_idx:
                l_map = torch.full_like(b.logsnr, 10.0)
                start_blocks.append(ContextBlock(
                    content=b.content, logsnr=l_map, type='latent', causal=True,
                    shape_meta=b.shape_meta, group_id=b.group_id, id=b.id
                ))
                fixed_data.append(b.content)
            else:
                l_map = torch.full_like(b.logsnr, -4.0)
                alpha, sigma = logsnr_to_alpha_sigma(l_map)
                eps = torch.randn_like(b.content)
                z_start = b.content * alpha + eps * sigma
                start_blocks.append(ContextBlock(
                    content=z_start, logsnr=l_map, type='latent', causal=True,
                    shape_meta=b.shape_meta, group_id=b.group_id, id=b.id
                ))
                fixed_data.append(None)
                
        z_final = spatial_euler_solver(components, start_blocks, 10.0, 50, config.get('mode', 'naive'), config, fixed_data)
        
        results.append({
            'context': seq[suffix_idx-1].content,
            'gt': seq[suffix_idx].content,
            'recon': z_final[suffix_idx]
        })
        
    fig, axes = plt.subplots(len(results), 3, figsize=(10, 4*len(results)))
    if len(results) == 1: axes = axes.reshape(1, -1)
    
    for i, res in enumerate(results):
        axes[i, 0].imshow(res['context'].permute(1,2,0).cpu().numpy()); axes[i,0].set_title("Context")
        axes[i, 1].imshow(res['gt'].permute(1,2,0).cpu().numpy()); axes[i,1].set_title("GT")
        axes[i, 2].imshow(res['recon'].permute(1,2,0).cpu().numpy()); axes[i,2].set_title("Recon")
        for ax in axes[i]: ax.axis('off')
        
    return fig

def plot_dset_reconstruction(result_dict, logger, name="reconstruction", show_map=False):
    x0 = result_dict['x0'].cpu()
    noisy = result_dict['noisy_input'].cpu()
    recon = result_dict['reconstruction'].cpu()
    
    cols = 4 if show_map else 3
    n = x0.shape[0]
    fig, axes = plt.subplots(n, cols, figsize=(3*cols, 2 * n))
    if n == 1: axes = axes.reshape(1, -1)
    
    for i in range(n):
        # GT
        axes[i, 0].imshow(x0[i].permute(1,2,0).numpy())
        axes[i, 0].axis('off')
        if i==0: axes[i,0].set_title("Ground Truth")
        
        # Input
        axes[i, 1].imshow(noisy[i].permute(1,2,0).clamp(0,1).numpy())
        axes[i, 1].axis('off')
        if i==0: axes[i,1].set_title("Noisy Input")
        
        # Output
        axes[i, 2].imshow(recon[i].permute(1,2,0).numpy())
        axes[i, 2].axis('off')
        if i==0: axes[i,2].set_title("Reconstruction")
        
        if show_map:
            lmap = result_dict['logsnr_map'][i].squeeze().cpu().numpy()
            axes[i, 3].imshow(lmap, cmap='viridis')
            axes[i, 3].axis('off')
            if i==0: axes[i,3].set_title("Split Map")

    plt.tight_layout()
    logger.save_figure(fig, name)

# ==============================================================================
# 5. Main Execution
# ==============================================================================

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    logger = ExperimentLogger(output_dir="./experiments_mix")
    device = torch.device('cuda')

    # --- Configuration ---
    # Define the curriculum:
    # 40% Uniform Checkerboard (Standard)
    # 60% Split-Screen Torus (Hard Geometry + Boundary Conditions)
    dataset_mix = {
        'uniform_checker': {
            'type': 'checkerboard',
            'ratio': 0.2,
            'noise_mode': 'uniform',
            'noise_params': {'min_snr': -4.0, 'max_snr': 2.0}
        },
        'split_checker': {
            'type': 'checkerboard',
            'ratio': 0.3,
            'noise_mode': 'split',
            'noise_params': {'min_snr': -5.0, 'max_snr': 2.0, 'angle_range_deg': 30.0}
        },
        'split_torus': {
            'type': 'torus',
            'ratio': 0.3,
            'noise_mode': 'split',
            'noise_params': {'min_snr': -5.0, 'max_snr': 2.0, 'angle_range_deg': 270.0}
        },
        'uniform_torus': {
            'type': 'torus',
            'ratio': 0.2,
            'noise_mode': 'uniform',
            'noise_params': {'min_snr': -5.0, 'max_snr': 5.0}
        },
        'video_causal_zoom': {
        'type': 'video',  # Triggers VideoFolderIterator
        'ratio': 0.8,     
        # Dataset-specific parameters passed to the Iterator
        'params': {
            'path': "C:/dox/recordings/rl_capture/capture_run_1760343426/videos",
            'time_sampler': {
                'min_pct': 0.001,
                'max_pct': 0.05,
                'stride': None  # Set to 1 for contiguous video modeling
            },
            'sequence_structure': [
                {'res': 32, 'noise_mode': 'split', 'noise_params': {'min_snr': 2.0, 'max_snr': 6.0}},
                {'res': 32, 'noise_mode': 'split', 'noise_params': {'min_snr': 2.0, 'max_snr': 6.0}},
                {'res': 32, 'noise_mode': 'split', 'noise_params': {'min_snr': 2.0, 'max_snr': 6.0}},
                {'res': 64, 'noise_mode': 'split', 'noise_params': {'min_snr': -5.0, 'max_snr': 5.0}}
            ]
        }
    },
    }

    base_config = {
        'ae_steps': 500,
        'steps': 1000,
        'distill_steps': 0,
        'buckets': [(16, 64), (32, 32), (64, 8)],
        'lambda_coeff': 0.2, # Regularization strength for lambda reconstruction
        'dataset_mix': dataset_mix
    }

    # --- Model Init ---
    print("🔧 Initializing ZC Model Stack...")
    embed_dim = 256
    model = coolerLDTformer(dim=embed_dim, depth=4, num_heads=8, topo_dim=3).to(device)
    model = torch.compile(model, dynamic=True)
    
    span_emb = SpanEmbedder(model.text_embed, model.patch_embedder)
    span_unemb = SpanUnembedder(model.text_head, model.patch_unembedder)
    
    # Dummy PageTable (needed for mask construction interface, though unused in logic)
    page_table = PageTable(num_blocks=1024, block_size=128, max_batch_size=128, max_logical_blocks=1024, device=device)
    
    components = (model, span_emb, span_unemb, None, page_table)

    # Iterator for Validation Sampling
    val_iterator = CompositeIterator(device, config=dataset_mix)

    # --- Run A: Naive Mode ---
    print("🚀 Starting Run A: Naive")
    model.param_init()
    config_n = {**base_config, 'mode': 'naive'}
    
    df_ae_n = train_autoembed(components, config_n)
    df_train_n = train_denoise(components, config_n)
    params_n = model.dump()

    # Sample A
    print("🎨 Sampling Naive (Post-Train)...")
    sample_res = [32, 64]
    for res in sample_res:
        res_n_strat = sample_viz_dset(components, val_iterator, {'mode':'naive', 'res':res})
        plot_dset_reconstruction(res_n_strat, logger, f"naive_train_stratified_{res}")
        
        res_n_split = sample_viz_split_topology(components, val_iterator, {'mode':'naive', 'res':res})
        plot_dset_reconstruction(res_n_split, logger, f"naive_train_split_{res}", show_map=True)

        # Requires a video iterator config
        if 'video_causal_zoom' in dataset_mix: # Or whatever key you used
            print("🎨 Sampling Prefix Sequence (Video)...")
            fig_prefix = sample_viz_causal_prefix_fig(components, val_iterator, {'mode':'naive'})
            logger.save_figure(fig_prefix, f"naive_video_prefix_generation_{res}")

    # --- Run B: Factorized Mode ---
    print("🚀 Starting Run B: Factorized")
    model.flush()
    model.param_init()
    config_f = {**base_config, 'mode': 'factorized'}
    
    df_ae_f = train_autoembed(components, config_f)
    df_train_f = train_denoise(components, config_f)
    params_f = model.dump()
    
    # Sample B
    print("🎨 Sampling Factorized (Post-Train)...")
    sample_res = [32, 64]
    for res in sample_res:
        res_f_strat = sample_viz_dset(components, val_iterator, {'mode':'factorized', 'res':res})
        plot_dset_reconstruction(res_f_strat, logger, f"factorized_train_stratified_{res}")
        
        res_f_split = sample_viz_split_topology(components, val_iterator, {'mode':'factorized', 'res':res})
        plot_dset_reconstruction(res_f_split, logger, f"factorized_train_split_{res}", show_map=True)

        # Requires a video iterator config
        if 'video_causal_zoom' in dataset_mix: # Or whatever key you used
            print("🎨 Sampling Prefix Sequence (Video)...")
            fig_prefix = sample_viz_causal_prefix(components, val_iterator, {'mode':'factorized'})
            logger.save_figure(fig_prefix, "factorized_video_prefix_generation_{res}")

    # --- Plotting ---
    print("\n📈 Plotting Results...")
    plot_losses(df_train_n, df_train_f, logger, metric='loss_v', title='Velocity Prediction Loss (MSE)')
    plot_losses(df_train_n, df_train_f, logger, metric='loss_lambda', title='Lambda Reconstruction Loss (L1)')
    
    #distillation is stinky, makes models collapse on image-semantics-unrelated outputs with extreme consistency
    """
    # --- Distillation (Optional) ---
    # redacted bc it was a perturbation study to prove our models train from diffusion gradients alone.
    """
    print(f"\n✅ Experiment Complete. Results in {logger.run_dir}")