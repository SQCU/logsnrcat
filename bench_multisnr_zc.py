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
from ld_tformer import SpanEmbedder, SpanUnembedder, build_dual_masks
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

def run_model_forward(components, z_list, logsnr_list, spans):
    """
    Unified forward pass for heterogeneous lists of tensors.
    
    Args:
        z_list: List[Tensor] of shape [C, H, W]
        logsnr_list: List[Tensor] of shape [1, H, W]
        spans: List[Dict] (metadata objects)
        
    Returns:
        decoded: List[Dict] keys=['image_vpreds', 'image_logsnrs', ...]
        aux_loss: Tensor (scalar)
    """
    model, span_embedder, span_unembedder, _, page_table = components
    device = model.text_embed.weight.device 
    
    # 1. Embed (Pass lists directly to SpanEmbedder)
    # SpanEmbedder handles the iteration and mixed-resolution logic
    z_flat, span_objects, _ = span_embedder.embed(
        spans, 
        text_tokens=[None]*len(spans), 
        images=z_list, 
        logsnr_maps=logsnr_list
    )
    
    # 2. Topology
    topo_embeds, _ = render_topology_embeddings(spans, 3, device)
    
    # 3. Masking (ZC Mode)
    # Construct Identity Page Table for the flat buffer
    L_total = z_flat.shape[0]
    block_size = page_table.block_size
    num_blocks = (L_total + block_size - 1) // block_size
    flat_page_table = torch.arange(num_blocks, device=device, dtype=torch.long)
    
    # Note: build_dual_masks must support span.doc_id for causal blocking
    block_masks = build_dual_masks(
        span_objects, topo_embeds, topo_embeds,
        page_table, flat_page_table, None
    )
    
    # 4. Transformer
    # Auto-scale RoPE base for very long sequences (e.g. video batches)
    base_ref_len = 64.0
    rope_scale = max(1.0, L_total / base_ref_len)
    
    z_out, aux_loss = model(
        z_flat.unsqueeze(0),
        topo_embeds.unsqueeze(0),
        slot_mapping=None,
        block_masks=block_masks,
        scale=rope_scale
    )
    
    # 5. Unembed
    # Returns list of dicts because output resolutions vary
    decoded = span_unembedder.decode(z_out.squeeze(0), span_objects)
    
    return decoded, aux_loss

def predict_velocity_list(components, z_list, logsnr_list, spans, mode='naive'):
    """
    High-level prediction wrapper for lists.
    Applies factorization scaling if needed.
    
    Returns:
        v_final_list: List[Tensor] [3, H, W]
        pred_logsnr_list: List[Tensor] [1, H, W]
        aux_loss: scalar
    """
    decoded, aux_loss = run_model_forward(components, z_list, logsnr_list, spans)
    
    v_final_list = []
    pred_logsnr_list = []
    
    for d in decoded:
        v_raw = d['image_vpreds']   # [3, H, W]
        pred_l = d['image_logsnrs'] # [1, H, W]
        
        if mode == 'factorized':
            # Apply sigma scaling
            sigma_p = torch.sqrt(torch.sigmoid(-pred_l))
            v_final = v_raw * sigma_p
        else:
            v_final = v_raw
            
        v_final_list.append(v_final)
        pred_logsnr_list.append(pred_l)
        
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
    """
    Phase 0: Train Embedder/Unembedder (Denoising Autoencoder Warmup).
    Ensures z_t -> Embed -> Unembed -> z_t works before Transformer training.
    """
    mode = config['mode']
    steps = config['ae_steps']
    buckets = config['buckets']
    
    print(f"\n--- Training: Auto-Encoder ({mode.upper()}) ---")
    model = components[0]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    
    iterator = CompositeIterator(model.text_embed.weight.device, config=config['dataset_mix'])
    manager = BucketManager(buckets)
    history = []
    
    pbar = tqdm(range(steps), desc="train-ae")
    for i in pbar:
        opt.zero_grad()
        res, bs = manager.next_bucket()
        
        # Data
        x0, logsnr_map = iterator.generate_batch(bs, res, num_tiles=4.0)
        
        # Noise
        z_t, _, _ = euler_forward_step(x0, logsnr_map)
        
        # Forward (Bypass Transformer logic, use span tools directly)
        # Note: We use run_model_forward but strictly it uses the Transformer. 
        # For true AE warmup we might want to skip the blocks, but 
        # let's train the whole stack as an identity pass for simplicity 
        # (or define a simplified forward if strictly needed).
        # Actually, using the full stack is fine; it learns to be an identity at layer 0+N.
        
        # Using run_model_forward to get decoded outputs
        base_spans = get_image_spans(res)
        v_raw, pred_logsnr, aux = run_model_forward(components, z_t, logsnr_map, base_spans)
        
        # Losses
        # 1. Image Reconstruction (Identity)
        # The output of the unembedder (v_raw) should match the input (z_t)
        # because we aren't doing diffusion logic, just compression logic.
        loss_img = F.mse_loss(v_raw, z_t)
        
        # 2. Metadata Reconstruction
        loss_meta = F.l1_loss(pred_logsnr, logsnr_map)
        
        total_loss = loss_img + 0.1 * loss_meta
        total_loss.backward()
        opt.step()
        scheduler.step()
        
        history.append({
            'step': i, 'res': res, 
            'loss_ae': loss_img.item(), 'loss_meta': loss_meta.item()
        })
        if i % 100 == 0:
            pbar.set_postfix({'ae': f'{loss_img.item():.4f}', 'meta': f'{loss_meta.item():.4f}'})
            
    return pd.DataFrame(history)

def train_denoise(components, config, logger=None):
    mode = config['mode']
    steps = config['steps']
    lambda_coeff = config.get('lambda_coeff', 0.2)
    
    print(f"\n--- Training: Denoise ({mode.upper()}) [Unified List Mode] ---")
    
    # Setup
    model = components[0]
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    scheduler = OneCycleLR(opt, max_lr=5e-4, total_steps=steps, pct_start=0.1)
    
    # Iterator now expected to implement generate_batch_list
    iterator = CompositeIterator(model.text_embed.weight.device, config=config['dataset_mix'])
    
    # We define a helper to get batch size from config or manager
    # Assuming config['buckets'] exists or we pick a fixed BS for list mode
    bs = 8 
    
    history = []
    pbar = tqdm(range(steps), desc=f"train-{mode}")
    
    for i in pbar:
        opt.zero_grad()
        
        # 1. Get Unified Data Lists
        # images_list: List[Tensor], logsnrs_list: List[Tensor], spans_meta: List[Dict]
        # (This supports mix of 32px, 64px, single images, sequences, etc.)
        images_list, logsnrs_list, spans_meta = iterator.generate_batch_list(bs)
        
        # 2. Noise Injection (Map over list)
        z_t_list = []
        v_true_list = []
        
        for img, lsnr in zip(images_list, logsnrs_list):
            # euler_forward_step handles broadcasting if lsnr is [1, H, W]
            z, v, _ = euler_forward_step(img, lsnr)
            z_t_list.append(z)
            v_true_list.append(v)
            
        # 3. Model Prediction
        v_pred_list, pred_logsnr_list, aux_loss = predict_velocity_list(
            components, z_t_list, logsnrs_list, spans_meta, mode
        )
        
        # 4. Loss Computation
        total_loss_v = 0.0
        total_loss_lam = 0.0
        count = len(v_pred_list)
        
        for idx in range(count):
            # MSE on Velocity
            total_loss_v += F.mse_loss(v_pred_list[idx], v_true_list[idx])
            
            # L1 on Lambda (Grounding)
            total_loss_lam += F.l1_loss(pred_logsnr_list[idx], logsnrs_list[idx])
            
        loss_v = total_loss_v / count
        loss_lam = total_loss_lam / count
        
        total_loss = loss_v + lambda_coeff * loss_lam + aux_loss
        
        total_loss.backward()
        opt.step()
        scheduler.step()
        
        # 5. Logging
        stats = {
            'step': i, 
            'loss_total': total_loss.item(),
            'loss_v': loss_v.item(),
            'loss_lambda': loss_lam.item()
        }
        history.append(stats)
        
        if i % 100 == 0:
            pbar.set_postfix({'v': f'{loss_v.item():.4f}', 'lam': f'{loss_lam.item():.4f}'})
            
    return pd.DataFrame(history)

def distill_consistency(components, config, logger=None):
    """
    Phase 2: Distillation.
    """
    mode = config['mode']
    steps = config['distill_steps']
    # Scale down buckets for memory
    buckets = [(r, max(1, b//2)) for r, b in config['buckets']]
    
    print(f"\n--- Distilling: {mode.upper()} ---")
    model = components[0]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scheduler = OneCycleLR(opt, max_lr=1e-4, total_steps=steps, pct_start=0.1)
    
    iterator = CompositeIterator(model.text_embed.weight.device, config=config['dataset_mix'])
    manager = BucketManager(buckets)
    history = []
    
    pbar = tqdm(range(steps), desc=f"distill-{mode}")
    for i in pbar:
        opt.zero_grad()
        res, bs = manager.next_bucket()
        
        # 1. Data & Noise
        x0, _ = iterator.generate_batch(bs, res, num_tiles=4.0) # Ignore data logsnr, we generate trajectory
        base_spans = get_image_spans(res)
        
        # 2. Consistency Trajectory
        # Pick random start/end times
        l_start = torch.rand(bs, device=x0.device) * 5.0 - 5.0 # [-5, 0] approx
        l_end = l_start + 2.0 # short hop
        
        # Map scalars to maps
        map_start = l_start.view(bs, 1, 1, 1).expand(bs, 1, res, res)
        map_end = l_end.view(bs, 1, 1, 1).expand(bs, 1, res, res)
        
        # Start State
        z_start, _, _ = euler_forward_step(x0, map_start)
        
        # 3. Student Predictions
        # A. One Step
        v_1, _, aux1 = predict_velocity(components, z_start, map_start, base_spans, mode)
        z_next_1 = euler_reverse_step(z_start, v_1, map_start, map_end)
        
        # B. Two Step (Target) - detached
        with torch.no_grad():
            # Midpoint
            l_mid = (l_start + l_end) / 2.0
            map_mid = l_mid.view(bs, 1, 1, 1).expand(bs, 1, res, res)
            
            v_mid_1, _, _ = predict_velocity(components, z_start, map_start, base_spans, mode)
            z_mid = euler_reverse_step(z_start, v_mid_1, map_start, map_mid)
            
            v_mid_2, _, _ = predict_velocity(components, z_mid, map_mid, base_spans, mode)
            z_next_target = euler_reverse_step(z_mid, v_mid_2, map_mid, map_end)
            
        # 4. Loss
        loss_cons = F.mse_loss(z_next_1, z_next_target.detach())
        loss_total = loss_cons + aux1
        
        loss_total.backward()
        opt.step()
        scheduler.step()
        
        history.append({'step': i, 'res': res, 'loss_cons': loss_cons.item()})
        if i % 100 == 0:
            pbar.set_postfix({'cons': f'{loss_cons.item():.4f}'})
            
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
def sample_viz_dset(components, iterator, config):
    """
    Stratified Sampling using the Spatial Solver.
    Generates items with random global noise levels, then solves them all to clean.
    """
    model, _, _, _, _ = components
    model.eval()
    
    res = config.get('res', 32)
    n_samples = config.get('num_samples', 8)
    mode = config.get('mode', 'naive')
    device = model.text_embed.weight.device
    
    # 1. Get Data (Ignore iterator noise, we generate our own stratification)
    x0, _ = iterator.generate_batch(n_samples, res, num_tiles=4.0)
    
    # 2. Generate Stratified Start Conditions
    min_snr = config.get('min_logsnr', -4.0)
    max_snr = config.get('max_logsnr', 1.0)
    
    # [B] -> [B, 1, H, W]
    start_vals = torch.rand(n_samples, device=device) * (max_snr - min_snr) + min_snr
    start_vals, sort_idx = torch.sort(start_vals)
    x0 = x0[sort_idx]
    
    logsnr_start_map = start_vals.view(n_samples, 1, 1, 1).expand(-1, -1, res, res)
    
    # 3. Noise Data
    alpha, sigma = logsnr_to_alpha_sigma(logsnr_start_map)
    eps = torch.randn_like(x0)
    z_start = x0 * alpha + eps * sigma
    
    # 4. Solve
    # Everyone marches from their specific start_snr to +10.0
    z_final = spatial_euler_solver(
        components, 
        z_start, 
        logsnr_start_map, 
        target_logsnr=10.0, 
        steps=50, 
        mode=mode, 
        config=config
    )
    
    model.train()
    
    mse = F.mse_loss(z_final, x0, reduction='none').mean(dim=[1,2,3])
    
    return {
        'x0': x0,
        'noisy_input': z_start, # For viz
        'reconstruction': z_final,
        'mse': mse,
        'start_snr': start_vals
    }

@torch.no_grad()
def sample_viz_split_topology(components, iterator, config):
    """
    Split-Screen / Spatial Noise Test.
    Uses the iterator's complex spatial logsnr_map as the starting condition.
    """
    model, _, _, _, _ = components
    model.eval()
    
    res = config.get('res', 32)
    n_samples = config.get('num_samples', 8)
    mode = config.get('mode', 'naive')
    
    # 1. Get Data AND Spatial Map from Iterator
    x0, logsnr_start_map = iterator.generate_batch(n_samples, res, num_tiles=4.0)
    
    # 2. Noise Data (Spatially!)
    alpha, sigma = logsnr_to_alpha_sigma(logsnr_start_map)
    eps = torch.randn_like(x0)
    z_start = x0 * alpha + eps * sigma
    
    # 3. Solve
    # The 'Noisy' pixels traverse from -5 to +10.
    # The 'Clean' pixels traverse from +2 to +10.
    # The solver handles the interpolation per-pixel.
    z_final = spatial_euler_solver(
        components, 
        z_start, 
        logsnr_start_map, 
        target_logsnr=10.0, 
        steps=50, 
        mode=mode, 
        config=config
    )
    
    model.train()
    
    return {
        'x0': x0,
        'noisy_input': z_start,
        'reconstruction': z_final,
        'logsnr_map': logsnr_start_map
    }

@torch.no_grad()
def sample_viz_causal_prefix(components, iterator, config):
    """
    Tests Causal Prefix Generation:
    1. Grabs a sequence (e.g. 4 frames).
    2. Pins frames 0..N-1 as Clean Context (Prefix).
    3. Noises frame N (Suffix).
    4. Solves frame N while attending to the fixed Prefix.
    """
    model, _, _, _, _ = components
    model.eval()
    
    # Config
    mode = config.get('mode', 'naive')
    res = config.get('res', 32)
    seq_len = 4
    n_sequences = 2 # Total batches
    suffix_idx = 3 # The index to generate (0-based) -> Last frame
    
    # 1. Get Data (Unified List)
    # We request a specific sequence structure from the video iterator if available
    # Or just grab items and chunk them if using the generic iterator
    # Assuming iterator returns a flat list where every `seq_len` items are a group.
    
    # Hack: Force iterator to give us sequences if it's a Video iterator
    # For CompositeIterator, we rely on the config passed during init to produce sequences.
    # Here we assume the iterator produces coherent blocks.
    
    flat_images, flat_logsnrs, _ = iterator.generate_batch_list(n_sequences * seq_len)
    
    # We only care about the images, we will synthesize our own noise maps for the test
    # Chunk into sequences
    sequences = [flat_images[i:i+seq_len] for i in range(0, len(flat_images), seq_len)]
    
    results = []
    
    for seq_idx, seq_imgs in enumerate(sequences):
        if len(seq_imgs) < seq_len: continue 
        
        # Setup Lists for Solver
        z_init_list = []
        logsnr_start_list = []
        fixed_data_list = [] # None = Evolve, Tensor = Pin
        
        device = seq_imgs[0].device
        
        for t, img in enumerate(seq_imgs):
            if t < suffix_idx:
                # --- PREFIX (Context) ---
                # State: Clean Image
                z_init_list.append(img) 
                
                # Noise Map: "Clean" (High SNR) implies we trust this input
                # We start it at Target SNR so the solver doesn't try to move it much anyway,
                # but 'fixed_data' ensures it stays bit-exact.
                l_map = torch.full((1, img.shape[1], img.shape[2]), 10.0, device=device)
                logsnr_start_list.append(l_map)
                
                # Constraint: PIN THIS
                fixed_data_list.append(img)
                
            else:
                # --- SUFFIX (Target) ---
                # State: Random Noise
                # We need to construct the noise level we WANT to start solving from.
                start_snr = -4.0 # Standard noisy starting point
                
                l_map = torch.full((1, img.shape[1], img.shape[2]), start_snr, device=device)
                logsnr_start_list.append(l_map)
                
                # Create noisy latent z_T
                alpha, sigma = logsnr_to_alpha_sigma(l_map)
                eps = torch.randn_like(img)
                z_t = img * alpha + eps * sigma # Or just pure noise if alpha~0
                
                z_init_list.append(z_t)
                
                # Constraint: None (Let it evolve)
                fixed_data_list.append(None)
        
        # Run Solver on this sequence group
        # The solver processes the list as one "batch" of distinct items
        # but the attention masking (via spans/group_id) binds them.
        z_final_list = spatial_euler_solver(
            components, 
            z_init_list, 
            logsnr_start_list, 
            target_logsnr=10.0, 
            steps=50, 
            mode=mode, 
            config=config,
            fixed_data=fixed_data_list # <--- The magic key
        )
        
        # Store result (just the target frame vs GT)
        target_gt = seq_imgs[suffix_idx]
        target_recon = z_final_list[suffix_idx]
        
        results.append({
            'gt': target_gt,
            'recon': target_recon,
            'context': seq_imgs[suffix_idx-1] # Previous frame for Ref
        })
        
        if len(results) >= n_sequences: break

    # Plot
    fig, axes = plt.subplots(len(results), 3, figsize=(10, 4*len(results)))
    if len(results) == 1: axes = axes.reshape(1, -1)
    
    for i, res in enumerate(results):
        # 1. Context (Last Prefix Frame)
        axes[i, 0].imshow(res['context'].permute(1,2,0).cpu().numpy())
        axes[i, 0].set_title("Context (Frame t-1)")
        axes[i, 0].axis('off')
        
        # 2. GT Target
        axes[i, 1].imshow(res['gt'].permute(1,2,0).cpu().numpy())
        axes[i, 1].set_title("Ground Truth (Frame t)")
        axes[i, 1].axis('off')
        
        # 3. Generated Target
        axes[i, 2].imshow(res['recon'].permute(1,2,0).cpu().numpy())
        axes[i, 2].set_title("Generated (Frame t)")
        axes[i, 2].axis('off')
        
    model.train()
    return fig

@torch.no_grad()
def spatial_euler_solver(components, z_list, logsnr_start_list, target_logsnr, steps, mode, config, fixed_data=None):
    """
    Unified solver for Tensors OR Lists of Tensors.
    Supports 'Fixed' latents for Inpainting/Prefix generation.
    
    Args:
        z_list: List[Tensor] (Initial state, mixed noise)
        logsnr_start_list: List[Tensor] (Starting noise map per item)
        fixed_data: Optional List[Tensor]. If an entry is NOT None, that latent 
                    is reset to this value at every step (e.g. clean prefix).
    """
    device = z_list[0].device
    # Create base spans for the resolutions present in the list
    # (Recomputed per step effectively via predict_velocity wrapper, but we need resolution metadata)
    # Actually, predict_velocity_list constructs spans internally or expects them.
    # We need to construct spans for the current z state.
    
    # Pre-compute spans since resolution doesn't change during sampling
    from bench_multires_zc import get_image_spans # Ensure available
    spans_meta = []
    for i, z in enumerate(z_list):
        H = z.shape[-1]
        spans_meta.append({
            'type': 'latent', 'len': (H//2)**2, 'shape': (H//2, H//2), 
            'causal': True, 'group_id': i, 'id': f"sample_{i}"
        })

    # Time parameter tau: 0.0 -> 1.0
    taus = torch.linspace(0.0, 1.0, steps + 1, device=device)

    # Helper to broadcast target
    def get_target_map(start_map):
        if isinstance(target_logsnr, (float, int)):
            return torch.full_like(start_map, target_logsnr)
        return target_logsnr # Assume tensor match

    target_maps = [get_target_map(m) for m in logsnr_start_list]

    for i in range(steps):
        tau_curr = taus[i]
        tau_next = taus[i+1]

        # 1. Interpolate Schedule (Per Item)
        lsnr_curr_list = []
        lsnr_next_list = []
        for start, end in zip(logsnr_start_list, target_maps):
            lsnr_curr_list.append((1 - tau_curr) * start + tau_curr * end)
            lsnr_next_list.append((1 - tau_next) * start + tau_next * end)

        # 2. Predict Velocity (Unified List Pass)
        v_pred_list, _, _ = predict_velocity_list(components, z_list, lsnr_curr_list, spans_meta, mode)

        # 3. Spatial Euler Step & Constraint Enforcement
        z_next_list = []
        for idx, (z, v, l_curr, l_next) in enumerate(zip(z_list, v_pred_list, lsnr_curr_list, lsnr_next_list)):
            
            # Check Constraint: If this frame is fixed (Prefix), don't step it.
            if fixed_data is not None and fixed_data[idx] is not None:
                # Keep it pinned to the ground truth/clean state
                z_next_list.append(fixed_data[idx])
            else:
                # Update Suffix
                z_new = euler_reverse_step(z, v, l_curr, l_next)
                z_next_list.append(z_new)
        
        z_list = z_next_list
        
    return [z.clamp(0, 1) for z in z_list]

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
            'path': "C:/dox/recordings/rl_capture/capture_run_1760343426/videos",  # Absolute or Relative path
            
            # The heart of the causal architecture:
            # Defines a single sequence of 4 frames [t-3, t-2, t-1, t]
            'sequence_structure': [
                # --- Context Frames (Past) ---
                # Low Res (32px), High SNR (Clean-ish), Mild Split Topology
                # These provide semantic grounding without burning compute on pixels.
                {
                    'res': 32, 
                    'noise_mode': 'split', 
                    'noise_params': {'min_snr': 2.0, 'max_snr': 6.0, 'angle_range_deg': 15.0}
                },
                {
                    'res': 32, 
                    'noise_mode': 'split', 
                    'noise_params': {'min_snr': 2.0, 'max_snr': 6.0, 'angle_range_deg': 15.0}
                },
                {
                    'res': 32, 
                    'noise_mode': 'split', 
                    'noise_params': {'min_snr': 2.0, 'max_snr': 6.0, 'angle_range_deg': 15.0}
                },
                
                # --- Target Frame (Present) ---
                # High Res (64px), Full Noise Range, Aggressive Split Topology
                # This is the actual denoising task.
                {
                    'res': 64, 
                    'noise_mode': 'split', 
                    'noise_params': {'min_snr': -5.0, 'max_snr': 5.0, 'angle_range_deg': 45.0}
                }
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
            fig_prefix = sample_viz_causal_prefix(components, val_iterator, {'mode':'naive'})
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
    print("\n🔮 Distillation Phase...")
    model.param_load(params_n)
    df_dist_n = distill_consistency(components, config_n)
    
    print("🎨 Sampling Naive (Post-Distill)...")
    res_nd = sample_viz_dset(components, val_iterator, {'mode':'naive', 'res':32})
    plot_dset_reconstruction(res_nd, logger, "naive_distill_stratified")
    
    model.param_load(params_f)
    df_dist_f = distill_consistency(components, config_f)
    
    print("🎨 Sampling Factorized (Post-Distill)...")
    res_fd = sample_viz_dset(components, val_iterator, {'mode':'factorized', 'res':32})
    plot_dset_reconstruction(res_fd, logger, "factorized_distill_stratified")
    
    plot_losses(df_dist_n, df_dist_f, logger, metric='loss_cons', title='Consistency Loss')
    """
    print(f"\n✅ Experiment Complete. Results in {logger.run_dir}")