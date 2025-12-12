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

def run_model_forward(components, z, logsnr_map, spans):
    """
    Low-level wrapper: Embedding -> Transformer -> Unembedding.
    Returns raw outputs (v_raw, pred_logsnr) and aux_loss.
    """
    model, span_embedder, span_unembedder, _, page_table = components
    B = z.shape[0]
    device = z.device
    
    # 1. Metadata Construction
    batch_spans_meta = []
    images = [z[i] for i in range(B)]
    # Handle spatial logsnr broadcasting for the embedder list
    logsnr_list = [logsnr_map[i] for i in range(B)]
    
    for i in range(B):
        item_spans = [s.copy() for s in spans]
        for s in item_spans: s['id'] = f"req_{i}"
        batch_spans_meta.extend(item_spans)
        
    # 2. Embed
    z_flat, span_objects, _ = span_embedder.embed(
        batch_spans_meta, 
        text_tokens=[None]*B, 
        images=images, 
        logsnr_maps=logsnr_list
    )
    
    # 3. Topology
    topo_embeds, _ = render_topology_embeddings(batch_spans_meta, 3, device)
    
    # 4. Masking (ZC Mode - Identity Heap)
    L_total = z_flat.shape[0]
    block_size = page_table.block_size
    num_blocks = (L_total + block_size - 1) // block_size
    flat_page_table = torch.arange(num_blocks, device=device, dtype=torch.long)
    
    block_masks = build_dual_masks(
        span_objects, topo_embeds, topo_embeds,
        page_table, flat_page_table, None
    )
    
    # 5. Transformer
    base_ref_len = 64.0
    rope_scale = max(1.0, L_total / base_ref_len)
    
    z_out, aux_loss = model(
        z_flat.unsqueeze(0),
        topo_embeds.unsqueeze(0),
        slot_mapping=None,
        block_masks=block_masks,
        scale=rope_scale
    )
    
    # 6. Unembed
    decoded = span_unembedder.decode(z_out.squeeze(0), span_objects)
    
    # Stack results
    v_raw = torch.stack([d['image_vpreds'] for d in decoded])
    pred_logsnr = torch.stack([d['image_logsnrs'] for d in decoded])
    
    return v_raw, pred_logsnr, aux_loss

def predict_velocity(components, z, logsnr_map, spans, mode='naive'):
    """
    High-level prediction logic.
    Applies factorization if mode='factorized'.
    Returns final velocity, predicted logsnr (for loss), and aux loss.
    """
    v_raw, pred_logsnr, aux_loss = run_model_forward(components, z, logsnr_map, spans)
    
    if mode == 'factorized':
        # Factorized: v_final = v_raw * sigma(predicted_noise_level)
        # This allows the model to output 'direction' and 'magnitude' separately
        sigma_p = torch.sqrt(torch.sigmoid(-pred_logsnr))
        v_final = v_raw * sigma_p
    else:
        # Naive: v_raw is the velocity.
        # Note: We still return pred_logsnr because we want to train it!
        v_final = v_raw
        
    return v_final, pred_logsnr, aux_loss

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
    """
    Phase 1: Flow Matching / Diffusion Training.
    Handles 'Naive' and 'Factorized' modes via config.
    """
    mode = config['mode']
    steps = config['steps']
    buckets = config['buckets']
    lambda_coeff = config.get('lambda_coeff', 0.2)
    
    print(f"\n--- Training: Denoise ({mode.upper()}) ---")
    model = components[0]
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    scheduler = OneCycleLR(opt, max_lr=5e-4, total_steps=steps, pct_start=0.1)
    
    iterator = CompositeIterator(model.text_embed.weight.device, config=config['dataset_mix'])
    manager = BucketManager(buckets)
    history = []
    
    pbar = tqdm(range(steps), desc=f"train-{mode}")
    for i in pbar:
        opt.zero_grad()
        res, bs = manager.next_bucket()
        
        # 1. Data (Spatial Noise Map included!)
        x0, logsnr_map = iterator.generate_batch(bs, res, num_tiles=4.0)
        
        # 2. Noise
        z_t, v_true, _ = euler_forward_step(x0, logsnr_map)
        
        # 3. Predict
        base_spans = get_image_spans(res)
        v_pred, pred_logsnr, aux_loss = predict_velocity(components, z_t, logsnr_map, base_spans, mode)
        
        # 4. Losses
        # A. Velocity Matching (MSE or NLL)
        # For simplicity, using MSE. If probabilistic needed, switch to NLL here.
        loss_v = F.mse_loss(v_pred, v_true)
        
        # B. Lambda Reconstruction (Anti-Cheat / Grounding)
        # Even Naive mode trains this, though it doesn't use it for v scaling.
        # This ensures the model "knows" the noise level.
        loss_lambda = F.l1_loss(pred_logsnr, logsnr_map)
        
        total_loss = loss_v + lambda_coeff * loss_lambda + aux_loss
        
        total_loss.backward()
        opt.step()
        scheduler.step()
        
        # 5. Logging
        stats = {
            'step': i, 'res': res,
            'loss_total': total_loss.item(),
            'loss_v': loss_v.item(),
            'loss_lambda': loss_lambda.item()
        }
        history.append(stats)
        
        if i % 100 == 0:
            pbar.set_postfix({'v': f'{loss_v.item():.4f}', 'lam': f'{loss_lambda.item():.4f}'})
            
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
# 5. Main Execution
# ==============================================================================

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    logger = ExperimentLogger(output_dir="./experiments_refactor")
    device = torch.device('cuda')

    # --- Configuration ---
    # Define the curriculum:
    # 40% Uniform Checkerboard (Standard)
    # 60% Split-Screen Torus (Hard Geometry + Boundary Conditions)
    dataset_mix = {
        'uniform_checker': {
            'type': 'checkerboard',
            'ratio': 0.4,
            'noise_mode': 'uniform',
            'noise_params': {'min_snr': -4.0, 'max_snr': 2.0}
        },
        'split_torus': {
            'type': 'torus',
            'ratio': 0.6,
            'noise_mode': 'split',
            'noise_params': {'min_snr': -5.0, 'max_snr': 2.0, 'angle_range_deg': 45.0}
        }
    }

    base_config = {
        'ae_steps': 1000,
        'steps': 1000,
        'distill_steps': 1000,
        'buckets': [(16, 64), (32, 32)],
        'lambda_coeff': 0.2, # Regularization strength for lambda reconstruction
        'dataset_mix': dataset_mix
    }

    # --- Model Init ---
    print("🔧 Initializing ZC Model Stack...")
    embed_dim = 256
    model = coolerLDTformer(dim=embed_dim, depth=8, num_heads=8, topo_dim=3).to(device)
    model = torch.compile(model, dynamic=True)
    
    span_emb = SpanEmbedder(model.text_embed, model.patch_embedder)
    span_unemb = SpanUnembedder(model.text_head, model.patch_unembedder)
    
    # Dummy PageTable (needed for mask construction interface, though unused in logic)
    page_table = PageTable(num_blocks=1024, block_size=128, max_batch_size=128, max_logical_blocks=1024, device=device)
    
    components = (model, span_emb, span_unemb, None, page_table)

    # --- Run A: Naive Mode ---
    print("🚀 Starting Run A: Naive")
    model.param_init()
    config_n = {**base_config, 'mode': 'naive'}
    
    df_ae_n = train_autoembed(components, config_n)
    df_train_n = train_denoise(components, config_n)
    params_n = model.dump()
    
    # --- Run B: Factorized Mode ---
    print("🚀 Starting Run B: Factorized")
    model.flush()
    model.param_init()
    config_f = {**base_config, 'mode': 'factorized'}
    
    df_ae_f = train_autoembed(components, config_f)
    df_train_f = train_denoise(components, config_f)
    params_f = model.dump()

    # --- Plotting ---
    print("\n📈 Plotting Results...")
    plot_losses(df_train_n, df_train_f, logger, metric='loss_v', title='Velocity Prediction Loss (MSE)')
    plot_losses(df_train_n, df_train_f, logger, metric='loss_lambda', title='Lambda Reconstruction Loss (L1)')
    
    # --- Distillation (Optional) ---
    print("\n🔮 Distillation Phase...")
    model.param_load(params_n)
    df_dist_n = distill_consistency(components, config_n)
    
    model.param_load(params_f)
    df_dist_f = distill_consistency(components, config_f)
    
    plot_losses(df_dist_n, df_dist_f, logger, metric='loss_cons', title='Consistency Loss')

    print(f"\n✅ Experiment Complete. Results in {logger.run_dir}")