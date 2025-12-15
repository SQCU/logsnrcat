# src/train.py - Training loops, loss, schedule
import math
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR

from .model import (
    coolerLDTformerZC, SpanEmbedder, SpanUnembedder, 
    build_dual_masks, ContextBlock, render_topology_embeddings, PageTable
)
from .data import CompositeIterator
from .utils import run_model_forward, predict_velocity_from_blocks
from .sample import euler_forward_step, euler_reverse_step
from .config import sanitize_config

def calculate_global_max_resolution(config: Dict[str, Any]) -> int:
    """
    Scans config to find the maximum resolution required for video caching.
    Considers both explicit bucket definitions and sequence relative scaling.
    """
    max_res = 32 # Floor
    
    # 1. Check Buckets (if enabled)
    bucketing = config['training'].get('bucketing', {'enabled': False})
    if bucketing.get('enabled', False):
        buckets = bucketing.get('image_buckets', [])
        if buckets:
            max_bucket_res = max(b['resolution'] for b in buckets)
        else:
            max_bucket_res = 32
    else:
        max_bucket_res = 0 # Not driving resolution
        
    # 2. Check Sequence Structures
    # We need to find the max 'res' (absolute) OR max 'relative_res'
    dataset_mix = config.get('dataset_mix', {})
    
    max_seq_res_abs = 0
    max_seq_rel = 1.0
    
    for split_name, split_cfg in dataset_mix.items():
        if split_cfg.get('type') == 'video':
            params = split_cfg.get('params', {}) or {} # Handle None
            seq_struct = params.get('sequence_structure', [])
            for frame in seq_struct:
                # Track absolute max defined in config
                max_seq_res_abs = max(max_seq_res_abs, frame.get('res', 32))
                # Track relative max
                max_seq_rel = max(max_seq_rel, frame.get('relative_res', 1.0))
    
    # 3. Compute Global Max
    if bucketing.get('enabled', False):
        # If bucketing, the demand is Bucket * Relative
        # We assume the worst case: Biggest Bucket * Biggest Relative Factor
        res_from_buckets = int(max_bucket_res * max_seq_rel)
        # Ensure we cover at least that
        max_res = max(max_res, res_from_buckets)
    else:
        # If no bucketing, we rely on the absolute definitions
        max_res = max(max_res, max_seq_res_abs)
        
    # Alignment (Optional, but safe)
    if max_res % 2 != 0: max_res += 1
    
    return max_res

# ==============================================================================
# 3. Training Loops (Config Driven)
# ==============================================================================

from .bucket_manager import build_bucket_manager_from_config

def train_autoembed(components, config, logger=None):
    # 1. Enforce Dictionary Type
    config = sanitize_config(config)
    
    # 2. Strict Access (No defaults allowed here - define them in Pydantic schema)
    mode = config['training']['mode']
    steps = config['training']['ae_steps']
    bs = config['training']['batch_size']
    
    
    # 4. Optimizer Params (Strict)
    opt_cfg = config['training']['ae_optimizer'] # distinct AE optimizer config
    lr = opt_cfg['lr']
    wd = opt_cfg['weight_decay']
    max_lr = opt_cfg['max_lr']
    pct_start = opt_cfg['pct_start']

    
    print(f"\n--- Training: Auto-Encoder ({mode.upper()}) ---")
    model = components[0]
    # 3. Build Manager
    # We explicitly look up the stride from the model instance, or the config dict
    model_stride = model.patch_embedder.stride
    bucket_mgr = build_bucket_manager_from_config(config, model_stride=model_stride)
    

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
    fused=True)
    scheduler = OneCycleLR(opt, max_lr=max_lr, total_steps=steps, pct_start=pct_start)
    # FIX: Calculate cache size
    caching_res = calculate_global_max_resolution(config)
    
    iterator = CompositeIterator(
        model.text_embed.weight.device, 
        config=config['dataset_mix'],
        caching_resolution=caching_res # Pass it down
    )
    history = []
    bucketing_enabled = config['training']['bucketing']['enabled']
    # BF16 usually doesn't need a GradScaler, but FP16 does.
    # We can use it conditionally.
    dtype = config['dtype']
    use_amp = (dtype == torch.bfloat16) or (dtype == torch.float16)
    # FIX: Use new torch.amp API
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16)) 
    pbar = tqdm(range(steps), desc="train-ae")
    for i in pbar:
        opt.zero_grad()
            
               # --- AUTOCAST BLOCK ---
        with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
            # 1. Get Clean Blocks
            # 1. NEW: Sample Bucket
            if bucketing_enabled:
                bucket = bucket_mgr.sample_bucket()
                curr_res = bucket.resolution
                curr_bs = bucket.batch_size
            else:
                curr_res = 32 # Fallback
                curr_bs = bs
            
            # 2. Generate Data with Dynamic Resolution
            clean_blocks = iterator.generate_batch_list(curr_bs, resolution=curr_res)
            
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
        if dtype == torch.float16:
            scaler.scale(total_loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
        else:
            # BF16 or FP32: Standard backward
            total_loss.backward()
            opt.step()
            scheduler.step()
        
        history.append({'step': i,'loss': loss_img.item(), 'loss_ae': loss_img.item() if count else 0})
        if i % config['logging']['log_interval'] == 0: pbar.set_postfix({'ae': f'{loss_img:.4f}'})
            
    return pd.DataFrame(history)

def train_denoise(components, config, logger=None):
    # 1. Enforce Dictionary Type
    config = sanitize_config(config)
    
    # 2. Strict Access (No defaults allowed here - define them in Pydantic schema)
    mode = config['training']['mode']
    steps = config['training']['ae_steps']
    bs = config['training']['batch_size']

    lambda_coeff = config['training']['lambda_coeff']
    
    
    # 4. Optimizer Params (Strict)
    opt_cfg = config['training']['ae_optimizer'] # distinct AE optimizer config
    lr = opt_cfg['lr']
    wd = opt_cfg['weight_decay']
    max_lr = opt_cfg['max_lr']
    pct_start = opt_cfg['pct_start']


    
    print(f"\n--- Training: Denoiser ({mode.upper()}) ---")
    model = components[0]
    # 3. Build Manager
    # We explicitly look up the stride from the model instance, or the config dict
    model_stride = model.patch_embedder.stride
    bucket_mgr = build_bucket_manager_from_config(config, model_stride=model_stride)
    

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
    fused=True)
    scheduler = OneCycleLR(opt, max_lr=max_lr, total_steps=steps, pct_start=pct_start)
    # FIX: Calculate cache size
    caching_res = calculate_global_max_resolution(config)
    
    iterator = CompositeIterator(
        model.text_embed.weight.device, 
        config=config['dataset_mix'],
        caching_resolution=caching_res # Pass it down
    )
    history = []
    # Pre-fetch bucketing flag to avoid lookup in loop
    bucketing_enabled = config['training']['bucketing']['enabled']
    # BF16 usually doesn't need a GradScaler, but FP16 does.
    # We can use it conditionally.
    dtype = config['dtype']
    use_amp = (dtype == torch.bfloat16) or (dtype == torch.float16)
    # FIX: Use new torch.amp API
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16)) 
    pbar = tqdm(range(steps), desc="train-ae")
    for i in pbar:
        opt.zero_grad()
                # --- AUTOCAST BLOCK ---
        with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
            # 1. Clean Data
            # 1. NEW: Sample Bucket
            if bucketing_enabled:
                bucket = bucket_mgr.sample_bucket()
                curr_res = bucket.resolution
                curr_bs = bucket.batch_size
            else:
                curr_res = 32 # Fallback
                curr_bs = bs   
            
            # 2. Generate Data with Dynamic Resolution
            clean_blocks = iterator.generate_batch_list(curr_bs, resolution=curr_res)
            
            # 2. Noise Injection
            noisy_blocks = []
            targets_v = []
            targets_l = []
            
            for b in clean_blocks:
                if b.type == 'latent':
                    z, v, _ = euler_forward_step(b.content, b.logsnr)
                    # Pass source/id/metadata through to noisy block
                    noisy_blocks.append(ContextBlock(
                        content=z, logsnr=b.logsnr, type='latent', causal=b.causal,
                        shape_meta=b.shape_meta, group_id=b.group_id, id=b.id,
                        source=getattr(b, 'source', 'unknown') # Capture source tag
                    ))
                    targets_v.append(v)
                    targets_l.append(b.logsnr)
                else:
                    noisy_blocks.append(b)
                    
            # 3. Predict
            v_preds, l_preds, aux = predict_velocity_from_blocks(components, noisy_blocks, mode)
            
            # 4. Loss & Harvest
            loss_v_accum = 0.0
            loss_lam_accum = 0.0
            valid_samples = 0
            step_stats = []
            
            # Iterate over results to compute gradient AND harvest stats
            for j, (block, vp, lp) in enumerate(zip(noisy_blocks, v_preds, l_preds)):
                if vp is not None:
                    # -- A. Velocity Loss (Exploded View) --
                    # [C, H, W] -> [C, H, W]
                    # Calculate elementwise squared error
                    sq_err_v = F.mse_loss(vp,targets_v[valid_samples], reduction="none")
                    
                    # 1. For Gradient (Mean Reduction)
                    loss_val = sq_err_v.mean()
                    loss_v_accum += loss_val
                    
                    # 2. For Stats (Variance + Detached Mean)
                    # We harvest the "shape" of the error distribution for this item
                    stat_mse = loss_val.detach().item()
                    stat_var = sq_err_v.var().detach().item()
                    
                    # -- B. Lambda Loss --
                    loss_lam_val = F.l1_loss(lp, targets_l[valid_samples])
                    loss_lam_accum += loss_lam_val
                    
                    # -- C. Metadata Extraction --
                    # Calculate resolution area
                    h_res = block.shape_meta[0] * block.shape_meta[1]
                    # Mean LogSNR (Time proxy)
                    h_lsnr = targets_l[valid_samples].mean().item()
                    
                    step_stats.append({
                        'step': i,
                        'source': getattr(block, 'source', 'unknown'),
                        'resolution': h_res,
                        'logsnr': h_lsnr,
                        'loss': stat_mse,
                        'loss_var': stat_var,
                        'loss_lambda': loss_lam_val.detach().item()
                    })
                    
                    valid_samples += 1
            
            if valid_samples > 0:
                loss_v_accum /= valid_samples
                loss_lam_accum /= valid_samples
            
            total_loss = loss_v_accum + lambda_coeff * loss_lam_accum + aux
                # --- BACKWARD ---
        if dtype == torch.float16:
            scaler.scale(total_loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
        else:
            # BF16 or FP32: Standard backward
            total_loss.backward()
            opt.step()
            scheduler.step()
            
        # Extend history with per-item stats (Rows = Batch Size * Steps)
        history.extend(step_stats)
        
        if i % config['logging']['log_interval']== 0: 
            # Log average of this batch for progress bar
            avg_v = sum(s['loss'] for s in step_stats) / max(1, len(step_stats))
            pbar.set_postfix({'v': f'{avg_v:.4f}'})
            
    return pd.DataFrame(history)
