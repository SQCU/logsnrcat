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
from .data_iterator import CompositeIterator
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

def train_autoembed(components, config, iterator, logger=None):
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
                    noise = torch.randn_like(b.content)
                    z_t, _, _ = euler_forward_step(b.content, b.logsnr, noise)
                    
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
            latent_cursor = 0
            for j, res in enumerate(decoded):
                if 'image_vpreds' in res:
                    loss_img += F.mse_loss(res['image_vpreds'], target_imgs[latent_cursor])
                    loss_meta += F.l1_loss(res['image_logsnrs'], target_lsnrs[latent_cursor])
                    latent_cursor += 1
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

def train_denoise(components, config, iterator, logger=None):
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

    history = []
    # Pre-fetch bucketing flag to avoid lookup in loop
    bucketing_enabled = config['training']['bucketing']['enabled']
    dtype = config['dtype']
    use_amp = (dtype == torch.bfloat16) or (dtype == torch.float16)
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
                    noise = torch.randn_like(b.content)
                    z, v, _ = euler_forward_step(b.content, b.logsnr, noise)
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
                    
            # 3. Model Forward (Raw Decoded Output)
            # we have text logits now so we are no longer using the get v from blocks wrapper
            decoded_results, aux_loss = run_model_forward(components, noisy_blocks)
            
            # 4. Loss Calculation
            loss_v_accum = 0.0
            loss_lam_accum = 0.0
            loss_text_accum = 0.0
            
            valid_latent_samples = 0
            valid_text_samples = 0
            step_stats = []
            
            # We iterate through the aligned lists of Blocks and Results
            latent_cursor = 0
            
            for block, res in zip(noisy_blocks, decoded_results):
                # --- A. Latent Diffusion Loss ---
                if block.type == 'latent':
                    if 'image_vpreds' in res:
                        v_raw = res['image_vpreds']
                        pred_l = res['image_logsnrs']
                        
                        # Factorization Logic (if enabled)
                        if mode == 'factorized':
                            sigma_p = torch.sqrt(torch.sigmoid(-pred_l))
                            v_pred = v_raw * sigma_p
                        else:
                            v_pred = v_raw
                            
                        target_v = targets_v[latent_cursor]
                        target_l = targets_l[latent_cursor]
                        latent_cursor += 1
                        
                        # Velocity MSE
                        sq_err_v = F.mse_loss(v_pred, target_v, reduction="none")
                        loss_v_accum += sq_err_v.mean()
                        
                        # Lambda L1
                        loss_lam_accum += F.l1_loss(pred_l, target_l)
                        
                        # Stats
                        step_stats.append({
                            'step': i,
                            'source': getattr(block, 'source', 'unknown'),
                            'type': 'latent',
                            'loss': sq_err_v.mean().detach().item(),
                            'loss_var': sq_err_v.var().detach().item(),
                            'logsnr': target_l.mean().detach().item(),
                            'resolution': block.content.shape[-1] * block.content.shape[-2]
                        })
                        valid_latent_samples += 1
                
                # --- B. Text Autoregressive Loss ---
                elif block.type == 'text':
                    if 'text_logits' in res:
                        logits = res['text_logits'] # [L, Vocab]
                        tokens = block.content      # [L]
                        
                        # Shift: Logit[t] predicts Token[t+1]
                        # Input: A B C
                        # Target: B C D (where D is from next block? For MVP we ignore cross-block prediction)
                        
                        shift_logits = logits[:-1, :].contiguous()
                        shift_targets = tokens[1:].contiguous()
                        
                        if shift_targets.numel() > 0:
                            loss_t = F.cross_entropy(shift_logits, shift_targets)
                            loss_text_accum += loss_t
                            
                            step_stats.append({
                                'step': i,
                                'source': getattr(block, 'source', 'text'),
                                'type': 'text',
                                'loss': loss_t.detach().item(),
                                'loss_var': 0.0 # Placeholder
                            })
                            valid_text_samples += 1

            # Normalize Accumulators
            if valid_latent_samples > 0:
                loss_v_accum /= valid_latent_samples
                loss_lam_accum /= valid_latent_samples
            
            if valid_text_samples > 0:
                loss_text_accum /= valid_text_samples
            
            # Weighted Sum
            total_loss = loss_v_accum + (lambda_coeff * loss_lam_accum) + loss_text_accum + aux_loss
            
        if dtype == torch.float16:
            scaler.scale(total_loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
        else:
            total_loss.backward()
            opt.step()
            scheduler.step()
            
        history.extend(step_stats)
        
        if i % config['logging']['log_interval'] == 0: 
            pbar.set_postfix({
                'v': f'{loss_v_accum:.3f}', 
                'txt': f'{loss_text_accum:.3f}'
            })
            
    return pd.DataFrame(history)
