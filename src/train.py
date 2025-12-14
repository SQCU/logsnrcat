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
    build_dual_masks, ContextBlock, render_topology_embeddings
)
from .data import CompositeIterator
from .utils import PageTable

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
    model, span_embedder, span_unembedder, page_table = components
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
        page_table, flat_page_table, None,
        window_size=getattr(model, 'window_size', 10.0)
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
    model, _, _, _ = components
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
    # Optimizer params from config
    lr = config.get('lr', 5e-4)
    wd = config.get('weight_decay', 0.1)
    max_lr = config.get('max_lr', lr)
    pct_start = config.get('pct_start', 0.1)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
    fused=True)
    scheduler = OneCycleLR(opt, max_lr=max_lr, total_steps=steps, pct_start=pct_start)
    iterator = CompositeIterator(model.text_embed.weight.device, config=config['dataset_mix'])
    history = []
        # BF16 usually doesn't need a GradScaler, but FP16 does.
    # We can use it conditionally.
    dtype = config.get('dtype', torch.float32)
    use_amp = (dtype == torch.bfloat16) or (dtype == torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16)) 
    pbar = tqdm(range(steps), desc="train-ae")
    for i in pbar:
        opt.zero_grad()
            
               # --- AUTOCAST BLOCK ---
        with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
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
        if i % config.get('log_interval', 100) == 0: pbar.set_postfix({'ae': f'{loss_img:.4f}'})
            
    return pd.DataFrame(history)

def train_denoise(components, config, logger=None):
    mode = config['mode']; steps = config['steps']; lambda_coeff = config.get('lambda_coeff', 0.2)
    bs = config.get('batch_size', 8)
    
    # Optimizer params from config
    lr = config.get('lr', 5e-4)
    wd = config.get('weight_decay', 0.1)
    max_lr = config.get('max_lr', lr)
    pct_start = config.get('pct_start', 0.1)
    
    model = components[0]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
    fused=True)
    scheduler = OneCycleLR(opt, max_lr=max_lr, total_steps=steps, pct_start=pct_start)
    iterator = CompositeIterator(model.text_embed.weight.device, config=config['dataset_mix'])
    
    history = []
    pbar = tqdm(range(steps), desc=f"train-{mode}")
    
    # BF16 usually doesn't need a GradScaler, but FP16 does.
    # We can use it conditionally.
    dtype = config.get('dtype', torch.float32)
    use_amp = (dtype == torch.bfloat16) or (dtype == torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16)) 

    for i in pbar:
        opt.zero_grad()
                # --- AUTOCAST BLOCK ---
        with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
            # 1. Clean Data
            clean_blocks = iterator.generate_batch_list(bs)
            
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
        
        if i % config.get('log_interval', 100) == 0: 
            # Log average of this batch for progress bar
            avg_v = sum(s['loss'] for s in step_stats) / max(1, len(step_stats))
            pbar.set_postfix({'v': f'{avg_v:.4f}'})
            
    return pd.DataFrame(history)
