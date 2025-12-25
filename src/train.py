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

# src/train.py - Add near top

class OnlineVarianceTracker:
    """
    Tracks per-logsnr-bucket variance online using Welford's algorithm.
    Uses these statistics to normalize gradients in real-time.
    """
    def __init__(
        self, 
        num_buckets: int = 20,
        snr_min: float = -4.0,
        snr_max: float = 6.0,
        ema_decay: float = 0.99,
        warmup_steps: int = 100,
        device: torch.device = None,
        **kwargs  # Absorbs 'enabled' and any future config fields
    ):
        self.num_buckets = num_buckets
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.ema_decay = ema_decay
        self.warmup_steps = warmup_steps
        self.device = device or torch.device('cuda')
        
        # Bucket edges
        self.bucket_edges = torch.linspace(snr_min, snr_max, num_buckets + 1, device=self.device)
        self.bucket_centers = (self.bucket_edges[:-1] + self.bucket_edges[1:]) / 2
        
        # Running stats (EMA)
        self.running_mean = torch.ones(num_buckets, device=self.device)
        self.running_var = torch.ones(num_buckets, device=self.device)
        self.counts = torch.zeros(num_buckets, device=self.device)
        self.step = 0
        
    def get_bucket_indices(self, logsnr_map: torch.Tensor) -> torch.Tensor:
        """Maps logsnr values to bucket indices [0, num_buckets-1]"""
        # Clamp to range
        clamped = logsnr_map.clamp(self.snr_min, self.snr_max - 1e-6)
        # Normalize to [0, num_buckets)
        normalized = (clamped - self.snr_min) / (self.snr_max - self.snr_min)
        indices = (normalized * self.num_buckets).long().clamp(0, self.num_buckets - 1)
        return indices
    
    @torch.no_grad()
    def update(self, logsnr_map: torch.Tensor, sq_err: torch.Tensor):
        self.step += 1
        
        # sq_err is [*, C, H, W], logsnr_map is [*, 1, H, W]
        # Reduce sq_err channels to match logsnr_map's singleton channel
        if sq_err.shape[1] != logsnr_map.shape[1]:
            sq_err = sq_err.mean(dim=1, keepdim=True)
        
        # Broadcast logsnr to match sq_err if needed, then flatten both
        logsnr_broadcast = logsnr_map.expand_as(sq_err)
        
        logsnr_flat = logsnr_broadcast.reshape(-1)
        err_flat = sq_err.reshape(-1)
        
        bucket_idx = self.get_bucket_indices(logsnr_flat)
        # Update each bucket
        for b in range(self.num_buckets):
            mask = (bucket_idx == b)
            if mask.sum() == 0:
                continue
                
            bucket_err = err_flat[mask]
            batch_mean = bucket_err.mean()
            batch_var = bucket_err.var() if mask.sum() > 1 else self.running_var[b]
            
            # EMA update
            self.running_mean[b] = self.ema_decay * self.running_mean[b] + (1 - self.ema_decay) * batch_mean
            self.running_var[b] = self.ema_decay * self.running_var[b] + (1 - self.ema_decay) * batch_var
            self.counts[b] += mask.sum()
    
    def get_weight_map(self, logsnr_map: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        if self.step < self.warmup_steps:
            return torch.ones(target_shape, device=logsnr_map.device, dtype=logsnr_map.dtype)
        
        logsnr_flat = logsnr_map.reshape(-1)
        bucket_idx = self.get_bucket_indices(logsnr_flat)
        pixel_var = self.running_var[bucket_idx]
        
        weights_flat = 1.0 / (pixel_var.sqrt() + 1e-6)
        weights_flat = weights_flat / (weights_flat.mean() + 1e-8)
        
        # Reshape to logsnr shape, then broadcast to target
        weights = weights_flat.reshape(logsnr_map.shape)
        return weights.expand(target_shape)
    
    def get_stats_dict(self) -> dict:
        """Returns current statistics for logging."""
        return {
            'bucket_centers': self.bucket_centers.cpu().tolist(),
            'bucket_means': self.running_mean.cpu().tolist(),
            'bucket_vars': self.running_var.cpu().tolist(),
            'bucket_counts': self.counts.cpu().tolist(),
            'step': self.step,
            'warmup_complete': self.step >= self.warmup_steps
        }

def compute_online_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    logsnr_map: torch.Tensor,
    tracker: OnlineVarianceTracker,
    update_stats: bool = True
) -> tuple[torch.Tensor, dict]:
    """
    Computes MSE with online variance-based weighting.
    
    The tracker learns the variance structure of THIS run and uses
    it to normalize gradients in real-time.
    """
    # 1. Pre-reduction squared error
    sq_err = (pred - target) ** 2
    
    # 2. Update tracker with unweighted error (before any correction)
    if update_stats:
        tracker.update(logsnr_map, sq_err)
    
    # 3. Get correction weights from current running estimates
    weights = tracker.get_weight_map(logsnr_map, sq_err.shape)
    
    # 4. Apply correction
    weighted_sq_err = sq_err * weights
    
    # 5. Stats for logging
    with torch.no_grad():
        stats = {
            'loss_unweighted': sq_err.mean().item(),
            'loss_weighted': weighted_sq_err.mean().item(),
            'weight_mean': weights.mean().item(),
            'weight_min': weights.min().item(),
            'weight_max': weights.max().item(),
            'correction_ratio': weighted_sq_err.mean().item() / (sq_err.mean().item() + 1e-8),
            **{f'var_bucket_{i}': v for i, v in enumerate(tracker.running_var.cpu().tolist())}
        }
    
    # 6. Final reduction
    loss = weighted_sq_err.mean()
    
    return loss, stats

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
            params = split_cfg['params']
            seq_struct = params['sequence_structure']
            for frame in seq_struct:
                # Track absolute max defined in config
                max_seq_res_abs = max(max_seq_res_abs, frame['res'])
                # Track relative max
                max_seq_rel = max(max_seq_rel, frame['relative_res'])
    
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
    # this is bad. why are we doing this? oh yes, we are doing this because this is an immature single-device training script.
    device = config['device']
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
    steps = config['training']['steps']
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
    device = config['device']
    var_cfg = config['training']['online_variance_correction']
    if var_cfg['enabled']:
        variance_tracker = OnlineVarianceTracker(device=device, **var_cfg)
    else:
        variance_tracker = None

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
                        # Inside the latent block processing:
                        if variance_tracker is not None:
                            loss_v, loss_stats = compute_online_weighted_mse(
                                v_pred, target_v, target_l, variance_tracker
                            )
                            loss_v_accum += loss_v
                            # Stats
                            step_stats.append({
                                'step': i,
                                'source': getattr(block, 'source', 'unknown'),
                                'type': 'latent',
                                'loss': loss_stats['loss_unweighted'],
                                'loss_weighted': loss_stats['loss_weighted'],
                                'loss_var': loss_stats.get('weight_max', 0) - loss_stats.get('weight_min', 0),
                                'logsnr': target_l.mean().detach().item(),
                                'resolution': block.content.shape[-1] * block.content.shape[-2],
                                'weight_mean': loss_stats['weight_mean'],
                                'correction_ratio': loss_stats['correction_ratio']
                            })
                        else:
                            sq_err_v = F.mse_loss(v_pred, target_v, reduction="none")
                            loss_v_accum += sq_err_v.mean()
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
                        
                        # Lambda L1
                        loss_lam_accum += F.l1_loss(pred_l, target_l)
                        # end of step paperwork
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
