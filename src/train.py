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
from .graph_runner import GraphRunner

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

    Returns stats with Python floats (no tensor sync overhead - values computed
    from already-computed tensors using cached graph values).
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

    # 5. Compute scalar losses (these will sync to CPU when accessed)
    # But since we need them for the loss anyway, the sync is unavoidable
    loss = weighted_sq_err.mean()
    loss_unweighted = sq_err.mean()

    # 6. Stats dict - use detached tensor values that can be converted later
    # For logging, we defer .item() calls to the logging interval
    with torch.no_grad():
        weight_mean = weights.mean()
        weight_min = weights.min()
        weight_max = weights.max()
        correction = loss / (loss_unweighted + 1e-8)

    # Return Python floats for stats that are already computed
    # These .item() calls are unavoidable for logging but happen once per call
    stats = {
        'loss_unweighted': loss_unweighted.detach(),  # Keep as tensor
        'loss_weighted': loss.detach(),
        'weight_mean': weight_mean.detach(),
        'weight_min': weight_min.detach(),
        'weight_max': weight_max.detach(),
        'correction_ratio': correction.detach(),
    }

    return loss, stats

def calculate_global_max_resolution(config: Dict[str, Any]) -> int:
    """
    Scans config to find the maximum resolution required for video caching.
    Considers both explicit bucket definitions and sequence relative scaling.
    """
    max_res = 32 # Floor

    # 1. Check Buckets (if enabled)
    bucketing = config['training']['bucketing']
    if bucketing['enabled']:
        buckets = bucketing['image_buckets']
        if buckets:
            max_bucket_res = max(b['resolution'] for b in buckets)
        else:
            max_bucket_res = 32
    else:
        max_bucket_res = 0 # Not driving resolution

    # 2. Check Sequence Structures
    # We need to find the max 'res' (absolute) OR max 'relative_res'
    dataset_mix = config['dataset_mix']

    max_seq_res_abs = 0
    max_seq_rel = 1.0

    for split_name, split_cfg in dataset_mix.items():
        if split_cfg['type'] == 'video':
            params = split_cfg['params']
            seq_struct = params['sequence_structure']
            for frame in seq_struct:
                # Track absolute max defined in config
                max_seq_res_abs = max(max_seq_res_abs, frame['res'])
                # Track relative max
                max_seq_rel = max(max_seq_rel, frame['relative_res'])

    # 3. Compute Global Max
    if bucketing['enabled']:
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
    """
    Train the autoencoder (sparse AE) in isolation.

    This trains ONLY the AE encoder/decoder to reconstruct clean images.
    No transformer, no noise injection - pure AE reconstruction.
    """
    config = sanitize_config(config)

    steps = config['training']['ae_steps']
    bs = config['training']['batch_size']
    device = config['device']
    dtype = config['dtype']

    # Check config for sparse AE - this is the source of truth
    sparse_ae_cfg = config['training']['sparse_ae']
    if not sparse_ae_cfg['enabled']:
        print("\n--- Skipping AE warmup (sparse_ae.enabled = false) ---")
        return pd.DataFrame()

    # Get components - SpanEmbedder wraps patch_emb, SpanUnembedder wraps patch_unembed
    _, span_emb, span_unemb, _ = components
    patch_emb = span_emb.patch_emb  # The actual patch embedder (SparseAEPatchEmbedder)
    patch_unemb = span_unemb.patch_unembed  # The actual patch unembedder

    # Get the sparse AE from the embedder wrapper
    if not hasattr(patch_emb, 'ae'):
        print("\n--- Skipping AE warmup (patch_emb has no .ae - check build_components) ---")
        return pd.DataFrame()

    sparse_ae = patch_emb.ae

    print(f"\n--- Training: Sparse AutoEncoder ---")
    print(f"    Levels: {sparse_ae.n_levels}, Code dim: {sparse_ae.code_dim}, "
          f"Sparsity k: {sparse_ae.k_per_patch}")

    # Optimizer for AE parameters only (not full model)
    # NOTE: patch_emb and patch_unemb both hold references to sparse_ae via self.ae
    # so we must NOT add their .parameters() directly (would duplicate sparse_ae params)
    # Instead: add sparse_ae params once, then only the wrapper-specific projection layers
    opt_cfg = config['training']['ae_optimizer']
    ae_params = list(sparse_ae.parameters())
    # Add only the wrapper-specific projection layers (not self.ae which is shared)
    ae_params += list(patch_emb.code_proj.parameters())
    ae_params += list(patch_emb.logsnr_proj.parameters())
    ae_params += list(patch_unemb.code_unproj.parameters())
    ae_params += list(patch_unemb.logsnr_decoder.parameters())

    opt = torch.optim.AdamW(ae_params, lr=opt_cfg['lr'], weight_decay=opt_cfg['weight_decay'], fused=True)
    scheduler = OneCycleLR(opt, max_lr=opt_cfg['max_lr'], total_steps=steps, pct_start=opt_cfg['pct_start'])

    # Build bucket manager
    model = components[0]
    model_stride = model.patch_embedder.stride
    bucket_mgr = build_bucket_manager_from_config(config, model_stride=model_stride)
    bucketing_enabled = config['training']['bucketing']['enabled']

    history = []
    use_amp = dtype in (torch.bfloat16, torch.float16)
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))
    log_interval = config['logging']['log_interval']

    pbar = tqdm(range(steps), desc="train-ae")
    for i in pbar:
        opt.zero_grad()

        with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
            # Sample bucket for resolution
            if bucketing_enabled:
                bucket = bucket_mgr.sample_bucket()
                curr_res = bucket.resolution
                curr_bs = bucket.batch_size
            else:
                curr_res = 64
                curr_bs = bs

            # Get clean image blocks
            clean_blocks = iterator.generate_batch_list(curr_bs, resolution=curr_res)

            # === Batch-flattening pattern ===
            # Group images by grid_shape for efficient batched processing
            # This builds masks once per unique resolution and processes all same-res images together
            latent_groups = {}  # grid_shape -> list of (block, img, logsnr)
            sources = []

            for b in clean_blocks:
                if b.type != 'latent':
                    continue

                img = b.content  # [C, H, W]
                lsnr = b.logsnr  # [1, H, W]
                sources.append(getattr(b, 'source', 'unknown'))

                # Compute grid_shape for grouping
                p = patch_emb.stride
                grid_shape = (img.shape[1] // p, img.shape[2] // p)

                if grid_shape not in latent_groups:
                    latent_groups[grid_shape] = []
                latent_groups[grid_shape].append((b, img, lsnr))

            if not latent_groups:
                continue

            # Process each grid_shape group as a batch
            loss_recon_accum = torch.tensor(0.0, device=device)
            loss_logsnr_accum = torch.tensor(0.0, device=device)
            sparsity_accum = 0.0
            n_latent = 0

            for grid_shape, group in latent_groups.items():
                # Stack into batch
                imgs = torch.stack([g[1] for g in group], dim=0)  # [B, C, H, W]
                logsnrs = torch.stack([g[2] for g in group], dim=0)  # [B, 1, H, W]
                batch_size = imgs.shape[0]

                # Use wrapper's cached masks for consistency with inference path
                encoder_masks, decoder_masks = patch_emb._get_masks(grid_shape, device)

                # Encode through sparse AE (batched)
                codes_list, _ = patch_emb.ae.encode(imgs, logsnrs,
                                                    grid_shape=grid_shape,
                                                    encoder_masks=encoder_masks,
                                                    decoder_masks=decoder_masks)

                # Concatenate codes across levels and project to embeddings
                codes_cat = torch.cat(codes_list, dim=-1)  # [B, N, code_dim * n_levels]
                z = patch_emb.code_proj(codes_cat)  # [B, N, embed_dim]

                # Decode through unembedder (batched)
                recon_with_logsnr = patch_unemb(z, grid_shape)  # [B, C+1, H, W]
                recon = recon_with_logsnr[:, :-1]  # [B, C, H, W]
                logsnr_pred = recon_with_logsnr[:, -1:]  # [B, 1, H, W]

                # Accumulate reconstruction loss
                sq_err = (recon - imgs) ** 2
                loss_recon_accum = loss_recon_accum + sq_err.mean() * batch_size

                # Accumulate logsnr loss
                loss_logsnr_accum = loss_logsnr_accum + F.l1_loss(logsnr_pred, logsnrs) * batch_size

                # Compute sparsity from codes
                nonzero_codes = (codes_cat != 0).sum()
                total_codes = codes_cat.numel()
                sparsity_accum += (1.0 - (nonzero_codes.item() / total_codes)) * batch_size
                n_latent += batch_size

            if n_latent == 0:
                continue

            # Average losses
            loss_recon = loss_recon_accum / n_latent
            loss_logsnr = loss_logsnr_accum / n_latent
            sparsity = sparsity_accum / n_latent

            # Total loss
            ae_cfg = config['training']['sparse_ae']
            total_loss = loss_recon + ae_cfg['logsnr_loss_weight'] * loss_logsnr

        # Backward
        if dtype == torch.float16:
            scaler.scale(total_loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            total_loss.backward()
            opt.step()
        scheduler.step()

        # Logging
        if i % log_interval == 0:
            loss_val = loss_recon.item()
            sparsity_val = sparsity if isinstance(sparsity, float) else sparsity.item()

            # Log per-source stats
            for src in set(sources):
                history.append({
                    'step': i,
                    'source': src,
                    'type': 'latent',
                    'loss': loss_val,
                    'resolution': curr_res * curr_res,
                    'sparsity': sparsity_val,
                    'loss_logsnr': loss_logsnr.item()
                })

            pbar.set_postfix({
                'recon': f'{loss_val:.4f}',
                'sparse': f'{sparsity_val:.1%}'
            })

    # --- End of training: collect reconstruction samples for visualization ---
    # This shows PURE AE encode/decode quality (no diffusion model involved)
    if logger is not None and steps > 0:
        print("Collecting AE reconstruction samples...")
        recon_samples = {'x0': [], 'noisy_input': [], 'reconstruction': [], 'logsnr_map': [], 'source': []}

        # Get the patch embedder/unembedder for direct AE encode/decode
        # SpanEmbedder.patch_emb and SpanUnembedder.patch_unembed are the actual modules
        patch_emb = span_emb.patch_emb  # SparseAEPatchEmbedder or ContextualPatchEmbedder
        patch_unemb = span_unemb.patch_unembed  # SparseAEPatchUnembedder or ContextualPatchUnembedder

        # Sample from each split
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
            for res in [64, 128, 256]:
                try:
                    sample_blocks = iterator.generate_batch_list(4, resolution=res)
                except:
                    continue

                for b in sample_blocks:
                    if b.type != 'latent':
                        continue

                    x0 = b.content  # [C, H, W]
                    logsnr = b.logsnr  # [1, H, W]

                    # Pure AE encode/decode (no noise, no diffusion)
                    # This tests the autoencoder reconstruction quality directly
                    z, grid_shape = patch_emb(x0, logsnr)  # Encode
                    recon_full = patch_unemb(z, grid_shape)  # Decode [C+1, H, W]
                    recon = recon_full[:3]  # RGB only, drop logsnr channel

                    recon_samples['x0'].append(x0)
                    recon_samples['noisy_input'].append(x0)  # For pure AE, input = clean
                    recon_samples['reconstruction'].append(recon)
                    recon_samples['logsnr_map'].append(logsnr)
                    recon_samples['source'].append(getattr(b, 'source', 'unknown'))

                    if len(recon_samples['x0']) >= 12:
                        break
                if len(recon_samples['x0']) >= 12:
                    break

        # Plot reconstructions
        if recon_samples['x0']:
            from .plotting import plot_dset_reconstruction
            plot_dset_reconstruction(recon_samples, logger, name="ae_reconstruction", show_map=True, show_error=True)
            print(f"  Saved {len(recon_samples['x0'])} AE reconstruction samples")

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
    page_table = components[3]
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

    # 4. CUDA Graph Runner (optional)
    # One graph per block layout config. Shape = max_blocks × block_size.
    graph_cfg = config['training']['graph_capture']
    if graph_cfg['enabled']:
        graph_runner = GraphRunner(model, page_table, config)
        warmup_steps_needed = graph_cfg['warmup_steps']
        print(f"[GraphRunner] CUDA Graph capture enabled, warmup={warmup_steps_needed} steps")
    else:
        graph_runner = None
        warmup_steps_needed = 0
    
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
            # Handle CUDA Graph warmup/capture/replay
            # During early steps, run with warmup_mode=True to accumulate warmups per bucket
            # Graph runner auto-captures after 3 warmups per bucket
            # After capture, subsequent calls use replay automatically
            use_warmup = (graph_runner is not None and i < warmup_steps_needed * 5)
            decoded_results, aux_loss = run_model_forward(
                components, noisy_blocks,
                graph_runner=graph_runner,
                warmup_mode=use_warmup
            )
            
            # 4. Loss Calculation
            # Keep accumulators as tensors to avoid CPU-GPU sync per block
            loss_v_accum = torch.tensor(0.0, device=device)
            loss_lam_accum = torch.tensor(0.0, device=device)
            loss_text_accum = torch.tensor(0.0, device=device)

            valid_latent_samples = 0
            valid_text_samples = 0

            # Deferred stats: only collect tensors, convert to CPU at log intervals
            # This avoids per-block CPU-GPU sync
            deferred_stats = []

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
                        if variance_tracker is not None:
                            loss_v, loss_stats = compute_online_weighted_mse(
                                v_pred, target_v, target_l, variance_tracker
                            )
                            loss_v_accum = loss_v_accum + loss_v
                            # Defer stats (keep tensors, sync at logging time)
                            deferred_stats.append({
                                'step': i,
                                'source': getattr(block, 'source', 'unknown'),
                                'type': 'latent',
                                'loss_tensor': loss_stats['loss_unweighted'],  # Now tensor
                                'loss_weighted_tensor': loss_stats['loss_weighted'],
                                'loss_var_tensor': loss_stats['weight_max'] - loss_stats['weight_min'],
                                'logsnr_tensor': target_l.mean().detach(),
                                'resolution': block.content.shape[-1] * block.content.shape[-2],
                                'weight_mean_tensor': loss_stats['weight_mean'],
                                'correction_ratio_tensor': loss_stats['correction_ratio']
                            })
                        else:
                            sq_err_v = F.mse_loss(v_pred, target_v, reduction="none")
                            loss_v_accum = loss_v_accum + sq_err_v.mean()
                            # Defer stats
                            deferred_stats.append({
                                'step': i,
                                'source': getattr(block, 'source', 'unknown'),
                                'type': 'latent',
                                'loss_tensor': sq_err_v.mean().detach(),  # Keep as tensor
                                'loss_var_tensor': sq_err_v.var().detach(),
                                'logsnr_tensor': target_l.mean().detach(),
                                'resolution': block.content.shape[-1] * block.content.shape[-2]
                            })

                        # Lambda L1
                        loss_lam_accum = loss_lam_accum + F.l1_loss(pred_l, target_l)
                        valid_latent_samples += 1

                # --- B. Text Autoregressive Loss ---
                elif block.type == 'text':
                    if 'text_logits' in res:
                        logits = res['text_logits']
                        tokens = block.content

                        shift_logits = logits[:-1, :].contiguous()
                        shift_targets = tokens[1:].contiguous()

                        if shift_targets.numel() > 0:
                            loss_t = F.cross_entropy(shift_logits, shift_targets)
                            loss_text_accum = loss_text_accum + loss_t

                            deferred_stats.append({
                                'step': i,
                                'source': getattr(block, 'source', 'text'),
                                'type': 'text',
                                'loss_tensor': loss_t.detach(),
                                'loss_var': 0.0
                            })
                            valid_text_samples += 1

            # Normalize Accumulators
            if valid_latent_samples > 0:
                loss_v_accum = loss_v_accum / valid_latent_samples
                loss_lam_accum = loss_lam_accum / valid_latent_samples

            if valid_text_samples > 0:
                loss_text_accum = loss_text_accum / valid_text_samples

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

        # Only sync to CPU at logging intervals (avoids per-step CPU-GPU sync)
        log_interval = config['logging']['log_interval']
        if i % log_interval == 0:
            # Now convert deferred tensors to CPU for logging
            step_stats = []
            for stat in deferred_stats:
                converted = {'step': stat['step'], 'source': stat['source'], 'type': stat['type']}

                # Handle tensor -> float conversion for all *_tensor keys
                for key in list(stat.keys()):
                    if key.endswith('_tensor'):
                        base_key = key[:-7]  # Remove '_tensor' suffix
                        val = stat[key]
                        converted[base_key] = val.item() if isinstance(val, torch.Tensor) else val
                    elif key not in ('step', 'source', 'type'):
                        converted[key] = stat[key]

                step_stats.append(converted)
            history.extend(step_stats)

            pbar.set_postfix({
                'v': f'{loss_v_accum.item():.3f}',
                'txt': f'{loss_text_accum.item():.3f}'
            })
            
    return pd.DataFrame(history)
