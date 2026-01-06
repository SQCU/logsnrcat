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
from .data_iterator import CompositeIterator, MultiResolutionPrefetcher
from .utils import run_model_forward, predict_velocity_from_blocks
from .sample import euler_forward_step, euler_reverse_step
from .config import sanitize_config
from .graph_runner import GraphRunner
from .optim_closure_bullshit import build_training_context
from .losses import scheduled_mse_bce_velocity_loss

def build_variance_tracker(config: dict, device: torch.device):
    """Build OnlineVarianceTracker from config, or None if disabled."""
    var_cfg = config['training']['online_variance_correction']
    if var_cfg['enabled']:
        return OnlineVarianceTracker(device=device, **var_cfg)
    return None


class DeferredStatsCollector:
    """
    Collects per-sample stats during training, converts tensors at logging intervals.

    Avoids CPU-GPU sync on every step by keeping tensors until flush().
    Deduplicates the stats collection boilerplate across training functions.

    Usage:
        collector = DeferredStatsCollector()

        # In training loop:
        collector.add(step=i, source='fractal', type='latent', resolution=4096,
                      loss_tensor=loss.detach(), logsnr_tensor=logsnr.mean().detach())

        # At logging intervals:
        if i % log_interval == 0:
            history.extend(collector.flush(extra_stats={'mse_weight': 0.5}))
    """
    def __init__(self):
        self._entries = []

    def add(self, step: int, source: str, type: str, resolution: int = None, **tensor_stats):
        """Add a stat entry. Keys ending in '_tensor' are converted to floats on flush."""
        entry = {'step': step, 'source': source, 'type': type}
        if resolution is not None:
            entry['resolution'] = resolution
        entry.update(tensor_stats)
        self._entries.append(entry)

    def flush(self, extra_stats: dict = None) -> list:
        """
        Convert tensors to floats and return entries. Clears internal buffer.

        Args:
            extra_stats: Optional dict of extra stats to add to each entry (e.g., schedule weights)

        Returns:
            List of converted stat dicts ready for DataFrame
        """
        converted_list = []
        base_keys = {'step', 'source', 'type', 'resolution'}
        for stat in self._entries:
            converted = {'step': stat['step'], 'source': stat['source'], 'type': stat['type']}
            if 'resolution' in stat:
                converted['resolution'] = stat['resolution']

            # Handle tensor -> float conversion for all *_tensor keys
            for key, val in stat.items():
                if key.endswith('_tensor'):
                    base_key = key[:-7]  # Remove '_tensor' suffix
                    converted[base_key] = val.item() if isinstance(val, torch.Tensor) else val
                elif key not in base_keys:
                    converted[key] = val

            # Add extra stats if provided
            if extra_stats:
                converted.update(extra_stats)

            converted_list.append(converted)

        self._entries.clear()
        return converted_list

    def __len__(self):
        return len(self._entries)


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

        # Vectorized bucket statistics using scatter operations
        # Accumulate sums and counts per bucket
        bucket_sums = torch.zeros(self.num_buckets, device=self.device, dtype=err_flat.dtype)
        bucket_sq_sums = torch.zeros(self.num_buckets, device=self.device, dtype=err_flat.dtype)
        bucket_counts = torch.zeros(self.num_buckets, device=self.device, dtype=err_flat.dtype)

        bucket_sums.scatter_add_(0, bucket_idx, err_flat)
        bucket_sq_sums.scatter_add_(0, bucket_idx, err_flat ** 2)
        bucket_counts.scatter_add_(0, bucket_idx, torch.ones_like(err_flat))

        # Compute batch means and variances (only for buckets with samples)
        has_samples = bucket_counts > 0
        batch_means = torch.where(has_samples, bucket_sums / bucket_counts.clamp(min=1), self.running_mean)
        # Var = E[X^2] - E[X]^2, with safety for single-sample buckets
        batch_vars = torch.where(
            bucket_counts > 1,
            (bucket_sq_sums / bucket_counts.clamp(min=1)) - batch_means ** 2,
            self.running_var
        )
        batch_vars = batch_vars.clamp(min=1e-8)  # Numerical stability

        # EMA update (only where we have samples)
        update_weight = (1 - self.ema_decay) * has_samples.float()
        self.running_mean = self.ema_decay * self.running_mean + update_weight * batch_means
        self.running_var = self.ema_decay * self.running_var + update_weight * batch_vars
        self.counts += bucket_counts
    
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


# ==============================================================================
# Training Loops (Config Driven)
# ==============================================================================

from .bucket_manager import build_bucket_manager_from_config

def train_autoembed(components, config, iterator, logger=None):
    """
    Train the autoencoder (sparse AE) in isolation.

    This trains ONLY the AE encoder/decoder to reconstruct clean images.
    No transformer, no noise injection - pure AE reconstruction.

    KEY DESIGN: Calls sparse_ae.forward() DIRECTLY (not through wrapper projections).
    The wrapper projections (latent_code_proj/unproj) are trained later in latent diffusion,
    not during AE warmup. AE warmup focuses solely on reconstruction quality.

    Loss functions are stateless and config-driven via src/losses.py.
    """
    from .losses import get_ae_loss_fn, group_blocks_by_grid, prepare_ae_batch, compute_ae_forward, get_k_for_step

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

    # Get model - sparse AE is now a submodule of the model
    model = components[0]

    # Check if model has sparse AE
    if not hasattr(model, 'sparse_ae') or model.sparse_ae is None:
        print("\n--- Skipping AE warmup (model has no sparse_ae) ---")
        return pd.DataFrame()

    sparse_ae = model.sparse_ae
    patch_emb = model.patch_embedder  # For visualization path only
    patch_unemb = model.patch_unembedder

    # Get loss function from config (stateless, config-driven)
    ae_loss_fn = get_ae_loss_fn(config)
    loss_type = sparse_ae_cfg['loss_type']
    loss_schedule_cfg = sparse_ae_cfg['loss_schedule']
    loss_schedule_enabled = loss_schedule_cfg['enabled']

    # K-annealing config
    k_start = sparse_ae_cfg.get('k_start')
    k_end = sparse_ae_cfg['k_per_patch']  # Final k
    k_anneal_steps = sparse_ae_cfg['k_anneal_steps']
    k_annealing_enabled = k_start is not None and k_start > k_end

    # Subspace routing entropy regularization (only when wavelet_gating=True)
    routing_entropy_weight = sparse_ae_cfg['routing_entropy_weight']
    wavelet_gating = sparse_ae_cfg['wavelet_gating']

    print(f"\n--- Training: Sparse AutoEncoder ---")
    print(f"    Levels: {sparse_ae.n_levels}, Code dim: {sparse_ae.code_dim}, "
          f"Sparsity k: {sparse_ae.k_per_patch}")
    if k_annealing_enabled:
        print(f"    K-annealing: {k_start} -> {k_end} over {k_anneal_steps} steps")
    if loss_schedule_enabled:
        mse_start = loss_schedule_cfg.get('mse_start', 1.0)
        bce_end = loss_schedule_cfg.get('bce_end', 0.9)
        schedule_type = loss_schedule_cfg.get('schedule', 'linear')
        print(f"    Loss: scheduled MSE→BCE ({schedule_type}), {mse_start:.0%} MSE → {bce_end:.0%} BCE")
    else:
        print(f"    Loss: {loss_type}")
    if wavelet_gating and routing_entropy_weight > 0:
        print(f"    Entropy regularization: weight={routing_entropy_weight}")

    # Build training context - model already owns sparse_ae, patch_embedder, patch_unembedder
    # No wrapper needed: model.parameters() includes all AE components
    ctx = build_training_context(model, config, total_steps=steps, role="ae")

    # Build bucket manager
    model_stride = model.patch_embedder.stride
    bucket_mgr = build_bucket_manager_from_config(config, model_stride=model_stride)
    bucketing_enabled = config['training']['bucketing']['enabled']

    # Build async prefetcher for non-blocking data generation
    # Gets first split name from iterator (or fallback to 'sprite_atlas')
    split_names = iterator.get_split_names()
    primary_split = split_names[0] if split_names else 'sprite_atlas'
    prefetcher = MultiResolutionPrefetcher(
        iterator=iterator,
        bucket_manager=bucket_mgr,
        split_name=primary_split,
        count=bs,
        buffer_per_resolution=4,
        seed=42,
        device=device
    )
    prefetcher.warmup(min_items_per_resolution=2)
    print(f"    Prefetcher ready: {len(bucket_mgr.buckets)} resolutions, buffer=4/res")

    history = []
    log_interval = config['logging']['log_interval']

    pbar = tqdm(range(steps), desc="train-ae")
    for i in pbar:
        ctx.zero_grad()

        with ctx.autocast():
            # Sample bucket for resolution
            if bucketing_enabled:
                bucket = bucket_mgr.sample_bucket()
                curr_res = bucket.resolution
                curr_bs = bucket.batch_size
            else:
                curr_res = 64
                curr_bs = bs

            # Get clean image blocks (async prefetch - near-instant)
            clean_blocks = prefetcher.get(curr_res)

            # Group by grid_shape for efficient batched processing
            latent_groups = group_blocks_by_grid(clean_blocks, sparse_ae.patch_size, device)

            if not latent_groups:
                continue

            # Process each grid_shape group as a batch
            total_loss = torch.tensor(0.0, device=device)
            loss_stats_accum = {}
            per_source_losses = {}  # Track per-source losses for meaningful plots
            n_latent = 0

            # Compute current k for k-annealing (None if disabled)
            current_k = get_k_for_step(i, k_start, k_end, k_anneal_steps) if k_annealing_enabled else None

            for grid_shape, group in latent_groups.items():
                # Prepare batched inputs
                prepared = prepare_ae_batch(group, sparse_ae, device)
                batch_size = prepared['images'].shape[0]
                batch_sources = prepared['sources']

                # DIRECT AE forward (no wrapper projection bottleneck!)
                output = compute_ae_forward(sparse_ae, prepared, k_override=current_k)

                # Stateless loss computation (pure reconstruction, no logsnr)
                # Pass step/total_steps for scheduled loss (ignored by non-scheduled losses)
                loss, stats = ae_loss_fn(output, prepared['images'], step=i, total_steps=steps)

                # Entropy regularization for subspace routing (wavelet_gating only)
                # High entropy = balanced subspace usage = good, so we SUBTRACT entropy
                if wavelet_gating and routing_entropy_weight > 0 and 'routing_entropy_mean' in output:
                    entropy_loss = -routing_entropy_weight * output['routing_entropy_mean']
                    loss = loss + entropy_loss
                    stats['entropy_loss'] = entropy_loss.item()
                    stats['routing_entropy'] = output['routing_entropy_mean'].item()
                    stats['wav_active'] = output['wav_active_mean'].item()
                    stats['amp_active'] = output['amp_active_mean'].item()

                # Compute per-sample MSE for per-source tracking (no grad needed)
                with torch.no_grad():
                    per_sample_mse = F.mse_loss(output['recon'], prepared['images'], reduction='none')
                    per_sample_mse = per_sample_mse.mean(dim=(1, 2, 3))  # [B]

                # Track per-source losses
                for idx, src in enumerate(batch_sources):
                    if src not in per_source_losses:
                        per_source_losses[src] = {'loss_sum': 0.0, 'count': 0, 'sparsity_sum': 0.0}
                    per_source_losses[src]['loss_sum'] += per_sample_mse[idx].item()
                    per_source_losses[src]['count'] += 1
                    per_source_losses[src]['sparsity_sum'] += stats['sparsity'].item()

                # Accumulate
                total_loss = total_loss + loss * batch_size
                for k, v in stats.items():
                    # Skip non-scalar stats (e.g. per_level list)
                    if isinstance(v, list):
                        continue
                    if k not in loss_stats_accum:
                        loss_stats_accum[k] = 0.0
                    if isinstance(v, torch.Tensor):
                        loss_stats_accum[k] += v.item() * batch_size
                    else:
                        loss_stats_accum[k] += v * batch_size
                n_latent += batch_size

            if n_latent == 0:
                continue

            # Average losses
            total_loss = total_loss / n_latent
            for k in loss_stats_accum:
                loss_stats_accum[k] /= n_latent

        # Backward and step (TrainingContext handles scaler, FP8, scheduling)
        ctx.backward(total_loss)
        ctx.step()

        # Logging with per-source losses (not batch-averaged)
        if i % log_interval == 0:
            # Log each source with its own average loss
            for src, data in per_source_losses.items():
                if data['count'] > 0:
                    src_loss = data['loss_sum'] / data['count']
                    src_sparsity = data['sparsity_sum'] / data['count']
                    entry = {
                        'step': i,
                        'source': src,
                        'type': 'latent',
                        'loss': src_loss,
                        'resolution': curr_res * curr_res,
                        'sparsity': src_sparsity,
                        'loss_logsnr': loss_stats_accum.get('logsnr_loss', 0.0)
                    }
                    # Add MSE/BCE loss schedule fields when enabled
                    if loss_schedule_enabled:
                        entry['mse_loss'] = loss_stats_accum.get('mse_loss', 0.0)
                        entry['bce_loss'] = loss_stats_accum.get('bce_loss', 0.0)
                        entry['mse_weight'] = loss_stats_accum.get('mse_weight', 1.0)
                        entry['bce_weight'] = loss_stats_accum.get('bce_weight', 0.0)
                        # Compute lerp progress for analysis
                        entry['lerp_t'] = i / max(steps - 1, 1)
                    # Add k-annealing field when enabled
                    if k_annealing_enabled:
                        entry['current_k'] = current_k
                    history.append(entry)

            # Progress bar shows batch-averaged loss
            loss_val = loss_stats_accum.get('recon_loss', total_loss.item())
            sparsity_val = loss_stats_accum.get('sparsity', 0.0)
            postfix = {
                'recon': f'{loss_val:.4f}',
                'sparse': f'{sparsity_val:.1%}'
            }
            if k_annealing_enabled:
                postfix['k'] = current_k
            pbar.set_postfix(postfix)

    prefetcher.stop()
    return pd.DataFrame(history)

def train_denoise(components, config, iterator, logger=None):
    """
    Train the denoiser (main LDTformer) on noisy images.

    Uses stateless loss functions from src/losses.py for velocity prediction.
    Supports both pixel-space and latent-space diffusion via config.
    """
    from .losses import get_denoise_loss_fn, logsnr_prediction_loss

    # 1. Enforce Dictionary Type
    config = sanitize_config(config)

    # 2. Strict Access (No defaults allowed here - define them in Pydantic schema)
    mode = config['training']['mode']
    steps = config['training']['steps']
    bs = config['training']['batch_size']
    lambda_coeff = config['training']['lambda_coeff']

    # Get config-driven loss function for velocity prediction
    denoise_loss_fn = get_denoise_loss_fn(config)

    print(f"\n--- Training: Denoiser ({mode.upper()}) ---")
    model, _, _, page_table = components
    device = config['device']
    variance_tracker = build_variance_tracker(config, device)

    # 3. Build Manager
    # We explicitly look up the stride from the model instance, or the config dict
    model_stride = model.patch_embedder.stride
    bucket_mgr = build_bucket_manager_from_config(config, model_stride=model_stride)

    # Build async prefetcher for non-blocking data generation
    split_names = iterator.get_split_names()
    primary_split = split_names[0] if split_names else 'sprite_atlas'
    prefetcher = MultiResolutionPrefetcher(
        iterator=iterator,
        bucket_manager=bucket_mgr,
        split_name=primary_split,
        count=bs,
        buffer_per_resolution=4,
        seed=42,
        device=device
    )
    prefetcher.warmup(min_items_per_resolution=2)
    print(f"    Prefetcher ready: {len(bucket_mgr.buckets)} resolutions, buffer=4/res")

    # 4. CUDA Graph Runner (optional)
    # One graph per block layout config. Shape = max_blocks × block_size.
    # GraphRunner creates its own static masks at max_ctx - no per-batch mask generation.
    graph_cfg = config['training']['graph_capture']
    if graph_cfg['enabled']:
        graph_runner = GraphRunner(
            model, page_table, config,
            window_size=getattr(model, 'window_size', 10.0)
        )
        warmup_steps_needed = graph_cfg['warmup_steps']
        print(f"[GraphRunner] CUDA Graph capture enabled, warmup={warmup_steps_needed} steps")
    else:
        graph_runner = None
        warmup_steps_needed = 0

    # Build training context (handles FP8, optimizer, scaler, autocast)
    ctx = build_training_context(model, config, total_steps=steps)

    history = []
    stats_collector = DeferredStatsCollector()
    bucketing_enabled = config['training']['bucketing']['enabled']
    log_interval = config['logging']['log_interval']

    pbar = tqdm(range(steps), desc="train-denoise")
    for i in pbar:
        ctx.zero_grad()
        with ctx.autocast():
            # 1. Clean Data
            # 1. NEW: Sample Bucket
            if bucketing_enabled:
                bucket = bucket_mgr.sample_bucket()
                curr_res = bucket.resolution
                curr_bs = bucket.batch_size
            else:
                curr_res = 32 # Fallback
                curr_bs = bs   
            
            # 2. Generate Data with Dynamic Resolution (async prefetch - near-instant)
            clean_blocks = prefetcher.get(curr_res)

            # 3. Noise Injection
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

            # Stats collection (uses shared DeferredStatsCollector)

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

                        # Velocity loss via stateless function (handles variance weighting)
                        loss_v, loss_stats = denoise_loss_fn(
                            v_pred, target_v, target_l,
                            variance_tracker=variance_tracker
                        )
                        loss_v_accum = loss_v_accum + loss_v

                        # Collect stats (deferred tensor->float conversion)
                        extra = {}
                        for k in ('loss_unweighted', 'weight_mean', 'weight_range', 'loss_var'):
                            if k in loss_stats:
                                extra[f'{k}_tensor'] = loss_stats[k]
                        stats_collector.add(
                            step=i, source=getattr(block, 'source', 'unknown'),
                            type='latent',
                            resolution=block.content.shape[-1] * block.content.shape[-2],
                            loss_tensor=loss_stats['loss'],
                            logsnr_tensor=target_l.mean().detach(),
                            **extra
                        )

                        # Lambda L1 loss via stateless function
                        loss_l, _ = logsnr_prediction_loss(pred_l, target_l)
                        loss_lam_accum = loss_lam_accum + loss_l
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

                            stats_collector.add(
                                step=i, source=getattr(block, 'source', 'text'),
                                type='text', loss_tensor=loss_t.detach(), loss_var=0.0
                            )
                            valid_text_samples += 1

            # Normalize Accumulators
            if valid_latent_samples > 0:
                loss_v_accum = loss_v_accum / valid_latent_samples
                loss_lam_accum = loss_lam_accum / valid_latent_samples

            if valid_text_samples > 0:
                loss_text_accum = loss_text_accum / valid_text_samples

            # Weighted Sum
            total_loss = loss_v_accum + (lambda_coeff * loss_lam_accum) + loss_text_accum + aux_loss

        # Backward and step (TrainingContext handles scaler, FP8, scheduling)
        ctx.backward(total_loss)
        ctx.step()

        # Only sync to CPU at logging intervals (avoids per-step CPU-GPU sync)
        if i % log_interval == 0:
            history.extend(stats_collector.flush())

            pbar.set_postfix({
                'v': f'{loss_v_accum.item():.3f}',
                'txt': f'{loss_text_accum.item():.3f}'
            })

    prefetcher.stop()
    return pd.DataFrame(history)


# =============================================================================
# Latent Diffusion Training (uses main LDTformer with 4D topology)
# =============================================================================

def train_latent_diffusion(
    components: Tuple,
    config: dict,
    iterator,
    logger=None,
):
    """
    Joint training of encoder + main LDTformer (as denoiser) + decoder.

    Uses the main LDTformer model for denoising in latent space with
    4D topology embeddings [highway, spatial_x, spatial_y, level].

    Pipeline:
    1. Encode clean images to pre_quant (continuous codes before FSQ)
    2. Flatten codes across levels: [B, n_patches * n_levels, code_dim]
    3. Add noise based on logsnr schedule
    4. Denoise with main LDTformer using level-aware SWA masks
    5. Unflatten, re-quantize through FSQ + sparsity
    6. Decode for reconstruction loss

    Gradients flow through: encoder <- LDTformer <- decoder
    This enables joint training of the full pipeline.

    Args:
        components: (model, span_emb, span_unemb, page_table)
        config: Experiment config dict
        iterator: Data iterator
        logger: Optional experiment logger

    Returns:
        DataFrame with training history
    """
    from .losses import group_blocks_by_grid
    from .embedders import prepare_latent_batch

    config = sanitize_config(config)
    model = components[0]
    device = config['device']
    dtype = config['dtype']

    # Get sparse AE components
    sparse_ae_cfg = config['training']['sparse_ae']
    if not sparse_ae_cfg['enabled']:
        print("\n--- Skipping latent diffusion (sparse_ae.enabled = false) ---")
        return pd.DataFrame()

    # Check topology config - must be "latent" mode
    topo_cfg = sparse_ae_cfg['topology']
    diffusion_space = topo_cfg['diffusion_space']
    if diffusion_space != 'latent':
        print(f"\n--- Skipping latent diffusion (diffusion_space = '{diffusion_space}', expected 'latent') ---")
        return pd.DataFrame()

    if not hasattr(model, 'sparse_ae') or model.sparse_ae is None:
        print("\n--- Skipping latent diffusion (model has no sparse_ae) ---")
        return pd.DataFrame()

    sparse_ae = model.sparse_ae
    patch_emb = model.patch_embedder
    patch_unemb = model.patch_unembedder
    n_levels = sparse_ae.n_levels
    code_dim = sparse_ae.code_dim

    # Topology geometry config (passed to latent_diffusion_forward_batch)
    topo_config = {
        'level_lambda': topo_cfg['level_lambda'],
        'level_scale': topo_cfg['level_scale'],
        'vertical_free': topo_cfg['vertical_attention_free'],
        'window_size': config['model']['window_size'],
    }

    # Training params from [training]
    steps = config['training']['steps']
    bs = config['training']['batch_size']
    lambda_coeff = config['training']['lambda_coeff']
    recon_weight = sparse_ae_cfg['ae_loss_weight']
    logsnr_weight = sparse_ae_cfg['logsnr_loss_weight']

    # Diffusion loss schedule config (MSE -> partial BCE for v-field)
    diffusion_loss_schedule = sparse_ae_cfg['diffusion_loss_schedule']
    use_scheduled_v_loss = diffusion_loss_schedule['enabled']
    if use_scheduled_v_loss:
        print(f"\n--- Diffusion Loss Schedule ENABLED ---")
        print(f"    V-field MSE: {diffusion_loss_schedule['mse_start']} -> {diffusion_loss_schedule['mse_end']}")
        print(f"    V-field BCE: {diffusion_loss_schedule['bce_start']} -> {diffusion_loss_schedule['bce_end']}")

    # AE loss schedule config (pinned at END values for reconstruction during diffusion)
    ae_loss_schedule = sparse_ae_cfg['loss_schedule']
    use_scheduled_recon_loss = ae_loss_schedule['enabled']
    if use_scheduled_recon_loss:
        # Pin at end values (warmup already lerped through the schedule)
        recon_mse_weight = ae_loss_schedule['mse_end']
        recon_bce_weight = ae_loss_schedule['bce_end']
        print(f"    Recon MSE: {recon_mse_weight:.2f} (pinned)")
        print(f"    Recon BCE: {recon_bce_weight:.2f} (pinned)")

    print(f"\n--- Training: Latent Diffusion (Main LDTformer) ---")
    print(f"    Steps: {steps}, n_levels: {n_levels}, code_dim: {code_dim}")
    print(f"    Level lambda: {topo_config['level_lambda']}, vertical_free: {topo_config['vertical_free']}")

    # Build training context - model already owns sparse_ae, patch_embedder, patch_unembedder
    # No wrapper needed: model.parameters() includes all components for joint training
    ctx = build_training_context(model, config, total_steps=steps)

    # Bucketing
    model_stride = patch_emb.stride
    bucket_mgr = build_bucket_manager_from_config(config, model_stride=model_stride)
    bucketing_enabled = config['training']['bucketing']['enabled']

    # Build async prefetcher for non-blocking data generation
    split_names = iterator.get_split_names()
    primary_split = split_names[0] if split_names else 'sprite_atlas'
    prefetcher = MultiResolutionPrefetcher(
        iterator=iterator,
        bucket_manager=bucket_mgr,
        split_name=primary_split,
        count=bs,
        buffer_per_resolution=4,
        seed=42,
        device=device
    )
    prefetcher.warmup(min_items_per_resolution=2)
    print(f"    Prefetcher ready: {len(bucket_mgr.buckets)} resolutions, buffer=4/res")

    # Variance tracker
    variance_tracker = build_variance_tracker(config, device)

    history = []
    stats_collector = DeferredStatsCollector()
    log_interval = config['logging']['log_interval']

    pbar = tqdm(range(steps), desc="train-latent-diff")
    for i in pbar:
        ctx.zero_grad()

        with ctx.autocast():
            # Sample bucket
            if bucketing_enabled:
                bucket = bucket_mgr.sample_bucket()
                curr_res = bucket.resolution
                curr_bs = bucket.batch_size
            else:
                curr_res = 64
                curr_bs = bs

            # Get clean blocks (async prefetch - near-instant)
            clean_blocks = prefetcher.get(curr_res)

            # Group by grid_shape for batched processing (reuse from losses.py)
            latent_groups = group_blocks_by_grid(clean_blocks, sparse_ae.patch_size, device)

            if not latent_groups:
                continue

            # Accumulators
            loss_v_accum = torch.tensor(0.0, device=device)
            loss_recon_accum = torch.tensor(0.0, device=device)
            loss_logsnr_accum = torch.tensor(0.0, device=device)
            last_v_stats = None  # Track v-field stats from last group for logging
            last_recon_stats = None  # Track recon stats from last group for logging
            n_latent = 0

            for grid_shape, group in latent_groups.items():
                # Prepare embeddings (extracted to embedders.py)
                batch = prepare_latent_batch(
                    group, grid_shape, sparse_ae, patch_emb, topo_config, device
                )
                B = batch['batch_size']
                target_v = batch['target_v']
                logsnr_flat = batch['logsnr_flat']
                alpha, sigma, noisy_codes = batch['alpha'], batch['sigma'], batch['noisy_codes']
                imgs = batch['imgs']

                # Apply model: h_input → LDTformer → v_pred, logsnr_pred
                h = model.forward_latent_diffusion(
                    batch['h_input'],
                    topo_embeds=batch['topo_embeds'].unsqueeze(0).expand(B, -1, -1),
                    block_mask=batch['latent_mask'],
                )
                v_pred = patch_unemb.latent_code_unproj(h)
                logsnr_pred = patch_unemb.logsnr_decoder(h)

                # V-field loss in latent space
                if use_scheduled_v_loss:
                    loss_v, v_stats = scheduled_mse_bce_velocity_loss(
                        v_pred, target_v, step=i, total_steps=steps,
                        schedule_cfg=diffusion_loss_schedule,
                        variance_tracker=variance_tracker, logsnr_map=logsnr_flat
                    )
                    last_v_stats = v_stats
                else:
                    loss_v = ((v_pred - target_v) ** 2).mean()
                loss_v_accum = loss_v_accum + loss_v * B

                # Reconstruction: recover clean codes, quantize, decode
                if recon_weight > 0:
                    clean_pred = alpha * noisy_codes - sigma * v_pred
                    clean_pred_stacked = clean_pred.view(B, batch['n_patches'], n_levels, code_dim)
                    prequant_pred_list = [clean_pred_stacked[:, :, lv, :] for lv in range(n_levels)]
                    cumulative_recon = sparse_ae.quantize_and_decode(
                        prequant_pred_list, grid_shape, batch['decoder_masks']
                    )

                    if use_scheduled_recon_loss:
                        recon_mse = F.mse_loss(cumulative_recon, imgs)
                        with torch.amp.autocast(device_type='cuda', enabled=False):
                            recon_clamped = cumulative_recon.float().clamp(1e-7, 1 - 1e-7)
                            imgs_clamped = imgs.float().clamp(1e-7, 1 - 1e-7)
                            recon_bce = F.binary_cross_entropy(recon_clamped, imgs_clamped)
                        recon_loss = recon_mse_weight * recon_mse + recon_bce_weight * recon_bce
                        last_recon_stats = {
                            'mse_loss': recon_mse.detach(), 'bce_loss': recon_bce.detach(),
                            'mse_weight': recon_mse_weight, 'bce_weight': recon_bce_weight
                        }
                    else:
                        recon_loss = F.mse_loss(cumulative_recon, imgs)
                        last_recon_stats = None
                    loss_recon_accum = loss_recon_accum + recon_loss * B

                # LogSNR prediction loss
                loss_logsnr = F.l1_loss(logsnr_pred, logsnr_flat)
                loss_logsnr_accum = loss_logsnr_accum + loss_logsnr * B
                n_latent += B

                # Per-sample stats for SNR binning
                with torch.no_grad():
                    per_sample_logsnr = logsnr_flat.mean(dim=1).squeeze(-1)
                    per_sample_loss = ((v_pred - target_v) ** 2).mean(dim=(1, 2))
                    res = curr_res * curr_res
                    for b_idx, block in enumerate(batch['blocks']):
                        extra = {}
                        if use_scheduled_v_loss and last_v_stats:
                            for k in ('loss_unweighted', 'weight_mean', 'loss_var'):
                                if k in last_v_stats:
                                    extra[f'{k}_tensor'] = last_v_stats[k]
                        stats_collector.add(
                            step=i, source=getattr(block, 'source', 'unknown'),
                            type='latent', resolution=res,
                            loss_tensor=per_sample_loss[b_idx].detach(),
                            logsnr_tensor=per_sample_logsnr[b_idx].detach(),
                            **extra
                        )

            if n_latent == 0:
                continue

            # Average losses
            loss_v = loss_v_accum / n_latent
            loss_recon = loss_recon_accum / n_latent
            loss_logsnr = loss_logsnr_accum / n_latent

            # Total loss
            total_loss = loss_v + recon_weight * loss_recon + logsnr_weight * loss_logsnr

        # Backward and step (TrainingContext handles scaler, FP8, scheduling)
        ctx.backward(total_loss)
        ctx.step()

        # Logging - per-sample entries for proper SNR binning
        if i % log_interval == 0:
            # Build extra_stats from schedule-specific values
            extra_stats = {}
            if use_scheduled_v_loss and last_v_stats is not None:
                extra_stats.update({
                    'mse_loss': float(last_v_stats['mse_loss']),
                    'bce_loss': float(last_v_stats['bce_loss']),
                    'mse_weight': last_v_stats['mse_weight'],
                    'bce_weight': last_v_stats['bce_weight'],
                    'lerp_t': i / max(steps - 1, 1),
                })
            if use_scheduled_recon_loss and last_recon_stats is not None:
                extra_stats.update({
                    'recon_mse_loss': float(last_recon_stats['mse_loss']),
                    'recon_bce_loss': float(last_recon_stats['bce_loss']),
                    'recon_mse_weight': last_recon_stats['mse_weight'],
                    'recon_bce_weight': last_recon_stats['bce_weight'],
                })
            history.extend(stats_collector.flush(extra_stats=extra_stats if extra_stats else None))

            postfix = {
                'v': f'{loss_v.item():.4f}',
                'rec': f'{loss_recon.item():.4f}' if recon_weight > 0 else 'N/A'
            }
            if use_scheduled_v_loss and last_v_stats is not None:
                postfix['v_bce'] = f'{last_v_stats["bce_weight"]:.2f}'
                # Show variance correction if active
                if 'weight_mean' in last_v_stats:
                    postfix['w_μ'] = f'{float(last_v_stats["weight_mean"]):.2f}'
            if use_scheduled_recon_loss and last_recon_stats is not None:
                postfix['r_bce'] = f'{last_recon_stats["bce_weight"]:.2f}'
            pbar.set_postfix(postfix)

    prefetcher.stop()
    return pd.DataFrame(history)
