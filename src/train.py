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
from .optim_utils import (
    build_optimizer_group, print_optimizer_summary, OptimizerGroup,
    prepare_model_for_training, HAS_FP8, get_linear_weight_dtype,
    build_optimizer_for_role, print_role_optimizer_summary,
    collect_fp8_modules, FP8Muon, FP8SGD, FP8AdamW, FP8Linear,
)

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

    KEY DESIGN: Calls sparse_ae.forward() DIRECTLY (not through wrapper projections).
    The wrapper projections (code_proj, code_unproj) exist for main transformer interface,
    not for AE training. Training through them adds an information bottleneck.

    Loss functions are stateless and config-driven via src/losses.py.
    """
    from .losses import get_ae_loss_fn, group_blocks_by_grid, prepare_ae_batch, compute_ae_forward

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
    loss_schedule_cfg = sparse_ae_cfg.get('loss_schedule', {})
    loss_schedule_enabled = isinstance(loss_schedule_cfg, dict) and loss_schedule_cfg.get('enabled', False)

    print(f"\n--- Training: Sparse AutoEncoder ---")
    print(f"    Levels: {sparse_ae.n_levels}, Code dim: {sparse_ae.code_dim}, "
          f"Sparsity k: {sparse_ae.k_per_patch}")
    if loss_schedule_enabled:
        mse_start = loss_schedule_cfg.get('mse_start', 1.0)
        bce_end = loss_schedule_cfg.get('bce_end', 0.9)
        schedule_type = loss_schedule_cfg.get('schedule', 'linear')
        print(f"    Loss: scheduled MSE→BCE ({schedule_type}), {mse_start:.0%} MSE → {bce_end:.0%} BCE")
    else:
        print(f"    Loss: {loss_type}")

    # Build AE module wrapper for optimizer
    # Wraps all AE-related parameters for unified optimizer handling
    # NOTE: We train sparse_ae directly but ALSO train projection layers
    # (they're used later during joint training with main transformer)
    class AEModule(nn.Module):
        """Wrapper to expose all AE parameters for optimizer."""
        def __init__(self, sparse_ae, patch_emb, patch_unemb):
            super().__init__()
            self.sparse_ae = sparse_ae
            self.code_proj = patch_emb.code_proj
            self.code_unproj = patch_unemb.code_unproj
            # logsnr projection is optional - SwiGLU variant is pure compression (no logsnr)
            if hasattr(patch_emb, 'logsnr_proj'):
                self.logsnr_proj = patch_emb.logsnr_proj
            if hasattr(patch_unemb, 'logsnr_decoder'):
                self.logsnr_decoder = patch_unemb.logsnr_decoder

    ae_module = AEModule(sparse_ae, patch_emb, patch_unemb)

    # Apply FP8 conversion if configured (W8A16: 8-bit weights, 16-bit activations)
    device = config['device']
    ae_module = prepare_model_for_training(ae_module, config, device=device)

    # Create FP8 optimizer for FP8Linear weights - must match the claimed optimizer
    fp8_modules = collect_fp8_modules(ae_module)
    fp8_optimizer = None
    if fp8_modules:
        ae_opt_cfg = config['training']['ae_optimizer']
        adamw_cfg = config['training']['optimizer']['adamw']
        fp8_optimizer = FP8AdamW(
            fp8_modules,
            lr=ae_opt_cfg['lr'],
            betas=tuple(adamw_cfg['betas']),
            weight_decay=ae_opt_cfg['weight_decay'],
        )
        n_fp8_params = sum(m.out_features * m.in_features for m in fp8_modules)
        print(f"[FP8] Created FP8AdamW for {len(fp8_modules)} modules ({n_fp8_params:,} params)")

    # Use role-based optimizer: simple AdamW for AE (no Muon)
    optimizer_group = build_optimizer_for_role(ae_module, "ae", config, total_steps=steps)
    print_role_optimizer_summary(optimizer_group, ae_module, "ae")

    # Build bucket manager
    model_stride = model.patch_embedder.stride
    bucket_mgr = build_bucket_manager_from_config(config, model_stride=model_stride)
    bucketing_enabled = config['training']['bucketing']['enabled']

    history = []
    use_amp = dtype in (torch.bfloat16, torch.float16)
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))
    log_interval = config['logging']['log_interval']

    pbar = tqdm(range(steps), desc="train-ae")
    for i in pbar:
        optimizer_group.zero_grad()

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

            # Group by grid_shape for efficient batched processing
            latent_groups = group_blocks_by_grid(clean_blocks, sparse_ae.patch_size, device)

            if not latent_groups:
                continue

            # Process each grid_shape group as a batch
            total_loss = torch.tensor(0.0, device=device)
            loss_stats_accum = {}
            per_source_losses = {}  # Track per-source losses for meaningful plots
            n_latent = 0

            for grid_shape, group in latent_groups.items():
                # Prepare batched inputs
                prepared = prepare_ae_batch(group, sparse_ae, device)
                batch_size = prepared['images'].shape[0]
                batch_sources = prepared['sources']

                # DIRECT AE forward (no wrapper projection bottleneck!)
                output = compute_ae_forward(sparse_ae, prepared)

                # Stateless loss computation (pure reconstruction, no logsnr)
                # Pass step/total_steps for scheduled loss (ignored by non-scheduled losses)
                loss, stats = ae_loss_fn(output, prepared['images'], step=i, total_steps=steps)

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

        # Backward
        if dtype == torch.float16:
            scaler.scale(total_loss).backward()
            for spec in optimizer_group.specs.values():
                scaler.step(spec.optimizer)
            scaler.update()
            optimizer_group.schedule_step()
        else:
            total_loss.backward()
            optimizer_group.step()
            optimizer_group.schedule_step()

        # Apply FP8 weight updates
        if fp8_optimizer is not None:
            fp8_optimizer.step()
            fp8_optimizer.zero_grad()

        # Logging with per-source losses (not batch-averaged)
        if i % log_interval == 0:
            # Log each source with its own average loss
            for src, data in per_source_losses.items():
                if data['count'] > 0:
                    src_loss = data['loss_sum'] / data['count']
                    src_sparsity = data['sparsity_sum'] / data['count']
                    history.append({
                        'step': i,
                        'source': src,
                        'type': 'latent',
                        'loss': src_loss,
                        'resolution': curr_res * curr_res,
                        'sparsity': src_sparsity,
                        'loss_logsnr': loss_stats_accum.get('logsnr_loss', 0.0)
                    })

            # Progress bar shows batch-averaged loss
            loss_val = loss_stats_accum.get('recon_loss', total_loss.item())
            sparsity_val = loss_stats_accum.get('sparsity', 0.0)
            pbar.set_postfix({
                'recon': f'{loss_val:.4f}',
                'sparse': f'{sparsity_val:.1%}'
            })

    # --- End of training: collect reconstruction samples for visualization ---
    if logger is not None and steps > 0:
        print("Collecting AE reconstruction samples...")
        recon_samples = {'x0': [], 'noisy_input': [], 'reconstruction': [], 'logsnr_map': [], 'source': []}

        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
            for res in [64, 128, 256]:
                try:
                    sample_blocks = iterator.generate_batch_list(4, resolution=res)
                except:
                    continue

                for b in sample_blocks:
                    if b.type != 'latent':
                        continue

                    x0 = b.content.unsqueeze(0)  # [1, C, H, W]
                    logsnr = b.logsnr.unsqueeze(0)  # [1, 1, H, W]
                    p = sparse_ae.patch_size
                    grid_shape = (x0.shape[2] // p, x0.shape[3] // p)

                    # DIRECT AE reconstruction (no projection bottleneck)
                    encoder_masks, decoder_masks = sparse_ae.build_masks(grid_shape, device)
                    output = sparse_ae(x0, logsnr_map=logsnr,
                                       encoder_masks=encoder_masks,
                                       decoder_masks=decoder_masks,
                                       grid_shape=grid_shape)
                    recon = output['recon'][0]  # [C, H, W]

                    recon_samples['x0'].append(b.content)
                    recon_samples['noisy_input'].append(b.content)
                    recon_samples['reconstruction'].append(recon)
                    recon_samples['logsnr_map'].append(b.logsnr)
                    recon_samples['source'].append(getattr(b, 'source', 'unknown'))

                    if len(recon_samples['x0']) >= 12:
                        break
                if len(recon_samples['x0']) >= 12:
                    break

        if recon_samples['x0']:
            from .plotting import plot_dset_reconstruction
            plot_dset_reconstruction(recon_samples, logger, name="ae_reconstruction", show_map=True, show_error=True)
            print(f"  Saved {len(recon_samples['x0'])} AE reconstruction samples")

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

    # Model now owns all components including sparse AE (if enabled)
    # So model.parameters() naturally includes everything needed for joint training

    # Optional FP8 conversion (transformer weights only, embeddings stay bf16)
    model = prepare_model_for_training(model, config, device=device)

    # Create FP8 optimizer for FP8Linear weights - must match the main optimizer type
    fp8_modules = collect_fp8_modules(model)
    fp8_optimizer = None
    if fp8_modules:
        opt_type = config['training']['optimizer']['type']
        n_fp8_params = sum(m.out_features * m.in_features for m in fp8_modules)

        if opt_type == "heterogeneous":
            # Muon for transformer weights in heterogeneous mode
            muon_cfg = config['training']['optimizer']['muon']
            fp8_optimizer = FP8Muon(
                fp8_modules,
                lr=muon_cfg['lr'],
                momentum=muon_cfg['momentum'],
                nesterov=muon_cfg['nesterov'],
                ns_steps=muon_cfg['ns_steps'],
            )
            print(f"[FP8] Created FP8Muon for {len(fp8_modules)} modules ({n_fp8_params:,} params)")
        else:
            # AdamW for all weights when using single optimizer
            adamw_cfg = config['training']['optimizer']['adamw']
            fp8_optimizer = FP8AdamW(
                fp8_modules,
                lr=adamw_cfg['lr'],
                betas=tuple(adamw_cfg['betas']),
                weight_decay=adamw_cfg['weight_decay'],
            )
            print(f"[FP8] Created FP8AdamW for {len(fp8_modules)} modules ({n_fp8_params:,} params)")

    # Build optimizer group for non-FP8 params (bias, embeddings, norms)
    optimizer_group = build_optimizer_group(model, config, total_steps=steps)
    print_optimizer_summary(optimizer_group, model)

    history = []
    # Pre-fetch bucketing flag to avoid lookup in loop
    bucketing_enabled = config['training']['bucketing']['enabled']
    dtype = config['dtype']
    use_amp = (dtype == torch.bfloat16) or (dtype == torch.float16)
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16)) 
    
    pbar = tqdm(range(steps), desc="train-denoise")
    for i in pbar:
        optimizer_group.zero_grad()
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

                        # Velocity loss via stateless function (handles variance weighting)
                        loss_v, loss_stats = denoise_loss_fn(
                            v_pred, target_v, target_l,
                            variance_tracker=variance_tracker
                        )
                        loss_v_accum = loss_v_accum + loss_v

                        # Defer stats (keep tensors, sync at logging time)
                        stat_entry = {
                            'step': i,
                            'source': getattr(block, 'source', 'unknown'),
                            'type': 'latent',
                            'loss_tensor': loss_stats['loss'],
                            'logsnr_tensor': target_l.mean().detach(),
                            'resolution': block.content.shape[-1] * block.content.shape[-2],
                        }
                        # Add variance-tracker specific stats if available
                        if 'loss_unweighted' in loss_stats:
                            stat_entry['loss_unweighted_tensor'] = loss_stats['loss_unweighted']
                        if 'weight_mean' in loss_stats:
                            stat_entry['weight_mean_tensor'] = loss_stats['weight_mean']
                        if 'weight_range' in loss_stats:
                            stat_entry['loss_var_tensor'] = loss_stats['weight_range']
                        if 'loss_var' in loss_stats:
                            stat_entry['loss_var_tensor'] = loss_stats['loss_var']
                        deferred_stats.append(stat_entry)

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
            # Step each optimizer in the group through scaler
            for spec in optimizer_group.specs.values():
                scaler.step(spec.optimizer)
            scaler.update()
            optimizer_group.schedule_step()
        else:
            total_loss.backward()
            optimizer_group.step()
            optimizer_group.schedule_step()

        # Apply FP8 weight updates (captured gradients -> FP8 storage)
        if fp8_optimizer is not None:
            fp8_optimizer.step()
            fp8_optimizer.zero_grad()

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
    from .context_manager import (
        render_latent_topology_embeddings,
        get_cached_latent_mask,
    )

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
    topo_cfg = sparse_ae_cfg.get('topology', {})
    diffusion_space = topo_cfg.get('diffusion_space', 'pixel')
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

    # Topology geometry config
    level_lambda = topo_cfg.get('level_lambda', 0.5)
    level_scale = topo_cfg.get('level_scale', 1.0)
    vertical_free = topo_cfg.get('vertical_attention_free', True)
    window_size = config['model']['window_size']

    # Training params from [training]
    steps = config['training']['steps']
    bs = config['training']['batch_size']
    lambda_coeff = config['training']['lambda_coeff']
    recon_weight = sparse_ae_cfg.get('ae_loss_weight', 0.1)
    logsnr_weight = sparse_ae_cfg.get('logsnr_loss_weight', 0.1)

    print(f"\n--- Training: Latent Diffusion (Main LDTformer) ---")
    print(f"    Steps: {steps}, n_levels: {n_levels}, code_dim: {code_dim}")
    print(f"    Level lambda: {level_lambda}, vertical_free: {vertical_free}")

    # Optimizer: encoder + model + decoder (joint training)
    # Build a combined module for heterogeneous optimization
    class LatentDiffusionModule(nn.Module):
        """Wrapper to enable heterogeneous optimizer classification."""
        def __init__(self, sparse_ae, main_model, patch_emb, patch_unemb):
            super().__init__()
            self.sparse_ae = sparse_ae
            self.denoiser = main_model  # Main LDTformer as denoiser
            # Use latent-specific projections (per-token, not concatenated)
            self.latent_code_proj = patch_emb.latent_code_proj
            self.logsnr_proj = patch_emb.logsnr_proj
            self.latent_code_unproj = patch_unemb.latent_code_unproj
            self.logsnr_decoder = patch_unemb.logsnr_decoder

    latent_diff_module = LatentDiffusionModule(sparse_ae, model, patch_emb, patch_unemb)

    # Apply FP8 conversion if configured (W8A16: 8-bit weights, 16-bit activations)
    latent_diff_module = prepare_model_for_training(latent_diff_module, config, device=device)

    # Create FP8 optimizer for FP8Linear weights - must match the main optimizer type
    fp8_modules = collect_fp8_modules(latent_diff_module)
    fp8_optimizer = None
    if fp8_modules:
        opt_type = config['training']['optimizer']['type']
        n_fp8_params = sum(m.out_features * m.in_features for m in fp8_modules)

        if opt_type == "heterogeneous":
            muon_cfg = config['training']['optimizer']['muon']
            fp8_optimizer = FP8Muon(
                fp8_modules,
                lr=muon_cfg['lr'],
                momentum=muon_cfg['momentum'],
                nesterov=muon_cfg['nesterov'],
                ns_steps=muon_cfg['ns_steps'],
            )
            print(f"[FP8] Created FP8Muon for {len(fp8_modules)} modules ({n_fp8_params:,} params)")
        else:
            adamw_cfg = config['training']['optimizer']['adamw']
            fp8_optimizer = FP8AdamW(
                fp8_modules,
                lr=adamw_cfg['lr'],
                betas=tuple(adamw_cfg['betas']),
                weight_decay=adamw_cfg['weight_decay'],
            )
            print(f"[FP8] Created FP8AdamW for {len(fp8_modules)} modules ({n_fp8_params:,} params)")

    optimizer_group = build_optimizer_group(latent_diff_module, config, total_steps=steps)
    print_optimizer_summary(optimizer_group, latent_diff_module)

    # Bucketing
    model_stride = patch_emb.stride
    bucket_mgr = build_bucket_manager_from_config(config, model_stride=model_stride)
    bucketing_enabled = config['training']['bucketing']['enabled']

    # Variance tracker
    var_cfg = config['training']['online_variance_correction']
    if var_cfg['enabled']:
        variance_tracker = OnlineVarianceTracker(device=device, **var_cfg)
    else:
        variance_tracker = None

    history = []
    use_amp = dtype in (torch.bfloat16, torch.float16)
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))
    log_interval = config['logging']['log_interval']

    pbar = tqdm(range(steps), desc="train-latent-diff")
    for i in pbar:
        optimizer_group.zero_grad()

        with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
            # Sample bucket
            if bucketing_enabled:
                bucket = bucket_mgr.sample_bucket()
                curr_res = bucket.resolution
                curr_bs = bucket.batch_size
            else:
                curr_res = 64
                curr_bs = bs

            # Get clean blocks
            clean_blocks = iterator.generate_batch_list(curr_bs, resolution=curr_res)

            # Group by grid_shape for batched processing
            latent_groups = {}
            for b in clean_blocks:
                if b.type != 'latent':
                    continue
                img = b.content
                lsnr = b.logsnr
                p = patch_emb.stride
                grid_shape = (img.shape[1] // p, img.shape[2] // p)
                if grid_shape not in latent_groups:
                    latent_groups[grid_shape] = []
                latent_groups[grid_shape].append((b, img, lsnr))

            if not latent_groups:
                continue

            # Accumulators
            loss_v_accum = torch.tensor(0.0, device=device)
            loss_recon_accum = torch.tensor(0.0, device=device)
            loss_logsnr_accum = torch.tensor(0.0, device=device)
            n_latent = 0

            for grid_shape, group in latent_groups.items():
                H_grid, W_grid = grid_shape
                n_patches = H_grid * W_grid
                total_tokens = n_patches * n_levels

                imgs = torch.stack([g[1] for g in group], dim=0)  # [B, C, H, W]
                logsnrs = torch.stack([g[2] for g in group], dim=0)  # [B, 1, H, W]
                batch_size = imgs.shape[0]
                p = sparse_ae.patch_size

                # Get encoder/decoder masks for sparse_ae
                encoder_masks, decoder_masks = patch_emb._get_masks(grid_shape, device)

                # Build latent diffusion topology (4D: highway, spatial_x, spatial_y, level)
                topo_embeds, level_ids, patch_ids = render_latent_topology_embeddings(
                    n_patches=n_patches,
                    n_levels=n_levels,
                    grid_shape=grid_shape,
                    device=device,
                    level_scale=level_scale,
                )

                # Get cached level-aware attention mask
                latent_mask = get_cached_latent_mask(
                    n_patches=n_patches,
                    n_levels=n_levels,
                    grid_shape=grid_shape,
                    window_size=window_size,
                    level_lambda=level_lambda,
                    vertical_free=vertical_free,
                    mode='local',
                    device=device,
                )

                # === ENCODE: Batched encoding with pre_quant ===
                codes_list, level_logsnrs, prequant_list = sparse_ae.encode_with_prequant(
                    imgs, logsnrs,
                    grid_shape=grid_shape,
                    encoder_masks=encoder_masks,
                    decoder_masks=decoder_masks
                )

                # Stack and flatten pre_quants across levels: [B, n_patches, n_levels, code_dim]
                pre_quant_stacked = torch.stack(prequant_list, dim=2)  # [B, N, L, D]
                pre_quant_flat = pre_quant_stacked.view(batch_size, total_tokens, code_dim)  # [B, N*L, D]

                # Expand logsnr to per-token (each level gets same logsnr per patch)
                logsnr_pooled = F.avg_pool2d(logsnrs, kernel_size=p, stride=p)
                logsnr_patches = logsnr_pooled.flatten(2).transpose(1, 2)  # [B, N, 1]
                logsnr_flat = logsnr_patches.repeat(1, n_levels, 1)  # [B, N*L, 1]

                # === DIFFUSION: Add noise to pre_quant, predict v-field ===
                noise = torch.randn_like(pre_quant_flat)

                # Compute alpha/sigma from logsnr
                alpha_sq = torch.sigmoid(logsnr_flat)  # [B, N*L, 1]
                sigma_sq = torch.sigmoid(-logsnr_flat)
                alpha = torch.sqrt(alpha_sq)
                sigma = torch.sqrt(sigma_sq)

                # Noisy codes: z_t = alpha * x_0 + sigma * noise
                noisy_codes = alpha * pre_quant_flat + sigma * noise

                # Target v-field: v = alpha * noise - sigma * x_0
                target_v = alpha * noise - sigma * pre_quant_flat

                # Project codes to model dim and add topology embedding
                # Use latent_code_proj for per-token (not concatenated) code projection
                h = patch_emb.latent_code_proj(noisy_codes)  # [B, N*L, model_dim]

                # Add logsnr conditioning
                logsnr_features = patch_emb.logsnr_proj(logsnr_flat)  # [B, N*L, model_dim]
                h = h + logsnr_features

                # Run through main LDTformer with 4D topology
                # The model expects: (tokens, topo_embeds, block_mask)
                h = model.forward_latent_diffusion(
                    h,
                    topo_embeds=topo_embeds.unsqueeze(0).expand(batch_size, -1, -1),
                    block_mask=latent_mask,
                )

                # Project back to code space (per-token, not concatenated)
                v_pred = patch_unemb.latent_code_unproj(h)  # [B, N*L, code_dim]
                logsnr_pred = patch_unemb.logsnr_decoder(h)  # [B, N*L, 1]

                # V-field loss in latent space
                sq_err_v = (v_pred - target_v) ** 2
                loss_v = sq_err_v.mean()
                loss_v_accum = loss_v_accum + loss_v * batch_size

                # === RECONSTRUCTION: Unflatten, re-quantize and decode ===
                if recon_weight > 0:
                    # Recover clean estimate from v-field: x_0 = alpha * z_t - sigma * v
                    clean_pred = alpha * noisy_codes - sigma * v_pred

                    # Unflatten back to [B, N, L, D]
                    clean_pred_stacked = clean_pred.view(batch_size, n_patches, n_levels, code_dim)

                    # Split to per-level list
                    prequant_pred_list = [clean_pred_stacked[:, :, lv, :] for lv in range(n_levels)]

                    # Batched quantize and decode
                    cumulative_recon = sparse_ae.quantize_and_decode(
                        prequant_pred_list, grid_shape, decoder_masks
                    )

                    # Reconstruction loss vs original image
                    recon_loss = F.mse_loss(cumulative_recon, imgs)
                    loss_recon_accum = loss_recon_accum + recon_loss * batch_size

                # LogSNR prediction loss
                loss_logsnr = F.l1_loss(logsnr_pred, logsnr_flat)
                loss_logsnr_accum = loss_logsnr_accum + loss_logsnr * batch_size

                n_latent += batch_size

            if n_latent == 0:
                continue

            # Average losses
            loss_v = loss_v_accum / n_latent
            loss_recon = loss_recon_accum / n_latent
            loss_logsnr = loss_logsnr_accum / n_latent

            # Total loss
            total_loss = loss_v + recon_weight * loss_recon + logsnr_weight * loss_logsnr

        # Backward
        if dtype == torch.float16:
            scaler.scale(total_loss).backward()
            for spec in optimizer_group.specs.values():
                scaler.step(spec.optimizer)
            scaler.update()
            optimizer_group.schedule_step()
        else:
            total_loss.backward()
            optimizer_group.step()
            optimizer_group.schedule_step()

        # Apply FP8 weight updates (captured gradients -> FP8 storage)
        if fp8_optimizer is not None:
            fp8_optimizer.step()
            fp8_optimizer.zero_grad()

        # Logging
        if i % log_interval == 0:
            history.append({
                'step': i,
                'source': 'combined',  # Latent diffusion operates on mixed data
                'type': 'latent_diff',
                'loss': loss_v.item(),  # Primary loss for plotting consistency
                'loss_v': loss_v.item(),
                'loss_recon': loss_recon.item() if recon_weight > 0 else 0.0,
                'loss_logsnr': loss_logsnr.item(),
                'loss_total': total_loss.item(),
                'resolution': curr_res * curr_res
            })

            pbar.set_postfix({
                'v': f'{loss_v.item():.4f}',
                'rec': f'{loss_recon.item():.4f}' if recon_weight > 0 else 'N/A'
            })

    return pd.DataFrame(history)
