# src/sample.py - Sampling solvers and visualization
import torch
from typing import List
from collections import defaultdict
import matplotlib.pyplot as plt

from .model import ContextBlock
from .utils import (
    get_schedule, logsnr_to_alpha_sigma,
    predict_velocity_from_blocks
)


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


class NoiseFactory:
    """
    Centralizes logic for applying diffusion noise to ContextBlocks.
    """
    @staticmethod
    def apply_noise(block: ContextBlock, target_logsnr: float = None, noise: torch.Tensor = None) -> ContextBlock:
        """Returns a NEW ContextBlock with noisy content and updated logsnr."""
        if block.type != 'latent':
            return block # Text passes through untouched

        device = block.content.device
        
        # Determine LogSNR Map
        if target_logsnr is not None:
            if isinstance(target_logsnr, (float, int)):
                logsnr_map = torch.full((1, *block.content.shape[-2:]), target_logsnr, device=device)
            else:
                logsnr_map = target_logsnr
        else:
            # Use existing if not overridden
            logsnr_map = block.logsnr if block.logsnr is not None else torch.zeros((1, *block.content.shape[-2:]), device=device)

        from src.utils import logsnr_to_alpha_sigma
        alpha, sigma = logsnr_to_alpha_sigma(logsnr_map)
        
        if noise is None:
            noise = torch.randn_like(block.content)
            
        z_t = block.content * alpha + noise * sigma
        
        # Copy metadata, update content
        return ContextBlock(
            content=z_t,
            type=block.type,
            causal=block.causal,
            logsnr=logsnr_map,
            group_id=block.group_id,
            id=block.id,
            source=getattr(block, 'source', 'unknown')
        )

@torch.no_grad()
def spatial_euler_solver(
    components, 
    start_blocks: List[ContextBlock], 
    target_logsnr: float, 
    steps: int, 
    mode: str, 
    fixed_indices: List[int] = [] 
) -> List[torch.Tensor]:
    """
    Evolves a list of ContextBlocks from their initial state/logsnr to target_logsnr.
    Blocks at indices specified in fixed_indices are excluded from the Euler update step.
    """
    if not start_blocks: return []
    
    device = start_blocks[0].content.device
    taus = torch.linspace(0.0, 1.0, steps + 1, device=device)
    
    # Initialize state
    z_list = [b.content for b in start_blocks]
    lsnr_start_list = [b.logsnr for b in start_blocks]
    # Target map logic - MUST handle None for text blocks
    target_maps = []
    for l in lsnr_start_list:
        if l is None:
            target_maps.append(None)
        elif isinstance(target_logsnr, (float, int)):
            target_maps.append(torch.full_like(l, target_logsnr))
        else:
            target_maps.append(target_logsnr)

    for i in range(steps):
        tau_curr = taus[i]
        tau_next = taus[i+1]

        lsnr_curr_list = []
        lsnr_next_list = []
        current_blocks = []
        
        # 1. Build Batch for Model with interpolated LogSNR
        for idx, (b, start, end) in enumerate(zip(start_blocks, lsnr_start_list, target_maps)):
            # Interpolate schedule map
            l_curr = (1 - tau_curr) * start + tau_curr * end
            l_next = (1 - tau_next) * start + tau_next * end
            
            lsnr_curr_list.append(l_curr)
            lsnr_next_list.append(l_next)
            
            # Construct block for prediction
            current_blocks.append(ContextBlock(
                content=z_list[idx], 
                logsnr=l_curr,
                type=b.type, 
                causal=b.causal, 
                shape_meta=b.shape_meta,
                group_id=b.group_id, 
                id=b.id
            ))

        # 2. Predict Velocity
        v_pred_list, _, _ = predict_velocity_from_blocks(components, current_blocks, mode)

        # 3. Update State
        z_next_list = []
        for idx, (z, v, l_curr, l_next) in enumerate(zip(z_list, v_pred_list, lsnr_curr_list, lsnr_next_list)):
            # Passthrough Logic
            #for like tokens i guess
            if l_s is None or l_e is None:
                lsnr_curr_list.append(None)
                lsnr_next_list.append(None)
                # Reuse existing block for metadata/text content
                curr_blocks.append(b) 
                continue 
            
            if v is None:
                # Text/Non-latent passes through
                z_next_list.append(z)
                continue
            
            if idx in fixed_indices:
                # Fixed latent: hold current value constant
                z_next_list.append(z)
            else:
                # Denoise latent
                z_new = euler_reverse_step(z, v, l_curr, l_next)
                z_next_list.append(z_new)
        
        z_list = z_next_list

    # Return list of tensors (clamped for validity)
    return [z.clamp(0, 1) for z in z_list]

@torch.no_grad()
def sample_viz_dset(components, iterator, config_dict, logger):
    n = config_dict['num_samples']
    # FIX: Pass resolution explicit from config
    res_arg = config_dict.get('res', 32)
    
    clean = iterator.generate_batch_list(n, resolution=res_arg)
    
    lat_indices = [i for i,b in enumerate(clean) if b.type == 'latent']
    if not lat_indices: return
    
    start_blocks = []
    for b in clean:
        if b.type == 'latent':
            nb = NoiseFactory.apply_noise(b)
            start_blocks.append(nb)
        else:
            start_blocks.append(b)
            
    results = spatial_euler_solver(
        components, start_blocks, 
        config_dict['target_logsnr'], config_dict['steps'], config_dict['mode']
    )
    
    x0 = [clean[i].content for i in lat_indices]
    noises = [start_blocks[i].content for i in lat_indices]
    recon = [results[i] for i in lat_indices]
    lmaps = [start_blocks[i].logsnr for i in lat_indices]
    
    path = logger.run_dir / f"stratified_{res_arg}.png"
    plot_dset_reconstruction(x0, noises, recon, lmaps, path)
   
    # RETURN LISTS instead of stacks
    #return {
    #    "x0": x0s,
    #    "noisy_input": noisy_inputs,
    #    "reconstruction": z_final,
    #    "logsnr_map": [b.logsnr for b in clean_blocks]
    #}


@torch.no_grad()
def sample_viz_split_topology(components, iterator, config, logger):
    """
    Standard evaluation: Take a batch, noise it (respecting split topology), denoise, plot.
    """
    # 1. Get Data
    n = config.get("num_samples", 8)
    res_arg = config.get('res', 32)
    clean_blocks = iterator.generate_batch_list(n, resolution=res_arg)
    # Filter for latents to check (text passes through)
    latent_indices = [i for i, b in enumerate(clean_blocks) if b.type == 'latent']
    
    if not latent_indices: return

    # 2. Prepare Noisy Input
    start_blocks = []
    noisy_tensors = []
    
    for b in clean_blocks:
        if b.type == 'latent':
            # Use the logsnr provided by iterator (Split Topology)
            nb = NoiseFactory.apply_noise(b, target_logsnr=b.logsnr)
            start_blocks.append(nb)
            noisy_tensors.append(nb.content)
        else:
            start_blocks.append(b)
            
    # 3. Solve
    z_final = spatial_euler_solver(
        components, start_blocks,
        target_logsnr=config.get("target_logsnr", 10.0),
        steps=config.get("steps", 50),
        mode=config.get("mode", "naive")
    )
    
    # 4. Plot
    # Extract just the latents for plotting
    x0s = [clean_blocks[i].content for i in latent_indices]
    noises = [noisy_tensors[i] if i < len(noisy_tensors) else None for i in range(len(latent_indices))] # logic fix needed if mixed
    # Actually, simpler to just zip through latent_indices
    
    # Re-gather results
    recon_latents = [z_final[i] for i in latent_indices]
    lmaps = [clean_blocks[i].logsnr for i in latent_indices]
    
    output_path = logger.run_dir / f"reconstruction_{config.get('mode')}.png"
    
    from .plotting import plot_dset_reconstruction
    plot_dset_reconstruction(x0s, noises, recon_latents, lmaps, output_path, show_map=True)



@torch.no_grad()
def sample_viz_causal_sweep(components, iterator, config, logger):
    """
    Causal Sweep: Generates sequences where prefix frames have variable noise,
    and suffix frames are fully masked (high noise) to be generated.
    """
    # 1. Config & Data Fetch
    N_sweeps = config.get('num_sweep_sequences', 4)
    M_len = config.get('sequence_length', 4)
    snr_start, snr_end = config.get('prefix_snr_range', (2.0, -4.0))
    source_name = config.get('video_source_name')
    res_arg = config.get('res', 32)
    
    # (Assume iterator filtering logic similar to before to get specific source)
    # For brevity, assuming iterator returns grouped blocks correctly
    flat_blocks = iterator.generate_batch_list(N_sweeps * M_len, resolution=res_arg) # Simplified
    
    # Group into sequences [N, M]
    sequences = []
    for i in range(0, len(flat_blocks), M_len):
        sequences.append(flat_blocks[i : i+M_len])
    
    sweep_snrs = torch.linspace(snr_start, snr_end, len(sequences))
    all_predictions = []
    
    # 2. Iterate Sweeps
    for i, seq in enumerate(sequences):
        prefix_snr = sweep_snrs[i].item()
        
        # Prepare Noisy State
        start_blocks = []
        fixed_indices = []
        
        for t, block in enumerate(seq):
            # Logic: Last frame is target (Gen), others are Context (Prefix)
            is_target = (t == M_len - 1)
            
            if is_target:
                # Target: High Noise (Standard Gen)
                nb = NoiseFactory.apply_noise(block, target_logsnr=prefix_snr) # Start from noise
                start_blocks.append(nb)
            else:
                # Context: Variable Noise
                nb = NoiseFactory.apply_noise(block, target_logsnr=prefix_snr)
                start_blocks.append(nb)
                fixed_indices.append(t) # Don't update this block during solver
        
        # 3. Solve
        final_contents = spatial_euler_solver(
            components, start_blocks,
            target_logsnr=10.0,
            steps=config.get('steps', 20),
            mode=config.get('mode', 'naive'),
            fixed_indices=fixed_indices
        )
        all_predictions.append(final_contents)

    # 4. Plot
    output_path = logger.run_dir / f"causal_sweep_{source_name}.png"
    plot_causal_sweep(sequences, all_predictions, sweep_snrs, output_path)