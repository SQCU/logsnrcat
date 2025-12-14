# src/sample.py - Sampling solvers and visualization
import torch
from typing import List
from collections import defaultdict
import matplotlib.pyplot as plt

from .model import ContextBlock
from .train import (
    get_schedule, logsnr_to_alpha_sigma, euler_reverse_step,
    predict_velocity_from_blocks
)


@torch.no_grad()
def spatial_euler_solver(components, start_blocks: List[ContextBlock], target_logsnr, steps, mode, config, fixed_data=None):
    """Evolves a list of ContextBlocks from their initial state/logsnr to target_logsnr."""
    device = start_blocks[0].content.device
    taus = torch.linspace(0.0, 1.0, steps + 1, device=device)
    
    z_list = [b.content for b in start_blocks]
    lsnr_start_list = [b.logsnr for b in start_blocks]
    
    def get_target_map(start_map):
        if isinstance(target_logsnr, (float, int)):
            return torch.full_like(start_map, target_logsnr)
        return target_logsnr

    target_maps = [get_target_map(m) for m in lsnr_start_list]

    for i in range(steps):
        tau_curr = taus[i]
        tau_next = taus[i+1]

        lsnr_curr_list = []
        lsnr_next_list = []
        current_blocks = []
        
        for idx, (b, start, end) in enumerate(zip(start_blocks, lsnr_start_list, target_maps)):
            l_curr = (1 - tau_curr) * start + tau_curr * end
            l_next = (1 - tau_next) * start + tau_next * end
            lsnr_curr_list.append(l_curr)
            lsnr_next_list.append(l_next)
            
            current_blocks.append(ContextBlock(
                content=z_list[idx], logsnr=l_curr,
                type=b.type, causal=b.causal, shape_meta=b.shape_meta,
                group_id=b.group_id, id=b.id
            ))

        v_pred_list, _, _ = predict_velocity_from_blocks(components, current_blocks, mode)

        z_next_list = []
        for idx, (z, v, l_curr, l_next) in enumerate(zip(z_list, v_pred_list, lsnr_curr_list, lsnr_next_list)):
            if v is None:
                z_next_list.append(z)
                continue
            if fixed_data is not None and fixed_data[idx] is not None:
                z_next_list.append(fixed_data[idx])
            else:
                z_new = euler_reverse_step(z, v, l_curr, l_next)
                z_next_list.append(z_new)
        
        z_list = z_next_list
        
    return [z.clamp(0, 1) for z in z_list]


@torch.no_grad()
def sample_viz_dset(components, iterator, config):
    """Sample and visualize reconstruction from stratified noise."""
    model, _, _, _ = components
    model.eval()
    
    res = config.get("res", 32)
    n = config.get("num_samples", 8)
    
    clean_blocks = iterator.generate_batch_list(n)
    clean_blocks = [b for b in clean_blocks if b.type == "latent"][:n]
    
    min_snr = config.get("min_logsnr", -4.0)
    max_snr = config.get("max_logsnr", 1.0)
    device = clean_blocks[0].content.device
    start_vals = torch.rand(len(clean_blocks), device=device) * (max_snr - min_snr) + min_snr
    start_vals, _ = torch.sort(start_vals)
    
    start_blocks = []
    noisy_inputs = []
    x0s = []
    
    for i, b in enumerate(clean_blocks):
        x0s.append(b.content)
        l_map = torch.full_like(b.logsnr, start_vals[i])
        alpha, sigma = logsnr_to_alpha_sigma(l_map)
        eps = torch.randn_like(b.content)
        z_start = b.content * alpha + eps * sigma
        
        start_blocks.append(ContextBlock(
            content=z_start, logsnr=l_map,
            type=b.type, causal=b.causal, shape_meta=b.shape_meta,
            group_id=b.group_id, id=b.id
        ))
        noisy_inputs.append(z_start)
    
    z_final = spatial_euler_solver(
        components, start_blocks,
        config.get("target_logsnr", 10.0),
        config.get("sampling_steps", 50),
        config.get("mode", "naive"), config
    )
    model.train()
    
    return {
        "x0": torch.stack(x0s),
        "noisy_input": torch.stack(noisy_inputs),
        "reconstruction": torch.stack(z_final),
        "logsnr_map": torch.stack([b.logsnr for b in clean_blocks])
    }


@torch.no_grad()
def sample_viz_split_topology(components, iterator, config):
    """Sample and visualize using split topology (block logsnr maps)."""
    model, _, _, _ = components
    model.eval()
    
    res = config.get("res", 32)
    n = config.get("num_samples", 8)
    clean_blocks = iterator.generate_batch_list(n)
    clean_blocks = [b for b in clean_blocks if b.type == "latent"][:n]
    
    start_blocks = []
    noisy_inputs = []
    x0s = []
    
    for b in clean_blocks:
        x0s.append(b.content)
        alpha, sigma = logsnr_to_alpha_sigma(b.logsnr)
        eps = torch.randn_like(b.content)
        z_start = b.content * alpha + eps * sigma
        
        start_blocks.append(ContextBlock(
            content=z_start, logsnr=b.logsnr,
            type=b.type, causal=b.causal, shape_meta=b.shape_meta,
            group_id=b.group_id, id=b.id
        ))
        noisy_inputs.append(z_start)
    
    z_final = spatial_euler_solver(
        components, start_blocks,
        config.get("target_logsnr", 10.0),
        config.get("sampling_steps", 50),
        config.get("mode", "naive"), config
    )
    model.train()
    
    return {
        "x0": torch.stack(x0s),
        "noisy_input": torch.stack(noisy_inputs),
        "reconstruction": torch.stack(z_final),
        "logsnr_map": torch.stack([b.logsnr for b in clean_blocks])
    }


@torch.no_grad()
def sample_viz_causal_sweep(components, iterator, config):
    """
    Demonstrates Information Flow by sweeping Prefix SNR.
    
    Layout per Sequence (2 rows):
    [Noisy P1] [Noisy P2] [Noisy P3] [Gen Suffix]
    [MSE P1]   [MSE P2]   [MSE P3]   [MSE Suffix]
    
    Args:
        config keys:
            num_sweep_sequences (N): Number of rows to plot.
            sequence_length (M): Total frames per row (default 4).
            prefix_snr_range: Tuple (start, end) e.g. (2.0, -4.0).
    """
    model, _, _, _ = components
    model.eval()
    device = model.text_embed.weight.device
    
    # 1. Config & Data Prep
    N = config.get('num_sweep_sequences', 4)
    M = config.get('sequence_length', 4)
    snr_start, snr_end = config.get('prefix_snr_range', (2.0, -4.0))
    
    # Fetch enough data to find N valid sequences of length M
    # We fetch a large batch and group by ID
    fetch_buffer = N * M * 2 
    raw_blocks = iterator.generate_batch_list(fetch_buffer)
    
    # Group into sequences
    from collections import defaultdict
    groups = defaultdict(list)
    for b in raw_blocks:
        if b.type == 'latent':
            groups[b.group_id].append(b)
            
    # Filter for valid length
    sequences = [sorted(g, key=lambda x: x.id) for g in groups.values() if len(g) >= M]
    sequences = sequences[:N]
    
    if len(sequences) < N:
        print(f"⚠️ Warning: Requested {N} sequences, found {len(sequences)}")
        N = len(sequences)

    # 2. Setup Plot
    # Figure: N * 2 Rows, M Columns
    fig, axes = plt.subplots(N * 2, M, figsize=(3 * M, 4 * N))
    plt.subplots_adjust(hspace=0.3, wspace=0.1)
    
    # Linear SNR schedule for the SWEEP (Row by Row)
    sweep_snrs = torch.linspace(snr_start, snr_end, N)
    
    print(f"🧪 Running Causal Information Sweep (Prefix SNR {snr_start} -> {snr_end})...")

    for i, seq in enumerate(sequences):
        prefix_snr = sweep_snrs[i].item()
        
        # Prepare Solver Inputs
        start_blocks = []
        fixed_data = [] # Constraints
        gt_visuals = [] # For MSE calc
        
        # Identify suffix index (last frame)
        suffix_idx = M - 1
        
        for t in range(M):
            block = seq[t]
            gt_visuals.append(block.content)
            
            if t < suffix_idx:
                # --- PREFIX (Information Source) ---
                # Noise level determined by sweep
                l_map = torch.full_like(block.logsnr, prefix_snr)
                alpha, sigma = logsnr_to_alpha_sigma(l_map)
                eps = torch.randn_like(block.content)
                z_t = block.content * alpha + eps * sigma
                
                # We FIX this latent. The solver will NOT update it.
                # It acts purely as a Key/Value source for the Suffix.
                start_blocks.append(ContextBlock(
                    content=z_t, logsnr=l_map, type='latent', causal=True,
                    shape_meta=block.shape_meta, group_id=block.group_id, id=block.id
                ))
                fixed_data.append(z_t)
                
            else:
                # --- SUFFIX (Information Sink) ---
                # Always starts at Pure Noise (-4.0)
                l_map = torch.full_like(block.logsnr, -4.0)
                alpha, sigma = logsnr_to_alpha_sigma(l_map)
                z_t = block.content * alpha + torch.randn_like(block.content) * sigma
                
                # We EVOLVE this latent.
                start_blocks.append(ContextBlock(
                    content=z_t, logsnr=l_map, type='latent', causal=True,
                    shape_meta=block.shape_meta, group_id=block.group_id, id=block.id
                ))
                fixed_data.append(None) # None = Solve me

        # Run Solver
        # Uses the config's sampling steps (e.g. 20 or 50)
        z_final = spatial_euler_solver(
            components, start_blocks, 
            target_logsnr=config.get('target_logsnr', 10.0),
            steps=config.get('sampling_steps', 20),
            mode=config.get('mode', 'naive'),
            config=config,
            fixed_data=fixed_data
        )
        
        # 3. Visualization Logic
        row_top = i * 2
        row_bot = i * 2 + 1
        
        for t in range(M):
            # A. Visual State
            # Prefix: Shows the noisy condition provided
            # Suffix: Shows the hallucinated result
            viz_tens = z_final[t].permute(1,2,0).cpu().clamp(0,1).numpy()
            
            ax_img = axes[row_top, t] if N > 1 else axes[t] # Handle N=1 edge case logic if needed
            ax_img.imshow(viz_tens)
            ax_img.axis('off')
            
            # Headers
            if t == 0:
                ax_img.set_title(f"Prefix SNR: {prefix_snr:.1f}\n(Input)", fontsize=10, loc='left')
            elif t == suffix_idx:
                ax_img.set_title("Generated Suffix\n(Output)", fontsize=10, fontweight='bold')
                
            # B. Error Field (MSE Heatmap)
            # Diff against Ground Truth
            gt = gt_visuals[t]
            # Since z_final[t] might be on GPU and gt on GPU, compute diff there
            diff = (z_final[t] - gt).pow(2).mean(dim=0).cpu().numpy() # [H, W]
            
            ax_err = axes[row_bot, t]
            # Use a hot colormap for error. Normalize 0 to 0.1 for sensitivity.
            im_err = ax_err.imshow(diff, cmap='inferno', vmin=0, vmax=0.1)
            ax_err.axis('off')
            
            if t == 0:
                ax_err.set_title("MSE Field vs GT", fontsize=10, loc='left')

    model.train()
    return fig