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


@torch.no_grad()
def spatial_euler_solver(components, start_blocks: List[ContextBlock], target_logsnr, steps, mode, config, fixed_data=None):
    """Evolves a list of ContextBlocks from their initial state/logsnr to target_logsnr."""
    if not start_blocks: return []
    
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
    
    n = config.get("num_samples", 8)
    # Fetch blocks (likely mixed resolution)
    clean_blocks = iterator.generate_batch_list(n)
    clean_blocks = [b for b in clean_blocks if b.type == "latent"][:n]
    
    if not clean_blocks: return {}

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
    
    # RETURN LISTS instead of stacks
    return {
        "x0": x0s,
        "noisy_input": noisy_inputs,
        "reconstruction": z_final,
        "logsnr_map": [b.logsnr for b in clean_blocks]
    }


@torch.no_grad()
def sample_viz_split_topology(components, iterator, config):
    """Sample and visualize using split topology (block logsnr maps)."""
    model, _, _, _ = components
    model.eval()
    
    n = config.get("num_samples", 8)
    clean_blocks = iterator.generate_batch_list(n)
    clean_blocks = [b for b in clean_blocks if b.type == "latent"][:n]
    
    if not clean_blocks: return {}
    
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
    
    # RETURN LISTS instead of stacks
    return {
        "x0": x0s,
        "noisy_input": noisy_inputs,
        "reconstruction": z_final,
        "logsnr_map": [b.logsnr for b in clean_blocks]
    }


@torch.no_grad()
def sample_viz_causal_sweep(components, iterator, config):
    """
    Demonstrates Information Flow by sweeping Prefix SNR.
    Requires 'video_source_name' to be explicitly defined in config.
    """
    model, _, _, _ = components
    model.eval()
    
    # 1. Strict Configuration Extraction
    N = config.get('num_sweep_sequences', 4)
    M = config.get('sequence_length', 4)
    snr_start, snr_end = config.get('prefix_snr_range', (2.0, -4.0))
    video_sequence_structure = config.get('video_sequence_structure')
    source_name = config.get('video_source_name') # Strict requirement
    
    if not video_sequence_structure:
        print("⚠️ Causal Sweep skipped: 'video_sequence_structure' missing.")
        return None
    
    if not source_name:
        raise ValueError("sample_viz_causal_sweep requires explicit 'video_source_name'")

    # 2. Strict Iterator Lookup (No searching/guessing)
    target_split = None
    for split in iterator.splits:
        if split['name'] == source_name:
            target_split = split
            break
            
    if target_split is None:
        raise ValueError(f"Causal Sweep Error: Requested video source '{source_name}' not found in iterator.")
        
    if target_split['type'] != 'video':
        raise ValueError(f"Causal Sweep Error: Source '{source_name}' is type '{target_split['type']}', expected 'video'.")

    video_iterator_instance = target_split['iterator']

    # 3. Fetch Data (Directly from the specific sub-iterator)
    flat_blocks = video_iterator_instance.generate_batch_list(N, video_sequence_structure)
    
    # 4. Re-group sequences
    sequences = []
    current_seq = []
    if flat_blocks:
        curr_gid = flat_blocks[0].group_id
        for b in flat_blocks:
            if b.group_id == curr_gid:
                current_seq.append(b)
            else:
                if len(current_seq) == M: sequences.append(sorted(current_seq, key=lambda x: x.id))
                current_seq = [b]
                curr_gid = b.group_id
        if len(current_seq) == M: sequences.append(sorted(current_seq, key=lambda x: x.id))
    
    sequences = sequences[:N]
    if not sequences:
        print(f"⚠️ Video iterator {source_name} returned no valid sequences.")
        return None

    # 5. Visualization Setup
    N_actual = len(sequences)
    fig, axes = plt.subplots(N_actual * 2, M, figsize=(3 * M, 4 * N_actual))
    if N_actual == 1 and M == 1: axes = np.array([[axes]])
    elif N_actual == 1: axes = axes.reshape(2, M)
    
    plt.subplots_adjust(hspace=0.3, wspace=0.1)
    sweep_snrs = torch.linspace(snr_start, snr_end, N_actual)
    
    print(f"🧪 Running Causal Sweep on '{source_name}' (Prefix SNR {snr_start} -> {snr_end})...")

    for i, seq in enumerate(sequences):
        prefix_snr = sweep_snrs[i].item()
        suffix_idx = M - 1
        
        start_blocks = []
        fixed_data = []
        gt_visuals = []
        
        for t in range(M):
            block = seq[t]
            gt_visuals.append(block.content)
            
            if t < suffix_idx:
                # Prefix (Source)
                l_map = torch.full_like(block.logsnr, prefix_snr)
                alpha, sigma = logsnr_to_alpha_sigma(l_map)
                z_t = block.content * alpha + torch.randn_like(block.content) * sigma
                
                start_blocks.append(ContextBlock(
                    content=z_t, logsnr=l_map, type='latent', causal=True,
                    shape_meta=block.shape_meta, group_id=block.group_id, id=block.id
                ))
                fixed_data.append(z_t)
            else:
                # Suffix (Sink) - Pure Noise
                l_map = torch.full_like(block.logsnr, -4.0)
                z_t = torch.randn_like(block.content) # Pure noise assumption for generation
                
                start_blocks.append(ContextBlock(
                    content=z_t, logsnr=l_map, type='latent', causal=True,
                    shape_meta=block.shape_meta, group_id=block.group_id, id=block.id
                ))
                fixed_data.append(None)

        # Solver
        z_final = spatial_euler_solver(
            components, start_blocks, 
            target_logsnr=config.get('target_logsnr', 10.0),
            steps=config.get('sampling_steps', 20),
            mode=config.get('mode', 'naive'),
            config=config,
            fixed_data=fixed_data
        )
        
        # Plotting
        row_top = i * 2
        row_bot = i * 2 + 1
        
        for t in range(M):
            # Image
            ax_img = axes[row_top, t] if N_actual > 1 else axes[t]
            viz = z_final[t].detach().cpu().permute(1,2,0).clamp(0,1).numpy()
            ax_img.imshow(viz)
            ax_img.axis('off')
            
            if t == 0: ax_img.set_title(f"Prefix: {prefix_snr:.1f}", fontsize=9)
            
            # Error
            gt = gt_visuals[t]
            diff = (z_final[t] - gt).pow(2).mean(dim=0).cpu().numpy()
            ax_err = axes[row_bot, t]
            ax_err.imshow(diff, cmap='inferno', vmin=0, vmax=0.1)
            ax_err.axis('off')

    model.train()
    return fig