# src/sample.py - Sampling solvers and visualization
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional
import math
from collections import defaultdict
import matplotlib.pyplot as plt

from .model import ContextBlock
from .utils import (
    logsnr_to_alpha_sigma,
    run_model_forward,
    run_model_forward_sampling
)

def euler_forward_step(x0, logsnr, noise=None):
    """
    Diffuses x0 -> z_t. Returns z_t and the target velocity v_true.
    """
    if noise is None:
        print("take total responsibility for data you pass to samplers, dunkass")
        return (f"i hope this gives you a type error", f"never should have come here")
    # shape errors by function users must be punished by explicit errors.
    # scalar snr fields are a special case: broadcast them yourself if you want to use them.
    alpha, sigma = logsnr_to_alpha_sigma(logsnr)
    
    z_t = x0 * alpha + noise * sigma
    v_true = alpha * noise - sigma * x0
    return z_t, v_true, noise

def euler_reverse_step(z_t, v_pred, logsnr_from, logsnr_to):
    """
    Denoises z_t -> z_{t-1}.
    """
    # shape errors by function users must be punished by explicit errors.
    # scalar snr fields are a special case: broadcast them yourself if you want to use them.
    alpha_from, sigma_from = logsnr_to_alpha_sigma(logsnr_from)
    alpha_to, sigma_to = logsnr_to_alpha_sigma(logsnr_to)
    
    # Reconstruct x0 (prediction)
    x0_pred = alpha_from * z_t - sigma_from * v_pred
    # Reconstruct eps (prediction)
    eps_pred = sigma_from * z_t + alpha_from * v_pred
    
    # Step to next level
    z_next = alpha_to * x0_pred + sigma_to * eps_pred
    return z_next

# ==============================================================================
# 1. Functional Primitives (Pure, Stateless)
# ==============================================================================

def discrete_logit_step(logits: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    """
    Samples a token ID from logits.
    Args:
        logits: [B, Vocab]
        config: Dict containing 'temperature', 'top_p', 'top_k'
    Returns:
        [B, 1] LongTensor of token IDs
    """
    temp = config.get('temperature', 1.0)
    top_p = config.get('top_p', 0.9)
    top_k = config.get('top_k', 0) # 0 = disable

    # 1. Temperature
    logits = logits / max(temp, 1e-6)

    # 2. Top-K
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v[:, -1].unsqueeze(1)
        logits = torch.where(logits < pivot, -float('Inf'), logits)

    # 3. Top-P (Nucleus)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter back to original indices
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = torch.where(indices_to_remove, -float('Inf'), logits)

    probs = F.softmax(logits, dim=-1)
    token_id = torch.multinomial(probs, num_samples=1)
    return token_id

def spatial_euler_step(
    z_t: torch.Tensor, 
    v_pred: torch.Tensor, 
    logsnr_curr: torch.Tensor, 
    logsnr_next: torch.Tensor
) -> torch.Tensor:
    """
    Pure Algebra: Denoises z_t -> z_{t-1} using flow matching Euler step.
    Handles spatially varying schedules (spatial_euler) naturally via tensor broadcasting.
    
    Args:
        z_t: [C, H, W] or [B, C, H, W]
        v_pred: Velocity prediction from model
        logsnr_curr: LogSNR at time t
        logsnr_next: LogSNR at time t-1
    """
    alpha_curr, sigma_curr = logsnr_to_alpha_sigma(logsnr_curr)
    alpha_next, sigma_next = logsnr_to_alpha_sigma(logsnr_next)
    
    # Predict x0 and epsilon
    # v = alpha * eps - sigma * x0
    # z = alpha * x0 + sigma * eps
    # -> x0 = alpha * z - sigma * v
    x0_pred = alpha_curr * z_t - sigma_curr * v_pred
    eps_pred = sigma_curr * z_t + alpha_curr * v_pred
    
    # Recompose at next step
    z_next = alpha_next * x0_pred + sigma_next * eps_pred
    return z_next



# ==============================================================================
# 3. Constructors
# ==============================================================================

def construct_tau_field(
    targets: List[Dict[str, Any]], 
    steps: int, 
    device: torch.device
) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Pre-calculates the diffusion schedule (Tau Field) for all targets.
    Returns a list of steps, where each step contains a list of (curr, next) logsnr tuples 
    corresponding to the targets.
    """
    taus = torch.linspace(0.0, 1.0, steps + 1, device=device)
    schedule_field = []

    for i in range(steps):
        tau_curr = taus[i]
        tau_next = taus[i+1]
        
        step_targets = []
        for tgt in targets:
            start = tgt.get('start_map')
            end = tgt.get('end_map')
            
            if start is None or end is None:
                raise ValueError(f"Target {tgt.get('idx')} missing start_map or end_map. Cannot construct Tau Field.")
                
            # Linear interpolation in LogSNR space
            # Note: This defines the integration topology.
            l_curr = (1 - tau_curr) * start + tau_curr * end
            l_next = (1 - tau_next) * start + tau_next * end
            
            step_targets.append((l_curr, l_next))
        
        schedule_field.append(step_targets)
        
    return schedule_field

def create_noisy_latent_span(
    x0: torch.Tensor,
    span_config: Dict[str, Any], 
    group_id: int,
    device: torch.device,
    id_tag: str = "gen"
) -> Tuple[ContextBlock, torch.Tensor, torch.Tensor]:
    """
    Creates a new Latent ContextBlock initialized by forward-noising an input x0.
    Args:
        x0: The tensor to be noised (e.g. Zeros for pure gen, or a reference image).
        span_config: Config containing 'start_logsnr', 'target_logsnr'.
    """
    # 1. Parse Config
    # We infer shape from x0 to ensure consistency
    shape = x0.shape
    start_snr = span_config['start_logsnr']
    target_snr = span_config['target_logsnr']
    
    # 2. Construct Maps
    # Explicit broadcast of scalar SNR to spatial map
    start_map = torch.full((1, *shape[-2:]), start_snr, device=device)
    end_map = torch.full((1, *shape[-2:]), target_snr, device=device)
    
    # 3. Create Content (Noise Injection)
    # z_start = alpha * x0 + sigma * eps
    alpha, sigma = logsnr_to_alpha_sigma(start_map)
    noise = torch.randn_like(x0)
    
    z_start = x0 * alpha + noise * sigma
    
    block = ContextBlock(
        content=z_start,
        type='latent',
        causal=True,
        shape_meta=shape,
        logsnr=start_map,
        group_id=group_id,
        id=id_tag
    )
    
    return block, start_map, end_map


# ==============================================================================
# 4. Bland Logitslop Heading!!!
# ==============================================================================

class MultiTurnContext:
    """
    The Single Source of Truth for the simulation state.
    Wraps the list of ContextBlocks and handles updates.

    Embedding Cache Management:
        - Blocks track their own cached embeddings
        - update_content() invalidates the block's embedding cache
        - update_metadata() (logsnr only) also invalidates for latents
        - append_span() adds new blocks without embeddings
    """
    def __init__(self, initial_blocks: List[ContextBlock]):
        # Deep copy to ensure we don't mutate the original dataset lists
        self.blocks = [
            ContextBlock(
                content=b.content.clone() if isinstance(b.content, torch.Tensor) else b.content,
                type=b.type,
                causal=b.causal,
                shape_meta=b.shape_meta,
                logsnr=b.logsnr.clone() if b.logsnr is not None else None,
                group_id=b.group_id,
                id=b.id,
                source=b.source,
                # Don't copy cached embeddings - they'll be computed fresh
                cached_embedding=None,
                embedding_content_hash=None
            ) for b in initial_blocks
        ]

    def update_content(self, idx: int, new_content: torch.Tensor):
        """Update content and invalidate cached embedding."""
        if not isinstance(new_content, torch.Tensor):
            raise TypeError(f"Content update must be Tensor, got {type(new_content)}")
        self.blocks[idx].content = new_content
        # Content changed - embedding is now invalid
        self.blocks[idx].invalidate_embedding()

    def update_metadata(self, idx: int, new_logsnr: torch.Tensor):
        """Update logsnr and invalidate cached embedding for latents."""
        self.blocks[idx].logsnr = new_logsnr
        # For latents, logsnr is part of the embedding input
        if self.blocks[idx].type == 'latent':
            self.blocks[idx].invalidate_embedding()

    def append_span(self, span: ContextBlock):
        """Append a new span (embeddings will be computed on next forward)."""
        self.blocks.append(span)

    def get_spans(self) -> List[ContextBlock]:
        return self.blocks

    def get_active_indices(self) -> List[int]:
        """Return indices of blocks that need unembedding (typically latents being sampled)."""
        return [i for i, b in enumerate(self.blocks) if b.type == 'latent']


@torch.no_grad()
def taufield_spatial_sampling(
    components,
    context: MultiTurnContext,
    targets: List[Dict[str, Any]],
    sampling_config: Dict[str, Any]
):
    """
    Executes a spatial sampling job.
    Driven by the Tau Field (Schedule) and explicit Targets.
    Does NOT invent noise. Does NOT guess targets.

    Optimizations:
    - Uses incremental embedding (cached embeddings for unchanged blocks)
    - Selective unembedding (only decode target latents, not text/stable blocks)
    """
    if not targets:
        return # Nothing to do

    steps = sampling_config.get('steps', 50)
    device = context.blocks[targets[0]['idx']].content.device

    # Indices we need to unembed (just the target latents)
    unembed_indices = [tgt['idx'] for tgt in targets]

    # 1. Construct the Field
    tau_field = construct_tau_field(targets, steps, device)

    # 2. Integration Loop
    for step_idx, step_schedule in enumerate(tau_field):

        # A. Sync Context Metadata
        # update_metadata invalidates embedding cache for affected blocks
        for i, (l_curr, _) in enumerate(step_schedule):
            tgt_idx = targets[i]['idx']
            context.update_metadata(tgt_idx, l_curr)

        # B. Observation (Model Forward)
        # Uses incremental embedding - only re-embeds blocks whose content/logsnr changed
        # Only unembeds the target latents (not text prefixes etc)
        current_blocks = context.get_spans()
        _, decoded_outputs, _, num_recomputed = run_model_forward_sampling(
            components, current_blocks, unembed_indices=unembed_indices
        )

        # C. Integration (Step)
        for i, (l_curr, l_next) in enumerate(step_schedule):
            tgt_idx = targets[i]['idx']

            # Extract Velocity from decoded outputs
            output_dict = decoded_outputs[tgt_idx]
            if output_dict is None or 'image_vpreds' not in output_dict:
                raise RuntimeError(f"Model did not return v-predictions for latent target at index {tgt_idx}.")

            v_pred = output_dict['image_vpreds']

            # Extract Current State
            z_curr = context.blocks[tgt_idx].content

            # Pure Functional Step
            z_next = spatial_euler_step(z_curr, v_pred, l_curr, l_next)

            # Mutate State (this invalidates embedding for this block)
            context.update_content(tgt_idx, z_next)

    # 3. Finalize
    for tgt in targets:
        context.update_metadata(tgt['idx'], tgt['end_map'])

    
@torch.no_grad()
def discrete_logit_sampling(
    components,
    context: MultiTurnContext,
    targets: List[Dict[str, Any]],
    sampling_config: Dict[str, Any]
):
    """
    Autoregressive Text Sampling.
    Grows text spans token-by-token using the model's text head.

    Optimizations:
    - Uses incremental embedding (caches embeddings for unchanged blocks)
    - Selective unembedding (only decode target text spans)
    """
    if not targets:
        return

    max_new_tokens = sampling_config.get('max_new_tokens', 128)
    eos_token_id = sampling_config.get('eos_token_id', None)

    # Indices we need to unembed (just the target text spans)
    unembed_indices = [tgt['idx'] for tgt in targets]

    # Track completion status for batch generation
    is_finished = {tgt['idx']: False for tgt in targets}

    for _ in range(max_new_tokens):
        # 1. Early Exit
        if all(is_finished.values()):
            break

        # 2. Observe (Optimized Forward Pass)
        # Uses incremental embedding - only re-embeds blocks whose content changed
        # Only unembeds the target text spans (not latent prefixes etc)
        current_blocks = context.get_spans()
        _, decoded_outputs, _, _ = run_model_forward_sampling(
            components, current_blocks, unembed_indices=unembed_indices
        )
        
        # 3. Sample & Mutate
        for tgt in targets:
            idx = tgt['idx']
            if is_finished[idx]: 
                continue
            
            output_dict = decoded_outputs[idx]
            
            if 'text_logits' not in output_dict:
                raise RuntimeError(f"Target {idx} produced no text_logits. Ensure target is a text block.")
            
            # Logits: [Seq_Len, Vocab_Size]
            # We predict the *next* token based on the *last* token's hidden state.
            logits_seq = output_dict['text_logits']
            next_token_logits = logits_seq[-1, :].unsqueeze(0) # [1, Vocab]
            
            # Sample (Stateless)
            # returns [1, 1]
            token_tensor = discrete_logit_step(next_token_logits, sampling_config)
            token_val = token_tensor.item()
            
            # Update Context (Append)
            current_tensor = context.blocks[idx].content
            # [L] + [1] -> [L+1]
            new_tensor = torch.cat([current_tensor, token_tensor.flatten()])
            
            context.update_content(idx, new_tensor)
            
            # Update Metadata (Keep shape_meta consistent for SpanEmbedder)
            context.blocks[idx].shape_meta = (new_tensor.shape[0],)
            
            # Check EOS
            if eos_token_id is not None and token_val == eos_token_id:
                is_finished[idx] = True


# ==============================================================================
# 6. Dispatch & Session Management
# ==============================================================================

SAMPLER_REGISTRY = {
    "taufield_euler": taufield_spatial_sampling,
    "discrete_logit": discrete_logit_sampling
}

# Default configuration acting as documentation and fallback
multiturn_session_default_config = {
    # Latent Diffusion Defaults
    "latent_defaults": {
        "steps": 50,
        "start_logsnr": -4.0,
        "target_logsnr": 10.0,
        "default_shape": (3, 32, 32),
        "channels": 3
    },
    # Text AR Defaults
    "text_defaults": {
        "max_new_tokens": 128,
        "temperature": 1.0,
        "top_p": 0.9,
        "top_k": 0
    },
    # System behavior
    "strict_mode": True,
    "device_fallback": "cuda"
}

def execute_multiturn_session(
    components,
    input_ctx: MultiTurnContext,
    sampling_queries: List[Dict[str, Any]],
    mturn_sesh_cfg: Dict[str, Any] = multiturn_session_default_config
) -> List[ContextBlock]:
    """
    The High-Level Controller.
    Parses configurations into structural mutations and sampling jobs.
    """
    # 1. Device resolution
    if input_ctx.get_spans():
        device = input_ctx.blocks[0].content.device 
    else:
        device = torch.device(mturn_sesh_cfg.get("device_fallback", "cuda"))


    for q_idx, query in enumerate(sampling_queries):
        active_targets = []
        mutation_type = query.get('mutation')

        # === 1. Structural Mutations ===
        
        # Calculate Next Group ID
        current_spans = input_ctx.get_spans()
        if current_spans:
            next_group_id = max(s.group_id for s in current_spans) + 1
        else:
            next_group_id = 0

        if mutation_type == 'append_latent':
            span_cfg = query['new_span_config']
            
            # Determine x0 Source
            source_type = query.get('append_source', 'zeros') # Default to zeros if unspecified
            
            if source_type == 'zeros':
                shape = tuple(span_cfg['shape'])
                x0 = torch.zeros(shape, device=device)
            elif source_type == 'copy_from':
                src_idx = query.get('append_index', -1)
                # Resolve index to absolute
                if src_idx < 0: search_idx = len(current_spans) + src_idx
                else: search_idx = src_idx
                
                # Heuristic: Find nearest latent block backwards from search_idx
                # This prevents copying text blocks (LongTensor) which crash randn_like
                x0 = None
                for i in range(search_idx, -1, -1):
                    if current_spans[i].type == 'latent':
                        x0 = current_spans[i].content.clone()
                        break
                
                if x0 is None:
                    # Fallback if no latent found: use configured shape or error
                    if 'shape' in span_cfg:
                        x0 = torch.zeros(tuple(span_cfg['shape']), device=device)
                    else:
                        raise ValueError(f"append_source='copy_from' at idx {src_idx} failed: No preceding latent block found and no explicit shape in config.")
            else:
                raise ValueError(f"Unknown append_source: {source_type}")

            new_span, start_map, end_map = create_noisy_latent_span(
                x0,
                span_cfg, 
                group_id=next_group_id,
                device=device,
                id_tag=f"gen_lat_{q_idx}"
            )
            
            input_ctx.append_span(new_span)
            
            new_idx = len(input_ctx.get_spans()) - 1
            active_targets.append({
                'idx': new_idx,
                'start_map': start_map,
                'end_map': end_map
            })

        elif mutation_type == 'append_text':
            span_cfg = query['new_span_config']
            init_content = span_cfg['initial_content']
            if not isinstance(init_content, torch.Tensor):
                init_content = torch.tensor(init_content, dtype=torch.long, device=device)
            else:
                init_content = init_content.to(device)
            
            new_span = ContextBlock(
                content=init_content,
                type='text',
                causal=True,
                group_id=next_group_id,
                id=f"gen_txt_{q_idx}"
            )
            input_ctx.append_span(new_span)
            new_idx = len(input_ctx.get_spans()) - 1
            active_targets.append({'idx': new_idx})

        elif mutation_type == 'inplace':
            target_indices = query.get('target_indices', [])
            if not target_indices:
                raise ValueError("Mutation 'inplace' requires 'target_indices'.")
            active_targets = target_indices

        elif mutation_type is not None:
             raise ValueError(f"Unknown mutation type: {mutation_type}")

        # === 2. Dispatch to Sampler ===
        sampler_name = query.get('sampler')
        if not sampler_name:
            raise ValueError(f"Query {q_idx} missing 'sampler' key.")
            
        sampler_fn = SAMPLER_REGISTRY.get(sampler_name)
        if not sampler_fn:
            raise ValueError(f"Unknown sampler: {sampler_name}")
        
        sampling_cfg = query['sampling_config']
        sampler_fn(components, input_ctx, active_targets, sampling_cfg)
            
    return input_ctx.get_spans()

@torch.no_grad()
def sample_viz_dset(components, iterator, config_dict, logger):
    """
    Visualization Wrapper: Inplace Refinement of Dataset Samples.
    Plots: [GT (Clean), Noisy Input, Reconstruction, LogSNR Map]
    """
    n = config_dict.get('num_samples', 4)
    res = config_dict.get('res', 32)
    
    # 1. Fetch Clean Data (Ground Truth)
    clean_blocks = iterator.generate_batch_list(n, resolution=res)
    if not clean_blocks: return

    # 2. Prepare Session Context
    # We create the context, but we also keep a separate reference to the 
    # original clean tensor values for plotting GT.
    ctx = MultiTurnContext(clean_blocks)
    
    # GT Extraction (Latents only)
    x0_gt = [b.content.clone() for b in clean_blocks if b.type == 'latent']
    
    # 3. Setup: Manual Noise Injection for "Inplace" test
    # We mutate the Context to be noisy. This creates our "Noisy Input".
    target_indices = []
    start_snr = -4.0
    target_snr = config_dict.get('target_logsnr', 10.0)
    
    # Store noisy inputs for visualization
    noisy_inputs_vis = []
    
    latent_idx = 0
    for i, b in enumerate(ctx.blocks):
        if b.type == 'latent':
            device = b.content.device
            
            # Map construction
            H, W = b.content.shape[-2:]
            start_map = torch.full((1, H, W), start_snr, device=device)
            end_map = torch.full((1, H, W), target_snr, device=device)
            
            # Explicit Noise Injection
            alpha, sigma = logsnr_to_alpha_sigma(start_map)
            z_clean = b.content # This is clean because we just made ctx from clean_blocks
            z_noisy = z_clean * alpha + torch.randn_like(z_clean) * sigma
            
            # Mutate block to be noisy for the session start
            b.content = z_noisy
            b.logsnr = start_map
            
            # Capture for Vis
            noisy_inputs_vis.append(z_noisy.clone())
            
            target_indices.append({
                'idx': i,
                'start_map': start_map,
                'end_map': end_map
            })
            latent_idx += 1

    # 4. Construct Query
    query = {
        "mutation": "inplace",
        "target_indices": target_indices,
        "sampler": "taufield_euler",
        "sampling_config": {
            "steps": config_dict.get('steps', 50),
            "mode": config_dict.get('mode', 'naive')
        }
    }
    
    # 5. Execute
    final_blocks = execute_multiturn_session(components, ctx, [query])
    
    # 6. Harvest Results
    recon = [b.content for b in final_blocks if b.type == 'latent']
    maps = [b.logsnr for b in final_blocks if b.type == 'latent']
    
    plot_data = {
        "x0": x0_gt,
        "noisy_input": noisy_inputs_vis,
        "reconstruction": recon,
        "logsnr_map": maps
    }
    
    from .plotting import plot_dset_reconstruction
    plot_dset_reconstruction(plot_data, logger, name=f"stratified_{res}", show_map=True)

@torch.no_grad()
def sample_viz_causal_sweep(components, iterator, config, logger):
    """
    Demonstrates Autoregressive Latent Generation (Append Logic).
 
    Iterates over EACH dataset split separately, generating homogeneous sequences.
    For each split, sweeps through SNR values to show denoising quality vs noise level.
 
    This is eval code: we measure "given N-1 context frames, can we predict frame N?"
    Works for video (temporal induction) AND functional datasets (pattern induction).
    """
    # === Config (REQUIRED - no defaults that hide misconfiguration) ===
    n_sweeps = config['sweep_count']
    seq_len = config['sweep_length']
    snr_start, snr_end = config['sweep_range']
    res_arg = config['res']
    steps = config.get('steps', 50)
 
    # Get available splits from the iterator
    split_names = iterator.get_split_names()
 
    from .plotting import plot_causal_sweep_v2
 
    # === Process each split independently ===
    for split_name in split_names:
        print(f"  Causal sweep: {split_name} @ {res_arg}px")
 
        # --- 1. Generate homogeneous data from this split ---
        # Request enough sequences: n_sweeps sequences, each seq_len items
        # For functional iterators: each "item" may be (text, latent) pair
        # For video: each "item" is one frame
        try:
            all_blocks = iterator.generate_from_split(
                split_name,
                count=n_sweeps * seq_len,
                resolution=res_arg
            )
        except KeyError as e:
            print(f"    Skipping {split_name}: {e}")
            continue
        except Exception as e:
            print(f"    Error generating from {split_name}: {e}")
            continue
 
        if not all_blocks:
            print(f"    Skipping {split_name}: no blocks generated")
            continue
 
        # --- 2. Group into sequences ---
        # Two modes:
        # - Video splits: group by group_id (natural sequences)
        # - Functional splits: compose synthetic sequences from independent samples
        from collections import defaultdict
        groups = defaultdict(list)
        for b in all_blocks:
            groups[b.group_id].append(b)
 
        # Check if this split produces multi-latent sequences naturally
        # If any group has >=2 latents, use group_id-based sequencing
        has_natural_sequences = any(
            len([b for b in seq if b.type == 'latent']) >= 2
            for seq in groups.values()
        )
 
        sequences = []
        if has_natural_sequences:
            # Video-style: use group_id as sequence boundary
            for gid in sorted(groups.keys()):
                seq = groups[gid]
                latents_in_seq = [b for b in seq if b.type == 'latent']
                if len(latents_in_seq) >= 2:
                    sequences.append({
                        'all_blocks': seq,
                        'latents': latents_in_seq,
                        'group_id': gid
                    })
        else:
            # Functional-style: compose synthetic sequences from independent samples
            # Each group typically has [text, latent] or just [latent]
            # We compose seq_len groups into one synthetic sequence
            all_latents = [b for b in all_blocks if b.type == 'latent']
            all_texts = [b for b in all_blocks if b.type == 'text']
 
            # Create synthetic sequences of seq_len latents each
            # Pair each latent with its text if available (same group_id)
            latent_to_text = {}
            for t in all_texts:
                latent_to_text[t.group_id] = t
 
            for i in range(0, len(all_latents) - seq_len + 1, seq_len):
                chunk_latents = all_latents[i:i + seq_len]
                if len(chunk_latents) < seq_len:
                    break
 
                # Build synthetic sequence with text prefixes
                synthetic_blocks = []
                for lat in chunk_latents:
                    if lat.group_id in latent_to_text:
                        synthetic_blocks.append(latent_to_text[lat.group_id])
                    synthetic_blocks.append(lat)
 
                sequences.append({
                    'all_blocks': synthetic_blocks,
                    'latents': chunk_latents,
                    'group_id': f"synthetic_{i}"
                })
 
        if len(sequences) < n_sweeps:
            print(f"    Note: {split_name} produced {len(sequences)} sequences (config wants {n_sweeps})")
 
        sequences = sequences[:n_sweeps]  # Trim to requested count
        if not sequences:
            continue
 
        # --- 3. Calculate sweep SNRs ---
        sweep_snrs = torch.linspace(snr_start, snr_end, len(sequences))
 
        # --- 4. Run inference for each sequence ---
        all_results = []  # List of dicts with metadata for plotting
 
        for seq_idx, seq_data in enumerate(sequences):
            current_snr = sweep_snrs[seq_idx].item()
            latents = seq_data['latents']
            all_seq_blocks = seq_data['all_blocks']
 
            # Split: all but last latent as context, last latent as GT target
            context_latents = latents[:-1]
            gt_target = latents[-1]
 
            # Build context: include ALL blocks (text + latent) except the target latent
            # This preserves text conditioning if present
            target_id = gt_target.id
            prefix_blocks = [b for b in all_seq_blocks if b.id != target_id]
 
            if not prefix_blocks:
                continue
 
            device = prefix_blocks[0].content.device
 
            # Clone GT for comparison
            gt_content = gt_target.content.clone()
            gt_shape = gt_target.shape_meta
            # Save clean context latents BEFORE noise injection (for MSE comparison)
            ctx_latents_gt = [lat.content.clone() for lat in context_latents]
 
            # --- 5. Setup context with noise injection ---
            ctx = MultiTurnContext(prefix_blocks)
 
            # Noise the prefix latents to current_snr
            for b in ctx.blocks:
                if b.type == 'latent':
                    b_map = torch.full((1, *b.shape_meta), current_snr, device=device)
                    alpha, sigma = logsnr_to_alpha_sigma(b_map)
                    noise = torch.randn_like(b.content)
                    b.content = b.content * alpha + noise * sigma
                    b.logsnr = b_map
 
            # --- 6. Build query: find last latent in context, use as init ---
            # Filter context to latents and find the last one by position
            ctx_latent_indices = [i for i, b in enumerate(ctx.blocks) if b.type == 'latent']
            if not ctx_latent_indices:
                print(f"    Skipping seq {seq_idx}: no latent in context")
                continue
 
            last_latent_idx = ctx_latent_indices[-1]
            last_latent_shape = ctx.blocks[last_latent_idx].shape_meta
 
            query = {
                "mutation": "append_latent",
                "append_source": "copy_from",
                "append_index": last_latent_idx,  # Explicit index, not -1
                "sampler": "taufield_euler",
                "new_span_config": {
                    "shape": last_latent_shape,  # Explicit shape from source
                    "start_logsnr": current_snr,
                    "target_logsnr": 10.0
                },
                "sampling_config": {
                    "steps": steps,
                    "mode": config.get('mode', 'naive')
                }
            }
 
            # --- 7. Execute ---
            final_blocks = execute_multiturn_session(components, ctx, [query])
 
            # --- 8. Extract results with explicit typing ---
            # The generated block is the last one appended
            generated_block = final_blocks[-1]
            pred_content = generated_block.content
 
            # Get noisy context latents for visualization
            ctx_latent_contents = [b.content for b in final_blocks[:-1] if b.type == 'latent']
 
            # Store result with explicit metadata
            all_results.append({
                'snr': current_snr,
                'gt': gt_content,
                'pred': pred_content,
                'ctx_latents': ctx_latent_contents,      # Noisy versions (what model sees)
                'ctx_latents_gt': ctx_latents_gt,        # Clean versions (for MSE baseline)
                'shape': gt_shape,
                'seq_idx': seq_idx
            })
 
        # --- 9. Plot with explicit metadata ---
        if all_results:
            output_path = logger.run_dir / f"causal_sweep_{split_name}_{res_arg}px.png"
            plot_causal_sweep_v2(all_results, output_path, split_name)