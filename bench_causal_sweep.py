# bench_causal_sweep.py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List
from dataclasses import replace

# Reuse existing infrastructure
from ld_tformer import coolerLDTformerZC, SpanEmbedder, SpanUnembedder, ContextBlock
from ld_tformer_embedding_functional import render_topology_embeddings
from dataset import safe_create_decoder, get_logsnr_batch
from diffusion_utils import logsnr_to_alpha_sigma
from bench_multisnr_zc import spatial_euler_solver

from ld_tformer import build_dual_masks, Span
from memory_manager import PageTable

# --- Visualizer Tools ---

def probe_householder_interference(components, blocks, device):
    """
    Isolates the learned positional bias by passing 'flat' content 
    through the learned RnRoPE layer.
    
    Returns: The Attention Pattern induced PURELY by geometry + learned twists.
    """
    model = components[0]
    
    # 1. Get Real Topology
    # (Reuse the safe hybrid extraction from before)
    class MockSpan:
        def __init__(self, b): self.shape, self.doc_id = b.shape_meta, b.group_id
    mock_spans = [MockSpan(b) for b in blocks]
    
    # We need GPU topo for the RoPE kernel
    topo_embeds, _ = render_topology_embeddings(mock_spans, 3, device)
    
    # 2. Construct "Flat" Content
    # [1, 1, L, Head_Dim] -> Broadcasts to all heads/batches
    # We use a fixed random vector to avoid orthogonal-identity artifacts
    # but reuse it for EVERY token.
    L = topo_embeds.shape[0]
    head_dim = model.layers[0].attn.head_dim
    num_heads = model.layers[0].attn.num_heads
    
    # A single "Concept" vector, repeated L times
    # This represents "Perfect Semantic Match" everywhere.
    torch.manual_seed(42)
    concept_vec = torch.randn(1, 1, 1, head_dim, device=device)
    q_flat = concept_vec.expand(1, num_heads, L, head_dim)
    k_flat = concept_vec.expand(1, num_heads, L, head_dim)
    
    # 3. Apply the Learned Twist (Layer 0's RoPE)
    # This applies Householder(in) -> RoPE(freqs) -> Householder(out)
    with torch.no_grad():
        # Scale=1.0 for visualization
        q_rot, k_rot = model.layers[0].attn.rope(q_flat, k_flat, topo_embeds, scale=1.0)
        
        # 4. Compute Self-Attention Scores (Batch 0, Head 0)
        # score = q_rot @ k_rot.T
        # Shape: [1, H, L, D] @ [1, H, D, L] -> [1, H, L, L]
        attn_scores = torch.matmul(q_rot, k_rot.transpose(-1, -2))
        
        # Extract Head 0 (or mean over heads?)
        # Let's show Head 0 to see specific basis alignments. 
        # Averaging might wash out the "Z-shapes".
        interference_pattern = attn_scores[0, 0]
        
    return interference_pattern.cpu()

def visualize_mechanism(components, blocks: List[ContextBlock], device, res=16):
    model, span_embedder, _, _, _ = components
    
    with torch.no_grad():
        # 1. Real Embedding & Topology
        z_flat, span_objects, _ = span_embedder.embed(blocks)
        topo_embeds, _ = render_topology_embeddings(span_objects, 3, device)
        L = topo_embeds.shape[0]
        
        # 2. Real PageTable
        block_size = 128
        num_blocks = (L + block_size - 1) // block_size
        page_table = PageTable(num_blocks+1, block_size, 1, num_blocks+1, device)
        flat_page_table = torch.arange(num_blocks, device=device, dtype=torch.long)
        
        # 3. Build Mask
        local_mask, _ = build_dual_masks(
            span_objects, topo_embeds, topo_embeds, 
            page_table, flat_page_table, None, window_size=10.0
        )
        
        # 4. Safe Densification & Normalization
        try:
            raw_mask = local_mask.to('cpu').to_dense()[0, 0]
        except:
            raw_mask = local_mask.to_dense()[0, 0].cpu()
            
        # If mask is block-level, upsample to token-level
        if raw_mask.shape[0] != L:
            scale_factor = L // raw_mask.shape[0]
            # Kronecker-like expansion: 8x8 -> 1024x1024
            raw_mask = raw_mask.repeat_interleave(scale_factor, dim=0).repeat_interleave(scale_factor, dim=1)
            # Clip to exact length (in case of rounding)
            raw_mask = raw_mask[:L, :L]

        # 5. Topological Distance
        dist_mat = torch.cdist(topo_embeds, topo_embeds, p=2.0).cpu()
        
        # 6. Downsample for Display (Target ~256px)
        viz_step = max(1, L // 256)
        idx = torch.arange(0, L, viz_step)
        
        dist_vis = dist_mat[idx][:, idx]
        mask_vis = raw_mask[idx][:, idx].float()
    
    return dist_vis, mask_vis


# --- Sweep Experiment ---
def run_causal_sweep(video_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🧪 Starting Causal Information Sweep on {device}")
    
    # 1. Data Prep
    try:
        decoder = safe_create_decoder(video_path, device)
        raw_frames = decoder.get_frames_at([0, 1, 2, 3]).data.float().to(device) / 255.0
        frames = F.interpolate(raw_frames, size=(32, 32), mode='area')
    except Exception as e:
        print(f"❌ Data load failed: {e}")
        return

    # 2. Setup Components
    model = coolerLDTformerZC(dim=256, depth=2).to(device)
    model.eval()
    
    page_table = PageTable(
        num_blocks=1024, block_size=128, 
        max_batch_size=1, max_logical_blocks=1024, 
        device=device
    )
    
    components = (
        model, 
        SpanEmbedder(model.text_embed, model.patch_embedder), 
        SpanUnembedder(model.text_head, model.patch_unembedder),
        None, 
        page_table 
    )

    # 3. Mechanism Visualization
    print("🔍 Visualizing Attention Topology...")
    
    # Init dummy LogSNR to satisfy type signature
    dummy_lsnr = torch.zeros((1, 32, 32), device=device)
    
    dummy_blocks = [
        ContextBlock(frames[0], logsnr=dummy_lsnr, group_id=1, id="ctx0"),
        ContextBlock(frames[1], logsnr=dummy_lsnr, group_id=1, id="ctx1"),
        ContextBlock(frames[2], logsnr=dummy_lsnr, group_id=1, id="ctx2"),
        ContextBlock(frames[3], logsnr=dummy_lsnr, group_id=1, id="tgt")
    ]
    
    dist_mat, mask_mat = visualize_mechanism(components, dummy_blocks, device)
    
    fig_mech, ax_mech = plt.subplots(1, 2, figsize=(12, 5))
    ax_mech[0].imshow(dist_mat, cmap='viridis_r')
    ax_mech[0].set_title("Topological Distance\n(Real Coordinates)")
    ax_mech[1].imshow(mask_mat, cmap='gray')
    ax_mech[1].set_title("Causal Attention Mask\n(Real Kernel Output)")
    plt.savefig("test_video/mechanism_check.png")
    plt.close()

    # Call the probe
    interference = probe_householder_interference(components, dummy_blocks, device)
    
    # Plotting
    L = interference.shape[0]
    scale = max(1, L // 512) # High res to see the moire
    idx = torch.arange(0, L, scale)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(interference[idx][:, idx], cmap='RdBu_r') # Red=Pos, Blue=Neg
    plt.title("Learned Householder Interference Pattern\n(Head 0, Layer 0)")
    plt.colorbar(label="Rotational Similarity")
    plt.savefig("test_video/rope_interference.png")
    plt.close()

    # 4. The Loop
    print("🔄 Running Noise Sweep...")
    context_snrs = [10.0, 5.0, 0.0, -2.0, -5.0]
    sweep_results = []
    
    for c_snr in context_snrs:
        blocks = []
        fixed_data = [] 
        
        # Build Sequence
        for t in range(3):
            l_map = torch.full((1, 32, 32), c_snr, device=device)
            alpha, sigma = logsnr_to_alpha_sigma(l_map)
            eps = torch.randn_like(frames[t])
            z = frames[t] * alpha + eps * sigma
            
            blocks.append(ContextBlock(
                content=z, logsnr=l_map, group_id=999, id=f"ctx_{t}",
                shape_meta=(16,16)
            ))
            fixed_data.append(z)

        # Target (Frame 3)
        tgt_start_snr = -4.0
        l_map_tgt = torch.full((1, 32, 32), tgt_start_snr, device=device)
        alpha, sigma = logsnr_to_alpha_sigma(l_map_tgt)
        z_tgt = frames[3] * alpha + torch.randn_like(frames[3]) * sigma
        
        blocks.append(ContextBlock(
            content=z_tgt, logsnr=l_map_tgt, group_id=999, id="tgt",
            shape_meta=(16,16)
        ))
        fixed_data.append(None)
        
        # Solve
        z_final = spatial_euler_solver(
            components, blocks, target_logsnr=10.0, steps=20, 
            mode='naive', config={}, fixed_data=fixed_data
        )
        
        sweep_results.append({
            'snr': c_snr,
            'input_context_sample': blocks[2].content,
            'recon_target': z_final[3]
        })
        print(f"   > Context SNR {c_snr:.1f} complete.")

    # 5. Plot Sweep
    print("🎨 Plotting Sweep Results...")
    n = len(sweep_results)
    fig, axes = plt.subplots(3, n, figsize=(3*n, 9))
    
    gt_img = frames[3].permute(1,2,0).cpu().numpy()
    
    for i, res in enumerate(sweep_results):
        ctx = res['input_context_sample'].permute(1,2,0).cpu().clamp(0,1).numpy()
        axes[0, i].imshow(ctx)
        axes[0, i].set_title(f"Ctx LogSNR {res['snr']}")
        axes[0, i].axis('off')
        
        recon = res['recon_target'].detach().permute(1,2,0).cpu().clamp(0,1).numpy()
        axes[1, i].imshow(recon)
        axes[1, i].set_title("Reconstruction")
        axes[1, i].axis('off')
        
        axes[2, i].imshow(gt_img)
        axes[2, i].set_title("Target GT")
        axes[2, i].axis('off')

    plt.tight_layout()
    os.makedirs("test_video", exist_ok=True)
    plt.savefig("test_video/causal_sweep.png")
    print("✅ Sweep complete. See test_video/causal_sweep.png")

if __name__ == "__main__":
    SEARCH_DIR = "C:/dox/recordings/rl_capture/capture_run_1760343426/videos"
    video_file = None
    if os.path.exists(SEARCH_DIR):
        files = list(os.listdir(SEARCH_DIR))
        mp4s = [f for f in files if f.endswith(".mp4")]
        if mp4s: video_file = os.path.join(SEARCH_DIR, mp4s[0])
            
    if video_file:
        run_causal_sweep(video_file)
    else:
        print("⚠️ No video found.")