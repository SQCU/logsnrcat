#!/usr/bin/env python3
"""
visualize_batch.py

Advanced Integration Test for N-Span Mixed-Modality Batches.
Generates 3 proof-of-capability plots for complex batch compositions.
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
import math
from collections import namedtuple

# Reuse project components
from src.data_iterator import build_checkerboard_iterator
from src.data_functional import (
    generate_checkerboard_query, render_checkerboard, serialize_query
)
from src.model import coolerLDTformerZC, SpanEmbedder, render_topology_embeddings, ContextBlock

# ==============================================================================
# 1. Manual Block Construction (The "Scenario Builder")
# ==============================================================================

def create_manual_block(content, b_type, causal, group_id, uid, device):
    """Helper to wrap tensor content into a ContextBlock."""
    if isinstance(content, dict): # Handle raw query dicts
        content = serialize_query(content).to(device)
    
    # Auto-generate dummy logsnr for latents to pass strict checks
    logsnr = None
    if b_type == 'latent':
        # [C, H, W] -> [1, H, W]
        h, w = content.shape[-2:]
        logsnr = torch.zeros((1, h, w), device=device, dtype=content.dtype)

    return ContextBlock(
        content=content,
        type=b_type,
        causal=causal,
        logsnr=logsnr,
        group_id=group_id,
        id=str(uid)
    )

def build_scenario_1(device, seed=200):
    """
    Scenario 1: Text -> 4 Latents -> Text
    "One batch of text:{query1...}, latent1...latent4, text: fizzbuzz..."
    """
    blocks = []
    gid = seed
    
    # 1. Compound Text Prefix (4 Queries)
    queries = []
    for i in range(4):
        q = generate_checkerboard_query(seed + i, {'force_high_contrast': True})
        queries.append(q)
    
    # Join JSONs with newlines for a single "Text Span"
    combined_json = "\n".join([json.dumps(q) for q in queries])
    tokens = torch.tensor([ord(c) for c in combined_json], dtype=torch.long, device=device)
    
    blocks.append(create_manual_block(tokens, 'text', True, gid, f"q_combined", device))
    
    # 2. 4 Latent Spans (Renders)
    for i, q in enumerate(queries):
        # Varied resolutions just for fun? Prompt says "embedded image" implying standard.
        # Let's stick to 32x32 for visibility in plot, or 64x64.
        img = render_checkerboard(q, 32, device)
        blocks.append(create_manual_block(img, 'latent', False, gid, f"img_{i}", device))
        
    # 3. Text Suffix ("fizzbuzz" x 8)
    fizz = "fizzbuzz " * 8
    fizz_tokens = torch.tensor([ord(c) for c in fizz], dtype=torch.long, device=device)
    blocks.append(create_manual_block(fizz_tokens, 'text', True, gid, "fizz", device))
    
    return blocks, "Scenario 1: Text(Queries) -> 4x Image -> Text(Fizzbuzz)"

def build_scenario_2(device, seed=300):
    """
    Scenario 2: Variable Resolutions
    [128x48, 48x128, 48x128, 128x128]
    """
    blocks = []
    gid = seed
    
    # Query is irrelevant, just need pixels
    q = generate_checkerboard_query(seed, {})
    
    resolutions = [(128, 48), (48, 128), (48, 128), (128, 128)]
    
    for i, (h, w) in enumerate(resolutions):
        # render_checkerboard takes 'resolution' as int usually, 
        # let's hack it or resize. The functional render is square-only in snippet.
        # We'll render square max and crop/resize.
        sq_res = max(h, w)
        img_sq = render_checkerboard(q, sq_res, device)
        # Center crop/slice
        img = img_sq[:, :h, :w]
        
        blocks.append(create_manual_block(img, 'latent', False, gid, f"var_res_{i}", device))
        
    return blocks, "Scenario 2: Variable Resolutions [128x48, 48x128, 48x128, 128x128]"

def build_scenario_3(device, seed=400):
    """
    Scenario 3: Interleaved
    Image -> Text -> Image -> Text
    """
    blocks = []
    gid = seed
    
    q1 = generate_checkerboard_query(seed, {})
    q2 = generate_checkerboard_query(seed+1, {})
    
    # Img 1
    blocks.append(create_manual_block(render_checkerboard(q1, 32, device), 'latent', False, gid, "img1", device))
    # Text 1
    blocks.append(create_manual_block(q1, 'text', True, gid, "txt1", device))
    # Img 2
    blocks.append(create_manual_block(render_checkerboard(q2, 32, device), 'latent', False, gid, "img2", device))
    # Text 2
    blocks.append(create_manual_block(q2, 'text', True, gid, "txt2", device))
    
    return blocks, "Scenario 3: Interleaved (Image -> Text -> Image -> Text)"

# ==============================================================================
# 2. Pipeline Processing
# ==============================================================================

def run_pipeline_analysis(blocks, device):
    """
    Runs embedding and topological analysis.
    Constructs the ACTUAL Block-Causal Mask for visualization using the model's internal logic.
    """
    # 1. Setup Model Components (Real)
    # Note: We use the actual class definitions to ensure we test the real logic
    from src.model import coolerLDTformerZC, SpanEmbedder, render_topology_embeddings, build_dual_masks
    from collections import namedtuple
    
    # Tiny model config, just enough to drive the embeddings
    model = coolerLDTformerZC(
        dim=64, depth=1, num_heads=4, topo_dim=3,
        vocab_size=256, context_size=4, stride=2
    ).to(device)
    span_emb = SpanEmbedder(model.text_embed, model.patch_embedder)
    
    # 2. Embed Spans (Raw Data -> Z, Metadata)
    z_flat, span_objects, _ = span_emb.embed(blocks)

    # 3. Render Topology (Metadata -> Coords) - dtype from model
    dtype = model.text_embed.weight.dtype
    topo_embeds, doc_ids = render_topology_embeddings(span_objects, max_dims=3, device=device, dtype=dtype)
    
    # 4. Setup PageTable Mocks for build_dual_masks
    # The mask builder requires these structures to resolve physical addresses, 
    # even in ZC (Zero-Copy) mode.
    L = z_flat.shape[0]
    block_size = 128
    num_blocks = (L + block_size - 1) // block_size
    
    PageTableMock = namedtuple('PageTable', ['block_size'])
    page_table = PageTableMock(block_size=block_size)
    
    # Identity mappings for training (Active matches Heap 1:1)
    flat_page_table = torch.arange(num_blocks, device=device, dtype=torch.long)
    inverse_page_table = torch.arange(num_blocks, device=device, dtype=torch.long)
    
    # 5. Retrieve the Mask Closure (The Source of Truth)
    # We call the debug version of the mask builder to get the internal closure
    _, _, debug_dict = build_dual_masks(
        spans=span_objects,
        topo_active=topo_embeds,
        topo_heap=topo_embeds, # Self-attention
        page_table=page_table,
        flat_page_table=flat_page_table,
        inverse_page_table=inverse_page_table,
        window_size=10.0, # Arbitrary large window for global check
        return_mask_closures=True
    )
    
    # This is the actual python function passed to flex_attention
    mask_mod = debug_dict['mask_mod_global']
    
    # 6. Materialize the Mask Tensor
    # Construct a full LxL grid of indices to evaluate the closure
    q_idx = torch.arange(L, device=device).unsqueeze(1).expand(L, L)
    k_idx = torch.arange(L, device=device).unsqueeze(0).expand(L, L)
    
    # Evaluate the closure.
    # We pass b=0, h=0 as the logic is currently batch/head invariant for this check.
    # The closure will lookup doc_ids, span_ids, and causal_modes internally.
    final_mask = mask_mod(0, 0, q_idx, k_idx)
    
    return {
        "spans": span_objects,
        "mask": final_mask,
        "topo": topo_embeds,
        "L": L
    }

# ==============================================================================
# 3. Generalized Visualization
# ==============================================================================

def plot_scenario(title, blocks, pipeline_data, output_filename):
    spans = pipeline_data['spans']
    mask = pipeline_data['mask']
    topo = pipeline_data['topo']
    L = pipeline_data['L']
    
    fig = plt.figure(figsize=(24, 12))
    plt.suptitle(title, fontsize=16, weight='bold')
    
    # Grid Layout: 
    # Left: Content Timeline (Strip)
    # Middle: Mask
    # Right: Topology
    
    gs = fig.add_gridspec(2, 4)
    
    # --- 1. Content Timeline (Top Left & Bottom Left) ---
    # We display thumbnails of the blocks in sequence
    ax_timeline = fig.add_subplot(gs[:, 0])
    ax_timeline.axis('off')
    ax_timeline.set_title("Sequence Content", fontsize=12)
    
    y_cursor = 1.0
    margin = 0.05
    slot_height = (1.0 - (len(blocks)+1)*margin) / len(blocks)
    
    for i, b in enumerate(blocks):
        # Create inset axes for each block
        ax_ins = ax_timeline.inset_axes([0.1, 1.0 - (i+1)*(slot_height+margin), 0.8, slot_height])
        
        if b.type == 'latent':
            # Image
            img = b.content.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
            ax_ins.imshow(img, aspect='auto') # aspect auto to fill slot
            info = f"Img {i}\n{list(b.content.shape[-2:])}"
        else:
            # Text
            try:
                txt_bytes = bytes(b.content.cpu().tolist())
                # Truncate for display
                txt_str = txt_bytes.decode('utf-8')
                disp_str = (txt_str[:50] + '...') if len(txt_str) > 50 else txt_str
                disp_str = disp_str.replace('\n', ' ')
            except:
                disp_str = "Bytes..."
            
            ax_ins.text(0.5, 0.5, f"Txt {i}\n{disp_str}", ha='center', va='center', fontsize=8, wrap=True)
            ax_ins.set_facecolor('#f0f0f0')
            ax_ins.set_xticks([])
            ax_ins.set_yticks([])
            info = f"Txt {i}\nL={b.content.shape[0]}"
            
        # Label
        ax_timeline.text(0.0, 1.0 - (i+1)*(slot_height+margin) + slot_height/2, f"#{i}", va='center', fontsize=10, weight='bold')
        ax_ins.set_xticks([])
        ax_ins.set_yticks([])
        
        # Color code border based on causality
        color = 'red' if b.causal else 'blue'
        for spine in ax_ins.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    # --- 2. Causal Mask (Center) ---
    ax_mask = fig.add_subplot(gs[:, 1:3])
    mask_np = mask.float().cpu().numpy()
    ax_mask.imshow(mask_np, cmap='Greys_r', interpolation='nearest', origin='upper')
    ax_mask.set_title(f"Attention Mask ({L}x{L})")
    
    # Draw span boundaries
    cursor = 0
    ticks = []
    labels = []
    
    for i, s in enumerate(spans):
        length = s.end_idx - s.start_idx
        center = cursor + length/2
        
        # Grid lines
        ax_mask.axvline(cursor - 0.5, color='red', linestyle='-', linewidth=0.5, alpha=0.3)
        ax_mask.axhline(cursor - 0.5, color='red', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Diagonal block annotation
        rect_color = 'red' if s.causal else 'blue'
        rect = patches.Rectangle((cursor-0.5, cursor-0.5), length, length, 
                                 linewidth=1, edgecolor=rect_color, facecolor='none')
        ax_mask.add_patch(rect)
        
        # Label
        mode = "AR" if s.causal else "Bi-Dir"
        ax_mask.text(center, center, f"{i}\n{mode}", color=rect_color, 
                     ha='center', va='center', fontsize=8, fontweight='bold', alpha=0.7)
        
        cursor += length
        
    ax_mask.set_xlabel("Key Index")
    ax_mask.set_ylabel("Query Index")

    # --- 3. Topology (Right) ---
    ax_topo = fig.add_subplot(gs[0, 3])
    topo_np = topo.detach().cpu().numpy()
    
    # Plot Y vs X
    cursor = 0
    for i, s in enumerate(spans):
        l = s.end_idx - s.start_idx
        chunk = topo_np[cursor : cursor+l]
        
        if s.type == 'text':
            ax_topo.scatter(chunk[:, 2], chunk[:, 1], label=f"#{i} Txt", alpha=0.6, s=50)
        else:
            # Subsample large images for scatter
            step = max(1, l // 200)
            ax_topo.scatter(chunk[::step, 2], chunk[::step, 1], label=f"#{i} Img", alpha=0.4, s=10, marker='s')
            
        cursor += l
        
    ax_topo.set_title("Spatial Topology (Y vs X)")
    ax_topo.invert_yaxis()
    ax_topo.grid(True, alpha=0.3)
    ax_topo.legend(fontsize='small')
    
    # --- 4. Highway (Bottom Right) ---
    ax_time = fig.add_subplot(gs[1, 3])
    highway = topo_np[:, 0]
    ax_time.plot(highway, label="Global Time")
    
    cursor = 0
    for i, s in enumerate(spans):
        length = s.end_idx - s.start_idx
        ax_time.axvline(cursor, color='gray', linestyle='--', alpha=0.3)
        cursor += length
        
    ax_time.set_title("Highway Coordinate")
    ax_time.set_xlabel("Token Index")
    
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Saved {output_filename}")


# ==============================================================================
# 4. Main Driver
# ==============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Capability Proofs on {device} ---")
    
    # 1. Run Scenario 1
    blocks1, title1 = build_scenario_1(device)
    data1 = run_pipeline_analysis(blocks1, device)
    plot_scenario(title1, blocks1, data1, "proof_scenario_1.png")
    
    # 2. Run Scenario 2
    blocks2, title2 = build_scenario_2(device)
    data2 = run_pipeline_analysis(blocks2, device)
    plot_scenario(title2, blocks2, data2, "proof_scenario_2.png")
    
    # 3. Run Scenario 3
    blocks3, title3 = build_scenario_3(device)
    data3 = run_pipeline_analysis(blocks3, device)
    plot_scenario(title3, blocks3, data3, "proof_scenario_3.png")

    print("\nAll proofs generated.")

if __name__ == "__main__":
    main()