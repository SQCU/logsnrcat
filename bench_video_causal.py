# bench_video_causal.py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import math
from dataclasses import dataclass, field
from typing import List, Union, Tuple, Dict

# --- Dependency Check ---
try:
    from torchcodec.decoders import VideoDecoder
    HAS_TORCHCODEC = True
except ImportError:
    HAS_TORCHCODEC = False
    print("⚠️ torchcodec not found. Please `pip install torchcodec` to run this benchmark.")

# Import model components
from ld_tformer import coolerLDTformerZC, SpanEmbedder, SpanUnembedder
from dataset import get_logsnr_batch, safe_create_decoder

# --- 1. Utilities & Data Structures ---

def logsnr_to_alpha_sigma(logsnr):
    """Safe logsnr conversion."""
    sigmoid_lsnr = torch.sigmoid(logsnr)
    sigmoid_neg_lsnr = torch.sigmoid(-logsnr)
    alpha = torch.sqrt(sigmoid_lsnr)
    sigma = torch.sqrt(sigmoid_neg_lsnr)
    return alpha, sigma

@dataclass
class ContextBlock:
    """
    Atomic unit for a multimodal sequence.
    Binds raw data to its topological metadata.
    """
    content: torch.Tensor             # [3, H, W]
    type: str = 'latent'              # Video is all latent for now
    causal: bool = True               # Video is inherently causal
    
    # Metadata
    shape_meta: Tuple[int, int] = field(default_factory=tuple)
    logsnr: torch.Tensor = None       # [1, H, W]
    
    def __post_init__(self):
        # Auto-infer shape from content if missing
        if not self.shape_meta and isinstance(self.content, torch.Tensor):
            h, w = self.content.shape[-2:]
            self.shape_meta = (h // 2, w // 2)

# --- 2. The Benchmark Logic ---

def test_causal_video_embedding(video_path):
    if not HAS_TORCHCODEC:
        return

    # Determine execution device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Testing Video Pipeline from: {video_path}")
    print(f"   Execution Device: {device}")
    
    # A. Load Raw Video Data (Efficiently via TorchCodec)
    # ----------------------
    try:
        # 1. Open Header (Lightweight)
        # Note: safe_create_decoder may return a CPU decoder if NVDEC is unavailable
        decoder = safe_create_decoder(video_path, device=device)
        total_frames = decoder.metadata.num_frames
        
        if total_frames is None:
            print("❌ Could not determine frame count.")
            return
            
        if total_frames < 4:
            print(f"❌ Video too short (need 4 frames, got {total_frames})")
            return

        # 2. Random Access Decode (O(1) Memory)
        indices = [0, 1, 2, 3]
        
        # get_frames_at returns a FrameBatch. 
        # The data might be on CPU if hardware decoding failed.
        # FIX: Explicitly move to target device immediately.
        # double fix: we added a wrapping closure function thingy to fix 'tensor not on device expected from context' errors!
        raw_sequence = decoder.get_frames_at(indices).data.float() / 255.0
        
        print(f"✅ Efficiently decoded {len(indices)} frames")
        
    except Exception as e:
        print(f"❌ Failed to decode video: {e}")
        return
    
    # B. Script the Context (Define Blocks)
    # -------------------------------------
    print("📝 Scripting the Causal Sequence...")
    
    sequence_def: List[ContextBlock] = []
    
    # Frames 0-2: "Context" -> Resize to 32px, High SNR (Clean-ish)
    for t in range(3):
        # Interpolate expects [N, C, H, W] or [1, C, H, W]
        img_32 = F.interpolate(raw_sequence[t:t+1], size=(32, 32), mode='area').squeeze(0)
        
        # High SNR for context (signal dominance)
        lsnr_map = get_logsnr_batch('split', 1, 32, 32, device, 
                                  {'min_snr': 2.0, 'max_snr': 6.0, 'angle_range_deg': 15.0}).squeeze(0)
        
        block = ContextBlock(
            content=img_32,
            logsnr=lsnr_map,
            shape_meta=(16, 16) # 32 // 2
        )
        sequence_def.append(block)

    # Frame 3: "Target" -> Keep at 64px, Low SNR (Noisy)
    img_64 = F.interpolate(raw_sequence[3:4], size=(64, 64), mode='area').squeeze(0)
    
    # Low/Full SNR for target
    lsnr_target = get_logsnr_batch('split', 1, 64, 64, device, 
                                 {'min_snr': -5.0, 'max_snr': 5.0, 'angle_range_deg': 45.0}).squeeze(0)
    
    block_target = ContextBlock(
        content=img_64,
        logsnr=lsnr_target,
        shape_meta=(32, 32) # 64 // 2
    )
    sequence_def.append(block_target)

    # C. Unzip to Pipeline Inputs
    # ---------------------------
    print("🔄 Pre-processing (Unzipping blocks)...")
    
    spans_meta = []
    latent_inputs = [] # List[Tensor]
    logsnr_inputs = [] # List[Tensor]
    noisy_inputs = []  # List[Tensor] for visualization
    
    global_cursor = 0
    group_uuid = 101 # Arbitrary ID for this video sequence
    
    for i, block in enumerate(sequence_def):
        h_grid, w_grid = block.shape_meta
        token_count = h_grid * w_grid
        
        # 1. Noise the image based on logsnr
        # Now both block.content and block.logsnr are guaranteed to be on 'device'
        alpha, sigma = logsnr_to_alpha_sigma(block.logsnr)
        eps = torch.randn_like(block.content)
        z_t = block.content * alpha + eps * sigma
        
        # 2. Add to lists
        latent_inputs.append(z_t)
        logsnr_inputs.append(block.logsnr)
        noisy_inputs.append(z_t) # Keep track for viz
        
        # 3. Build Metadata
        spans_meta.append({
            'type': 'latent',
            'len': token_count,
            'shape': block.shape_meta,
            'causal': True,
            'group_id': group_uuid,
            'id': f"frame_{i}"
        })
        
        print(f"   Frame {i}: {block.content.shape} -> {token_count} tokens")
        global_cursor += token_count
        
    print(f"📊 Total Sequence Length: {global_cursor} tokens")

    # D. Model Initialization
    # -----------------------
    # Using small dim for speed, verifying shapes/flow mainly
    model = coolerLDTformerZC(dim=256, depth=2, num_heads=4, topo_dim=3).to(device)
    model.eval()
    
    span_emb = SpanEmbedder(model.text_embed, model.patch_embedder)
    span_unemb = SpanUnembedder(model.text_head, model.patch_unembedder)

    # E. Forward Pass (Embed -> Unembed)
    # ----------------------------------
    print("🚀 Running Embedder...")
    
    # Note: text_tokens is list of None because all blocks are latent
    z_flat, span_objects, _ = span_emb.embed(
        spans_meta,
        text_tokens=[None]*len(spans_meta),
        images=latent_inputs,
        logsnr_maps=logsnr_inputs
    )
    
    print(f"   Embedded shape: {z_flat.shape}")
    
    print("🚀 Running Unembedder...")
    decoded = span_unemb.decode(z_flat, span_objects)

    # F. Visualization
    # ----------------
    print("🎨 Visualizing...")
    
    fig, axes = plt.subplots(4, 3, figsize=(10, 14))
    
    for i in range(4):
        # 1. Ground Truth
        gt = sequence_def[i].content.permute(1,2,0).cpu().numpy()
        axes[i, 0].imshow(gt)
        axes[i, 0].set_title(f"GT Frame {i} ({gt.shape[0]}px)")
        axes[i, 0].axis('off')
        
        # 2. Noisy Input (What the model saw)
        noisy_viz = noisy_inputs[i].permute(1,2,0).cpu().clamp(0,1).numpy()
        axes[i, 1].imshow(noisy_viz)
        axes[i, 1].set_title("Noisy Input")
        axes[i, 1].axis('off')
        
        # 3. Prediction (Denoised estimate)
        v_pred = decoded[i]['image_vpreds'] # [3, H, W]
        lsnr = logsnr_inputs[i]
        
        alpha, sigma = logsnr_to_alpha_sigma(lsnr)
        z_in = latent_inputs[i]
        x0_pred = alpha * z_in - sigma * v_pred
        
        recon = x0_pred.permute(1,2,0).detach().cpu().clamp(0,1).numpy()
        axes[i, 2].imshow(recon)
        axes[i, 2].set_title("Reconstruction (Init)")
        axes[i, 2].axis('off')

    plt.tight_layout()
    os.makedirs("test_video", exist_ok=True)
    out_path = "test_video/causal_rewrite.png"
    plt.savefig(out_path)
    print(f"✅ Saved visualization to {out_path}")

if __name__ == "__main__":
    # Auto-find a video in the typical directory if available
    SEARCH_DIR = "C:/dox/recordings/rl_capture/capture_run_1760343426/videos"
    video_file = None
    
    if os.path.exists(SEARCH_DIR):
        files = list(os.listdir(SEARCH_DIR))
        mp4s = [f for f in files if f.endswith(".mp4")]
        if mp4s:
            video_file = os.path.join(SEARCH_DIR, mp4s[0])
            
    if video_file:
        test_causal_video_embedding(video_file)
    else:
        print(f"⚠️ No video found in {SEARCH_DIR}. Update path in script.")