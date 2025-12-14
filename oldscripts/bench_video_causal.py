# bench_video_causal.py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from typing import List, Union
from ld_tformer import coolerLDTformerZC, SpanEmbedder, SpanUnembedder, ContextBlock
from dataset import get_logsnr_batch, safe_create_decoder
from diffusion_utils import  logsnr_to_alpha_sigma

# --- Dependency Check ---
try:
    from torchcodec.decoders import VideoDecoder
    HAS_TORCHCODEC = True
except ImportError:
    HAS_TORCHCODEC = False

def test_causal_video_embedding(video_path):
    if not HAS_TORCHCODEC: return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Testing Pipeline on {device}")
    
    # 1. Load Data via Safe Decoder
    try:
        decoder = safe_create_decoder(video_path, device)
        indices = [0, 1, 2, 3]
        raw_sequence = decoder.get_frames_at(indices).data.float().to(device) / 255.0
    except Exception as e:
        print(f"❌ Error: {e}"); return
        
    # 2. Script ContextBlocks
    sequence_def = []
    # Context
    for t in range(3):
        img = F.interpolate(raw_sequence[t:t+1], size=(32,32), mode='area').squeeze(0)
        lsnr = get_logsnr_batch('split', 1, 32, 32, device, {'min_snr':2.0}).squeeze(0)
        sequence_def.append(ContextBlock(img, logsnr=lsnr, group_id=101, id=f"ctx_{t}"))
    # Target
    img = F.interpolate(raw_sequence[3:4], size=(64,64), mode='area').squeeze(0)
    lsnr = get_logsnr_batch('split', 1, 64, 64, device, {'min_snr':-5.0}).squeeze(0)
    sequence_def.append(ContextBlock(img, logsnr=lsnr, group_id=101, id="tgt"))
    
    # 3. Noise & Unzip
    noisy_blocks = []
    latent_inputs = [] # For recon
    for b in sequence_def:
        alpha, sigma = logsnr_to_alpha_sigma(b.logsnr)
        eps = torch.randn_like(b.content)
        z = b.content * alpha + eps * sigma
        noisy_blocks.append(ContextBlock(z, logsnr=b.logsnr, type='latent', causal=True, 
                                         shape_meta=b.shape_meta, group_id=b.group_id, id=b.id))
        latent_inputs.append(z)

    # 4. Model
    model = coolerLDTformerZC(dim=256, depth=2).to(device)
    span_emb = SpanEmbedder(model.text_embed, model.patch_embedder)
    span_unemb = SpanUnembedder(model.text_head, model.patch_unembedder)
    
    # 5. Pipeline
    z_flat, spans, _ = span_emb.embed(noisy_blocks)
    decoded = span_unemb.decode(z_flat, spans)
    
    # 6. Viz
    fig, axes = plt.subplots(4, 3, figsize=(10, 14))
    for i in range(4):
        # GT
        axes[i,0].imshow(sequence_def[i].content.permute(1,2,0).cpu().numpy())
        # Noisy
        axes[i,1].imshow(latent_inputs[i].permute(1,2,0).cpu().clamp(0,1).numpy())
        # Recon
        v = decoded[i]['image_vpreds']
        alpha, sigma = logsnr_to_alpha_sigma(sequence_def[i].logsnr)
        x0 = alpha * latent_inputs[i] - sigma * v
        axes[i,2].imshow(x0.permute(1,2,0).detach().cpu().clamp(0,1).numpy())
    
    plt.savefig("test_video/causal_final.png")
    print("✅ Done")

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