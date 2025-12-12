# bench_video_causal.py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from pathlib import Path

from dataset import VideoFolderIterator, get_logsnr_batch
from diffusion_utils import logsnr_to_alpha_sigma
from ld_tformer import coolerLDTformerZC, SpanEmbedder, SpanUnembedder, Span
# No longer importing run_embed_step, we do it raw

def test_causal_video_embedding(video_path):
    device = torch.device('cuda')
    print(f"🔧 Testing Video Pipeline from: {video_path}")
    
    # 1. Setup Model (Same config as main bench)
    embed_dim = 256
    model = coolerLDTformerZC(dim=embed_dim, depth=4, num_heads=8, topo_dim=3).to(device)
    model.eval() 
    
    span_emb = SpanEmbedder(model.text_embed, model.patch_embedder)
    span_unemb = SpanUnembedder(model.text_head, model.patch_unembedder)
    # No components tuple needed since we call span_emb directly
    
    # 2. Get Data: 4 Frames Per Sequence, 2 Sequences
    batch_size = 2
    seq_len = 4
    iterator = VideoFolderIterator(video_path, device=device)
    
    # Get raw 64x64 frames first [B*Seq, 3, 64, 64]
    frames_flat = iterator.generate_batch(batch_size, resolution=64, sequence_length=seq_len)
    
    # Reshape to [B, Seq, 3, H, W] to manipulate
    frames = frames_flat.view(batch_size, seq_len, 3, 64, 64)
    
    processed_images = []
    processed_logsnrs = []
    batch_spans = []
    
    print("\n🎞️ constructing causal batches...")
    
    for b in range(batch_size):
        # --- Prefix Frames (0, 1, 2) ---
        for t in range(3):
            # Downsample to 32
            img_32 = F.interpolate(frames[b, t:t+1], size=(32, 32), mode='area')
            processed_images.append(img_32.squeeze(0)) # [3, 32, 32]
            
            # High SNR (+1 to +5) -> "Clean-ish context"
            # get_logsnr_batch returns [1, 1, 32, 32], we want [1, 32, 32] for the embedder
            lsnr_map = get_logsnr_batch('split', 1, 32, 32, device, 
                                      {'min_snr': 1.0, 'max_snr': 5.0, 'angle_range_deg': 15.0})
            processed_logsnrs.append(lsnr_map.squeeze(0)) # [1, 32, 32]
            
            # Span Metadata
            batch_spans.append({
                'type': 'latent',
                'len': (32//2)**2, # 16*16 = 256 tokens
                'shape': (16, 16),
                'causal': True, 
                'id': f"vid_{b}_fr_{t}"
            })
            
        # --- Target Frame (3) ---
        img_64 = frames[b, 3] # [3, 64, 64]
        processed_images.append(img_64)
        
        # Full SNR (-5 to +5) -> "Denoising Target"
        lsnr_map_target = get_logsnr_batch('split', 1, 64, 64, device, 
                                         {'min_snr': -5.0, 'max_snr': 5.0, 'angle_range_deg': 45.0})
        processed_logsnrs.append(lsnr_map_target.squeeze(0)) # [1, 64, 64]
        
        batch_spans.append({
            'type': 'latent',
            'len': (64//2)**2, # 32*32 = 1024 tokens
            'shape': (32, 32),
            'causal': True,
            'id': f"vid_{b}_fr_TARGET"
        })

    # 4. Noise the inputs
    noisy_images = []
    for img, lsnr in zip(processed_images, processed_logsnrs):
        alpha, sigma = logsnr_to_alpha_sigma(lsnr)
        eps = torch.randn_like(img)
        z_t = img * alpha + eps * sigma
        noisy_images.append(z_t)

    # 5. Run Embedding
    print("🚀 Running Embedder (Mixed Res)...")
    
    # CRITICAL FIX: Pass the list of 3D tensors directly.
    # Do NOT unsqueeze them into [1, 3, H, W] batches.
    # The SpanEmbedder loop iterates and passes [3, H, W] to patch_emb.
    
    z_flat, span_objects, _ = span_emb.embed(
        batch_spans,
        text_tokens=[None]*len(batch_spans),
        images=noisy_images,        # List[Tensor[3, H, W]]
        logsnr_maps=processed_logsnrs # List[Tensor[1, H, W]]
    )
    
    print(f"✅ Embedding Successful. Tokens: {z_flat.shape}")
    
    # 6. Run Unembedding (Check Reconstruction)
    print("🚀 Running Unembedder...")
    
    outputs = span_unemb.decode(z_flat, span_objects)
    
    # 7. Visualize
    print("🎨 Visualizing Causal Stream...")
    
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    
    # We'll plot the first sequence (4 frames)
    for i in range(4): 
        # GT
        gt_img = processed_images[i].permute(1,2,0).cpu().numpy()
        axes[i, 0].imshow(gt_img)
        axes[i, 0].set_title(f"GT Frame {i} ({gt_img.shape[0]}px)")
        axes[i, 0].axis('off')
        
        # LogSNR Map
        lmap = processed_logsnrs[i].squeeze().cpu().numpy()
        axes[i, 1].imshow(lmap, cmap='viridis')
        axes[i, 1].set_title(f"LogSNR {lmap.min():.1f} to {lmap.max():.1f}")
        axes[i, 1].axis('off')
        
        # Reconstruction (Decoder Output)
        v_pred = outputs[i]['image_vpreds'].squeeze(0) # [3, H, W]
        lsnr = processed_logsnrs[i]
        alpha, sigma = logsnr_to_alpha_sigma(lsnr)
        
        z_in = noisy_images[i]
        x0_pred = alpha * z_in - sigma * v_pred
        
        recon = x0_pred.permute(1,2,0).clamp(0,1).detach().cpu().numpy()
        axes[i, 2].imshow(recon)
        axes[i, 2].set_title("Reconstruction")
        axes[i, 2].axis('off')
        
    os.makedirs("test_video", exist_ok=True)
    plt.tight_layout()
    plt.savefig("test_video/causal_chain.png")
    print("✅ Saved visualization to test_video/causal_chain.png")

if __name__ == "__main__":
    # Point this to your actual folder
    VIDEO_DIR = "C:/dox/recordings/rl_capture/capture_run_1760343426/videos"
    if os.path.exists(VIDEO_DIR):
        test_causal_video_embedding(VIDEO_DIR)
    else:
        print(f"Please create {VIDEO_DIR} or update path.")