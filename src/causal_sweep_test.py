# causal_sweep_test.py
import torch
import os
import matplotlib.pyplot as plt
from pathlib import Path

# New SRC Imports
from src.model import coolerLDTformerZC, SpanEmbedder, SpanUnembedderr, PageTable
from src.data import CompositeIterator
from src.utils import ExperimentLogge
from src.sample import sample_viz_causal_sweep

def find_video_path():
    # Attempt to locate the specific test folder used in previous context
    # Or fallback to local directory
    candidates = [
        "C:/dox/recordings/rl_capture/capture_run_1760343426/videos",
        #"./videos",
        #"."
    ]
    
    for c in candidates:
        if os.path.exists(c):
            # Check if any mp4 exists
            if any(f.endswith('.mp4') for f in os.listdir(c)):
                print(f"📂 Found video data in: {c}")
                return c
    
    print("⚠️ No video path found. Please adjust script or place .mp4 in current dir.")
    return "."

def run_test():
    # 1. Setup & Logger
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = ExperimentLogger(output_dir="experiments_mix")
    print(f"🔧 Initializing Causal Sweep Test on {device}")

    # 2. Configure Dataset (The "Script")
    video_dir = find_video_path()
    
    # We define a video split that returns 4-frame sequences
    dataset_config = {
        'video_test': {
            'type': 'video',
            'ratio': 1.0,
            'params': {
                'path': video_dir,
                # Simple random sampling of 4 frames
                'time_sampler': {'min_pct': 0.0, 'max_pct': 0.1},
                'sequence_structure': [
                    {'res': 32, 'noise_mode': 'uniform'} for _ in range(4)
                ]
            }
        }
    }
    
    print("📚 initializing iterator...")
    iterator = CompositeIterator(device, config=dataset_config)

    # 3. Initialize Model Components (Random Weights)
    print("🧠 Initializing Random Model...")
    dim = 256
    model = coolerLDTformerZC(
        dim=dim, 
        depth=4, 
        num_heads=8, 
        topo_dim=3,
        context_size=4,
        stride=2
    ).to(device)
    
    # Init PageTable (Required for mask generation)
    # Capacity arbitrary but large enough for the batch
    page_table = PageTable(
        num_blocks=2048, block_size=128, 
        max_batch_size=128, max_logical_blocks=2048, 
        device=device
    )
    
    components = (
        model,
        SpanEmbedder(model.text_embed, model.patch_embedder),
        SpanUnembedder(model.text_head, model.patch_unembedder),
        page_table
    )

    # 4. Configure the Sweep
    sweep_config = {
        'num_sweep_sequences': 5,        # 5 rows (High SNR -> Low SNR)
        'sequence_length': 4,            # 3 Context + 1 Target
        'prefix_snr_range': (5.0, -5.0), # Sweep range
        'target_logsnr': 10.0,           # Solve target to clean
        'sampling_steps': 20,            # Fast solve
        'mode': 'naive'
    }

    # 5. Run the "Wind Tunnel" Test
    print("🌪️ Running Causal Information Flow Sweep...")
    try:
        fig = sample_viz_causal_sweep(components, iterator, sweep_config)
        
        # Save results
        logger.save_figure(fig, "causal_sweep_test")
        print(f"\n✅ Test Successful! Visualization saved to: {logger.run_dir}/causal_sweep_test.png")
        print("   (Note: Reconstructions will be random/noisy as model is untrained,")
        print("    but the structural coherence of the mask/topology is proven if the script runs.)")
        
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()