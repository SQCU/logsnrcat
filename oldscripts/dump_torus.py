# dump_torus.py
import torch
import matplotlib.pyplot as plt
import os
import math
import sys

# Try importing from the new native file first
try:
    from dataset_torus_native import TorusIterator
except ImportError:
    print("Could not import 'dataset_torus_native'. Checking for 'dataset_torus'...")
    try:
        from dataset_torus import TorusIterator
    except ImportError:
        print("Error: Could not find dataset_torus_native.py or dataset_torus.py")
        sys.exit(1)

def save_grid(tensor_batch, resolution, name, save_dir):
    """
    Plots a grid of images from a (B, C, H, W) tensor.
    """
    # Detach, Move to CPU, permute to (B, H, W, C) for Matplotlib
    images = tensor_batch.detach().permute(0, 2, 3, 1).cpu().numpy()
    
    B = images.shape[0]
    
    # Calculate grid dimensions
    cols = 4
    rows = math.ceil(B / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    
    # Flatten axes for easy iteration, handling cases with 1 row or 1 col
    if isinstance(axes,  plt.Axes): axes = [axes] # Handle B=1
    ax_flat = axes.flat if hasattr(axes, 'flat') else axes
    
    for i, ax in enumerate(ax_flat):
        if i < B:
            # Clip to [0,1] to avoid matplotlib warnings if we overshoot slightly
            img_data = images[i].clip(0, 1)
            ax.imshow(img_data)
            ax.set_title(f"Idx {i}", fontsize=8)
        
        ax.axis('off')
        
    plt.suptitle(f"Native Torus: {resolution}x{resolution} px", fontsize=16)
    plt.tight_layout()
    
    filename = os.path.join(save_dir, f"{name}.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Initializing Native Torus Generator on {device} ---")
    
    # Initialize the native raymarcher
    try:
        iterator = TorusIterator(device=device)
    except Exception as e:
        print(f"Failed to init iterator: {e}")
        return

    output_dir = "torus_native_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configurations to test: (Resolution, BatchSize)
    configs = [
        (16, 16),   # The "Brutal" Aliasing Test
        (32, 16),   # The "Sparse" Texture Test
        (64, 8),    # The "Clear" Geometry Test
        (256, 4)    # The "Ground Truth" Reference
    ]
    
    print(f"Dumping samples to ./{output_dir}/ ...")
    
    for res, bs in configs:
        print(f"  Rendering {bs} samples at {res}x{res}...")
        
        try:
            # Generate the batch
            batch = iterator.generate_batch(bs, res)
            
            # Print stats to ensure we aren't generating empty black images
            print(f"    Range: [{batch.min():.3f}, {batch.max():.3f}]")

            # Save the grid
            save_grid(batch, res, f"native_{res}px", output_dir)
            
        except Exception as e:
            print(f"  Failed rendering {res}x{res}: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone! Images saved to 'torus_native_samples'.")

if __name__ == "__main__":
    main()