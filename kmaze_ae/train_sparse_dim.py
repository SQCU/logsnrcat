"""Train sparse per-dim hierarchical FSQ autoencoder.

Config: 128-dim 3-bit, 6 levels, 97% sparsity (4 dims per patch)
"""
import torch
import torch.nn.functional as F
import time
from pathlib import Path
from PIL import Image
import numpy as np

from model_sparse_dim import SparsePerDimFSQAutoencoder
from fractal import FractalQueue


def save_comparison_image(originals, level_recons, path, n_samples=4):
    originals = originals[:n_samples].detach().cpu()
    recon_0 = level_recons[0][:n_samples].detach().cpu()
    recon_01 = level_recons[1][:n_samples].detach().cpu()
    final = level_recons[-1][:n_samples].detach().cpu()

    originals = originals.clamp(0, 1).numpy().transpose(0, 2, 3, 1)
    recon_0 = recon_0.clamp(0, 1).numpy().transpose(0, 2, 3, 1)
    recon_01 = recon_01.clamp(0, 1).numpy().transpose(0, 2, 3, 1)
    final = final.clamp(0, 1).numpy().transpose(0, 2, 3, 1)

    rows = []
    for i in range(n_samples):
        row = np.concatenate([originals[i], recon_0[i], recon_01[i], final[i]], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)

    grid = (grid * 255).astype(np.uint8)
    Image.fromarray(grid).save(path)


def train():
    device = 'cuda:0'
    print(f"Using device: {device}")

    output_dir = Path("outputs_sparse_dim")
    output_dir.mkdir(exist_ok=True)

    # Sparse per-dim model: 128-dim 3-bit, 6 levels, 4 dims per patch (97% sparsity)
    model = SparsePerDimFSQAutoencoder(
        n_levels=6,
        patch_size=16,
        image_size=256,
        hidden_dim=256,
        code_dim=128,
        k_per_patch=4,  # keep 4 of 128 = 3.1% = 97% sparsity
        residual_scale=2.0
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Config: 128-dim 3-bit, 6 levels, 4 dims/patch (97% sparsity)")
    print(f"Effective bits per image: 256 patches × 4 dims × 3 bits × 6 levels = {256*4*3*6} bits")

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    print("Starting fractal generator workers...")
    fractal_queue = FractalQueue(size=256, n_workers=4, queue_size=256)

    batch_size = 32
    grad_clip = 1.0

    last_image_time = time.time()
    image_interval = 10

    step = 0
    start_time = time.time()

    print("Starting training (sparse per-dim)...")
    print("-" * 60)

    try:
        while True:
            batch = fractal_queue.get_batch(batch_size).to(device)

            output = model(batch)
            recon = output['recon']

            loss = F.mse_loss(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            step += 1
            current_time = time.time()

            if step % 10 == 0:
                elapsed = current_time - start_time
                sparsity = output['sparsity']
                print(f"Step {step:6d} | Loss: {loss.item():.6f} | Sparsity: {sparsity*100:.1f}% | Time: {elapsed:.1f}s")

            if current_time - last_image_time >= image_interval:
                img_path = output_dir / f"comparison_step{step:06d}.png"
                save_comparison_image(batch, output['level_recons'], str(img_path))
                print(f"  -> Saved {img_path}")
                last_image_time = current_time

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        print("Shutting down fractal workers...")
        fractal_queue.shutdown()

        ckpt_path = output_dir / "checkpoint_final.pt"
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    train()
