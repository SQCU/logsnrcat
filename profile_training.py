"""
Profile actual training pipeline (AE warmup + denoiser).

Runs a short training pass (100 AE steps, 100 denoise steps) with CUDA profiling
to identify bottlenecks in the real pipeline vs isolated benchmarks.
"""

import torch
import os
import sys
from pathlib import Path
from torch.profiler import profile, record_function, ProfilerActivity
import time

# Prevent wandb from starting
os.environ['WANDB_MODE'] = 'disabled'

# Force non-interactive matplotlib backend
import matplotlib
matplotlib.use('Agg')

def main():
    print("Training Pipeline Profiler")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")

    # Apply patches
    from src.patches import apply_all as apply_patches
    apply_patches()

    # Enable flex attention backward caching
    torch._inductor.config.unsafe_marked_cacheable_functions['torch.ops.higher_order.flex_attention_backward'] = True

    # Import after env setup
    from src.config import load_config
    from src.data_iterator import CompositeIterator
    from src.train import train_autoembed, train_denoise, train_latent_diffusion
    from src.plotting import ExperimentLogger
    from main import build_components

    # Load config
    config_path = Path("configs/sparse_ae_swiglu_shared.toml")
    print(f"Loading config: {config_path}")
    cfg = load_config(config_path)

    # Override for short profiling run
    cfg['training']['ae_steps'] = 25
    cfg['training']['steps'] = 25
    cfg['logging']['log_interval'] = 5
    cfg['logging']['output_dir'] = './profile_traces'
    cfg['logging']['sample_after_training'] = False
    cfg['logging']['eval_server']['enabled'] = False
    cfg['sampling']['subspace_sensitivity']['enabled'] = False

    # Enable compile for realistic production performance
    cfg['training']['compile'] = True

    # Set dtype
    dtype_str = cfg['training']['precision']
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    dtype = dtype_map[dtype_str]
    cfg['dtype'] = dtype
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    cfg['device'] = device

    print(f"\nProfiling config:")
    print(f"  AE steps: {cfg['training']['ae_steps']}")
    print(f"  Denoise steps: {cfg['training']['steps']}")
    print(f"  Batch size: {cfg['training']['batch_size']}")
    print(f"  Compile: {cfg['training']['compile']}")
    print(f"  Precision: {dtype_str}")

    # Build components
    print("\nBuilding model and components...")
    components = build_components(cfg, device)
    model = components[0]

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    if hasattr(model, 'sparse_ae') and model.sparse_ae is not None:
        ae_params = sum(p.numel() for p in model.sparse_ae.parameters())
        print(f"Sparse AE parameters: {ae_params:,}")

    os.makedirs('./profile_traces', exist_ok=True)

    # Setup logging and data
    val_iterator = CompositeIterator(
        device,
        config=cfg['dataset_mix'],
        caching_resolution=cfg['training']['bucketing']['caching_resolution']
    )
    logger = ExperimentLogger(output_dir='./profile_traces')

    print("\n" + "=" * 70)
    print("PHASE 1: Profiling AE Warmup (25 steps)")
    print("=" * 70)

    # Profile AE training (includes compilation overhead in first few steps)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as ae_prof:
        df_ae = train_autoembed(components, cfg, val_iterator, logger)

    # Export trace
    ae_prof.export_chrome_trace("./profile_traces/ae_trace.json")

    # Print AE profile summary
    print("\n" + "=" * 70)
    print("AE TRAINING PROFILE SUMMARY")
    print("=" * 70)

    print("\nTop 30 CUDA operations by total GPU time:")
    print(ae_prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=30
    ))

    # Export AE trace
    ae_prof.export_chrome_trace("./profile_traces/ae_trace.json")
    print("\nExported AE trace to: profile_traces/ae_trace.json")

    # Memory summary
    print("\n" + "-" * 50)
    print("Memory operations (>1ms CPU):")
    for event in ae_prof.key_averages():
        if event.cpu_time_total > 1000 and ('mem' in event.key.lower() or 'empty' in event.key.lower() or 'zero' in event.key.lower() or 'copy' in event.key.lower()):
            print(f"  {event.key[:60]:60s} CPU: {event.cpu_time_total/1000:6.2f}ms  Calls: {event.count}")

    print("\n" + "=" * 70)
    print("PHASE 2: Profiling Denoiser Training (25 steps)")
    print("=" * 70)

    # Profile denoiser training
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as denoise_prof:
        # Select training function based on config
        sparse_ae_cfg = cfg['training'].get('sparse_ae', {})
        diffusion_space = sparse_ae_cfg.get('topology', {}).get('diffusion_space', 'pixel')

        if diffusion_space == 'latent':
            df_train = train_latent_diffusion(components, cfg, val_iterator, logger)
        else:
            df_train = train_denoise(components, cfg, val_iterator, logger)

    # Print denoiser profile summary
    print("\n" + "=" * 70)
    print("DENOISER TRAINING PROFILE SUMMARY")
    print("=" * 70)

    print("\nTop 30 CUDA operations by total GPU time:")
    print(denoise_prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=30
    ))

    # Export denoiser trace
    denoise_prof.export_chrome_trace("./profile_traces/denoise_trace.json")
    print("\nExported denoiser trace to: profile_traces/denoise_trace.json")

    # Sync analysis
    print("\n" + "-" * 50)
    print("Synchronization points:")
    for event in denoise_prof.key_averages():
        if 'sync' in event.key.lower() or 'Synchronize' in event.key:
            print(f"  {event.key[:60]:60s} CPU: {event.cpu_time_total/1000:6.2f}ms  Calls: {event.count}")

    # Combined analysis
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Calculate totals
    ae_cuda_total = sum(e.device_time_total for e in ae_prof.key_averages() if e.device_time_total) / 1000
    denoise_cuda_total = sum(e.device_time_total for e in denoise_prof.key_averages() if e.device_time_total) / 1000

    print(f"\nAE phase (25 steps):")
    print(f"  Total GPU time: {ae_cuda_total:.1f}ms")
    print(f"  Avg per step: {ae_cuda_total/25:.2f}ms")

    print(f"\nDenoiser phase (25 steps):")
    print(f"  Total GPU time: {denoise_cuda_total:.1f}ms")
    print(f"  Avg per step: {denoise_cuda_total/25:.2f}ms")

    print(f"\nTrace files exported to ./profile_traces/")
    print("  - ae_trace.json")
    print("  - denoise_trace.json")
    print("\nOpen traces in chrome://tracing or edge://tracing")


if __name__ == "__main__":
    main()
