"""
Lightweight timing breakdown - no trace export, just wall-clock analysis.
Measures where time goes in the training step without profiler overhead.
"""

import torch
import torch.nn.functional as F
import os
import time
from pathlib import Path

os.environ['WANDB_MODE'] = 'disabled'

import matplotlib
matplotlib.use('Agg')

def main():
    print("Training Step Breakdown")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")

    from src.patches import apply_all as apply_patches
    apply_patches()

    torch._inductor.config.unsafe_marked_cacheable_functions['torch.ops.higher_order.flex_attention_backward'] = True

    from src.config import load_config
    from src.data_iterator import CompositeIterator
    from src.optim_utils import build_optimizer_group
    from main import build_components

    # Load config
    cfg = load_config(Path("configs/sparse_ae_swiglu_shared.toml"))

    # Short run, compile enabled
    cfg['training']['ae_steps'] = 30
    cfg['training']['steps'] = 0
    cfg['training']['compile'] = True
    cfg['logging']['output_dir'] = './profile_traces'
    cfg['logging']['eval_server']['enabled'] = False

    dtype_str = cfg['training']['precision']
    dtype = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[dtype_str]
    cfg['dtype'] = dtype
    cfg['device'] = torch.device('cuda')
    torch.set_float32_matmul_precision("high")

    print(f"\nConfig: compile={cfg['training']['compile']}, precision={dtype_str}")

    # Build
    print("\nBuilding model...")
    components = build_components(cfg, cfg['device'])
    model = components[0]
    ae = model.sparse_ae

    print(f"AE parameters: {sum(p.numel() for p in ae.parameters()):,}")

    # Data iterator
    val_iterator = CompositeIterator(
        cfg['device'],
        config=cfg['dataset_mix'],
        caching_resolution=cfg['training']['bucketing']['caching_resolution']
    )

    # Optimizer
    ae_opt_cfg = cfg['training']['ae_optimizer']
    ae_opt = torch.optim.AdamW(ae.parameters(), lr=ae_opt_cfg['lr'], weight_decay=ae_opt_cfg['weight_decay'])

    use_amp = dtype in (torch.bfloat16, torch.float16)
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))

    # Warmup compilation
    print("\nWarming up (compilation)...")
    ae.train()
    for i in range(10):
        # Use generate_from_split to get consistent resolution without waste
        # sprite_atlas respects resolution kwarg, outputs at requested size
        blocks = val_iterator.generate_from_split('sprite_atlas', count=4, resolution=64)
        if len(blocks) == 0:
            continue
        images = torch.stack([b.content for b in blocks]).to(cfg['device'])

        with torch.amp.autocast('cuda', dtype=dtype, enabled=use_amp):
            out = ae(images)
            recon = out['recon']
            loss = F.mse_loss(recon, images)

        ae_opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(ae_opt)
        scaler.update()

        if i == 0:
            print(f"  Step {i}: compiling...")
        elif i == 5:
            print(f"  Step {i}: still warming...")

    torch.cuda.synchronize()
    print("  Warmup complete.")

    # Timed breakdown
    print("\n" + "=" * 70)
    print("STEADY-STATE TIMING BREAKDOWN (20 steps)")
    print("=" * 70)

    timings = {
        'data_gen': [],
        'to_device': [],
        'forward': [],
        'loss': [],
        'backward': [],
        'optimizer': [],
        'total': [],
    }

    # Pre-generate a batch for consistent timing
    torch.cuda.synchronize()

    for step in range(20):
        t_total_start = time.perf_counter()

        # Data generation - use generate_from_split for consistent resolution
        t0 = time.perf_counter()
        blocks = val_iterator.generate_from_split('sprite_atlas', count=4, resolution=64)
        t1 = time.perf_counter()
        timings['data_gen'].append(t1 - t0)

        if len(blocks) == 0:
            continue

        # To device
        t0 = time.perf_counter()
        images = torch.stack([b.content for b in blocks]).to(cfg['device'])
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings['to_device'].append(t1 - t0)

        # Forward
        t0 = time.perf_counter()
        with torch.amp.autocast('cuda', dtype=dtype, enabled=use_amp):
            out = ae(images)
            recon = out['recon']
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings['forward'].append(t1 - t0)

        # Loss
        t0 = time.perf_counter()
        with torch.amp.autocast('cuda', dtype=dtype, enabled=use_amp):
            loss = F.mse_loss(recon, images)
            if 'aux_loss' in out:
                loss = loss + 0.01 * out['aux_loss']
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings['loss'].append(t1 - t0)

        # Backward
        t0 = time.perf_counter()
        ae_opt.zero_grad()
        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings['backward'].append(t1 - t0)

        # Optimizer
        t0 = time.perf_counter()
        scaler.step(ae_opt)
        scaler.update()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings['optimizer'].append(t1 - t0)

        t_total_end = time.perf_counter()
        timings['total'].append(t_total_end - t_total_start)

    # Print results
    print(f"\n{'Phase':<15} {'Mean (ms)':<12} {'Std (ms)':<12} {'%':>8}")
    print("-" * 50)

    total_mean = sum(sum(v) for v in timings.values() if v) / len(timings['total'])

    for name, times in timings.items():
        if times:
            mean_ms = sum(times) / len(times) * 1000
            std_ms = (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5 * 1000
            pct = (sum(times) / len(times)) / (timings['total'][0] if timings['total'] else 1) * 100
            print(f"{name:<15} {mean_ms:<12.2f} {std_ms:<12.2f} {pct:>7.1f}%")

    total_mean_ms = sum(timings['total']) / len(timings['total']) * 1000
    batches_per_sec = 1000 / total_mean_ms

    print("-" * 50)
    print(f"\nTotal: {total_mean_ms:.1f}ms/step = {batches_per_sec:.2f} batches/sec")

    # Test async prefetcher
    print("\n" + "=" * 70)
    print("ASYNC PREFETCHER TIMING (20 steps)")
    print("=" * 70)

    from src.data_iterator import AsyncPrefetcher

    prefetcher = AsyncPrefetcher(
        iterator=val_iterator,
        split_name='sprite_atlas',
        count=4,
        resolution=64,
        buffer_size=8,
        seed=42,
        device=cfg['device']
    )

    # Warmup prefetcher - let it fill the buffer
    print("\nWarming up prefetcher buffer...")
    prefetcher.warmup(min_items=4)
    print(f"  Buffer filled: {prefetcher.stats['buffer_fill']}/{prefetcher.buffer_size}")

    async_timings = {
        'data_get': [],
        'to_device': [],
        'forward': [],
        'loss': [],
        'backward': [],
        'optimizer': [],
        'total': [],
    }

    torch.cuda.synchronize()

    for step in range(20):
        t_total_start = time.perf_counter()

        # Data fetch from prefetcher (should be near-instant)
        t0 = time.perf_counter()
        blocks = prefetcher.get()
        t1 = time.perf_counter()
        async_timings['data_get'].append(t1 - t0)

        if len(blocks) == 0:
            continue

        # To device
        t0 = time.perf_counter()
        images = torch.stack([b.content for b in blocks]).to(cfg['device'])
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        async_timings['to_device'].append(t1 - t0)

        # Forward
        t0 = time.perf_counter()
        with torch.amp.autocast('cuda', dtype=dtype, enabled=use_amp):
            out = ae(images)
            recon = out['recon']
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        async_timings['forward'].append(t1 - t0)

        # Loss
        t0 = time.perf_counter()
        with torch.amp.autocast('cuda', dtype=dtype, enabled=use_amp):
            loss = F.mse_loss(recon, images)
            if 'aux_loss' in out:
                loss = loss + 0.01 * out['aux_loss']
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        async_timings['loss'].append(t1 - t0)

        # Backward
        t0 = time.perf_counter()
        ae_opt.zero_grad()
        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        async_timings['backward'].append(t1 - t0)

        # Optimizer
        t0 = time.perf_counter()
        scaler.step(ae_opt)
        scaler.update()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        async_timings['optimizer'].append(t1 - t0)

        t_total_end = time.perf_counter()
        async_timings['total'].append(t_total_end - t_total_start)

    # Print async results
    print(f"\n{'Phase':<15} {'Mean (ms)':<12} {'Std (ms)':<12} {'%':>8}")
    print("-" * 50)

    async_total_mean = sum(async_timings['total']) / len(async_timings['total'])
    for name, times in async_timings.items():
        if times:
            mean_ms = sum(times) / len(times) * 1000
            std_ms = (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5 * 1000
            pct = (sum(times) / len(times)) / async_total_mean * 100
            print(f"{name:<15} {mean_ms:<12.2f} {std_ms:<12.2f} {pct:>7.1f}%")

    async_total_ms = async_total_mean * 1000
    async_batches_per_sec = 1000 / async_total_ms

    print("-" * 50)
    print(f"\nAsync Total: {async_total_ms:.1f}ms/step = {async_batches_per_sec:.2f} batches/sec")
    print(f"Prefetcher stats: {prefetcher.stats}")

    # Speedup comparison
    speedup = total_mean_ms / async_total_ms
    data_time_saved = (sum(timings['data_gen']) / len(timings['data_gen']) -
                       sum(async_timings['data_get']) / len(async_timings['data_get'])) * 1000
    print(f"\nSpeedup: {speedup:.2f}x ({data_time_saved:.1f}ms saved on data generation)")

    prefetcher.stop()

    # Compare to baseline expectation
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    fwd_mean = sum(timings['forward']) / len(timings['forward']) * 1000
    bwd_mean = sum(timings['backward']) / len(timings['backward']) * 1000

    print(f"\nForward/Backward ratio: {bwd_mean/fwd_mean:.2f}x")
    print(f"  (Expected ~2-3x for MoE due to grouped_gemm backward)")

    # MoE specific timing
    print("\n" + "=" * 70)
    print("MOE-SPECIFIC TIMING")
    print("=" * 70)

    # Time just the MoE layers
    print("\nIsolating MoE layer timing...")

    # Get a sample batch
    blocks = val_iterator.generate_from_split('sprite_atlas', count=4, resolution=64)
    if len(blocks) == 0:
        print("No images generated, skipping MoE timing")
        return
    images = torch.stack([b.content for b in blocks]).to(cfg['device'])

    # Find MoE modules
    moe_modules = []
    for name, mod in ae.named_modules():
        if 'SigmoidMoE' in type(mod).__name__:
            moe_modules.append((name, mod))

    if moe_modules:
        print(f"Found {len(moe_modules)} MoE modules")

        moe = moe_modules[0][1]

        # Create dummy input matching expected shape [B, L, D]
        B, C, H, W = images.shape
        L = (H//16) * (W//16)  # sequence length (patches)
        dummy = torch.randn(B, L, 256, device=cfg['device'], dtype=dtype)

        # Warmup with autocast
        with torch.amp.autocast('cuda', dtype=dtype, enabled=use_amp):
            for _ in range(5):
                _, _ = moe(dummy)
        torch.cuda.synchronize()

        # Time forward (no grads needed)
        times_fwd = []
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype, enabled=use_amp):
            for _ in range(20):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out, aux = moe(dummy)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times_fwd.append(t1 - t0)

        # Time backward (needs grads)
        times_bwd = []
        dummy_grad = dummy.detach().requires_grad_(True)
        with torch.amp.autocast('cuda', dtype=dtype, enabled=use_amp):
            for _ in range(20):
                out, aux = moe(dummy_grad)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                (out.sum() + aux).backward()
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times_bwd.append(t1 - t0)
                moe.zero_grad()
                dummy_grad.grad = None

        fwd_moe = sum(times_fwd) / len(times_fwd) * 1000
        bwd_moe = sum(times_bwd) / len(times_bwd) * 1000

        print(f"\nSingle MoE layer (batch={dummy.shape[0]}, seq_len={dummy.shape[1]}, dim={dummy.shape[2]}):")
        print(f"  Forward:  {fwd_moe:.3f}ms")
        print(f"  Backward: {bwd_moe:.3f}ms (includes forward)")
        print(f"  Ratio: {bwd_moe/fwd_moe:.2f}x")

        # Count MoE layers in full model
        n_moe = len(moe_modules)
        print(f"\nTotal MoE contribution estimate ({n_moe} layers):")
        print(f"  Forward:  {fwd_moe * n_moe:.1f}ms")
        print(f"  Backward: {bwd_moe * n_moe:.1f}ms")
    else:
        print("No MoE modules found!")


if __name__ == "__main__":
    main()
