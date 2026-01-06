"""
Profile MoE forward/backward to find pipeline stalls and sync points.

Outputs:
1. Chrome trace (open in chrome://tracing)
2. Summary table of operations by duration
3. CUDA sync events and memory ops
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
import os

# Import our MoE implementation
from src.blocks import SigmoidMoE, HAS_GROUPED_GEMM

def create_test_model(device, dtype):
    """Create a small model with MoE layers for profiling."""

    class TestModel(nn.Module):
        def __init__(self, dim=512, hidden_dim=2048, num_experts=8, num_active=2, num_layers=2):
            super().__init__()
            self.layers = nn.ModuleList([
                SigmoidMoE(dim, hidden_dim, num_experts, num_active)
                for _ in range(num_layers)
            ])
            self.norm = nn.RMSNorm(dim)

        def forward(self, x):
            aux_losses = []
            for layer in self.layers:
                out, aux = layer(x)
                x = x + out
                aux_losses.append(aux)
            return self.norm(x), sum(aux_losses)

    return TestModel().to(device, dtype)


def profile_model(model, x, num_warmup=5, num_profile=10, backward=True):
    """Profile forward (and optionally backward) passes."""

    # Warmup
    for _ in range(num_warmup):
        out, aux = model(x)
        if backward:
            (out.sum() + aux).backward()
            model.zero_grad()

    torch.cuda.synchronize()

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for _ in range(num_profile):
            out, aux = model(x)
            if backward:
                (out.sum() + aux).backward()
                model.zero_grad()
            # Mark iteration boundary
            prof.step()

    return prof


def analyze_sync_points(prof):
    """Find CUDA synchronization events that might cause stalls."""
    print("\n" + "="*70)
    print("CUDA SYNCHRONIZATION ANALYSIS")
    print("="*70)

    events = prof.key_averages()

    # Look for sync-related operations
    sync_keywords = ['synchronize', 'cudaStreamSync', 'cudaDeviceSync',
                     'cudaMemcpy', 'to(', '.cpu()', '.cuda()', 'copy_']

    sync_events = []
    for evt in events:
        name_lower = evt.key.lower()
        if any(kw.lower() in name_lower for kw in sync_keywords):
            sync_events.append(evt)

    if sync_events:
        print("\nPotential sync points found:")
        for evt in sorted(sync_events, key=lambda e: -e.cpu_time_total):
            device_time = evt.device_time_total
            print(f"  {evt.key[:60]:60s} "
                  f"GPU: {device_time/1000:8.2f}ms "
                  f"CPU: {evt.cpu_time_total/1000:8.2f}ms "
                  f"Calls: {evt.count}")
    else:
        print("\nNo obvious sync points found in event names")

    return sync_events


def analyze_gaps(prof):
    """Analyze gaps between CUDA kernels (potential pipeline stalls)."""
    print("\n" + "="*70)
    print("KERNEL TIMELINE ANALYSIS")
    print("="*70)

    # Get top CUDA operations by time
    events = prof.key_averages()
    cuda_events = [e for e in events if e.device_time_total > 0]
    cuda_events.sort(key=lambda e: -e.device_time_total)

    print("\nTop 20 GPU operations by total time:")
    print(f"{'Operation':<55} {'GPU ms':>10} {'CPU ms':>10} {'Calls':>8}")
    print("-" * 85)

    for evt in cuda_events[:20]:
        name = evt.key[:55]
        print(f"{name:<55} {evt.device_time_total/1000:10.2f} "
              f"{evt.cpu_time_total/1000:10.2f} {evt.count:8d}")

    # Calculate total GPU time and estimate gaps
    total_gpu = sum(e.device_time_total for e in cuda_events)
    print(f"\nTotal GPU kernel time: {total_gpu/1000:.2f}ms")


def analyze_memory_ops(prof):
    """Find memory operations that might cause stalls."""
    print("\n" + "="*70)
    print("MEMORY OPERATION ANALYSIS")
    print("="*70)

    events = prof.key_averages()

    # Memory-related keywords
    mem_keywords = ['alloc', 'free', 'malloc', 'memcpy', 'memset',
                    'empty', 'zeros', 'ones', 'clone', 'contiguous']

    mem_events = []
    for evt in events:
        name_lower = evt.key.lower()
        if any(kw in name_lower for kw in mem_keywords):
            if evt.cpu_time_total > 100:  # Only significant ops (>0.1ms)
                mem_events.append(evt)

    if mem_events:
        print("\nSignificant memory operations (>0.1ms):")
        for evt in sorted(mem_events, key=lambda e: -e.cpu_time_total)[:15]:
            print(f"  {evt.key[:55]:55s} CPU: {evt.cpu_time_total/1000:8.2f}ms  Calls: {evt.count}")
    else:
        print("\nNo significant memory operations found")


def export_traces(prof, output_dir="profile_traces"):
    """Export traces for external analysis."""
    os.makedirs(output_dir, exist_ok=True)

    # Chrome trace
    trace_path = os.path.join(output_dir, "moe_trace.json")
    prof.export_chrome_trace(trace_path)
    print(f"\nExported Chrome trace to: {trace_path}")
    print("  Open in chrome://tracing or edge://tracing")

    # Stacks for flame graph
    stacks_path = os.path.join(output_dir, "moe_stacks.txt")
    prof.export_stacks(stacks_path, metric="self_cuda_time_total")
    print(f"Exported CUDA stacks to: {stacks_path}")

    return trace_path


def main():
    print("MoE Pipeline Profiler")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"grouped_gemm available: {HAS_GROUPED_GEMM}")

    device = torch.device('cuda')
    dtype = torch.bfloat16

    # Test configuration
    batch_size = 4
    seq_len = 1024
    dim = 512

    print(f"\nTest config: batch={batch_size}, seq={seq_len}, dim={dim}")
    print(f"Model: 2 MoE layers, 8 experts, 2 active")

    # Create model and input
    model = create_test_model(device, dtype)
    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype, requires_grad=True)

    print("\nProfiling forward + backward passes...")

    with torch.amp.autocast('cuda', dtype=dtype):
        prof = profile_model(model, x, num_warmup=5, num_profile=10, backward=True)

    # Analysis
    analyze_sync_points(prof)
    analyze_gaps(prof)
    analyze_memory_ops(prof)

    # Export
    trace_path = export_traces(prof)

    # Print summary table
    print("\n" + "="*70)
    print("FULL OPERATION TABLE")
    print("="*70)
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))


if __name__ == "__main__":
    main()
