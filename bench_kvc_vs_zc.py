#!/usr/bin/env python3
"""
bench_kvc_vs_zc.py - KV Cache vs Zero-Copy Performance Benchmark

Demonstrates:
1. VRAM usage: ZC (with grad) vs KVC (no grad, paged memory)
2. Throughput: Serial latency and batch throughput
3. Correctness: Hidden state norm comparison for identical contexts
4. Relevance: KV caching applies to diffusion, not just AR-LLM

This benchmark uses random-init models intentionally - the performance
characteristics are identical to trained models, and we avoid:
- API lock-in to specific checkpoints
- Pretending that models are precious artifacts
- Confusion between "model quality" and "systems performance"

Usage:
    python bench_kvc_vs_zc.py
    python bench_kvc_vs_zc.py --plot-only  # if results.json exists
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from src.config import load_config
from src.model import (
    coolerLDTformerZC, coolerLDTformerKVC,
    SpanEmbedder, SpanUnembedder, PageTable,
    ContextBlock, Span,
    render_topology_embeddings, build_dual_masks, update_kv_cache
)
from src.utils import KVTManager, logsnr_to_alpha_sigma


# =============================================================================
# Memory Measurement Utilities
# =============================================================================

def get_gpu_memory_mb() -> Dict[str, float]:
    """Returns current GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "max_allocated": 0}

    return {
        "allocated": torch.cuda.memory_allocated() / 1024**2,
        "reserved": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated": torch.cuda.max_memory_allocated() / 1024**2,
    }


def reset_memory_stats():
    """Reset peak memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


# =============================================================================
# Test Data Generation
# =============================================================================

def create_test_context(
    num_text_tokens: int,
    num_latents: int,
    latent_res: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> List[ContextBlock]:
    """
    Creates a test context: [text prefix] + [N latent spans]

    This mimics a real inference scenario:
    - Text prompt (random tokens, but structure matters)
    - Multiple latent generations (e.g., video frames, multi-view)
    """
    blocks = []
    group_id = 0

    # Text prefix (random tokens)
    if num_text_tokens > 0:
        text_tokens = torch.randint(0, 1000, (num_text_tokens,), device=device)
        blocks.append(ContextBlock(
            content=text_tokens,
            type='text',
            causal=True,
            shape_meta=(num_text_tokens,),
            group_id=group_id,
            id="text_prefix"
        ))

    # Latent spans
    for i in range(num_latents):
        # Random "noisy" latent at some logsnr
        content = torch.randn(3, latent_res, latent_res, device=device, dtype=dtype)
        logsnr = torch.full((1, latent_res, latent_res), -2.0, device=device, dtype=dtype)

        blocks.append(ContextBlock(
            content=content,
            type='latent',
            causal=False,  # Bidirectional within latent
            shape_meta=(latent_res, latent_res),
            logsnr=logsnr,
            group_id=group_id,
            id=f"latent_{i}"
        ))

    return blocks


# =============================================================================
# ZC (Zero-Copy) Forward Pass
# =============================================================================

def run_zc_forward(
    model: coolerLDTformerZC,
    span_emb: SpanEmbedder,
    span_unemb: SpanUnembedder,
    page_table: PageTable,
    blocks: List[ContextBlock],
    with_grad: bool = True
) -> Tuple[torch.Tensor, List[Dict], float]:
    """
    Standard ZC forward pass (training-style).
    Returns: (z_out, decoded_outputs, aux_loss)
    """
    device = model.text_embed.weight.device
    dtype = model.text_embed.weight.dtype  # Infer dtype from model

    context_manager = torch.enable_grad() if with_grad else torch.no_grad()

    with context_manager:
        # 1. Embed
        z_flat, span_objects, _ = span_emb.embed(blocks)

        # 2. Topology (dtype from model)
        topo_embeds, _ = render_topology_embeddings(span_objects, 3, device, dtype=dtype)

        # 3. Masking
        L_total = z_flat.shape[0]
        block_size = page_table.block_size
        num_blocks = (L_total + block_size - 1) // block_size
        flat_page_table = torch.arange(num_blocks, device=device, dtype=torch.long)

        block_masks = build_dual_masks(
            span_objects, topo_embeds, topo_embeds,
            page_table, flat_page_table, None,
            window_size=model.window_size
        )

        # 4. Forward
        rope_scale = max(1.0, L_total / 64.0)
        z_out, aux_loss = model(
            z_flat.unsqueeze(0),
            topo_embeds.unsqueeze(0),
            slot_mapping=None,
            block_masks=block_masks,
            scale=rope_scale
        )

        # 5. Decode
        decoded = span_unemb.decode(z_out.squeeze(0), span_objects)

    return z_out, decoded, aux_loss


# =============================================================================
# KVC (KV-Cache) Forward Pass
# =============================================================================

def run_kvc_forward(
    model: coolerLDTformerKVC,
    span_emb: SpanEmbedder,
    span_unemb: SpanUnembedder,
    kvt_manager: KVTManager,
    page_table: PageTable,
    blocks: List[ContextBlock],
    req_id: int = 0
) -> Tuple[torch.Tensor, List[Dict], float]:
    """
    KVC forward pass with paged attention.
    Returns: (z_out, decoded_outputs, aux_loss)
    """
    device = model.text_embed.weight.device
    dtype = model.text_embed.weight.dtype  # Infer dtype from model

    with torch.no_grad():
        # 1. Embed
        z_flat, span_objects, content_hashes = span_emb.embed(blocks)

        # 2. Topology (dtype from model)
        topo_embeds, _ = render_topology_embeddings(span_objects, 3, device, dtype=dtype)

        # 3. Allocate in KVT Manager
        kvt_manager.allocate_and_write_sequence(req_id, content_hashes, topo_embeds)

        # 4. Get paging info
        flat_page_table, inverse_page_table = kvt_manager.get_flat_page_mapping([req_id])
        block_tables = [kvt_manager.req_tables[req_id]]
        seq_lengths = [kvt_manager.req_lengths[req_id]]
        slot_mapping = kvt_manager.get_slot_mapping(block_tables, seq_lengths)

        # 5. Build masks
        topo_heap = kvt_manager.get_topo_view()
        block_masks = build_dual_masks(
            span_objects, topo_embeds, topo_heap,
            page_table, flat_page_table, inverse_page_table,
            window_size=model.window_size
        )

        # 6. Forward with KV caches
        L_total = z_flat.shape[0]
        rope_scale = max(1.0, L_total / 64.0)

        # Get per-layer caches
        k_caches = [kvt_manager.get_flat_kv_view(i)[0] for i in range(len(model.layers))]
        v_caches = [kvt_manager.get_flat_kv_view(i)[1] for i in range(len(model.layers))]

        z_out, aux_loss = model(
            z_flat.unsqueeze(0),
            topo_embeds.unsqueeze(0),
            k_caches, v_caches,
            slot_mapping,
            block_masks,
            scale=rope_scale
        )

        # 7. Decode
        decoded = span_unemb.decode(z_out.squeeze(0), span_objects)

        # 8. Free request
        kvt_manager.free_request(req_id)

    return z_out, decoded, aux_loss


# =============================================================================
# Benchmark Functions
# =============================================================================

@dataclass
class BenchmarkResult:
    name: str
    vram_allocated_mb: float
    vram_peak_mb: float
    latency_ms: float
    throughput_tokens_per_sec: float
    output_norm: float
    num_tokens: int


def benchmark_zc(
    cfg: Dict,
    device: torch.device,
    dtype: torch.dtype,
    num_text: int,
    num_latents: int,
    latent_res: int,
    with_grad: bool,
    warmup_runs: int = 2,
    timed_runs: int = 5
) -> BenchmarkResult:
    """Benchmark ZC model."""

    # Build model
    model = coolerLDTformerZC(
        dim=cfg['model']['dim'],
        depth=cfg['model']['depth'],
        num_heads=cfg['model']['num_heads'],
        topo_dim=cfg['model']['topo_dim'],
        mlp_depth=cfg['model']['mlp_depth'],
        vocab_size=cfg['model']['vocab_size'],
        global_layer_interval=cfg['model']['global_layer_interval'],
        num_experts=cfg['model']['num_experts'],
        num_active=cfg['model']['num_active'],
        rope_base=cfg['model']['rope_base'],
        mlp_ratio=cfg['model']['mlp_ratio'],
        jitter_noise=0.0,  # No jitter for deterministic benchmark
        context_size=cfg['model']['patch_embedder']['context_size'],
        stride=cfg['model']['patch_embedder']['stride'],
        fourier_dim=cfg['model']['patch_embedder']['fourier_dim'],
        window_size=cfg['model']['window_size']
    ).to(device=device, dtype=dtype)

    span_emb = SpanEmbedder(model.text_embed, model.patch_embedder)
    span_unemb = SpanUnembedder(model.text_head, model.patch_unembedder)
    page_table = PageTable(
        num_blocks=cfg['page_table']['num_blocks'],
        block_size=cfg['page_table']['block_size'],
        max_batch_size=cfg['page_table']['max_batch_size'],
        max_logical_blocks=cfg['page_table']['max_logical_blocks'],
        device=device
    )

    # Create test data
    blocks = create_test_context(num_text, num_latents, latent_res, device, dtype)

    # Calculate token count
    num_tokens = num_text + num_latents * (latent_res // cfg['model']['patch_embedder']['stride'])**2

    # Warmup
    for _ in range(warmup_runs):
        z_out, _, _ = run_zc_forward(model, span_emb, span_unemb, page_table, blocks, with_grad)
        if with_grad:
            z_out.sum().backward()
            model.zero_grad()

    # Reset and measure
    reset_memory_stats()
    torch.cuda.synchronize()

    latencies = []
    output_norm = 0.0

    for i in range(timed_runs):
        start = time.perf_counter()
        z_out, _, _ = run_zc_forward(model, span_emb, span_unemb, page_table, blocks, with_grad)
        if with_grad:
            z_out.sum().backward()
            model.zero_grad()
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

        if i == 0:
            output_norm = z_out.norm().item()

    mem = get_gpu_memory_mb()

    # Cleanup
    del model, span_emb, span_unemb, page_table, blocks
    torch.cuda.empty_cache()

    avg_latency = np.mean(latencies)

    return BenchmarkResult(
        name=f"ZC ({'grad' if with_grad else 'no_grad'})",
        vram_allocated_mb=mem['allocated'],
        vram_peak_mb=mem['max_allocated'],
        latency_ms=avg_latency,
        throughput_tokens_per_sec=num_tokens / (avg_latency / 1000),
        output_norm=output_norm,
        num_tokens=num_tokens
    )


def benchmark_kvc(
    cfg: Dict,
    device: torch.device,
    dtype: torch.dtype,
    num_text: int,
    num_latents: int,
    latent_res: int,
    warmup_runs: int = 2,
    timed_runs: int = 5,
    headroom_multiplier: float = 2.0  # Extra space for multi-turn / diffusion steps
) -> BenchmarkResult:
    """Benchmark KVC model."""

    # Calculate token count FIRST to right-size allocations
    block_size = cfg['page_table']['block_size']
    latent_tokens_per_span = (latent_res // cfg['model']['patch_embedder']['stride'])**2
    num_tokens = num_text + num_latents * latent_tokens_per_span

    # Right-size the KV cache: actual blocks needed + headroom for multi-turn
    needed_blocks = (num_tokens + block_size - 1) // block_size
    max_blocks = max(needed_blocks * int(headroom_multiplier), 8)  # minimum 8 blocks

    print(f"  KVC allocation: {num_tokens} tokens -> {needed_blocks} blocks needed, "
          f"allocating {max_blocks} (headroom={headroom_multiplier}x)")

    # Build model
    model = coolerLDTformerKVC(
        dim=cfg['model']['dim'],
        depth=cfg['model']['depth'],
        num_heads=cfg['model']['num_heads'],
        topo_dim=cfg['model']['topo_dim'],
        mlp_depth=cfg['model']['mlp_depth'],
        vocab_size=cfg['model']['vocab_size'],
        global_layer_interval=cfg['model']['global_layer_interval'],
        num_experts=cfg['model']['num_experts'],
        num_active=cfg['model']['num_active'],
        rope_base=cfg['model']['rope_base'],
        mlp_ratio=cfg['model']['mlp_ratio'],
        jitter_noise=0.0,
        context_size=cfg['model']['patch_embedder']['context_size'],
        stride=cfg['model']['patch_embedder']['stride'],
        fourier_dim=cfg['model']['patch_embedder']['fourier_dim'],
        window_size=cfg['model']['window_size']
    ).to(device=device, dtype=dtype)

    span_emb = SpanEmbedder(model.text_embed, model.patch_embedder)
    span_unemb = SpanUnembedder(model.text_head, model.patch_unembedder)

    page_table = PageTable(
        num_blocks=max_blocks,  # Right-sized
        block_size=block_size,
        max_batch_size=cfg['page_table']['max_batch_size'],
        max_logical_blocks=max_blocks,  # Match
        device=device
    )

    # KVT Manager for paged attention - RIGHT-SIZED
    kvt_manager = KVTManager(
        max_blocks=max_blocks,
        block_size=block_size,
        kv_dim=cfg['model']['dim'],
        layers=cfg['model']['depth'],
        heads=cfg['model']['num_heads'],
        topo_dim=cfg['model']['topo_dim'],
        device=device,
        dtype=dtype
    )

    # Create test data
    blocks = create_test_context(num_text, num_latents, latent_res, device, dtype)

    # Warmup
    for i in range(warmup_runs):
        z_out, _, _ = run_kvc_forward(model, span_emb, span_unemb, kvt_manager, page_table, blocks, req_id=i)

    # Reset and measure
    reset_memory_stats()
    torch.cuda.synchronize()

    latencies = []
    output_norm = 0.0

    for i in range(timed_runs):
        req_id = warmup_runs + i
        start = time.perf_counter()
        z_out, _, _ = run_kvc_forward(model, span_emb, span_unemb, kvt_manager, page_table, blocks, req_id=req_id)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

        if i == 0:
            output_norm = z_out.norm().item()

    mem = get_gpu_memory_mb()

    # Cleanup
    del model, span_emb, span_unemb, page_table, kvt_manager, blocks
    torch.cuda.empty_cache()

    avg_latency = np.mean(latencies)

    return BenchmarkResult(
        name="KVC (paged)",
        vram_allocated_mb=mem['allocated'],
        vram_peak_mb=mem['max_allocated'],
        latency_ms=avg_latency,
        throughput_tokens_per_sec=num_tokens / (avg_latency / 1000),
        output_norm=output_norm,
        num_tokens=num_tokens
    )


# =============================================================================
# Plotting
# =============================================================================

def plot_results(results: List[BenchmarkResult], output_path: Path):
    """Generate comparison plots."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('KVC vs ZC Performance Comparison\n(Random Init Model - Systems Benchmark)', fontsize=14)

    names = [r.name for r in results]
    colors = ['#e74c3c', '#3498db', '#2ecc71'][:len(results)]

    # 1. VRAM Usage
    ax = axes[0, 0]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, [r.vram_allocated_mb for r in results], width, label='Allocated', color=colors, alpha=0.7)
    ax.bar(x + width/2, [r.vram_peak_mb for r in results], width, label='Peak', color=colors, alpha=0.4, hatch='//')
    ax.set_ylabel('VRAM (MB)')
    ax.set_title('Memory Usage')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. Latency
    ax = axes[0, 1]
    bars = ax.bar(names, [r.latency_ms for r in results], color=colors, alpha=0.7)
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Forward Pass Latency')
    ax.grid(axis='y', alpha=0.3)
    for bar, r in zip(bars, results):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{r.latency_ms:.1f}ms', ha='center', va='bottom', fontsize=9)

    # 3. Throughput
    ax = axes[1, 0]
    bars = ax.bar(names, [r.throughput_tokens_per_sec for r in results], color=colors, alpha=0.7)
    ax.set_ylabel('Tokens/sec')
    ax.set_title('Throughput')
    ax.grid(axis='y', alpha=0.3)
    for bar, r in zip(bars, results):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{r.throughput_tokens_per_sec:.0f}', ha='center', va='bottom', fontsize=9)

    # 4. Output Norm (Correctness Sanity Check)
    ax = axes[1, 1]
    bars = ax.bar(names, [r.output_norm for r in results], color=colors, alpha=0.7)
    ax.set_ylabel('L2 Norm')
    ax.set_title('Output Hidden State Norm\n(should be similar across methods)')
    ax.grid(axis='y', alpha=0.3)

    # Add note about what we're measuring
    norm_mean = np.mean([r.output_norm for r in results])
    norm_std = np.std([r.output_norm for r in results])
    ax.axhline(y=norm_mean, color='gray', linestyle='--', alpha=0.5)
    ax.text(0.02, 0.98, f'μ={norm_mean:.2f}, σ={norm_std:.2f}',
            transform=ax.transAxes, fontsize=9, va='top')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def print_results_table(results: List[BenchmarkResult]):
    """Print results as ASCII table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Method':<20} {'VRAM (MB)':<12} {'Peak (MB)':<12} {'Latency':<12} {'Tokens/s':<12} {'Norm':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r.name:<20} {r.vram_allocated_mb:<12.1f} {r.vram_peak_mb:<12.1f} "
              f"{r.latency_ms:<12.2f} {r.throughput_tokens_per_sec:<12.0f} {r.output_norm:<10.2f}")
    print("=" * 80)

    # Analysis
    if len(results) >= 2:
        zc_grad = next((r for r in results if 'grad' in r.name and 'no_grad' not in r.name), None)
        kvc = next((r for r in results if 'KVC' in r.name), None)

        if zc_grad and kvc:
            print("\nANALYSIS:")
            print(f"  VRAM Reduction (KVC vs ZC+grad): {(1 - kvc.vram_peak_mb/zc_grad.vram_peak_mb)*100:.1f}%")
            print(f"  Speedup (KVC vs ZC+grad): {zc_grad.latency_ms/kvc.latency_ms:.2f}x")
            print(f"  Throughput Gain: {kvc.throughput_tokens_per_sec/zc_grad.throughput_tokens_per_sec:.2f}x")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="KVC vs ZC Benchmark")
    parser.add_argument("--config", default="configs/multisnr_default.toml", help="Config file")
    parser.add_argument("--num-text", type=int, default=64, help="Number of text tokens in prefix")
    parser.add_argument("--num-latents", type=int, default=4, help="Number of 16x16 latent spans")
    parser.add_argument("--latent-res", type=int, default=16, help="Latent resolution (before patching)")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16", help="Data type")
    parser.add_argument("--output-dir", default="./benchmark_results", help="Output directory")
    parser.add_argument("--plot-only", action="store_true", help="Only plot from existing results.json")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    results_file = output_dir / "kvc_vs_zc_results.json"
    plot_file = output_dir / "kvc_vs_zc_comparison.png"

    if args.plot_only:
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
            results = [BenchmarkResult(**r) for r in data['results']]
            plot_results(results, plot_file)
            print_results_table(results)
        else:
            print(f"No results file found at {results_file}")
        return

    # Load config
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print("=" * 60)
    print("KVC vs ZC Benchmark")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Context: {args.num_text} text tokens + {args.num_latents}x {args.latent_res}x{args.latent_res} latents")
    print(f"Model: {cfg['model']['dim']}d, {cfg['model']['depth']}L, {cfg['model']['num_heads']}H")
    print("=" * 60)

    results = []

    # 1. ZC with gradients (training mode)
    print("\n[1/3] Benchmarking ZC (with gradients)...")
    try:
        r = benchmark_zc(cfg, device, dtype, args.num_text, args.num_latents, args.latent_res, with_grad=True)
        results.append(r)
        print(f"  -> {r.latency_ms:.2f}ms, {r.vram_peak_mb:.1f}MB peak")
    except Exception as e:
        print(f"  -> FAILED: {e}")

    # 2. ZC without gradients (inference baseline)
    print("\n[2/3] Benchmarking ZC (no gradients)...")
    try:
        r = benchmark_zc(cfg, device, dtype, args.num_text, args.num_latents, args.latent_res, with_grad=False)
        results.append(r)
        print(f"  -> {r.latency_ms:.2f}ms, {r.vram_peak_mb:.1f}MB peak")
    except Exception as e:
        print(f"  -> FAILED: {e}")

    # 3. KVC (paged attention)
    print("\n[3/3] Benchmarking KVC (paged attention)...")
    try:
        r = benchmark_kvc(cfg, device, dtype, args.num_text, args.num_latents, args.latent_res)
        results.append(r)
        print(f"  -> {r.latency_ms:.2f}ms, {r.vram_peak_mb:.1f}MB peak")
    except Exception as e:
        print(f"  -> FAILED: {e}")

    if not results:
        print("\nNo benchmarks completed successfully!")
        return

    # Save results
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'num_text': args.num_text,
                'num_latents': args.num_latents,
                'latent_res': args.latent_res,
                'dtype': args.dtype,
                'model_dim': cfg['model']['dim'],
                'model_depth': cfg['model']['depth']
            },
            'results': [r.__dict__ for r in results]
        }, f, indent=2)
    print(f"\nSaved results to {results_file}")

    # Plot and print
    plot_results(results, plot_file)
    print_results_table(results)

    print(f"\n✓ Benchmark complete. Artifacts in {output_dir}/")


if __name__ == "__main__":
    main()
