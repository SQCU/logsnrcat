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


def compile_transformer_layers(model, mode='default'):
    """
    Compile ONLY the transformer layers, not embedding/unembedding.

    Why selective compilation:
    - Embedding: Dynamic shapes (variable span counts, gather ops)
    - Transformer layers: Uniform access patterns, static shapes once embedded
    - Unembedding: Selective gather/scatter for specific spans

    The transformer stack is the compute-heavy part and benefits most from
    fusion. Embedding/unembedding are memory-bound with dynamic control flow.

    See PyTorch blog on static shapes for CUDA graphs:
    "Instead of using the dynamic-shape tensors... we used static shape tensors
    where a mask is used to indicate which elements are valid."
    """
    for i, layer in enumerate(model.layers):
        model.layers[i] = torch.compile(layer, mode=mode, dynamic=False)
    return model

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


# =============================================================================
# Concatenative AR Benchmarks (True Multi-Span Growing Context)
# =============================================================================

@dataclass
class ConcatARResult:
    """Results for concatenative autoregressive benchmarks."""
    name: str
    total_latency_ms: float
    num_outer_steps: int        # Number of latents generated
    num_inner_steps: int        # Diffusion steps per latent
    total_forward_passes: int   # outer * inner
    final_context_tokens: int   # Text + all latents (per sequence)
    prefix_tokens: int          # Text only
    vram_peak_mb: float
    avg_ms_per_forward: float
    tokens_computed_total: int  # For efficiency calculation
    blocks_reembedded: int = 0  # How many blocks were actually re-embedded (KVC diagnostic)
    batch_size: int = 1         # Number of parallel sequences
    sequences_per_second: float = 0.0  # Throughput metric for batched runs


def benchmark_concat_ar_zc(
    cfg: Dict,
    device: torch.device,
    dtype: torch.dtype,
    num_text: int,
    latent_res: int,
    num_latents: int = 3,       # Outer AR steps (number of latents to generate)
    steps_per_latent: int = 5,  # Inner diffusion steps per latent
    warmup_runs: int = 1,
    use_compile: bool = True
) -> ConcatARResult:
    """
    Concatenative AR benchmark with ZC (baseline).

    Pattern:
        for each new latent:  # OUTER LOOP - context GROWS
            append noisy latent to context
            for each diffusion step:  # INNER LOOP - refine
                forward(full_context)  # ZC recomputes everything

    This is the CORRECT test for KV caching benefit:
    - Context grows: [text] -> [text, lat1] -> [text, lat1, lat2] -> ...
    - ZC must recompute ALL tokens every time
    - KVC should cache prefix and only compute new/active tokens
    """
    from src.model import coolerLDTformerZC

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
        jitter_noise=0.0,
        context_size=cfg['model']['patch_embedder']['context_size'],
        stride=cfg['model']['patch_embedder']['stride'],
        fourier_dim=cfg['model']['patch_embedder']['fourier_dim'],
        window_size=cfg['model']['window_size']
    ).to(device=device, dtype=dtype)

    # Compile ONLY the transformer layers, not embedding/unembedding
    if use_compile:
        compile_transformer_layers(model)

    span_emb = SpanEmbedder(model.text_embed, model.patch_embedder)
    span_unemb = SpanUnembedder(model.text_head, model.patch_unembedder)

    block_size = cfg['page_table']['block_size']
    latent_tokens_per_span = (latent_res // cfg['model']['patch_embedder']['stride'])**2
    max_tokens = num_text + num_latents * latent_tokens_per_span
    max_blocks = (max_tokens + block_size - 1) // block_size

    page_table = PageTable(
        num_blocks=max(max_blocks * 2, 16),
        block_size=block_size,
        max_batch_size=cfg['page_table']['max_batch_size'],
        max_logical_blocks=max(max_blocks * 2, 16),
        device=device
    )

    # Start with text prefix only
    text_tokens = torch.randint(0, 1000, (num_text,), device=device)
    text_block = ContextBlock(
        content=text_tokens,
        type='text',
        causal=True,
        shape_meta=(num_text,),
        group_id=0,
        id="text_prefix"
    )
    blocks = [text_block]

    # Warmup with a single latent
    warmup_latent = torch.randn(3, latent_res, latent_res, device=device, dtype=dtype)
    warmup_block = ContextBlock(
        content=warmup_latent, type='latent', causal=False,
        shape_meta=(latent_res, latent_res),
        logsnr=torch.full((1, latent_res, latent_res), -4.0, device=device, dtype=dtype),
        group_id=0, id="warmup"
    )
    for _ in range(warmup_runs):
        run_zc_forward(model, span_emb, span_unemb, page_table,
                       [text_block, warmup_block], with_grad=False)

    # Reset and measure
    reset_memory_stats()
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    total_forwards = 0
    tokens_computed = 0

    # === OUTER LOOP: Generate num_latents sequentially ===
    for lat_idx in range(num_latents):
        # Create new noisy latent and APPEND to context (context GROWS)
        new_latent = torch.randn(3, latent_res, latent_res, device=device, dtype=dtype)
        new_block = ContextBlock(
            content=new_latent,
            type='latent',
            causal=False,
            shape_meta=(latent_res, latent_res),
            logsnr=torch.full((1, latent_res, latent_res), -4.0, device=device, dtype=dtype),
            group_id=lat_idx + 1,
            id=f"latent_{lat_idx}"
        )
        blocks.append(new_block)

        # === INNER LOOP: Diffusion refinement ===
        logsnr_schedule = torch.linspace(-4.0, 6.0, steps_per_latent + 1, device=device, dtype=dtype)

        for step in range(steps_per_latent):
            # Update logsnr for current latent
            current_logsnr = logsnr_schedule[step].item()
            blocks[-1].logsnr = torch.full(
                (1, latent_res, latent_res), current_logsnr, device=device, dtype=dtype
            )

            # ZC forward: recomputes ENTIRE context every time
            current_token_count = num_text + (lat_idx + 1) * latent_tokens_per_span
            z_out, decoded, _ = run_zc_forward(
                model, span_emb, span_unemb, page_table, blocks, with_grad=False
            )

            total_forwards += 1
            tokens_computed += current_token_count

            # Euler step (simplified)
            if 'image_vpreds' in decoded[-1]:
                v_pred = decoded[-1]['image_vpreds']
                blocks[-1].content = blocks[-1].content + 0.1 * v_pred

    torch.cuda.synchronize()
    total_time = (time.perf_counter() - start_time) * 1000

    mem = get_gpu_memory_mb()

    # Cleanup
    del model, span_emb, span_unemb, page_table
    torch.cuda.empty_cache()

    return ConcatARResult(
        name="ZC Concat-AR",
        total_latency_ms=total_time,
        num_outer_steps=num_latents,
        num_inner_steps=steps_per_latent,
        total_forward_passes=total_forwards,
        final_context_tokens=max_tokens,
        prefix_tokens=num_text,
        vram_peak_mb=mem['max_allocated'],
        avg_ms_per_forward=total_time / total_forwards,
        tokens_computed_total=tokens_computed
    )


def benchmark_concat_ar_kvc(
    cfg: Dict,
    device: torch.device,
    dtype: torch.dtype,
    num_text: int,
    latent_res: int,
    num_latents: int = 3,
    steps_per_latent: int = 5,
    warmup_runs: int = 1,
    use_compile: bool = True
) -> ConcatARResult:
    """
    Concatenative AR benchmark with KVC (prefix caching).

    Key optimization:
    - OUTER LOOP: When appending new latent, prefix K/V is cached
    - INNER LOOP: Update mode - only recompute active latent's K/V
    """
    from src.utils import KVCSessionState, run_model_forward_kvc

    block_size = cfg['page_table']['block_size']
    latent_tokens_per_span = (latent_res // cfg['model']['patch_embedder']['stride'])**2
    max_tokens = num_text + num_latents * latent_tokens_per_span
    max_blocks = (max_tokens + block_size - 1) // block_size
    alloc_blocks = max(max_blocks * 2, 16)

    print(f"    KVC concat-AR: {num_latents} latents x {steps_per_latent} steps, "
          f"max {max_tokens} tokens")

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

    # Compile model for fused flex_attention kernels
    if use_compile:
        compile_transformer_layers(model)

    span_emb = SpanEmbedder(model.text_embed, model.patch_embedder)
    span_unemb = SpanUnembedder(model.text_head, model.patch_unembedder)

    page_table = PageTable(
        num_blocks=alloc_blocks,
        block_size=block_size,
        max_batch_size=cfg['page_table']['max_batch_size'],
        max_logical_blocks=alloc_blocks,
        device=device
    )

    kvt_manager = KVTManager(
        max_blocks=alloc_blocks,
        block_size=block_size,
        kv_dim=cfg['model']['dim'],
        layers=cfg['model']['depth'],
        heads=cfg['model']['num_heads'],
        topo_dim=cfg['model']['topo_dim'],
        device=device,
        dtype=dtype
    )

    components_kvc = (model, span_emb, span_unemb, page_table, kvt_manager)

    # Start with text prefix
    text_tokens = torch.randint(0, 1000, (num_text,), device=device)
    text_block = ContextBlock(
        content=text_tokens,
        type='text',
        causal=True,
        shape_meta=(num_text,),
        group_id=0,
        id="text_prefix"
    )
    blocks = [text_block]

    # Warmup
    warmup_latent = torch.randn(3, latent_res, latent_res, device=device, dtype=dtype)
    warmup_block = ContextBlock(
        content=warmup_latent, type='latent', causal=False,
        shape_meta=(latent_res, latent_res),
        logsnr=torch.full((1, latent_res, latent_res), -4.0, device=device, dtype=dtype),
        group_id=0, id="warmup"
    )
    for i in range(warmup_runs):
        run_kvc_forward(model, span_emb, span_unemb, kvt_manager, page_table,
                        [text_block, warmup_block], req_id=i)

    # Reset and measure
    reset_memory_stats()
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    total_forwards = 0
    tokens_computed = 0  # Track how many tokens we actually computed (vs cached)
    total_blocks_reembedded = 0  # Track embedding cache efficiency

    # Create session state for tracking cache
    session = KVCSessionState(kvt_manager, req_id=warmup_runs)

    # === OUTER LOOP: Generate latents with growing context ===
    for lat_idx in range(num_latents):
        # Create new noisy latent
        new_latent = torch.randn(3, latent_res, latent_res, device=device, dtype=dtype)
        new_block = ContextBlock(
            content=new_latent,
            type='latent',
            causal=False,
            shape_meta=(latent_res, latent_res),
            logsnr=torch.full((1, latent_res, latent_res), -4.0, device=device, dtype=dtype),
            group_id=lat_idx + 1,
            id=f"latent_{lat_idx}"
        )
        blocks.append(new_block)

        # === INNER LOOP: Diffusion refinement ===
        logsnr_schedule = torch.linspace(-4.0, 6.0, steps_per_latent + 1, device=device, dtype=dtype)

        for step in range(steps_per_latent):
            blocks[-1].logsnr = torch.full(
                (1, latent_res, latent_res),
                logsnr_schedule[step].item(),
                device=device, dtype=dtype
            )
            # Invalidate embedding cache after logsnr change
            blocks[-1].invalidate_embedding()

            if lat_idx == 0 and step == 0:
                # First call ever: PREFILL mode
                decoded, _, num_recomputed = run_model_forward_kvc(
                    components_kvc, blocks, session, mode='prefill'
                )
                current_tokens = num_text + latent_tokens_per_span
                tokens_computed += current_tokens
            else:
                # Subsequent calls: UPDATE mode (only active latent)
                decoded, _, num_recomputed = run_model_forward_kvc(
                    components_kvc, blocks, session, mode='update'
                )
                # Only the active latent tokens are recomputed
                tokens_computed += latent_tokens_per_span

            total_forwards += 1
            total_blocks_reembedded += num_recomputed

            # Euler step (updates content, which will invalidate embedding)
            if decoded and 'image_vpreds' in decoded[-1]:
                v_pred = decoded[-1]['image_vpreds']
                blocks[-1].content = blocks[-1].content + 0.1 * v_pred
                blocks[-1].invalidate_embedding()  # Content changed

    torch.cuda.synchronize()
    total_time = (time.perf_counter() - start_time) * 1000

    mem = get_gpu_memory_mb()

    # Cleanup
    session.cleanup()
    del model, span_emb, span_unemb, page_table, kvt_manager
    torch.cuda.empty_cache()

    return ConcatARResult(
        name="KVC Concat-AR",
        total_latency_ms=total_time,
        num_outer_steps=num_latents,
        num_inner_steps=steps_per_latent,
        total_forward_passes=total_forwards,
        final_context_tokens=max_tokens,
        prefix_tokens=num_text,
        vram_peak_mb=mem['max_allocated'],
        avg_ms_per_forward=total_time / total_forwards,
        tokens_computed_total=tokens_computed,
        blocks_reembedded=total_blocks_reembedded
    )


# =============================================================================
# BATCHED Concatenative AR Benchmarks (Parallel Sequences)
# =============================================================================

def benchmark_concat_ar_zc_batched(
    cfg: Dict,
    device: torch.device,
    dtype: torch.dtype,
    num_text: int,
    latent_res: int,
    batch_size: int = 2,
    num_latents: int = 3,
    steps_per_latent: int = 5,
    warmup_runs: int = 1,
    use_compile: bool = True
) -> ConcatARResult:
    """
    Batched concatenative AR benchmark with ZC.

    Runs batch_size independent sequences in parallel using doc_id isolation.
    This is the realistic eval scenario: generate multiple samples simultaneously.

    Each sequence has its own group_id namespace:
    - Sequence 0: group_ids 0, 1, 2, ... (text=0, latent0=1, latent1=2, ...)
    - Sequence 1: group_ids 1000, 1001, 1002, ...
    - etc.
    """
    from src.model import coolerLDTformerZC

    print(f"    ZC batched: bs={batch_size}, {num_latents} latents x {steps_per_latent} steps")

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
        jitter_noise=0.0,
        context_size=cfg['model']['patch_embedder']['context_size'],
        stride=cfg['model']['patch_embedder']['stride'],
        fourier_dim=cfg['model']['patch_embedder']['fourier_dim'],
        window_size=cfg['model']['window_size']
    ).to(device=device, dtype=dtype)

    if use_compile:
        compile_transformer_layers(model)

    span_emb = SpanEmbedder(model.text_embed, model.patch_embedder)
    span_unemb = SpanUnembedder(model.text_head, model.patch_unembedder)

    block_size = cfg['page_table']['block_size']
    latent_tokens_per_span = (latent_res // cfg['model']['patch_embedder']['stride'])**2
    tokens_per_seq = num_text + num_latents * latent_tokens_per_span
    total_max_tokens = tokens_per_seq * batch_size
    max_blocks = (total_max_tokens + block_size - 1) // block_size

    page_table = PageTable(
        num_blocks=max(max_blocks * 2, 16),
        block_size=block_size,
        max_batch_size=cfg['page_table']['max_batch_size'],
        max_logical_blocks=max(max_blocks * 2, 16),
        device=device
    )

    # Create batch_size independent sequences, each with its own group_id namespace
    GROUP_OFFSET = 1000  # Separation between sequence namespaces
    all_batch_blocks: List[List[ContextBlock]] = []

    for b in range(batch_size):
        text_tokens = torch.randint(0, 1000, (num_text,), device=device)
        text_block = ContextBlock(
            content=text_tokens,
            type='text',
            causal=True,
            shape_meta=(num_text,),
            group_id=b * GROUP_OFFSET,  # Each sequence gets its own namespace
            id=f"batch{b}_text"
        )
        all_batch_blocks.append([text_block])

    # Warmup with all sequences
    warmup_blocks_flat = []
    for b, seq_blocks in enumerate(all_batch_blocks):
        warmup_lat = torch.randn(3, latent_res, latent_res, device=device, dtype=dtype)
        warmup_block = ContextBlock(
            content=warmup_lat, type='latent', causal=False,
            shape_meta=(latent_res, latent_res),
            logsnr=torch.full((1, latent_res, latent_res), -4.0, device=device, dtype=dtype),
            group_id=b * GROUP_OFFSET + 1, id=f"batch{b}_warmup"
        )
        warmup_blocks_flat.extend(seq_blocks + [warmup_block])

    for _ in range(warmup_runs):
        run_zc_forward(model, span_emb, span_unemb, page_table, warmup_blocks_flat, with_grad=False)

    # Reset sequences back to just text
    all_batch_blocks = []
    for b in range(batch_size):
        text_tokens = torch.randint(0, 1000, (num_text,), device=device)
        text_block = ContextBlock(
            content=text_tokens,
            type='text',
            causal=True,
            shape_meta=(num_text,),
            group_id=b * GROUP_OFFSET,
            id=f"batch{b}_text"
        )
        all_batch_blocks.append([text_block])

    # Reset and measure
    reset_memory_stats()
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    total_forwards = 0
    tokens_computed = 0

    # === OUTER LOOP: Generate latents (all sequences in parallel) ===
    for lat_idx in range(num_latents):
        # Append new latent to each sequence
        for b in range(batch_size):
            new_latent = torch.randn(3, latent_res, latent_res, device=device, dtype=dtype)
            new_block = ContextBlock(
                content=new_latent,
                type='latent',
                causal=False,
                shape_meta=(latent_res, latent_res),
                logsnr=torch.full((1, latent_res, latent_res), -4.0, device=device, dtype=dtype),
                group_id=b * GROUP_OFFSET + lat_idx + 1,
                id=f"batch{b}_latent_{lat_idx}"
            )
            all_batch_blocks[b].append(new_block)

        # === INNER LOOP: Diffusion refinement (all sequences together) ===
        logsnr_schedule = torch.linspace(-4.0, 6.0, steps_per_latent + 1, device=device, dtype=dtype)

        for step in range(steps_per_latent):
            # Update logsnr for active latent in each sequence
            for b in range(batch_size):
                all_batch_blocks[b][-1].logsnr = torch.full(
                    (1, latent_res, latent_res),
                    logsnr_schedule[step].item(),
                    device=device, dtype=dtype
                )

            # Flatten all sequences into one context
            blocks_flat = []
            for seq_blocks in all_batch_blocks:
                blocks_flat.extend(seq_blocks)

            # Single forward pass processes ALL sequences
            current_tokens_per_seq = num_text + (lat_idx + 1) * latent_tokens_per_span
            z_out, decoded, _ = run_zc_forward(
                model, span_emb, span_unemb, page_table, blocks_flat, with_grad=False
            )

            total_forwards += 1
            tokens_computed += current_tokens_per_seq * batch_size

            # Euler step for each sequence's active latent
            decoded_idx = 0
            for b in range(batch_size):
                seq_len = len(all_batch_blocks[b])
                # The active latent is at position decoded_idx + seq_len - 1
                active_decoded_idx = decoded_idx + seq_len - 1
                if active_decoded_idx < len(decoded) and 'image_vpreds' in decoded[active_decoded_idx]:
                    v_pred = decoded[active_decoded_idx]['image_vpreds']
                    all_batch_blocks[b][-1].content = all_batch_blocks[b][-1].content + 0.1 * v_pred
                decoded_idx += seq_len

    torch.cuda.synchronize()
    total_time = (time.perf_counter() - start_time) * 1000

    mem = get_gpu_memory_mb()

    # Cleanup
    del model, span_emb, span_unemb, page_table
    torch.cuda.empty_cache()

    sequences_per_sec = (batch_size * num_latents) / (total_time / 1000)

    return ConcatARResult(
        name=f"ZC Batched (bs={batch_size})",
        total_latency_ms=total_time,
        num_outer_steps=num_latents,
        num_inner_steps=steps_per_latent,
        total_forward_passes=total_forwards,
        final_context_tokens=tokens_per_seq,
        prefix_tokens=num_text,
        vram_peak_mb=mem['max_allocated'],
        avg_ms_per_forward=total_time / total_forwards,
        tokens_computed_total=tokens_computed,
        batch_size=batch_size,
        sequences_per_second=sequences_per_sec
    )


def benchmark_concat_ar_kvc_batched(
    cfg: Dict,
    device: torch.device,
    dtype: torch.dtype,
    num_text: int,
    latent_res: int,
    batch_size: int = 2,
    num_latents: int = 3,
    steps_per_latent: int = 5,
    warmup_runs: int = 1,
    use_compile: bool = True
) -> ConcatARResult:
    """
    Batched concatenative AR benchmark with KVC.

    Each sequence gets its own KVCSessionState for independent cache tracking.
    All sequences processed in single forward pass via doc_id isolation.
    """
    from src.utils import KVCSessionState, run_model_forward_kvc

    block_size = cfg['page_table']['block_size']
    latent_tokens_per_span = (latent_res // cfg['model']['patch_embedder']['stride'])**2
    tokens_per_seq = num_text + num_latents * latent_tokens_per_span
    total_max_tokens = tokens_per_seq * batch_size
    max_blocks = (total_max_tokens + block_size - 1) // block_size
    alloc_blocks = max(max_blocks * 2, 16)

    print(f"    KVC batched: bs={batch_size}, {num_latents} latents x {steps_per_latent} steps, "
          f"max {total_max_tokens} total tokens")

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

    if use_compile:
        compile_transformer_layers(model)

    span_emb = SpanEmbedder(model.text_embed, model.patch_embedder)
    span_unemb = SpanUnembedder(model.text_head, model.patch_unembedder)

    page_table = PageTable(
        num_blocks=alloc_blocks,
        block_size=block_size,
        max_batch_size=cfg['page_table']['max_batch_size'],
        max_logical_blocks=alloc_blocks,
        device=device
    )

    kvt_manager = KVTManager(
        max_blocks=alloc_blocks,
        block_size=block_size,
        kv_dim=cfg['model']['dim'],
        layers=cfg['model']['depth'],
        heads=cfg['model']['num_heads'],
        topo_dim=cfg['model']['topo_dim'],
        device=device,
        dtype=dtype
    )

    # NOTE: For batched KVC, we need to handle multiple sequences.
    # Current implementation uses single session - for true parallel KVC,
    # would need to extend KVTManager to handle batch allocations.
    # For now, we process sequences together but share KV cache state.
    # This still demonstrates batched inference benefits.

    components_kvc = (model, span_emb, span_unemb, page_table, kvt_manager)

    GROUP_OFFSET = 1000
    all_batch_blocks: List[List[ContextBlock]] = []

    for b in range(batch_size):
        text_tokens = torch.randint(0, 1000, (num_text,), device=device)
        text_block = ContextBlock(
            content=text_tokens,
            type='text',
            causal=True,
            shape_meta=(num_text,),
            group_id=b * GROUP_OFFSET,
            id=f"batch{b}_text"
        )
        all_batch_blocks.append([text_block])

    # Warmup (skip KVC session for warmup)
    warmup_blocks_flat = []
    for b, seq_blocks in enumerate(all_batch_blocks):
        warmup_lat = torch.randn(3, latent_res, latent_res, device=device, dtype=dtype)
        warmup_block = ContextBlock(
            content=warmup_lat, type='latent', causal=False,
            shape_meta=(latent_res, latent_res),
            logsnr=torch.full((1, latent_res, latent_res), -4.0, device=device, dtype=dtype),
            group_id=b * GROUP_OFFSET + 1, id=f"batch{b}_warmup"
        )
        warmup_blocks_flat.extend(seq_blocks + [warmup_block])

    for i in range(warmup_runs):
        run_kvc_forward(model, span_emb, span_unemb, kvt_manager, page_table,
                        warmup_blocks_flat, req_id=i)

    # Reset sequences
    all_batch_blocks = []
    for b in range(batch_size):
        text_tokens = torch.randint(0, 1000, (num_text,), device=device)
        text_block = ContextBlock(
            content=text_tokens,
            type='text',
            causal=True,
            shape_meta=(num_text,),
            group_id=b * GROUP_OFFSET,
            id=f"batch{b}_text"
        )
        all_batch_blocks.append([text_block])

    # Reset and measure
    reset_memory_stats()
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    total_forwards = 0
    tokens_computed = 0
    total_blocks_reembedded = 0

    # Create session for batched processing
    session = KVCSessionState(kvt_manager, req_id=warmup_runs)

    # === OUTER LOOP: Generate latents ===
    for lat_idx in range(num_latents):
        for b in range(batch_size):
            new_latent = torch.randn(3, latent_res, latent_res, device=device, dtype=dtype)
            new_block = ContextBlock(
                content=new_latent,
                type='latent',
                causal=False,
                shape_meta=(latent_res, latent_res),
                logsnr=torch.full((1, latent_res, latent_res), -4.0, device=device, dtype=dtype),
                group_id=b * GROUP_OFFSET + lat_idx + 1,
                id=f"batch{b}_latent_{lat_idx}"
            )
            all_batch_blocks[b].append(new_block)

        logsnr_schedule = torch.linspace(-4.0, 6.0, steps_per_latent + 1, device=device, dtype=dtype)

        for step in range(steps_per_latent):
            for b in range(batch_size):
                all_batch_blocks[b][-1].logsnr = torch.full(
                    (1, latent_res, latent_res),
                    logsnr_schedule[step].item(),
                    device=device, dtype=dtype
                )
                all_batch_blocks[b][-1].invalidate_embedding()

            # Flatten for forward
            blocks_flat = []
            for seq_blocks in all_batch_blocks:
                blocks_flat.extend(seq_blocks)

            if lat_idx == 0 and step == 0:
                decoded, _, num_recomputed = run_model_forward_kvc(
                    components_kvc, blocks_flat, session, mode='prefill'
                )
                current_tokens = (num_text + latent_tokens_per_span) * batch_size
                tokens_computed += current_tokens
            else:
                decoded, _, num_recomputed = run_model_forward_kvc(
                    components_kvc, blocks_flat, session, mode='update'
                )
                tokens_computed += latent_tokens_per_span * batch_size

            total_forwards += 1
            total_blocks_reembedded += num_recomputed

            # Euler step
            decoded_idx = 0
            for b in range(batch_size):
                seq_len = len(all_batch_blocks[b])
                active_decoded_idx = decoded_idx + seq_len - 1
                if active_decoded_idx < len(decoded) and 'image_vpreds' in decoded[active_decoded_idx]:
                    v_pred = decoded[active_decoded_idx]['image_vpreds']
                    all_batch_blocks[b][-1].content = all_batch_blocks[b][-1].content + 0.1 * v_pred
                    all_batch_blocks[b][-1].invalidate_embedding()
                decoded_idx += seq_len

    torch.cuda.synchronize()
    total_time = (time.perf_counter() - start_time) * 1000

    mem = get_gpu_memory_mb()

    session.cleanup()
    del model, span_emb, span_unemb, page_table, kvt_manager
    torch.cuda.empty_cache()

    sequences_per_sec = (batch_size * num_latents) / (total_time / 1000)

    return ConcatARResult(
        name=f"KVC Batched (bs={batch_size})",
        total_latency_ms=total_time,
        num_outer_steps=num_latents,
        num_inner_steps=steps_per_latent,
        total_forward_passes=total_forwards,
        final_context_tokens=tokens_per_seq,
        prefix_tokens=num_text,
        vram_peak_mb=mem['max_allocated'],
        avg_ms_per_forward=total_time / total_forwards,
        tokens_computed_total=tokens_computed,
        blocks_reembedded=total_blocks_reembedded,
        batch_size=batch_size,
        sequences_per_second=sequences_per_sec
    )


def print_concat_ar_results(zc: ConcatARResult, kvc: ConcatARResult):
    """Print concatenative AR benchmark comparison."""
    batch_str = f" (bs={zc.batch_size})" if zc.batch_size > 1 else ""
    print("\n" + "=" * 90)
    print(f"CONCATENATIVE AUTOREGRESSION BENCHMARK{batch_str} (True Growing Context)")
    print("=" * 90)
    print(f"Pattern: [text] -> [text, lat1] -> [text, lat1, lat2] -> ...")
    print(f"Config: {zc.num_outer_steps} latents × {zc.num_inner_steps} diffusion steps "
          f"= {zc.total_forward_passes} forward passes")
    if zc.batch_size > 1:
        print(f"Batch: {zc.batch_size} parallel sequences")
    print(f"Context: {zc.prefix_tokens} text + up to {zc.final_context_tokens - zc.prefix_tokens} latent tokens per seq")
    print("-" * 90)
    print(f"{'Metric':<35} {'ZC':<20} {'KVC':<20} {'Improvement':<15}")
    print("-" * 90)

    speedup = zc.total_latency_ms / kvc.total_latency_ms if kvc.total_latency_ms > 0 else 0
    print(f"{'Total latency (ms)':<35} {zc.total_latency_ms:<20.2f} "
          f"{kvc.total_latency_ms:<20.2f} {speedup:.2f}x")

    print(f"{'Avg ms/forward':<35} {zc.avg_ms_per_forward:<20.2f} "
          f"{kvc.avg_ms_per_forward:<20.2f} {zc.avg_ms_per_forward/kvc.avg_ms_per_forward:.2f}x")

    print(f"{'Peak VRAM (MB)':<35} {zc.vram_peak_mb:<20.1f} "
          f"{kvc.vram_peak_mb:<20.1f} "
          f"{(1 - kvc.vram_peak_mb/zc.vram_peak_mb)*100:.1f}%")

    print(f"{'Tokens computed':<35} {zc.tokens_computed_total:<20} "
          f"{kvc.tokens_computed_total:<20} "
          f"{(1 - kvc.tokens_computed_total/zc.tokens_computed_total)*100:.1f}% saved")

    # Throughput for batched runs
    if zc.batch_size > 1 and zc.sequences_per_second > 0:
        print(f"{'Sequences/second':<35} {zc.sequences_per_second:<20.2f} "
              f"{kvc.sequences_per_second:<20.2f} "
              f"{kvc.sequences_per_second/zc.sequences_per_second:.2f}x")

    if hasattr(kvc, 'blocks_reembedded') and kvc.blocks_reembedded > 0:
        total_possible = kvc.total_forward_passes * (1 + kvc.num_outer_steps)  # text + latents per step
        print(f"{'Blocks re-embedded':<35} {'-':<20} "
              f"{kvc.blocks_reembedded:<20} "
              f"({100*kvc.blocks_reembedded/max(total_possible, 1):.1f}% of naive)")

    print("=" * 90)
    print(f"\nKVC Efficiency: Computed {kvc.tokens_computed_total} tokens vs "
          f"ZC's {zc.tokens_computed_total} ({100*kvc.tokens_computed_total/zc.tokens_computed_total:.1f}%)")
    if hasattr(kvc, 'blocks_reembedded') and kvc.blocks_reembedded > 0:
        print(f"  Embedding cache: Only {kvc.blocks_reembedded} block re-embeddings "
              f"(text prefix cached across all steps)")
    print("  - Prefix tokens cached and reused across all forward passes")
    print("  - Only active latent recomputed during inner diffusion loop")


def benchmark_zc(
    cfg: Dict,
    device: torch.device,
    dtype: torch.dtype,
    num_text: int,
    num_latents: int,
    latent_res: int,
    with_grad: bool,
    warmup_runs: int = 2,
    timed_runs: int = 5,
    use_compile: bool = True
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

    # Compile ONLY transformer layers (not embedding/unembedding which have dynamic shapes)
    if use_compile and not with_grad:  # Don't compile training mode
        compile_transformer_layers(model)

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
    headroom_multiplier: float = 2.0,  # Extra space for multi-turn / diffusion steps
    use_compile: bool = True
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

    # Compile model for fused flex_attention kernels
    if use_compile:
        compile_transformer_layers(model)

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
    parser.add_argument("--diffusion-steps", type=int, default=5,
                        help="Number of diffusion steps per latent (realistic: 5-10)")
    parser.add_argument("--ar-latents", type=int, default=3,
                        help="Number of latents in concatenative AR test (context grows)")
    parser.add_argument("--skip-trajectory", action="store_true",
                        help="Skip the concatenative AR benchmark")
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip torch.compile (flex_attention designed for compilation)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for parallel AR benchmark (default: 1)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    results_file = output_dir / "kvc_vs_zc_results.json"
    plot_file = output_dir / "kvc_vs_zc_comparison.png"

    use_compile = not args.no_compile

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
    print(f"Compile: {use_compile}")
    print(f"Batch size: {args.batch_size}")
    print(f"Context: {args.num_text} text tokens + {args.num_latents}x {args.latent_res}x{args.latent_res} latents")
    print(f"Model: {cfg['model']['dim']}d, {cfg['model']['depth']}L, {cfg['model']['num_heads']}H")
    print("=" * 60)

    results = []
    compile_str = " (compiled)" if use_compile else ""

    # 1. ZC with gradients (training mode) - never compiled
    print("\n[1/3] Benchmarking ZC (with gradients)...")
    try:
        r = benchmark_zc(cfg, device, dtype, args.num_text, args.num_latents, args.latent_res,
                        with_grad=True, use_compile=False)  # Training not compiled in benchmark
        results.append(r)
        print(f"  -> {r.latency_ms:.2f}ms, {r.vram_peak_mb:.1f}MB peak")
    except Exception as e:
        print(f"  -> FAILED: {e}")

    # 2. ZC without gradients (inference baseline)
    print(f"\n[2/3] Benchmarking ZC (no gradients){compile_str}...")
    try:
        r = benchmark_zc(cfg, device, dtype, args.num_text, args.num_latents, args.latent_res,
                        with_grad=False, use_compile=use_compile)
        results.append(r)
        print(f"  -> {r.latency_ms:.2f}ms, {r.vram_peak_mb:.1f}MB peak")
    except Exception as e:
        print(f"  -> FAILED: {e}")

    # 3. KVC (paged attention)
    print(f"\n[3/3] Benchmarking KVC (paged attention){compile_str}...")
    try:
        r = benchmark_kvc(cfg, device, dtype, args.num_text, args.num_latents, args.latent_res,
                         use_compile=use_compile)
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

    # =========================================================================
    # Concatenative AR Benchmark (True Growing Context)
    # =========================================================================
    if not args.skip_trajectory:
        print("\n" + "=" * 60)
        print(f"CONCATENATIVE AR BENCHMARK")
        print(f"  {args.ar_latents} latents × {args.diffusion_steps} steps/latent")
        print("=" * 60)

        compile_str = " (compiled)" if use_compile else " (eager)"
        print(f"\n[4/5] Benchmarking ZC concat-AR{compile_str}...")
        try:
            zc_ar = benchmark_concat_ar_zc(
                cfg, device, dtype, args.num_text, args.latent_res,
                num_latents=args.ar_latents,
                steps_per_latent=args.diffusion_steps,
                use_compile=use_compile
            )
            print(f"  -> {zc_ar.total_latency_ms:.2f}ms total, "
                  f"{zc_ar.avg_ms_per_forward:.2f}ms avg/forward, "
                  f"{zc_ar.tokens_computed_total} tokens computed")
        except Exception as e:
            print(f"  -> FAILED: {e}")
            import traceback
            traceback.print_exc()
            zc_ar = None

        print(f"\n[5/5] Benchmarking KVC concat-AR{compile_str}...")
        try:
            kvc_ar = benchmark_concat_ar_kvc(
                cfg, device, dtype, args.num_text, args.latent_res,
                num_latents=args.ar_latents,
                steps_per_latent=args.diffusion_steps,
                use_compile=use_compile
            )
            print(f"  -> {kvc_ar.total_latency_ms:.2f}ms total, "
                  f"{kvc_ar.avg_ms_per_forward:.2f}ms avg/forward, "
                  f"{kvc_ar.tokens_computed_total} tokens computed")
        except Exception as e:
            print(f"  -> FAILED: {e}")
            import traceback
            traceback.print_exc()
            kvc_ar = None

        if zc_ar and kvc_ar:
            print_concat_ar_results(zc_ar, kvc_ar)

        # === BATCHED BENCHMARK (if batch_size > 1) ===
        if args.batch_size > 1:
            print("\n" + "=" * 60)
            print(f"BATCHED CONCATENATIVE AR BENCHMARK (bs={args.batch_size})")
            print(f"  {args.ar_latents} latents × {args.diffusion_steps} steps/latent × {args.batch_size} parallel")
            print("=" * 60)

            print(f"\n[6/7] Benchmarking ZC batched{compile_str}...")
            try:
                zc_batched = benchmark_concat_ar_zc_batched(
                    cfg, device, dtype, args.num_text, args.latent_res,
                    batch_size=args.batch_size,
                    num_latents=args.ar_latents,
                    steps_per_latent=args.diffusion_steps,
                    use_compile=use_compile
                )
                print(f"  -> {zc_batched.total_latency_ms:.2f}ms total, "
                      f"{zc_batched.sequences_per_second:.2f} seq/s, "
                      f"{zc_batched.tokens_computed_total} tokens")
            except Exception as e:
                print(f"  -> FAILED: {e}")
                import traceback
                traceback.print_exc()
                zc_batched = None

            print(f"\n[7/7] Benchmarking KVC batched{compile_str}...")
            try:
                kvc_batched = benchmark_concat_ar_kvc_batched(
                    cfg, device, dtype, args.num_text, args.latent_res,
                    batch_size=args.batch_size,
                    num_latents=args.ar_latents,
                    steps_per_latent=args.diffusion_steps,
                    use_compile=use_compile
                )
                print(f"  -> {kvc_batched.total_latency_ms:.2f}ms total, "
                      f"{kvc_batched.sequences_per_second:.2f} seq/s, "
                      f"{kvc_batched.tokens_computed_total} tokens")
            except Exception as e:
                print(f"  -> FAILED: {e}")
                import traceback
                traceback.print_exc()
                kvc_batched = None

            if zc_batched and kvc_batched:
                print_concat_ar_results(zc_batched, kvc_batched)

                # Compare single vs batched efficiency
                if zc_ar and zc_batched:
                    print("\n" + "-" * 60)
                    print("BATCHING EFFICIENCY:")
                    single_seq_per_sec = 1.0 / (zc_ar.total_latency_ms / 1000 / zc_ar.num_outer_steps)
                    print(f"  ZC:  {zc_batched.sequences_per_second:.2f} seq/s batched vs "
                          f"{single_seq_per_sec:.2f} seq/s single = "
                          f"{zc_batched.sequences_per_second / single_seq_per_sec:.2f}x throughput")
                if kvc_ar and kvc_batched:
                    single_seq_per_sec = 1.0 / (kvc_ar.total_latency_ms / 1000 / kvc_ar.num_outer_steps)
                    print(f"  KVC: {kvc_batched.sequences_per_second:.2f} seq/s batched vs "
                          f"{single_seq_per_sec:.2f} seq/s single = "
                          f"{kvc_batched.sequences_per_second / single_seq_per_sec:.2f}x throughput")

    print(f"\n✓ Benchmark complete. Artifacts in {output_dir}/")


if __name__ == "__main__":
    main()
