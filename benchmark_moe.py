"""
Benchmark: Dense MoE vs Sparse MoE (grouped_gemm)

Compares throughput of:
1. Dense MoE: All experts computed for all tokens, then weighted sum
2. Sparse MoE: Only selected experts computed via grouped_gemm permute/gmm/unpermute

Expected result: Sparse MoE should be ~(E/K)x faster for E experts, K active.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from dataclasses import dataclass

try:
    import grouped_gemm.ops
    HAS_GROUPED_GEMM = True
except ImportError:
    HAS_GROUPED_GEMM = False
    print("WARNING: grouped_gemm not available, sparse MoE benchmark will be skipped")


class DenseMoE(nn.Module):
    """
    Dense MoE baseline: computes ALL experts for ALL tokens, then masks.
    This is O(E*N) compute regardless of how many experts are selected.
    """
    def __init__(self, dim, hidden_dim, num_experts=8, num_active=2):
        super().__init__()
        self.num_experts = num_experts
        self.num_active = num_active
        self.hidden_dim = hidden_dim
        self.dim = dim

        self.router = nn.Linear(dim, num_experts)

        # Individual expert MLPs (inefficient but clear)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 2 * hidden_dim, bias=False),
                # SwiGLU applied manually in forward
            )
            for _ in range(num_experts)
        ])
        self.down_projs = nn.ModuleList([
            nn.Linear(hidden_dim, dim, bias=False)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        B, L, D = x.shape
        K = self.num_active

        # Routing
        router_logits = self.router(x)  # [B, L, E]
        scores = torch.sigmoid(router_logits)
        top_k_scores, top_k_indices = torch.topk(scores, K, dim=-1)

        # Normalize
        weights = top_k_scores / (top_k_scores.sum(dim=-1, keepdim=True) + 1e-6)

        # Dense: compute ALL experts (wasteful!)
        expert_outputs = []
        for e in range(self.num_experts):
            h = self.experts[e](x)  # [B, L, 2*H]
            h1, h2 = h.chunk(2, dim=-1)
            h = F.silu(h1) * h2  # SwiGLU
            out = self.down_projs[e](h)  # [B, L, D]
            expert_outputs.append(out)

        expert_outputs = torch.stack(expert_outputs, dim=-2)  # [B, L, E, D]

        # Gather selected experts and weight
        # top_k_indices: [B, L, K]
        indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        selected = torch.gather(expert_outputs, dim=2, index=indices_expanded)  # [B, L, K, D]

        # Weighted sum
        output = (selected * weights.unsqueeze(-1)).sum(dim=2)

        return output


class DenseMoEBatched(nn.Module):
    """
    Dense MoE with batched matmuls - more efficient than ModuleList but still O(E*N).
    Uses einsum for batched expert computation.
    """
    def __init__(self, dim, hidden_dim, num_experts=8, num_active=2):
        super().__init__()
        self.num_experts = num_experts
        self.num_active = num_active
        self.hidden_dim = hidden_dim
        self.dim = dim

        self.router = nn.Linear(dim, num_experts)

        # Stacked weights like sparse version
        self.w1 = nn.Parameter(torch.empty(num_experts, 2 * hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))

        nn.init.xavier_uniform_(self.w1.view(-1, dim))
        nn.init.xavier_uniform_(self.w2.view(-1, hidden_dim))

    def forward(self, x):
        B, L, D = x.shape
        K = self.num_active
        E = self.num_experts

        # Routing
        router_logits = self.router(x)
        scores = torch.sigmoid(router_logits)
        top_k_scores, top_k_indices = torch.topk(scores, K, dim=-1)
        weights = top_k_scores / (top_k_scores.sum(dim=-1, keepdim=True) + 1e-6)

        # Dense computation: x @ all experts
        # x: [B, L, D], w1: [E, 2H, D] -> [B, L, E, 2H]
        h = torch.einsum('bld,ehd->bleh', x, self.w1)

        # SwiGLU
        h1, h2 = h.chunk(2, dim=-1)
        h = F.silu(h1) * h2  # [B, L, E, H]

        # Down projection: [B, L, E, H] @ [E, D, H].T -> [B, L, E, D]
        expert_outputs = torch.einsum('bleh,edh->bled', h, self.w2)

        # Gather and weight
        indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        selected = torch.gather(expert_outputs, dim=2, index=indices_expanded)
        output = (selected * weights.unsqueeze(-1)).sum(dim=2)

        return output


class SparseMoE(nn.Module):
    """
    Sparse MoE via grouped_gemm: O(K*N) compute for K active experts.
    """
    def __init__(self, dim, hidden_dim, num_experts=8, num_active=2):
        super().__init__()
        self.num_experts = num_experts
        self.num_active = num_active
        self.hidden_dim = hidden_dim
        self.dim = dim

        self.router = nn.Linear(dim, num_experts)
        self.w1 = nn.Parameter(torch.empty(num_experts, 2 * hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))

        # Pre-allocated pinned memory buffer for expert counts
        self._expert_counts_cpu = torch.empty(num_experts, dtype=torch.int64, pin_memory=True)

        nn.init.xavier_uniform_(self.w1.view(-1, dim))
        nn.init.xavier_uniform_(self.w2.view(-1, hidden_dim))

    def forward(self, x):
        B, L, D = x.shape
        N = B * L
        K = self.num_active
        E = self.num_experts

        # Routing
        router_logits = self.router(x)
        scores = torch.sigmoid(router_logits)
        top_k_scores, top_k_indices = torch.topk(scores, K, dim=-1)
        # Use float32 for weights - grouped_gemm unpermute expects f32 probs
        weights = (top_k_scores / (top_k_scores.sum(dim=-1, keepdim=True) + 1e-6)).float()

        # Flatten
        x_flat = x.view(N, D).contiguous()
        indices_flat = top_k_indices.view(N, K).to(torch.int32).contiguous()
        weights_flat = weights.view(N, K)  # Already contiguous

        # Permute
        permuted_x, row_id_map = grouped_gemm.ops.permute(x_flat, indices_flat)

        # Expert counts on GPU via bincount, then async copy to pinned CPU buffer
        expert_counts_gpu = torch.bincount(indices_flat.view(-1).long(), minlength=E)
        self._expert_counts_cpu.copy_(expert_counts_gpu, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        # Clone before gmm - it saves batch_sizes for backward, can't reuse buffer
        expert_counts = self._expert_counts_cpu.clone()

        # Grouped GEMM: up projection
        h = grouped_gemm.ops.gmm(permuted_x, self.w1, expert_counts, trans_b=True)

        # SwiGLU
        h1, h2 = h.chunk(2, dim=-1)
        h = F.silu(h1) * h2

        # Grouped GEMM: down projection
        expert_out = grouped_gemm.ops.gmm(h, self.w2, expert_counts, trans_b=True)

        # Unpermute
        out_flat = grouped_gemm.ops.unpermute(expert_out, row_id_map, weights_flat)

        return out_flat.view(B, L, D)


@dataclass
class BenchmarkConfig:
    batch_size: int = 4
    seq_len: int = 1024
    dim: int = 512
    hidden_dim: int = 2048
    num_experts: int = 8
    num_active: int = 2
    warmup_iters: int = 10
    bench_iters: int = 50
    backward: bool = True


def benchmark_model(model, x, config: BenchmarkConfig, name: str):
    """Benchmark forward (and optionally backward) pass."""
    device = x.device

    # Warmup
    for _ in range(config.warmup_iters):
        out = model(x)
        if config.backward:
            loss = out.sum()
            loss.backward()
            model.zero_grad()

    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(config.bench_iters):
        out = model(x)
        if config.backward:
            loss = out.sum()
            loss.backward()
            model.zero_grad()
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / config.bench_iters) * 1000

    tokens_per_iter = config.batch_size * config.seq_len
    throughput = tokens_per_iter / (elapsed / config.bench_iters)

    return {
        'name': name,
        'avg_ms': avg_ms,
        'throughput_tok_per_sec': throughput,
        'total_tokens': tokens_per_iter,
    }


def run_benchmark(config: BenchmarkConfig):
    device = torch.device('cuda')
    dtype = torch.bfloat16

    print(f"\n{'='*60}")
    print(f"MoE Benchmark: {config.num_experts} experts, {config.num_active} active")
    print(f"Input: [{config.batch_size}, {config.seq_len}, {config.dim}]")
    print(f"Hidden: {config.hidden_dim}, Backward: {config.backward}")
    print(f"{'='*60}\n")

    # Create input
    x = torch.randn(config.batch_size, config.seq_len, config.dim,
                    device=device, dtype=dtype, requires_grad=config.backward)

    results = []

    # Dense MoE (batched einsum version)
    print("Benchmarking Dense MoE (batched)...")
    dense = DenseMoEBatched(
        config.dim, config.hidden_dim,
        config.num_experts, config.num_active
    ).to(device, dtype)

    with torch.amp.autocast('cuda', dtype=dtype):
        result = benchmark_model(dense, x, config, "Dense (batched)")
    results.append(result)
    print(f"  {result['avg_ms']:.2f} ms/iter, {result['throughput_tok_per_sec']/1e6:.2f}M tok/s")

    del dense
    torch.cuda.empty_cache()

    # Sparse MoE
    if HAS_GROUPED_GEMM:
        print("Benchmarking Sparse MoE (grouped_gemm)...")
        sparse = SparseMoE(
            config.dim, config.hidden_dim,
            config.num_experts, config.num_active
        ).to(device, dtype)

        with torch.amp.autocast('cuda', dtype=dtype):
            result = benchmark_model(sparse, x, config, "Sparse (grouped_gemm)")
        results.append(result)
        print(f"  {result['avg_ms']:.2f} ms/iter, {result['throughput_tok_per_sec']/1e6:.2f}M tok/s")

        del sparse
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")

    for r in results:
        print(f"{r['name']:25s}: {r['avg_ms']:7.2f} ms, {r['throughput_tok_per_sec']/1e6:6.2f}M tok/s")

    if len(results) >= 2:
        speedup = results[0]['avg_ms'] / results[1]['avg_ms']
        theoretical = config.num_experts / config.num_active
        print(f"\nSpeedup: {speedup:.2f}x (theoretical max: {theoretical:.1f}x)")
        print(f"Efficiency: {(speedup / theoretical) * 100:.1f}% of theoretical")

    return results


def main():
    print("MoE Throughput Benchmark: Dense vs Sparse (grouped_gemm)")
    print("="*60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"grouped_gemm available: {HAS_GROUPED_GEMM}")

    # Standard config
    configs = [
        BenchmarkConfig(batch_size=4, seq_len=1024, dim=512, hidden_dim=2048,
                       num_experts=8, num_active=2),
        BenchmarkConfig(batch_size=8, seq_len=2048, dim=768, hidden_dim=3072,
                       num_experts=8, num_active=2),
        BenchmarkConfig(batch_size=4, seq_len=4096, dim=1024, hidden_dim=4096,
                       num_experts=16, num_active=2),
    ]

    all_results = []
    for cfg in configs:
        results = run_benchmark(cfg)
        all_results.append((cfg, results))

    # Final summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")

    for cfg, results in all_results:
        if len(results) >= 2:
            speedup = results[0]['avg_ms'] / results[1]['avg_ms']
            theoretical = cfg.num_experts / cfg.num_active
            print(f"E={cfg.num_experts}, K={cfg.num_active}, dim={cfg.dim}: "
                  f"{speedup:.2f}x speedup ({(speedup/theoretical)*100:.0f}% of {theoretical:.0f}x theoretical)")


if __name__ == "__main__":
    main()
