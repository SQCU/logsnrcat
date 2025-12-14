"""
kv_cache_allocator.py

Memory-aware KV cache allocation helper.
Prevents catastrophic over-allocation by measuring actual available memory.
"""

import torch
from typing import Tuple, Optional
import math

class KVCacheAllocator:
    """
    Computes safe KV cache sizes based on available GPU memory.
    
    Philosophy:
    1. Measure what's already allocated (model weights, optimizer states)
    2. Reserve memory for batch processing (activations, gradients)
    3. Allocate remaining memory to KV cache
    """
    
    def __init__(
        self,
        device: torch.device,
        kv_cache_memory_fraction: float = 0.90,  # Use 90% of *available* memory
        safety_margin_gb: float = 1.0             # Reserve 1GB for CUDA ops
    ):
        self.device = device
        self.kv_cache_memory_fraction = kv_cache_memory_fraction
        self.safety_margin_gb = safety_margin_gb
        
    def get_memory_stats(self) -> dict:
        """Query current GPU memory state."""
        if self.device.type != 'cuda':
            return {'total': 0, 'allocated': 0, 'reserved': 0, 'free': 0}
            
        torch.cuda.synchronize(self.device)
        
        # Total capacity
        total = torch.cuda.get_device_properties(self.device).total_memory
        
        # What PyTorch has already allocated
        allocated = torch.cuda.memory_allocated(self.device)
        
        # What PyTorch has reserved from CUDA (includes fragmentation)
        reserved = torch.cuda.memory_reserved(self.device)
        
        # True free memory (conservative estimate)
        free = total - reserved
        
        return {
            'total_gb': total / 1e9,
            'allocated_gb': allocated / 1e9,
            'reserved_gb': reserved / 1e9,
            'free_gb': free / 1e9
        }
    
    def estimate_batch_memory(
        self,
        batch_size: int,
        max_seq_len: int,
        embed_dim: int,
        num_layers: int,
        dtype: torch.dtype = torch.float32
    ) -> float:
        """
        Estimate peak memory for batch processing (activations + gradients).
        
        Conservative estimate:
        - Forward activations: B × L × D per layer
        - Backward gradients: ~2× forward (gradients + intermediate states)
        - Attention workspace: B × H × L × L (for flex_attention materialization)
        """
        bytes_per_element = 4 if dtype == torch.float32 else 2
        
        # Residual stream per layer
        activations_per_layer = batch_size * max_seq_len * embed_dim
        total_activations = activations_per_layer * num_layers
        
        # Gradients (conservative 2× multiplier)
        total_gradients = total_activations * 2
        
        # Attention workspace (assumes 8 heads, sparse mask reduces this but be safe)
        num_heads = 8
        attn_workspace = batch_size * num_heads * max_seq_len * max_seq_len
        
        total_elements = total_activations + total_gradients + attn_workspace
        total_bytes = total_elements * bytes_per_element
        
        return total_bytes / 1e9  # Convert to GB
            
    def compute_safe_cache_blocks(
        self,
        block_size: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        dtype: torch.dtype = torch.float32,
        expected_batch_size: int = 128,
        expected_seq_len: int = 64,
        heap_utilization_factor: float = 1.5  # NEW: Only allocate 1.5× batch needs
    ) -> Tuple[int, dict]:
        """
        Computes maximum safe number of cache blocks.
        
        NEW STRATEGY: Limit heap size to avoid mask materialization explosion.
        FlexAttention materializes Q_LEN × KV_LEN during create_block_mask,
        so we can't have a huge heap even if GPU memory allows it.
        """
        bytes_per_element = 4 if dtype == torch.float32 else 2
        head_dim = embed_dim // num_heads
        
        # 1. Current memory state
        stats = self.get_memory_stats()
        
        # 2. Estimate batch processing overhead
        batch_memory_gb = self.estimate_batch_memory(
            expected_batch_size,
            expected_seq_len,
            embed_dim,
            num_layers,
            dtype
        )
        
        # 3. Compute minimum blocks needed for batch
        # FIX: Calculate blocks per request accounting for fragmentation
        blocks_per_req = math.ceil(expected_seq_len / block_size)
        min_blocks_needed = expected_batch_size * blocks_per_req
        
        # 4. CRITICAL: Limit heap to avoid mask materialization explosion
        # FlexAttention will materialize [Q_LEN, KV_LEN] during create_block_mask
        # With Q_LEN ≈ batch_size × seq_len, we need KV_LEN to be reasonable
        # 
        # Safe rule: KV_LEN should be at most 10× Q_LEN to keep mask < 1GB
        max_safe_heap_tokens = expected_batch_size * expected_seq_len * 10
        max_blocks_from_mask = max_safe_heap_tokens // block_size
        
        # 5. Compute blocks from memory budget (original logic)
        available_for_cache = stats['free_gb'] - self.safety_margin_gb - batch_memory_gb
        available_for_cache = max(0, available_for_cache)
        cache_budget_gb = available_for_cache * self.kv_cache_memory_fraction
        cache_budget_bytes = cache_budget_gb * 1e9
        
        elements_per_block = num_layers * num_heads * block_size * head_dim * 2
        bytes_per_block = elements_per_block * bytes_per_element
        max_blocks_from_memory = int(cache_budget_bytes / bytes_per_block)
        
        # 6. Take minimum of memory limit and mask safety limit
        max_blocks = min(max_blocks_from_memory, max_blocks_from_mask)
        
        # 7. Apply utilization factor (allocate some headroom but not too much)
        max_blocks = max(min_blocks_needed, int(min_blocks_needed * heap_utilization_factor))
        max_blocks = min(max_blocks, max_blocks_from_mask)  # Still respect mask limit
        
        if max_blocks < min_blocks_needed:
            print(f"⚠️  WARNING: Only {max_blocks} blocks available, but need {min_blocks_needed} for batch!")
            print(f"    Consider reducing batch size or sequence length.")
        
        # 8. Build report
        report = {
            'memory_stats': stats,
            'batch_overhead_gb': batch_memory_gb,
            'safety_margin_gb': self.safety_margin_gb,
            'available_for_cache_gb': available_for_cache,
            'cache_budget_gb': cache_budget_gb,
            'bytes_per_block': bytes_per_block,
            'max_blocks': max_blocks,
            'min_blocks_needed': min_blocks_needed,
            'capacity_tokens': max_blocks * block_size,
            'utilization': f"{(cache_budget_gb / stats['total_gb'] * 100):.1f}% of total GPU"
        }
        
        return max_blocks, report
    
    def compute_page_table_sizes(
        self,
        max_blocks: int,
        block_size: int,
        expected_batch_size: int,
        expected_seq_len: int,
        concurrent_requests_multiplier: float = 2.0
    ) -> Tuple[int, int, dict]:
        """
        Computes safe PageTable dimensions.
        
        Args:
            max_blocks: Number of physical blocks (from compute_safe_cache_blocks)
            block_size: Tokens per block
            expected_batch_size: Typical batch size
            expected_seq_len: Typical sequence length
            concurrent_requests_multiplier: Headroom for batching dynamics
        
        Returns:
            max_batch_size: Safe max_batch_size for PageTable
            max_logical_blocks: Safe max_logical_blocks per request
            report: Sizing details
        """
        # Max batch size: Allow for some headroom beyond expected
        max_batch_size = int(expected_batch_size * concurrent_requests_multiplier)
        
        # Max logical blocks per request: Longest possible sequence
        blocks_per_request = math.ceil(expected_seq_len / block_size)
        
        # Add headroom for variable-length sequences (e.g., images of different sizes)
        max_logical_blocks = int(blocks_per_request * 1.5)
        
        # PageTable memory overhead (int32 tensors)
        # page_table: [max_batch_size, max_logical_blocks]
        # physical_to_logical: [max_batch_size, num_blocks]
        page_table_elements = max_batch_size * max_logical_blocks
        inverse_table_elements = max_batch_size * max_blocks
        total_elements = page_table_elements + inverse_table_elements
        
        bytes_per_int32 = 4
        page_table_memory_gb = (total_elements * bytes_per_int32) / 1e9
        
        report = {
            'max_batch_size': max_batch_size,
            'max_logical_blocks': max_logical_blocks,
            'blocks_per_request': blocks_per_request,
            'page_table_memory_gb': page_table_memory_gb,
            'page_table_elements': page_table_elements,
            'inverse_table_elements': inverse_table_elements
        }
        
        return max_batch_size, max_logical_blocks, report
    
    def print_allocation_report(self, report: dict, page_table_report: Optional[dict] = None):
        """Pretty-print the allocation analysis."""
        print("\n" + "="*60)
        print("📊 KV Cache Allocation Report")
        print("="*60)
        
        mem = report['memory_stats']
        print(f"GPU Memory:")
        print(f"  Total:        {mem['total_gb']:.2f} GB")
        print(f"  Allocated:    {mem['allocated_gb']:.2f} GB (model + optimizer)")
        print(f"  Reserved:     {mem['reserved_gb']:.2f} GB (PyTorch heap)")
        print(f"  Free:         {mem['free_gb']:.2f} GB")
        
        print(f"\nMemory Reservations:")
        print(f"  Batch overhead:  {report['batch_overhead_gb']:.2f} GB")
        print(f"  Safety margin:   {report['safety_margin_gb']:.2f} GB")
        print(f"  Available:       {report['available_for_cache_gb']:.2f} GB")
        
        print(f"\nKV Cache Allocation:")
        print(f"  Budget:          {report['cache_budget_gb']:.2f} GB ({report['utilization']})")
        print(f"  Bytes per block: {report['bytes_per_block']:,}")
        print(f"  Max blocks:      {report['max_blocks']:,}")
        print(f"  Capacity:        {report['capacity_tokens']:,} tokens")
        
        print(f"\nBatch Requirements:")
        print(f"  Min blocks needed: {report['min_blocks_needed']}")
        
        if report['max_blocks'] >= report['min_blocks_needed']:
            headroom = report['max_blocks'] / report['min_blocks_needed']
            print(f"  ✅ Sufficient capacity ({headroom:.1f}× minimum)")
        else:
            print(f"  ❌ INSUFFICIENT CAPACITY!")
        
        if page_table_report:
            print(f"\nPage Table Sizing:")
            print(f"  Max batch size:      {page_table_report['max_batch_size']}")
            print(f"  Max logical blocks:  {page_table_report['max_logical_blocks']}")
            print(f"  Blocks per request:  {page_table_report['blocks_per_request']}")
            print(f"  Page table memory:   {page_table_report['page_table_memory_gb']:.3f} GB")
            print(f"    - Forward map:     {page_table_report['page_table_elements']:,} int32")
            print(f"    - Inverse map:     {page_table_report['inverse_table_elements']:,} int32")
        
        print("="*60 + "\n")


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def allocate_kv_cache_safely(
    device: torch.device,
    block_size: int,
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    dtype: torch.dtype = torch.float32,
    expected_batch_size: int = 128,
    expected_seq_len: int = 64,
    kv_cache_memory_fraction: float = 0.90,
    safety_margin_gb: float = 1.0,
    concurrent_requests_multiplier: float = 2.0,
    verbose: bool = True
) -> Tuple[int, int, int]:
    """
    One-shot function to compute safe sizes for KVTManager and PageTable.
    
    Usage:
        max_blocks, max_batch, max_logical = allocate_kv_cache_safely(
            device='cuda',
            block_size=128,
            embed_dim=256,
            num_layers=4,
            num_heads=8,
            expected_batch_size=128,
            expected_seq_len=64  # 8×8 grid after pooling
        )
        
        kvt_manager = KVTManager(
            max_blocks=max_blocks,
            block_size=block_size,
            kv_dim=embed_dim,
            layers=num_layers,
            heads=num_heads,
            topo_dim=3,
            device=device
        )
        
        page_table = PageTable(
            num_blocks=max_blocks,
            block_size=block_size,
            max_batch_size=max_batch,
            max_logical_blocks=max_logical,
            device=device
        )
    
    Returns:
        max_blocks: Number of physical blocks for cache
        max_batch_size: Safe batch size for PageTable
        max_logical_blocks: Safe logical blocks per request
    """
    allocator = KVCacheAllocator(
        device=device,
        kv_cache_memory_fraction=kv_cache_memory_fraction,
        safety_margin_gb=safety_margin_gb
    )
    
    # Compute cache blocks
    max_blocks, cache_report = allocator.compute_safe_cache_blocks(
        block_size=block_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dtype=dtype,
        expected_batch_size=expected_batch_size,
        expected_seq_len=expected_seq_len
    )
    
    # Compute page table sizes
    max_batch, max_logical, pt_report = allocator.compute_page_table_sizes(
        max_blocks=max_blocks,
        block_size=block_size,
        expected_batch_size=expected_batch_size,
        expected_seq_len=expected_seq_len,
        concurrent_requests_multiplier=concurrent_requests_multiplier
    )
    
    if verbose:
        allocator.print_allocation_report(cache_report, pt_report)
    
    return max_blocks, max_batch, max_logical