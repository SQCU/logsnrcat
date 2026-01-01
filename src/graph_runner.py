"""
CUDA Graph capture for transformer forward pass.

Graph shape is determined by memory block layout (max_blocks × block_size),
NOT by "sequence length" or "batch size". With paged attention:
- Tensors are allocated at max capacity upfront
- "Variable length" means different valid regions, not different shapes
- Mask CONTENTS change, mask SHAPES are fixed

One graph per block layout configuration. Currently we use one PageTable
config, so there's ONE graph. Bucketing would only matter if we had
multiple block size/count configurations for bin-packing different
context size classes.

Usage:
    runner = GraphRunner(model, page_table, config)

    # Warmup with REAL data (required - 3+ times)
    for batch in warmup_batches:
        z, topo, masks = prepare_data(batch)
        runner.warmup(z, topo, masks)

    runner.capture()

    # Training/inference loop:
    z, topo, masks = prepare_data(batch)
    z_out, aux = runner.replay(z, topo, masks)
"""

import torch
from torch.nn.attention.flex_attention import BlockMask
from typing import Tuple, Optional
from dataclasses import dataclass
import copy


@dataclass
class GraphBuffers:
    """Static buffers sized for max context. Updated in-place before replay."""
    # Input buffers - sized for max_ctx
    z: torch.Tensor              # [1, max_ctx, D]
    topo: torch.Tensor           # [1, max_ctx, topo_dim]

    # BlockMask tensor buffers - sized for max_blocks
    mask_local_kv_indices: torch.Tensor
    mask_local_full_kv_indices: torch.Tensor
    mask_local_kv_num_blocks: torch.Tensor
    mask_local_full_kv_num_blocks: torch.Tensor

    mask_global_kv_indices: torch.Tensor
    mask_global_full_kv_indices: torch.Tensor
    mask_global_kv_num_blocks: torch.Tensor
    mask_global_full_kv_num_blocks: torch.Tensor

    # Output buffer
    z_out: torch.Tensor          # [1, max_ctx, D]
    aux_out: torch.Tensor        # [1] scalar


class GraphRunner:
    """
    CUDA Graph runner for transformer forward pass.

    ONE graph per memory block configuration. Shape determined by:
    - page_table.num_blocks × page_table.block_size = max context capacity

    All "variable length" inputs are just different valid regions within
    fixed-size buffers. Masks control what's valid, not tensor shapes.
    """

    def __init__(self, model, page_table, config: dict):
        self.model = model
        self.page_table = page_table
        self.device = config['device']
        self.dtype = config['dtype']

        # Model config
        self.dim = model.text_embed.weight.shape[1]
        self.topo_dim = 3
        self.block_size = page_table.block_size

        # Max context from page table configuration
        self.max_ctx = page_table.num_blocks * page_table.block_size
        self.max_blocks = page_table.num_blocks

        # Graph state
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._buffers: Optional[GraphBuffers] = None
        self._static_masks: Optional[Tuple[BlockMask, BlockMask]] = None
        self._warmup_count = 0
        self._captured = False

        # Capture stream
        self._stream = torch.cuda.Stream(device=self.device)

        # Utilization tracking
        self._utilization_samples: list = []
        self._utilization_warned = False

    def _create_static_buffers(self, mask_local: BlockMask, mask_global: BlockMask) -> GraphBuffers:
        """Allocate static buffers at MAX context size.

        Mask buffers are allocated at max_blocks in the Q-block dimension (first dim),
        preserving the KV-blocks-per-Q dimension from the attention pattern.
        This allows variable sequence lengths while keeping attention pattern fixed.
        """
        # Get KV dimension from incoming mask (determined by attention pattern, not seq len)
        local_kv_dim = mask_local.kv_indices.shape[1] if mask_local.kv_indices.dim() > 1 else 1
        local_full_kv_dim = mask_local.full_kv_indices.shape[1] if mask_local.full_kv_indices.dim() > 1 else 1
        global_kv_dim = mask_global.kv_indices.shape[1] if mask_global.kv_indices.dim() > 1 else 1
        global_full_kv_dim = mask_global.full_kv_indices.shape[1] if mask_global.full_kv_indices.dim() > 1 else 1

        return GraphBuffers(
            # Input buffers at max capacity
            z=torch.zeros(1, self.max_ctx, self.dim, device=self.device, dtype=self.dtype),
            topo=torch.zeros(1, self.max_ctx, self.topo_dim, device=self.device, dtype=self.dtype),

            # Mask tensors at max_blocks in Q dimension, pattern-determined in KV dimension
            mask_local_kv_indices=torch.zeros(
                self.max_blocks, local_kv_dim, device=self.device, dtype=mask_local.kv_indices.dtype),
            mask_local_full_kv_indices=torch.zeros(
                self.max_blocks, local_full_kv_dim, device=self.device, dtype=mask_local.full_kv_indices.dtype),
            mask_local_kv_num_blocks=torch.zeros(
                self.max_blocks, device=self.device, dtype=mask_local.kv_num_blocks.dtype),
            mask_local_full_kv_num_blocks=torch.zeros(
                self.max_blocks, device=self.device, dtype=mask_local.full_kv_num_blocks.dtype),

            mask_global_kv_indices=torch.zeros(
                self.max_blocks, global_kv_dim, device=self.device, dtype=mask_global.kv_indices.dtype),
            mask_global_full_kv_indices=torch.zeros(
                self.max_blocks, global_full_kv_dim, device=self.device, dtype=mask_global.full_kv_indices.dtype),
            mask_global_kv_num_blocks=torch.zeros(
                self.max_blocks, device=self.device, dtype=mask_global.kv_num_blocks.dtype),
            mask_global_full_kv_num_blocks=torch.zeros(
                self.max_blocks, device=self.device, dtype=mask_global.full_kv_num_blocks.dtype),

            # Output buffer at max capacity
            z_out=torch.zeros(1, self.max_ctx, self.dim, device=self.device, dtype=self.dtype),
            aux_out=torch.zeros(1, device=self.device, dtype=self.dtype),
        )

    def _create_static_masks(self, mask_local: BlockMask, mask_global: BlockMask) -> Tuple[BlockMask, BlockMask]:
        """Create BlockMask objects that reference our static buffers.

        Updates both tensor attributes AND shape metadata to reflect max_blocks.
        """
        static_local = copy.copy(mask_local)
        static_local.kv_indices = self._buffers.mask_local_kv_indices
        static_local.full_kv_indices = self._buffers.mask_local_full_kv_indices
        static_local.kv_num_blocks = self._buffers.mask_local_kv_num_blocks
        static_local.full_kv_num_blocks = self._buffers.mask_local_full_kv_num_blocks
        # Update shape metadata if present (flex_attention BlockMask attrs)
        if hasattr(static_local, 'num_rows'):
            static_local.num_rows = self.max_blocks
        if hasattr(static_local, 'num_cols'):
            static_local.num_cols = self.max_blocks

        static_global = copy.copy(mask_global)
        static_global.kv_indices = self._buffers.mask_global_kv_indices
        static_global.full_kv_indices = self._buffers.mask_global_full_kv_indices
        static_global.kv_num_blocks = self._buffers.mask_global_kv_num_blocks
        static_global.full_kv_num_blocks = self._buffers.mask_global_full_kv_num_blocks
        if hasattr(static_global, 'num_rows'):
            static_global.num_rows = self.max_blocks
        if hasattr(static_global, 'num_cols'):
            static_global.num_cols = self.max_blocks

        return static_local, static_global

    def _copy_inputs_to_buffers(
        self,
        z: torch.Tensor,
        topo: torch.Tensor,
        masks: Tuple[BlockMask, BlockMask]
    ) -> int:
        """
        Copy data into static buffers. Returns valid length.

        Input z may be smaller than max_ctx - we copy into the valid region.
        Mask tensors may have fewer Q-blocks than max_blocks - we copy into
        the valid slice and zero the remainder (kv_num_blocks=0 means no attention).
        """
        mask_local, mask_global = masks
        valid_len = z.shape[1]

        # Copy input tensors into valid region (rest stays zero/stale, masked out)
        self._buffers.z[:, :valid_len].copy_(z)
        self._buffers.topo[:, :valid_len].copy_(topo)

        # Get actual number of Q-blocks from incoming masks
        actual_blocks_local = mask_local.kv_num_blocks.shape[0]
        actual_blocks_global = mask_global.kv_num_blocks.shape[0]

        # Zero out mask buffers first (kv_num_blocks=0 disables attention for OOB blocks)
        self._buffers.mask_local_kv_num_blocks.zero_()
        self._buffers.mask_local_full_kv_num_blocks.zero_()
        self._buffers.mask_global_kv_num_blocks.zero_()
        self._buffers.mask_global_full_kv_num_blocks.zero_()

        # Copy local mask into valid slice
        self._buffers.mask_local_kv_indices[:actual_blocks_local].copy_(mask_local.kv_indices)
        self._buffers.mask_local_full_kv_indices[:actual_blocks_local].copy_(mask_local.full_kv_indices)
        self._buffers.mask_local_kv_num_blocks[:actual_blocks_local].copy_(mask_local.kv_num_blocks)
        self._buffers.mask_local_full_kv_num_blocks[:actual_blocks_local].copy_(mask_local.full_kv_num_blocks)

        # Copy global mask into valid slice
        self._buffers.mask_global_kv_indices[:actual_blocks_global].copy_(mask_global.kv_indices)
        self._buffers.mask_global_full_kv_indices[:actual_blocks_global].copy_(mask_global.full_kv_indices)
        self._buffers.mask_global_kv_num_blocks[:actual_blocks_global].copy_(mask_global.kv_num_blocks)
        self._buffers.mask_global_full_kv_num_blocks[:actual_blocks_global].copy_(mask_global.full_kv_num_blocks)

        return valid_len

    def warmup(
        self,
        z: torch.Tensor,
        topo: torch.Tensor,
        masks: Tuple[BlockMask, BlockMask],
        scale: float = 1.0
    ) -> Tuple[torch.Tensor, float]:
        """
        Warmup run with REAL data. Required before capture (3+ times).

        First call allocates buffers. Subsequent calls warm up CUDA internals.
        Returns output so training can continue during warmup phase.
        """
        mask_local, mask_global = masks

        # First warmup: allocate buffers at max size
        if self._buffers is None:
            self._buffers = self._create_static_buffers(mask_local, mask_global)
            self._static_masks = self._create_static_masks(mask_local, mask_global)
            print(f"[GraphRunner] Allocated buffers: max_ctx={self.max_ctx}, "
                  f"max_blocks={self.max_blocks}, block_size={self.block_size}")

        # Copy data to static buffers
        valid_len = self._copy_inputs_to_buffers(z, topo, masks)

        # Run forward on capture stream
        self._stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self._stream):
            z_out, aux = self.model(
                self._buffers.z,
                self._buffers.topo,
                slot_mapping=None,
                block_masks=self._static_masks,
                scale=scale
            )
            self._buffers.z_out.copy_(z_out)
            if isinstance(aux, torch.Tensor):
                self._buffers.aux_out.copy_(aux.view(-1)[:1])
            else:
                self._buffers.aux_out.fill_(float(aux) if aux else 0.0)

        torch.cuda.current_stream().wait_stream(self._stream)
        self._warmup_count += 1

        # Return only the valid region
        return self._buffers.z_out[:, :valid_len].clone(), self._buffers.aux_out.item()

    def capture(self, scale: float = 1.0):
        """
        Capture model.forward() as a CUDA graph.

        Must call warmup() at least 3 times with real data first.
        Graph captures operations on max-sized static buffers.
        """
        if self._warmup_count < 3:
            raise RuntimeError(f"Need at least 3 warmups, got {self._warmup_count}")

        if self._buffers is None:
            raise RuntimeError("No buffers allocated - call warmup() first")

        print(f"[GraphRunner] Capturing graph: max_ctx={self.max_ctx}, "
              f"dim={self.dim}, dtype={self.dtype}")

        self._graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self._graph, stream=self._stream):
            z_out, aux = self.model(
                self._buffers.z,
                self._buffers.topo,
                slot_mapping=None,
                block_masks=self._static_masks,
                scale=scale
            )
            self._buffers.z_out.copy_(z_out)
            if isinstance(aux, torch.Tensor):
                self._buffers.aux_out.copy_(aux.view(-1)[:1])
            else:
                self._buffers.aux_out.fill_(float(aux) if aux else 0.0)

        self._captured = True
        print(f"[GraphRunner] Graph captured successfully")

    def replay(
        self,
        z: torch.Tensor,
        topo: torch.Tensor,
        masks: Tuple[BlockMask, BlockMask]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Replay captured graph with new data.

        Copies new data into static buffers, replays graph, returns valid region.
        Mask contents control what's valid - tensor shapes are fixed.
        """
        if not self._captured:
            raise RuntimeError("No graph captured - call capture() first")

        # Copy new data into static buffers
        valid_len = self._copy_inputs_to_buffers(z, topo, masks)

        # Replay graph (single CUDA call for entire transformer forward)
        self._graph.replay()

        # Track utilization for config recommendations
        self._check_utilization(valid_len)

        # Return view of valid region - no clone needed, backward completes before next replay
        z_out = self._buffers.z_out[:, :valid_len]
        aux = self._buffers.aux_out
        return z_out, aux

    def forward(
        self,
        z: torch.Tensor,
        topo: torch.Tensor,
        masks: Tuple[BlockMask, BlockMask],
        scale: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unified forward - uses replay if captured, else eager.
        """
        if self._captured:
            return self.replay(z, topo, masks)
        else:
            # Eager fallback
            z_out, aux = self.model(
                z, topo, slot_mapping=None, block_masks=masks, scale=scale
            )
            aux_tensor = aux if isinstance(aux, torch.Tensor) else torch.tensor(
                float(aux) if aux else 0.0, device=self.device, dtype=self.dtype
            )
            return z_out, aux_tensor

    @property
    def is_captured(self) -> bool:
        return self._captured

    @property
    def warmup_count(self) -> int:
        return self._warmup_count

    def _check_utilization(self, valid_len: int):
        """Track utilization and warn if consistently low."""
        if self._utilization_warned:
            return

        utilization = valid_len / self.max_ctx
        self._utilization_samples.append(utilization)

        # Check after 10 samples
        if len(self._utilization_samples) >= 10:
            avg_util = sum(self._utilization_samples) / len(self._utilization_samples)
            max_util = max(self._utilization_samples)

            if avg_util < 0.75:
                self._utilization_warned = True

                # Calculate recommended config
                # Round up to next power of 2 for block count
                avg_tokens = int(max_util * self.max_ctx * 1.25)  # 25% headroom
                recommended_blocks = (avg_tokens + self.block_size - 1) // self.block_size
                # Round to nice number
                nice_blocks = 1
                while nice_blocks < recommended_blocks:
                    nice_blocks *= 2

                print(f"\n[GraphRunner] ⚠️  LOW PAGE TABLE UTILIZATION")
                print(f"  Current config: num_blocks={self.max_blocks}, block_size={self.block_size}")
                print(f"  Max context: {self.max_ctx:,} slots")
                print(f"  Avg batch utilization: {avg_util:.1%} ({int(avg_util * self.max_ctx):,} tokens)")
                print(f"  Peak batch utilization: {max_util:.1%} ({int(max_util * self.max_ctx):,} tokens)")
                print(f"  ")
                print(f"  Recommendation: reduce [page_table] in config:")
                print(f"    num_blocks = {nice_blocks}  # (currently {self.max_blocks})")
                print(f"    block_size = {self.block_size}")
                print(f"  This would reduce graph capture overhead by {self.max_blocks // nice_blocks}x\n")

    def reset(self):
        """Clear captured graph and buffers."""
        self._graph = None
        self._buffers = None
        self._static_masks = None
        self._warmup_count = 0
        self._captured = False
        self._utilization_samples = []
        self._utilization_warned = False
