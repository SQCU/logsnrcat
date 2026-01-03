"""
CUDA Graph capture for transformer forward pass.

Graph shape is determined by memory block layout (max_blocks × block_size),
NOT by "sequence length" or "batch size". With paged attention:
- Tensors are allocated at max capacity upfront
- "Variable length" means different valid regions, not different shapes
- Mask SHAPES are fixed at max_ctx, content determines valid regions

One graph per block layout configuration. Currently we use one PageTable
config, so there's ONE graph.

Usage:
    runner = GraphRunner(model, page_table, config, window_size=10.0)

    # Warmup with REAL data (required - 3+ times)
    for batch in warmup_batches:
        z, topo = prepare_data(batch)
        runner.warmup(z, topo)

    runner.capture()

    # Training/inference loop:
    z, topo = prepare_data(batch)
    z_out, aux = runner.replay(z, topo)
"""

import torch
from torch.nn.attention.flex_attention import create_block_mask, BlockMask
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class GraphBuffers:
    """Static buffers sized for max context. Updated in-place before replay."""
    # Input buffers - sized for max_ctx
    z: torch.Tensor              # [1, max_ctx, D]
    topo: torch.Tensor           # [1, max_ctx, topo_dim]

    # Output buffer
    z_out: torch.Tensor          # [1, max_ctx, D]
    aux_out: torch.Tensor        # [1] scalar


class GraphRunner:
    """
    CUDA Graph runner for transformer forward pass.

    ONE graph per memory block configuration. Shape determined by:
    - page_table.num_blocks × page_table.block_size = max context capacity

    Creates STATIC masks at max_ctx during initialization. No mask resizing
    or per-batch mask generation - the pattern is fixed (causal + windowed).
    """

    def __init__(self, model, page_table, config: dict, window_size: float = 10.0):
        self.model = model
        self.page_table = page_table
        self.device = config['device']
        self.dtype = config['dtype']
        self.window_size = window_size

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
        self._warmup_count = 0
        self._captured = False

        # Capture stream
        self._stream = torch.cuda.Stream(device=self.device)

        # Utilization tracking
        self._utilization_samples: list = []
        self._utilization_warned = False

        # Create static masks at max_ctx - ONCE, at initialization
        self._static_masks = self._create_static_masks()
        print(f"[GraphRunner] Created static masks at max_ctx={self.max_ctx}, "
              f"window_size={window_size}")

    def _create_static_masks(self) -> Tuple[BlockMask, BlockMask]:
        """
        Create static BlockMasks at max_ctx with fixed pattern.

        Pattern:
        - Local: causal + spatial window (dist² < window_size²)
        - Global: causal only (full attention within causal constraint)

        These masks are created ONCE and reused for all forward passes.
        The valid region is determined by actual data copied into buffers,
        not by mask shape.
        """
        L = self.max_ctx
        win_sq = self.window_size * self.window_size

        # For static masks, we use simple causal pattern
        # Positions are just indices 0..L-1, no span structure needed

        def mask_mod_local(b, h, q_idx, kv_idx):
            """Causal + spatial window."""
            # Causal: can only attend to past (including self)
            causal_ok = q_idx >= kv_idx

            # Spatial window: assume positions are laid out as 1D for simplicity
            # For 2D spatial, we'd need topology - but for static masks,
            # we approximate with 1D distance
            dist = q_idx - kv_idx
            window_ok = (dist * dist) < win_sq

            return causal_ok & window_ok

        def mask_mod_global(b, h, q_idx, kv_idx):
            """Causal only (full attention within past)."""
            return q_idx >= kv_idx

        # Create masks at max_ctx
        local_mask = create_block_mask(
            mask_mod_local, B=None, H=None, Q_LEN=L, KV_LEN=L
        )
        global_mask = create_block_mask(
            mask_mod_global, B=None, H=None, Q_LEN=L, KV_LEN=L
        )

        return local_mask, global_mask

    def _create_buffers(self) -> GraphBuffers:
        """Allocate static buffers at max context size."""
        return GraphBuffers(
            z=torch.zeros(1, self.max_ctx, self.dim, device=self.device, dtype=self.dtype),
            topo=torch.zeros(1, self.max_ctx, self.topo_dim, device=self.device, dtype=self.dtype),
            z_out=torch.zeros(1, self.max_ctx, self.dim, device=self.device, dtype=self.dtype),
            aux_out=torch.zeros(1, device=self.device, dtype=self.dtype),
        )

    def _copy_inputs_to_buffers(self, z: torch.Tensor, topo: torch.Tensor) -> int:
        """
        Copy data into static buffers. Returns valid length.

        Input z may be smaller than max_ctx - we copy into the valid region.
        Rest of buffer is zeroed (or stale from previous, but masked out).
        """
        valid_len = z.shape[1]

        # Zero buffers first (ensures clean state for smaller inputs)
        self._buffers.z.zero_()
        self._buffers.topo.zero_()

        # Copy input tensors into valid region
        self._buffers.z[:, :valid_len].copy_(z)
        self._buffers.topo[:, :valid_len].copy_(topo)

        return valid_len

    def warmup(
        self,
        z: torch.Tensor,
        topo: torch.Tensor,
        scale: float = 1.0
    ) -> Tuple[torch.Tensor, float]:
        """
        Warmup run with REAL data. Required before capture (3+ times).

        First call allocates buffers. Subsequent calls warm up CUDA internals.
        Returns output so training can continue during warmup phase.
        """
        # First warmup: allocate buffers
        if self._buffers is None:
            self._buffers = self._create_buffers()
            print(f"[GraphRunner] Allocated buffers: max_ctx={self.max_ctx}, "
                  f"max_blocks={self.max_blocks}, block_size={self.block_size}")

        # Copy data to static buffers
        valid_len = self._copy_inputs_to_buffers(z, topo)

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
        topo: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Replay captured graph with new data.

        Copies new data into static buffers, replays graph, returns valid region.
        Mask is fixed at max_ctx - valid data region determined by input size.
        """
        if not self._captured:
            raise RuntimeError("No graph captured - call capture() first")

        # Copy new data into static buffers
        valid_len = self._copy_inputs_to_buffers(z, topo)

        # Replay graph (single CUDA call for entire transformer forward)
        self._graph.replay()

        # Track utilization for config recommendations
        self._check_utilization(valid_len)

        # Return view of valid region
        z_out = self._buffers.z_out[:, :valid_len]
        aux = self._buffers.aux_out
        return z_out, aux

    def forward(
        self,
        z: torch.Tensor,
        topo: torch.Tensor,
        scale: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unified forward - uses replay if captured, else eager with static masks.
        """
        if self._captured:
            return self.replay(z, topo)
        else:
            # Eager fallback - still use static masks
            z_out, aux = self.model(
                z, topo, slot_mapping=None, block_masks=self._static_masks, scale=scale
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
                avg_tokens = int(max_util * self.max_ctx * 1.25)  # 25% headroom
                recommended_blocks = (avg_tokens + self.block_size - 1) // self.block_size
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
        """Clear captured graph and buffers (but keep static masks)."""
        self._graph = None
        self._buffers = None
        self._warmup_count = 0
        self._captured = False
        self._utilization_samples = []
        self._utilization_warned = False
