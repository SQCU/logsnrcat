# src/paging.py - Paged KV cache management for inference
"""
Implements PageTable for translating between logical sequence positions
and physical KV cache locations. Enables efficient batched inference
with shared prefix caching.
"""

import copy
import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask


class PageTable:
    """
    Manages the mapping between Logical Blocks (Sequence) and Physical Blocks (Heap).
    Implements the 'convert_logical_block_mask' primitive for Paged FlexAttention.
    """
    def __init__(self,
                 num_blocks: int,
                 block_size: int,
                 max_batch_size: int,
                 max_logical_blocks: int,
                 device='cuda'):

        self.block_size = block_size
        self.device = device

        # [logical_batch_idx, logical_block_idx] -> physical_page_idx
        self.page_table = torch.full(
            (max_batch_size, max_logical_blocks), -1,
            dtype=torch.int32, device=device
        )

        # [logical_batch_idx, physical_page_idx] -> logical_page_idx
        # Used by the mask_mod to reverse-lookup logical positions
        self.physical_to_logical = torch.full(
            (max_batch_size, num_blocks), -1,
            dtype=torch.int32, device=device
        )

    def convert_logical_block_mask(
        self,
        logical_mask: BlockMask,
        batch_idx: torch.Tensor
    ) -> BlockMask:
        """
        Teleports a BlockMask from Logical Space to Physical Space.

        Args:
            logical_mask: Mask computed on logical sequences (e.g. 0..L).
            batch_idx: [B] Tensor mapping Kernel Batch Index -> Logical Request ID.
                       (Used to look up the specific Page Table row).

        Returns:
            A new BlockMask instance valid for the Paged KV Cache.
        """

        # 1. Identify Active Page Tables
        # Select the rows corresponding to the active requests in this kernel batch
        # shape: [B, Max_Logical_Blocks]
        active_page_table = self.page_table[batch_idx.long()]

        # 2. Extract Logical Indices (Sparse)
        # These are indices into the Logical Block sequence (0, 1, 2...)
        B_kernel = batch_idx.size(0)

        kv_indices = logical_mask.kv_indices
        full_kv_indices = logical_mask.full_kv_indices
        kv_num_blocks = logical_mask.kv_num_blocks
        full_kv_num_blocks = logical_mask.full_kv_num_blocks

        if kv_indices.size(0) == 1 and B_kernel > 1:
            kv_indices = kv_indices.expand(B_kernel, -1, -1, -1)
            full_kv_indices = full_kv_indices.expand(B_kernel, -1, -1, -1)
            kv_num_blocks = kv_num_blocks.expand(B_kernel, -1, -1)
            full_kv_num_blocks = full_kv_num_blocks.expand(B_kernel, -1, -1)

        # 3. Map to Physical Indices
        # Reshape page_table for broadcasting against H, Q dimensions
        # [B, 1, 1, Max_Log]
        pt_view = active_page_table.view(B_kernel, 1, 1, -1)

        # Gather Physical Indices for Partial Blocks
        phys_kv_indices = torch.gather(
            pt_view.expand(-1, kv_indices.size(1), kv_indices.size(2), -1),
            3,
            kv_indices.long()
        )

        # Gather Physical Indices for Full Blocks
        phys_full_kv_indices = torch.gather(
            pt_view.expand(-1, full_kv_indices.size(1), full_kv_indices.size(2), -1),
            3,
            full_kv_indices.long()
        )

        # 4. Wrap the Mask Mod
        # The kernel calls mask_mod(b, h, q, k_phys).
        # We must translate k_phys -> k_log to check the original geometry condition.

        original_mod = logical_mask.mask_mod
        block_size = self.block_size
        physical_to_logical = self.physical_to_logical

        def physical_mask_mod(b, h, q_idx, k_phys_idx):
            # 1. Get Logical Request ID
            # b is the kernel batch index (0..B-1)
            logical_req_id = batch_idx[b]

            # 2. Get Logical Block ID
            phys_block = k_phys_idx // block_size
            offset = k_phys_idx % block_size

            # physical_to_logical: [Max_Reqs, Max_Phys]
            log_block = physical_to_logical[logical_req_id, phys_block]

            # 3. Reconstruct Logical K Index
            log_k_idx = log_block * block_size + offset

            # 4. Delegate to Original Logic
            return original_mod(b, h, q_idx, log_k_idx)

        # 5. Construct New BlockMask
        # We clone the object (shallow copy) and overwrite the tensor attributes.
        physical_mask = copy.copy(logical_mask)

        physical_mask.kv_indices = phys_kv_indices.int()
        physical_mask.full_kv_indices = phys_full_kv_indices.int()
        physical_mask.kv_num_blocks = kv_num_blocks.int()
        physical_mask.full_kv_num_blocks = full_kv_num_blocks.int()
        physical_mask.mask_mod = physical_mask_mod

        return physical_mask

    def convert_flattened_block_mask(
        self,
        logical_mask: BlockMask,
        flat_page_table: torch.Tensor,     # [Total_Logical_Blocks] -> Phys_Block
        inverse_page_table: torch.Tensor   # [Capacity_Blocks] -> Log_Block
    ) -> BlockMask:
        """
        Teleports a BlockMask from Logical Space to Physical Space for B=1 (Flattened) execution.
        """
        block_size = self.block_size

        # 1. Map Logical Indices to Physical Indices (Sparse)
        # logical_mask.kv_indices has shape [1, H, Q_blocks, K_sparse_blocks]
        phys_kv_indices = flat_page_table[logical_mask.kv_indices.long()]

        # 2. Map Full Blocks (Dense)
        phys_full_kv_indices = flat_page_table[logical_mask.full_kv_indices.long()]

        # 3. Wrap Mask Mod
        original_mod = logical_mask.mask_mod

        def physical_mask_mod(b, h, q_idx, k_phys_idx):
            # Map Physical Heap Index -> Logical Sequence Index
            phys_block = k_phys_idx // block_size
            offset = k_phys_idx % block_size

            # Lookup
            log_block = inverse_page_table[phys_block]

            # Reconstruct Logical Index
            log_k_idx = log_block * block_size + offset

            return original_mod(b, h, q_idx, log_k_idx)

        # 4. Construct New BlockMask
        physical_mask = copy.copy(logical_mask)
        physical_mask.kv_indices = phys_kv_indices.int()
        physical_mask.full_kv_indices = phys_full_kv_indices.int()
        physical_mask.mask_mod = physical_mask_mod

        return physical_mask


def update_kv_cache(
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor
):
    """
    Writes new KV tokens into the persistent cache using scatter/indexing.

    Args:
        k_new: [B*L, H, 1, D] - Incoming K tokens
        v_new: [B*L, H, 1, D] - Incoming V tokens
        k_cache: [1, H, Capacity, D] - Physical Heap
        v_cache: [1, H, Capacity, D] - Physical Heap
        slot_mapping: [B*L] - Physical indices for the incoming stream
    """
    # Squeeze time dim: [B*L, H, 1, D] -> [B*L, H, D]
    k_src = k_new.squeeze(2)
    v_src = v_new.squeeze(2)

    # We want to write to k_cache[0, :, slot_mapping, :]
    # k_cache[0] shape: [H, Capacity, D]
    # k_src shape:      [BL, H, D] -> Permute to [H, BL, D] for alignment

    k_src_p = k_src.permute(1, 0, 2).to(k_cache.dtype)
    v_src_p = v_src.permute(1, 0, 2).to(v_cache.dtype)

    # Vectorized advanced indexing:
    # Dim 0 (H): Selects all heads (implicit broadcasting or slice)
    # Dim 1 (Capacity): Selects specific slots via slot_mapping
    # Dim 2 (D): Selects all features

    # Note: k_cache is [1, H, Cap, D], so we index at [0] first.
    k_cache[0, :, slot_mapping, :] = k_src_p
    v_cache[0, :, slot_mapping, :] = v_src_p
