"""
memory_manager.py

Implements the Content-Agnostic Paged Memory System.
Strictly separates Block Management (Hashes -> IDs) from Data Storage (Tensors).
"""

import torch
import xxhash
import numpy as np
from collections import deque
from typing import List, Dict, Tuple, Optional, Union, Any
import copy
from torch.nn.attention.flex_attention import BlockMask

# =========================================================
# 1. CORE BLOCK ABSTRACTIONS
# =========================================================

class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.block_hash = -1  # Unique signature of this block's specific content + causal history
        
    def link(self, block_hash: int):
        self.block_hash = block_hash

    def reset(self):
        self.ref_count = 0
        self.block_hash = -1

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: List[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: Dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        
    def _allocate_block(self) -> Block:
        if not self.free_block_ids:
            raise RuntimeError("OOM: BlockManager has no free blocks.")
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        block.reset()
        block.ref_count = 1
        return block

    def _deallocate_block(self, block_id: int):
        block = self.blocks[block_id]
        if block.block_hash != -1:
            if self.hash_to_block_id.get(block.block_hash) == block_id:
                del self.hash_to_block_id[block.block_hash]
        block.reset()
        self.free_block_ids.append(block_id)

    @staticmethod
    def compute_block_hash(content_hashes: Union[List[int], np.ndarray], prefix_hash: int = -1) -> int:
        h = xxhash.xxh64()
        if prefix_hash != -1:
            h.update(prefix_hash.to_bytes(8, 'little'))
        if isinstance(content_hashes, list):
            # CHANGED: Use uint64 to support full unsigned 64-bit hash values.
            # np.int64 raises OverflowError for values > 2^63-1.
            content_hashes = np.array(content_hashes, dtype=np.uint64)
        h.update(content_hashes.tobytes())
        return h.intdigest()

    def allocate(self, content_stream: List[int]) -> Tuple[List[int], List[int]]:
        num_items = len(content_stream)
        num_blocks = (num_items + self.block_size - 1) // self.block_size
        block_table = []
        newly_allocated = []
        prefix_hash = -1
        
        for i in range(num_blocks):
            start = i * self.block_size
            end = min(start + self.block_size, num_items)
            chunk = content_stream[start:end]
            is_full = (len(chunk) == self.block_size)
            
            if is_full:
                current_hash = self.compute_block_hash(chunk, prefix_hash)
            else:
                current_hash = -1
                
            block_id = -1
            if current_hash != -1:
                block_id = self.hash_to_block_id.get(current_hash, -1)
                
            if block_id != -1:
                self.blocks[block_id].ref_count += 1
                block_table.append(block_id)
                prefix_hash = current_hash 
            else:
                block = self._allocate_block()
                block_id = block.block_id
                if is_full:
                    block.link(current_hash)
                    self.hash_to_block_id[current_hash] = block_id
                    prefix_hash = current_hash
                else:
                    prefix_hash = -1
                block_table.append(block_id)
                newly_allocated.append(block_id)
                
        return block_table, newly_allocated

    def free(self, block_table: List[int]):
        for bid in block_table:
            block = self.blocks[bid]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(bid)

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
        # kv_indices shape: [B, H, Q_Blocks, K_Blocks_Sparse]
        # (Note: FlexAttention usually broadcasts B if masks are identical, 
        #  but for PagedAttention we assume uniqueness per batch item or handle broadcast).
        
        # We assume logical_mask batch dim matches active_page_table batch dim (B).
        # If logical_mask is shared (B=1) but we have multiple requests, expand it.
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
        # We need to gather the physical block IDs using the logical block IDs.
        # active_page_table: [B, Max_Log]
        # indices: [B, H, Q, K_Sparse]
        
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
        
        def physical_mask_mod(b, h, q_idx, k_phys_idx):
            # 1. Get Logical Request ID
            # b is the kernel batch index (0..B-1)
            logical_req_id = batch_idx[b]
            
            # 2. Get Logical Block ID
            phys_block = k_phys_idx // self.block_size
            offset = k_phys_idx % self.block_size
            
            # self.physical_to_logical: [Max_Reqs, Max_Phys]
            log_block = self.physical_to_logical[logical_req_id, phys_block]
            
            # 3. Reconstruct Logical K Index
            log_k_idx = log_block * self.block_size + offset
            
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
        
        # We retain BLOCK_SIZE and other metadata from logical_mask
        
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
        # 1. Map Logical Indices to Physical Indices (Sparse)
        # logical_mask.kv_indices has shape [1, H, Q_blocks, K_sparse_blocks]
        phys_kv_indices = flat_page_table[logical_mask.kv_indices.long()]
        
        # 2. Map Full Blocks (Dense)
        phys_full_kv_indices = flat_page_table[logical_mask.full_kv_indices.long()]

        # 3. Wrap Mask Mod
        original_mod = logical_mask.mask_mod
        
        def physical_mask_mod(b, h, q_idx, k_phys_idx):
            # Map Physical Heap Index -> Logical Sequence Index
            phys_block = k_phys_idx // self.block_size
            offset = k_phys_idx % self.block_size
            
            # Lookup
            log_block = inverse_page_table[phys_block]
            
            # Reconstruct Logical Index
            log_k_idx = log_block * self.block_size + offset
            
            return original_mod(b, h, q_idx, log_k_idx)

        # 4. Construct New BlockMask
        physical_mask = copy.copy(logical_mask)
        physical_mask.kv_indices = phys_kv_indices.int()
        physical_mask.full_kv_indices = phys_full_kv_indices.int()
        physical_mask.mask_mod = physical_mask_mod
        
        return physical_mask



# =========================================================
# 2. KVT MANAGER (Tensor Container)
# =========================================================

class KVTManager:
    """
    Holds the Physical Memory Tensors and delegates allocation to BlockManager.
    Does NOT know about geometry, spans, or embedding logic.
    """

    def __init__(self, 
                 max_blocks: int, 
                 block_size: int, 
                 kv_dim: int, 
                 layers: int, 
                 heads: int, 
                 topo_dim: int, 
                 device='cuda',
                 dtype=torch.float32):
        
        self.device = device
        self.dtype = dtype
        self.block_size = block_size
        self.block_manager = BlockManager(max_blocks, block_size)
        
        # --- Physical Memory ---
        # Feature Cache (Split K and V to prevent overwrite)
        self.head_dim = kv_dim // heads
        
        # CHANGED: Layout is now [Layers, Heads, Max_Blocks, Block_Size, Head_Dim]
        # This allows zero-copy flattening of (Blocks, Block_Size) -> Capacity
        self.k_cache = torch.zeros(
            (layers, heads, max_blocks, block_size, self.head_dim),
            dtype=self.dtype, device=device
        )
        self.v_cache = torch.zeros(
            (layers, heads, max_blocks, block_size, self.head_dim),
            dtype=self.dtype, device=device
        )
        
        # 2. Topology Cache (Coords)
        # [Max_Blocks, Block_Size, Topo_Dim]
        self.topo_cache = torch.zeros(
            (max_blocks, block_size, topo_dim),
            dtype=self.dtype, device=device
        )

        # Track request metadata
        self.req_tables: Dict[int, List[int]] = {}   # req_id -> block_table
        self.req_lengths: Dict[int, int] = {}        # req_id -> total_length
        self.req_highway_offset: Dict[int, int] = {} # req_id -> last highway val
        
    def allocate_and_write_sequence(
        self,
        req_id: int,
        content_hashes: List[int],
        topo_data: torch.Tensor  # [L_total, Topo_Dim] - GLOBAL COORDINATES
    ):
        """
        topo_data contains ABSOLUTE positions:
        - Highway counts from 0..L_total across the entire sequence
        - Spatial dims are per-span coordinates (only non-zero for images)
        """
        block_table, fresh_blocks = self.block_manager.allocate(content_hashes)
        
        # Write topology WITH GLOBAL COORDINATES
        self._write_topology_to_blocks(block_table, topo_data)
        
        self.req_tables[req_id] = block_table
        self.req_lengths[req_id] = len(content_hashes)
        self.req_highway_offset[req_id] = topo_data[-1, 0].item()  # Last highway value
        
        return block_table, fresh_blocks

    def extend_sequence(
        self,
        req_id: int,
        new_content_hashes: List[int],
        new_topo_data: torch.Tensor  # [L_new, Topo_Dim] - CONTINUES HIGHWAY
    ):
        """
        Extends an existing sequence with new tokens.
        new_topo_data must have highway values that CONTINUE from the last token.
        
        Example:
            If last token had highway=335, first new token must have highway=336
        """
        old_table = self.req_tables[req_id]
        old_length = self.req_lengths[req_id]
        
        # Allocate new blocks
        full_hashes = self.get_cached_hashes(req_id) + new_content_hashes
        new_table, fresh_blocks = self.block_manager.allocate(full_hashes)
        
        # Only write topology for NEW tokens
        # (old tokens already have their topology written)
        new_block_ids = new_table[len(old_table):]
        self._write_topology_to_blocks(new_block_ids, new_topo_data)
        
        # Update tracking
        self.req_tables[req_id] = new_table
        self.req_lengths[req_id] = old_length + len(new_content_hashes)
        self.req_highway_offset[req_id] = new_topo_data[-1, 0].item()

    def _write_topology_to_blocks(self, block_ids: List[int], topo_data: torch.Tensor):
        """Internal: Writes topology data into physical blocks."""
        cursor = 0
        total_len = topo_data.shape[0]
        
        for bid in block_ids:
            write_len = min(self.block_size, total_len - cursor)
            if write_len <= 0:
                break
            
            chunk = topo_data[cursor : cursor + write_len]
            
            # Write to cache
            self.topo_cache[bid, :write_len] = chunk
            
            # Zero out remainder if partial block
            if write_len < self.block_size:
                self.topo_cache[bid, write_len:] = 0
                
            cursor += write_len

    def free_sequence(self, block_table: List[int]):
        """Decrements ref counts and recycles blocks."""
        self.block_manager.free(block_table)
    
    def free_request(self, req_id: int):
        """
        Completely frees a request and cleans up all metadata.
        This is the proper way to release a request after inference.
        """
        if req_id not in self.req_tables:
            # Already freed or never allocated
            return
        
        # Free the blocks
        block_table = self.req_tables[req_id]
        self.block_manager.free(block_table)
        
        # Clean up metadata
        del self.req_tables[req_id]
        del self.req_lengths[req_id]
        del self.req_highway_offset[req_id]
    
    def get_attention_inputs(
        self,
        req_ids: List[int],
        layer_idx: int
    ) -> Dict[str, Any]:
        """
        Returns everything needed for a single attention layer forward pass.
        """
        # 1. Get K/V views
        k_cache, v_cache = self.get_flat_kv_view(layer_idx)
        
        # 2. Build slot mapping for active tokens
        block_tables = [self.req_tables[rid] for rid in req_ids]
        seq_lengths = [self.req_lengths[rid] for rid in req_ids]
        slot_mapping = self.get_slot_mapping(block_tables, seq_lengths)
        
        # 3. Batch index tensor
        batch_idx = torch.arange(len(req_ids), device=self.device)
        
        return {
            'k_cache': k_cache,
            'v_cache': v_cache,
            'topo_heap': self.get_topo_view(),           # For mask_mod K positions
            'topo_active': self.get_active_topo_slices(req_ids),  # For RoPE + mask_mod Q positions
            'slot_mapping': slot_mapping,
            'batch_idx': batch_idx,
            'block_tables': block_tables
        }
    
    def get_topo_view(self) -> torch.Tensor:
        """Returns flattened topology cache for ALL physical slots."""
        # [Max_Blocks, Block_Size, Topo_Dim] -> [Capacity, Topo_Dim]
        return self.topo_cache.flatten(0, 1)
    
    
    def get_active_topo_slices(
        self,
        req_ids: List[int]
    ) -> List[torch.Tensor]:
        """
        Returns topology for ONLY the active tokens in each request.
        """
        result = []
        for rid in req_ids:
            table = self.req_tables[rid]
            length = self.req_lengths[rid]
            
            # Gather topology from blocks
            topo_chunks = []
            cursor = 0
            
            for bid in table:
                chunk_len = min(self.block_size, length - cursor)
                if chunk_len <= 0:
                    break
                
                chunk = self.topo_cache[bid, :chunk_len]  # [chunk_len, Topo_Dim]
                topo_chunks.append(chunk)
                cursor += chunk_len
            
            result.append(torch.cat(topo_chunks, dim=0))  # [L_i, Topo_Dim]
        
        return result
    
    def get_flat_kv_view(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # View: [Heads, Blocks, Block_Size, Dim] -> [Heads, Capacity, Dim] -> [1, Heads, Capacity, Dim]
        # This view is compatible with BlockMask (Capacity) and flex_attention (B=1)
        k_cache = self.k_cache[layer_idx].flatten(1, 2).unsqueeze(0)
        v_cache = self.v_cache[layer_idx].flatten(1, 2).unsqueeze(0)
        return k_cache, v_cache

    def get_flat_page_mapping(self, req_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates 1D Page Tables for Flattened Execution (B=1).
        
        Returns:
            flat_page_table: [Total_Logical_Blocks] -> Physical_ID
            inverse_page_table: [Capacity] -> Logical_ID (or -1)
        """
        # 1. Concatenate all logical block tables
        all_blocks = []
        for rid in req_ids:
            all_blocks.extend(self.req_tables[rid])
            
        flat_page_table = torch.tensor(all_blocks, dtype=torch.long, device=self.device)
        
        # 2. Build Inverse (Heap -> Logical)
        capacity = len(self.block_manager.blocks)
        inverse_page_table = torch.full((capacity,), -1, dtype=torch.long, device=self.device)
        
        # Scatter logical indices (0, 1, 2...) into the physical slots
        logical_indices = torch.arange(len(all_blocks), device=self.device)
        inverse_page_table.index_copy_(0, flat_page_table, logical_indices)
        
        return flat_page_table, inverse_page_table

    def get_batch_mappings(self, req_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates the Page Table and Inverse Table for the Glue Layer.
        """
        batch_size = len(req_ids)
        
        tables = [self.req_tables[rid] for rid in req_ids]
        # Handle empty batches gracefully using default 0
        max_log = max([len(t) for t in tables] + [0])
        
        # FIX: Use len() for list-based block storage
        max_phys = len(self.block_manager.blocks)
        
        # Page Table (Logical -> Physical)
        page_table = torch.full((batch_size, max_log), -1, dtype=torch.int32, device=self.device)
        for i, t in enumerate(tables):
            if len(t) > 0:
                # Direct tensor creation handles the list of ints
                page_table[i, :len(t)] = torch.tensor(t, dtype=torch.int32, device=self.device)
            
        # Inverse Table (Physical -> Logical)
        phys_to_log = torch.full((batch_size, max_phys), -1, dtype=torch.int32, device=self.device)
        
        for i, t in enumerate(tables):
            if len(t) > 0:
                phys_ids = torch.tensor(t, dtype=torch.long, device=self.device)
                log_ids = torch.arange(len(t), dtype=torch.int32, device=self.device)
                # Scatter writes logical indices into physical slots
                phys_to_log[i].scatter_(0, phys_ids, log_ids)
            
        return page_table, phys_to_log
        
    def get_slot_mapping(self, block_tables: List[List[int]], seq_lengths: List[int]) -> torch.Tensor:
        slots = []
        for table, length in zip(block_tables, seq_lengths):
            for token_idx in range(length):
                block_idx_in_table = token_idx // self.block_size
                offset = token_idx % self.block_size
                physical_block_id = table[block_idx_in_table]
                physical_slot = physical_block_id * self.block_size + offset
                slots.append(physical_slot)
        return torch.tensor(slots, dtype=torch.int64, device=self.device)