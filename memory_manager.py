"""
memory_manager.py

Implements the Content-Agnostic Paged Memory System.
Strictly separates Block Management (Hashes -> IDs) from Data Storage (Tensors).
"""

import torch
import xxhash
import numpy as np
from collections import deque
from typing import List, Dict, Tuple, Optional, Union
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
    """
    Manages the lifecycle of Physical Block IDs based on Content Hashes.
    Implements Prefix Caching via Causal Hashing.
    """
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: List[Block] = [Block(i) for i in range(num_blocks)]
        
        # Maps a (PrefixHash + ContentHash) -> PhysicalBlockID
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
        """
        Computes a unique hash for a block based on its content and its predecessor.
        This enables 'Prefix Caching'.
        """
        h = xxhash.xxh64()
        if prefix_hash != -1:
            h.update(prefix_hash.to_bytes(8, 'little'))
        
        # Assume content_hashes is a list/array of 64-bit integers
        if isinstance(content_hashes, list):
            content_hashes = np.array(content_hashes, dtype=np.int64)
        h.update(content_hashes.tobytes())
        return h.intdigest()

    def allocate(self, content_stream: List[int]) -> Tuple[List[int], List[int]]:
        """
        Processes a stream of atomic content hashes.
        Returns:
            block_table: List of Physical Block IDs covering the stream.
            newly_allocated: List of Block IDs that are fresh (need data writing).
        """
        num_items = len(content_stream)
        num_blocks = (num_items + self.block_size - 1) // self.block_size
        
        block_table = []
        newly_allocated = []
        prefix_hash = -1
        
        for i in range(num_blocks):
            start = i * self.block_size
            end = min(start + self.block_size, num_items)
            chunk = content_stream[start:end]
            
            # Only cache full blocks
            is_full = (len(chunk) == self.block_size)
            
            # Compute Candidate Hash
            if is_full:
                current_hash = self.compute_block_hash(chunk, prefix_hash)
            else:
                current_hash = -1
                
            # Check Cache
            block_id = -1
            if current_hash != -1:
                block_id = self.hash_to_block_id.get(current_hash, -1)
                
            if block_id != -1:
                # HIT
                self.blocks[block_id].ref_count += 1
                block_table.append(block_id)
                prefix_hash = current_hash # Continue chain
            else:
                # MISS
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
                 device='cuda'):
        
        self.device = device
        self.block_size = block_size
        self.block_manager = BlockManager(max_blocks, block_size)
        
        # --- Physical Memory ---
        # 1. Feature Cache (KV)
        # [Layers, Max_Blocks, Heads, Block_Size, Head_Dim]
        self.head_dim = kv_dim // heads
        self.kv_cache = torch.zeros(
            (layers, max_blocks, heads, block_size, self.head_dim),
            dtype=torch.bfloat16, device=device
        )
        
        # 2. Topology Cache (Coords)
        # [Max_Blocks, Block_Size, Topo_Dim]
        self.topo_cache = torch.zeros(
            (max_blocks, block_size, topo_dim),
            dtype=torch.float32, device=device
        )

        # Track request metadata
        self.req_tables: Dict[int, List[int]] = {}  # req_id -> block_table
        self.req_lengths: Dict[int, int] = {}        # req_id -> total_length
        
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
    
    def get_attention_inputs(
        self,
        req_ids: List[int],
        layer_idx: int
    ) -> Dict[str, Any]:
        """
        Returns everything needed for a single attention layer forward pass.
        
        Returns dict with:
            'k_cache': [1, H, Capacity, D]
            'v_cache': [1, H, Capacity, D]
            'topo_heap': [Capacity, Topo_Dim] - FULL heap topology
            'slot_mapping': [sum(L_i)] - where to write new tokens
            'batch_idx': [B] - request IDs for page table lookup
            'block_tables': List[List[int]] - for building page table
        """
        # 1. Get K/V views
        k_cache, v_cache = self.get_flat_kv_view(layer_idx)
        
        # 2. Get FULL topology heap (not just active tokens)
        topo_heap = self.get_topo_view()  # [Capacity, Topo_Dim]
        
        # 3. Build slot mapping for active tokens
        block_tables = [self.req_tables[rid] for rid in req_ids]
        seq_lengths = [self.req_lengths[rid] for rid in req_ids]
        slot_mapping = self.get_slot_mapping(block_tables, seq_lengths)
        
        # 4. Batch index tensor
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
        Used when you need per-request topology (e.g., for RoPE on Q/K).
        
        Returns:
            List of [L_i, Topo_Dim] tensors
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
    
    # In KVTManager class...
    def get_flat_kv_view(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a flattened [1, 1, Total_Capacity, Dim] view of K and V caches
        for a specific layer, satisfying FlexAttention's input requirements.
        """
        # self.kv_cache: [Layers, Max_Blocks, Heads, Block_Size, Head_Dim]
        # We need to flatten Blocks, Heads, Block_Size into one dimension?
        # WAIT: Paged FlexAttention expects [1, 1, Total_Tokens, Dim].
        # It treats the heap as a massive sequence.
        # But our heap has 'Heads' dimension.
        # Standard FlexAttention expects:
        #   query: [B, H, 1, D]
        #   key:   [B, H, L, D] (Here B=1, H=1, L=Huge)
        #
        # If we have multi-head KV cache, we must be careful.
        # If Heads > 1, the physical heap is [Blocks, Heads, Block_Size, Head_Dim].
        # Flattening to [Total_Tokens, D] mixes Heads and Tokens if not careful.
        #
        # Correct View for Multi-Head Paged Attention:
        # We treat Heads as part of the Batch or keep them separate?
        # FlexAttention usually broadcasts over Heads.
        #
        # If we flatten to [1, 1, Blocks*Heads*Block_Size, Head_Dim], 
        # the indices in `page_table` must account for the Head stride? 
        # OR, we keep Heads dim: [1, Heads, Blocks*Block_Size, Head_Dim]?
        # 
        # Let's assume standard layout: [Blocks, Heads, Block_Size, Head_Dim]
        # We permute to [Heads, Blocks, Block_Size, Head_Dim] -> [Heads, Total_Tokens, D]
        # Then FlexAttn inputs:
        #   k: [1, Heads, Total_Tokens, D]
        
        k_cache = self.kv_cache[layer_idx] # [Max_Blocks, Heads, Block_Size, Head_Dim]
        
        # Permute to bring Heads out
        k_cache = k_cache.permute(1, 0, 2, 3) # [Heads, Max_Blocks, Block_Size, Head_Dim]
        k_flat = k_cache.flatten(1, 2)        # [Heads, Total_Capacity, Head_Dim]
        
        # Expand Batch dim to 1
        k_out = k_flat.unsqueeze(0) # [1, Heads, Total_Capacity, Head_Dim]
        
        # For this demo, assuming K and V are in same tensor or separate.
        # If self.kv_cache holds both, split them. 
        # (Assuming separate or caller handles splits).
        return k_out, k_out # Placeholder: return K and V views

    def get_batch_mappings(self, req_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates the Page Table and Inverse Table for the Glue Layer.
        
        Returns:
            page_table: [Batch, Max_Logical_Blocks]
            phys_to_log_table: [Batch, Max_Physical_Blocks]
        """
        batch_size = len(req_ids)
        
        # 1. Fetch tables
        tables = [self.block_manager.hash_to_block_id.get(rid, []) for rid in req_ids] 
        # Wait, block_manager.hash_to... is the cache map. 
        # We need the block_tables for these requests.
        # The 'content-agnostic' manager stores tables where?
        # Ah, the 'memory_manager.py' previous turn implemented 'allocate' returning IDs,
        # but didn't explicitly store the table per Request ID inside KVTManager.
        # KVTManager needs to track {req_id -> block_table}.
        # (Assuming self.req_tables exists)
        
        tables = [self.req_tables[rid] for rid in req_ids]
        max_log = max(len(t) for t in tables)
        max_phys = self.block_manager.blocks.shape[0] if hasattr(self.block_manager, 'blocks') else len(self.block_manager.blocks)
        
        # 2. Build Page Table (Logical -> Physical)
        page_table = torch.full((batch_size, max_log), -1, dtype=torch.int32, device=self.device)
        for i, t in enumerate(tables):
            page_table[i, :len(t)] = torch.tensor(t, dtype=torch.int32, device=self.device)
            
        # 3. Build Inverse Table (Physical -> Logical)
        # Initialize with -1
        phys_to_log = torch.full((batch_size, max_phys), -1, dtype=torch.int32, device=self.device)
        
        # Scatter logical indices
        # logical indices are just 0..len(t)
        # phys_to_log[b, page_table[b, i]] = i
        
        # Vectorized scatter:
        # We can use scatter_()
        # indices = page_table.long() # [B, Max_Log] (Physical IDs)
        # values = torch.arange(max_log).expand(B, -1)
        # phys_to_log.scatter_(1, indices, values) 
        # (Need to handle -1 in page_table to avoid invalid writes)
        
        for i, t in enumerate(tables):
            phys_ids = torch.tensor(t, dtype=torch.long, device=self.device)
            log_ids = torch.arange(len(t), dtype=torch.int32, device=self.device)
            phys_to_log[i].scatter_(0, phys_ids, log_ids)
            
        return page_table, phys_to_log
        
    # In KVTManager (memory_manager.py)
    def get_slot_mapping(self, block_tables: List[List[int]], seq_lengths: List[int]) -> torch.Tensor:
        """
        Generates flat physical token indices for scatter writes.
        
        Args:
            block_tables: List of [block_ids] per sequence
            seq_lengths: List of L_i per sequence
            
        Returns:
            [sum(seq_lengths)] - flat physical indices
        """
        slots = []
        for table, length in zip(block_tables, seq_lengths):
            # For each sequence, map token positions to physical slots
            # Token i in sequence -> Block (i // block_size), Offset (i % block_size)
            for token_idx in range(length):
                block_idx_in_table = token_idx // self.block_size
                offset = token_idx % self.block_size
                
                physical_block_id = table[block_idx_in_table]
                physical_slot = physical_block_id * self.block_size + offset
                
                slots.append(physical_slot)
        
        return torch.tensor(slots, dtype=torch.int64, device=self.device)