"""
src/utils.py - context management, sampling, logger, plotting

Originally memory_manager.py - Implements the Content-Agnostic Paged Memory System.
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
        self.req_content_hashes: Dict[int, List[int]] = {}  # req_id -> content hash stream
        
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
        self.req_content_hashes[req_id] = list(content_hashes)  # Store for extend_sequence

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
        self.req_content_hashes[req_id] = full_hashes  # Update stored hashes

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
    
    def get_cached_hashes(self, req_id: int) -> List[int]:
        """
        Returns the content hash stream for an existing request.
        Used by extend_sequence to maintain prefix-cache compatibility.
        """
        if req_id not in self.req_content_hashes:
            raise KeyError(f"Request {req_id} has no cached content hashes. Was it allocated?")
        return list(self.req_content_hashes[req_id])  # Return copy to prevent mutation

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
        if req_id in self.req_content_hashes:
            del self.req_content_hashes[req_id]
    
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

# ==============================================================================
# 1. Math & Physics Helpers (Deduplicated)
# ==============================================================================

def get_schedule(t, schedule_bounds: tuple = (5, -4)):
    """Linear LogSNR schedule."""
    return schedule_bounds[0] - t * (schedule_bounds[1] - schedule_bounds[0])

def logsnr_to_alpha_sigma(logsnr):
    """
    Returns alpha, sigma for a given logsnr.
    Handles broadcasting if logsnr is [B, 1, H, W].
    """
    # Ensure numerical stability
    sigmoid_lsnr = torch.sigmoid(logsnr)
    sigmoid_neg_lsnr = torch.sigmoid(-logsnr)
    alpha = torch.sqrt(sigmoid_lsnr)
    sigma = torch.sqrt(sigmoid_neg_lsnr)
    return alpha, sigma

def get_image_spans(resolution):
    latent_res = resolution // 2
    length = latent_res * latent_res
    return [{'type': 'latent', 'len': length, 'shape': (latent_res, latent_res), 'causal': False}]


# ==============================================================================
# 2. Model Wrappers 
# ==============================================================================

# Import necessary helpers from model.py
from .model import (
    render_topology_embeddings,
    build_dual_masks,
    ContextBlock,
    Span
)


# =============================================================================
# Factored Forward Pass (Embedding-Aware)
# =============================================================================

def run_forward_from_embeddings(
    model,
    z_flat: torch.Tensor,
    topo_embeds: torch.Tensor,
    span_objects: List[Span],
    page_table,
    unembed_fn=None,
    unembed_indices: Optional[List[int]] = None
):
    """
    Pure transformer forward from pre-computed embeddings.

    This is the core forward pass - no embedding, just:
    1. Build masks
    2. Run transformer
    3. Optionally unembed specific spans

    Args:
        model: The transformer model (ZC variant)
        z_flat: Pre-computed embeddings [L_total, D]
        topo_embeds: Topology embeddings [L_total, topo_dim]
        span_objects: List of Span metadata
        page_table: PageTable for masking
        unembed_fn: Optional unembedding function (span_unembedder.decode)
        unembed_indices: Which spans to unembed (None = all, [] = none)

    Returns:
        z_out: Transformer output [L_total, D]
        decoded: Unembedded outputs (or None if unembed_fn not provided)
        aux_loss: MoE auxiliary loss
    """
    device = z_flat.device
    dtype = z_flat.dtype

    # Build masks
    L_total = z_flat.shape[0]
    block_size = page_table.block_size
    num_blocks = (L_total + block_size - 1) // block_size
    flat_page_table = torch.arange(num_blocks, device=device, dtype=torch.long)

    block_masks = build_dual_masks(
        span_objects, topo_embeds, topo_embeds,
        page_table, flat_page_table, None,
        window_size=getattr(model, 'window_size', 10.0)
    )

    # Transformer forward
    rope_scale = max(1.0, L_total / 64.0)
    z_out, aux_loss = model(
        z_flat.unsqueeze(0),
        topo_embeds.unsqueeze(0),
        slot_mapping=None,
        block_masks=block_masks,
        scale=rope_scale
    )
    z_out = z_out.squeeze(0)

    # Selective unembedding
    decoded = None
    if unembed_fn is not None:
        if unembed_indices is None:
            # Unembed all spans
            decoded = unembed_fn(z_out, span_objects)
        elif len(unembed_indices) > 0:
            # Unembed only specified spans
            selected_spans = [span_objects[i] for i in unembed_indices]
            decoded = unembed_fn(z_out, selected_spans)
            # Map back to full list with None placeholders
            full_decoded = [None] * len(span_objects)
            for i, idx in enumerate(unembed_indices):
                full_decoded[idx] = decoded[i]
            decoded = full_decoded
        # else: unembed_indices is empty list, skip unembedding

    return z_out, decoded, aux_loss


def run_model_forward(components, blocks: List[ContextBlock], incremental: bool = False):
    """
    Unified forward pass for ZC model.

    Args:
        components: (model, span_embedder, span_unembedder, page_table)
        blocks: List of ContextBlocks
        incremental: If True, use cached embeddings where valid (for sampling)
                    If False, always embed fresh (for training)

    Returns:
        decoded: Unembedded outputs per span
        aux_loss: MoE auxiliary loss
    """
    model, span_embedder, span_unembedder, page_table = components
    device = model.text_embed.weight.device
    dtype = model.text_embed.weight.dtype

    # Embedding phase
    if incremental:
        z_flat, span_objects, _, num_recomputed = span_embedder.embed_incremental(blocks)
    else:
        z_flat, span_objects, _ = span_embedder.embed(blocks)

    # Topology
    topo_embeds, _ = render_topology_embeddings(span_objects, 3, device, dtype=dtype)

    # Forward + unembed all
    _, decoded, aux_loss = run_forward_from_embeddings(
        model, z_flat, topo_embeds, span_objects, page_table,
        unembed_fn=span_unembedder.decode,
        unembed_indices=None  # Unembed all
    )

    return decoded, aux_loss


def run_model_forward_sampling(
    components,
    blocks: List[ContextBlock],
    unembed_indices: Optional[List[int]] = None
):
    """
    Optimized forward pass for sampling.

    Uses incremental embedding and selective unembedding.

    Args:
        components: (model, span_embedder, span_unembedder, page_table)
        blocks: List of ContextBlocks (with cached embeddings)
        unembed_indices: Which spans need unembedding (typically just active latents)
                        None = all, [] = none (just want z_out)

    Returns:
        z_out: Transformer output [L_total, D]
        decoded: Unembedded outputs (only for specified indices)
        aux_loss: MoE auxiliary loss
        num_recomputed: How many blocks were re-embedded (for diagnostics)
    """
    model, span_embedder, span_unembedder, page_table = components
    device = model.text_embed.weight.device
    dtype = model.text_embed.weight.dtype

    # Incremental embedding - reuse cached where valid
    z_flat, span_objects, _, num_recomputed = span_embedder.embed_incremental(blocks)

    # Topology (TODO: also cache this)
    topo_embeds, _ = render_topology_embeddings(span_objects, 3, device, dtype=dtype)

    # Forward with selective unembedding
    z_out, decoded, aux_loss = run_forward_from_embeddings(
        model, z_flat, topo_embeds, span_objects, page_table,
        unembed_fn=span_unembedder.decode,
        unembed_indices=unembed_indices
    )

    return z_out, decoded, aux_loss, num_recomputed


def predict_velocity_from_blocks(components, blocks: List[ContextBlock], mode='naive'):
    """
    Wrapper that calls model and processes outputs (factorization, etc).
    """
    decoded, aux_loss = run_model_forward(components, blocks)
    
    v_final_list = []
    pred_logsnr_list = []
    
    for i, d in enumerate(decoded):
        if 'image_vpreds' in d:
            v_raw = d['image_vpreds']
            pred_l = d['image_logsnrs']
            
            if mode == 'factorized':
                sigma_p = torch.sqrt(torch.sigmoid(-pred_l))
                v_final = v_raw * sigma_p
            else:
                v_final = v_raw
            
            v_final_list.append(v_final)
            pred_logsnr_list.append(pred_l)
        else:
            # For text-only blocks, we might not have vpreds relevant to diffusion loss
            # Just append None or dummy
            v_final_list.append(None)
            pred_logsnr_list.append(None)

    return v_final_list, pred_logsnr_list, aux_loss


# =============================================================================
# KVC-Aware Forward Pass (Concatenative AR with Cache Hits)
# =============================================================================

class KVCSessionState:
    """
    Tracks the state of a KVC inference session.

    This enables true concatenative AR with prefix caching:
    - Prefill: First call processes all tokens, caches all K/V
    - Update: Inner diffusion loop updates active span's K/V in place
    - Extend: Outer AR loop appends new spans, reuses prefix cache
    """
    def __init__(self, kvt_manager: 'KVTManager', req_id: int):
        self.kvt_manager = kvt_manager
        self.req_id = req_id
        self.cached_prefix_len = 0  # How many tokens are stably cached
        self.total_len = 0          # Total sequence length
        self.active_span_start = 0  # Start of the active (mutable) span
        self.active_span_len = 0    # Length of the active span
        self.is_initialized = False

    def prefill(self, content_hashes: List[int], topo_data: torch.Tensor,
                active_start: int = 0, active_len: int = None):
        """
        Initial prefill: cache entire context.

        Args:
            content_hashes: Hashes for all tokens
            topo_data: [L_total, Topo_Dim] topology
            active_start: Where the active (mutable) span starts
            active_len: Length of active span (defaults to rest of sequence)
        """
        self.kvt_manager.allocate_and_write_sequence(
            self.req_id, content_hashes, topo_data
        )
        self.total_len = len(content_hashes)
        self.active_span_start = active_start
        self.active_span_len = active_len if active_len else (self.total_len - active_start)
        self.cached_prefix_len = active_start  # Everything before active span is stable
        self.is_initialized = True

    def extend(self, new_content_hashes: List[int], new_topo_data: torch.Tensor,
               new_active_len: int = None):
        """
        Extend context with new span (concatenative AR step).
        The previous active span becomes part of the cached prefix.

        Args:
            new_content_hashes: Hashes for NEW tokens only
            new_topo_data: [L_new, Topo_Dim] topology for new tokens
            new_active_len: Length of new active span (defaults to all new tokens)
        """
        if not self.is_initialized:
            raise RuntimeError("Cannot extend before prefill")

        # Previous active span is now part of stable prefix
        self.cached_prefix_len = self.total_len

        # Extend the sequence
        self.kvt_manager.extend_sequence(
            self.req_id, new_content_hashes, new_topo_data
        )

        old_total = self.total_len
        self.total_len = old_total + len(new_content_hashes)
        self.active_span_start = old_total
        self.active_span_len = new_active_len if new_active_len else len(new_content_hashes)

    def get_slot_mapping_for_active(self) -> torch.Tensor:
        """
        Returns slot mapping for only the active span.
        Used when updating K/V for just the active (mutable) tokens.
        """
        block_table = self.kvt_manager.req_tables[self.req_id]
        block_size = self.kvt_manager.block_size
        device = self.kvt_manager.device

        slots = []
        for token_idx in range(self.active_span_start,
                               self.active_span_start + self.active_span_len):
            block_idx = token_idx // block_size
            offset = token_idx % block_size
            physical_block = block_table[block_idx]
            physical_slot = physical_block * block_size + offset
            slots.append(physical_slot)

        return torch.tensor(slots, dtype=torch.long, device=device)

    def cleanup(self):
        """Free the request from KVT manager."""
        if self.is_initialized:
            self.kvt_manager.free_request(self.req_id)
            self.is_initialized = False


def run_model_forward_kvc(
    components_kvc,
    blocks: List['ContextBlock'],
    session_state: KVCSessionState,
    mode: str = 'prefill'
):
    """
    KVC-aware forward pass supporting true concatenative AR.

    Uses incremental embedding to avoid re-embedding unchanged blocks.

    Args:
        components_kvc: Tuple of (model_kvc, span_embedder, span_unembedder, page_table, kvt_manager)
        blocks: List of ContextBlocks (full context, with embedding caches)
        session_state: KVCSessionState tracking cache state
        mode: 'prefill' (first call), 'update' (inner loop), or 'extend' (outer AR loop)

    Returns:
        decoded: List of decoded outputs per span
        aux_loss: Auxiliary loss from MoE routing
        num_recomputed: Number of blocks that were re-embedded (for diagnostics)

    The key insight:
    - prefill: Process ALL tokens, cache ALL K/V
    - update: Process ACTIVE tokens only, update their K/V in cache, attend to FULL cache
    - extend: Process NEW tokens only, extend cache, attend to FULL cache

    Embedding optimization:
    - Uses embed_incremental() - only re-embeds blocks whose content/logsnr changed
    - ContextBlocks cache their embeddings between calls
    - In 'update' mode, typically only the active latent needs re-embedding
    """
    model, span_embedder, span_unembedder, page_table, kvt_manager = components_kvc
    device = model.text_embed.weight.device
    dtype = model.text_embed.weight.dtype

    # 1. Incremental embedding - reuse cached embeddings where valid
    # This is THE key optimization - avoid re-embedding unchanged blocks
    z_flat, span_objects, content_hashes, num_recomputed = span_embedder.embed_incremental(blocks)
    L_total = z_flat.shape[0]

    # 2. Topology for full context
    topo_embeds_full, _ = render_topology_embeddings(span_objects, 3, device, dtype=dtype)

    if mode == 'prefill':
        # === PREFILL MODE ===
        # Process all tokens, cache all K/V

        # Find where the active span starts (last span)
        active_start = span_objects[-1].start_idx if span_objects else 0
        active_len = L_total - active_start

        # Initialize cache with full context
        session_state.prefill(content_hashes, topo_embeds_full, active_start, active_len)

        # Get paging info for full context
        flat_page_table, inverse_page_table = kvt_manager.get_flat_page_mapping([session_state.req_id])
        block_tables = [kvt_manager.req_tables[session_state.req_id]]
        seq_lengths = [kvt_manager.req_lengths[session_state.req_id]]
        slot_mapping = kvt_manager.get_slot_mapping(block_tables, seq_lengths)

        # Build masks (Q=full, K=full for prefill)
        topo_heap = kvt_manager.get_topo_view()
        block_masks = build_dual_masks(
            span_objects, topo_embeds_full, topo_heap,
            page_table, flat_page_table, inverse_page_table,
            window_size=model.window_size
        )

        # Forward pass with all tokens
        rope_scale = max(1.0, L_total / 64.0)
        k_caches = [kvt_manager.get_flat_kv_view(i)[0] for i in range(len(model.layers))]
        v_caches = [kvt_manager.get_flat_kv_view(i)[1] for i in range(len(model.layers))]

        z_out, aux_loss = model(
            z_flat.unsqueeze(0),
            topo_embeds_full.unsqueeze(0),
            k_caches, v_caches,
            slot_mapping,
            block_masks,
            scale=rope_scale
        )

    elif mode == 'update':
        # === UPDATE MODE ===
        # Process only active span tokens, update their K/V, attend to full cache

        if not session_state.is_initialized:
            raise RuntimeError("Cannot update before prefill")

        # Extract only active tokens
        active_start = session_state.active_span_start
        active_end = active_start + session_state.active_span_len
        z_active = z_flat[active_start:active_end]
        topo_active = topo_embeds_full[active_start:active_end]

        # Slot mapping for active tokens only
        slot_mapping = session_state.get_slot_mapping_for_active()

        # Get paging info for full context (for K/V retrieval)
        flat_page_table, inverse_page_table = kvt_manager.get_flat_page_mapping([session_state.req_id])
        topo_heap = kvt_manager.get_topo_view()

        # Build masks: Q=active tokens, K=full heap
        # This requires spans for active tokens but heap for full context
        active_spans = [s for s in span_objects if s.end_idx > active_start]
        # Adjust span indices to be relative to active window
        adjusted_spans = []
        for s in active_spans:
            new_s = copy.copy(s)
            new_s.start_idx = max(0, s.start_idx - active_start)
            new_s.end_idx = min(session_state.active_span_len, s.end_idx - active_start)
            if new_s.end_idx > new_s.start_idx:
                adjusted_spans.append(new_s)

        block_masks = build_dual_masks(
            adjusted_spans, topo_active, topo_heap,
            page_table, flat_page_table, inverse_page_table,
            window_size=model.window_size
        )

        # Forward with active tokens only
        L_active = z_active.shape[0]
        rope_scale = max(1.0, session_state.total_len / 64.0)
        k_caches = [kvt_manager.get_flat_kv_view(i)[0] for i in range(len(model.layers))]
        v_caches = [kvt_manager.get_flat_kv_view(i)[1] for i in range(len(model.layers))]

        z_out_active, aux_loss = model(
            z_active.unsqueeze(0),
            topo_active.unsqueeze(0),
            k_caches, v_caches,
            slot_mapping,
            block_masks,
            scale=rope_scale
        )

        # Reconstruct full output (prefix is zeros, only active is valid)
        z_out = torch.zeros((1, L_total, z_flat.shape[-1]), device=device, dtype=dtype)
        z_out[0, active_start:active_end] = z_out_active.squeeze(0)

    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'prefill' or 'update'")

    # 3. Decode outputs
    decoded = span_unembedder.decode(z_out.squeeze(0), span_objects)

    return decoded, aux_loss, num_recomputed


