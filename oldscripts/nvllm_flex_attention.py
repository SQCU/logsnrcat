"""
nvllm_flex_attention.py

Physical Memory Operations for Paged Attention.
Contains the cache update primitives used by the Forward Pass.
"""

import torch

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
    # usage of slice(None) is equivalent to ':'
    
    k_cache[0, :, slot_mapping, :] = k_src_p
    v_cache[0, :, slot_mapping, :] = v_src_p