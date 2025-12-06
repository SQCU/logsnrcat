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
    k_new: [B*L, H, 1, D]
    k_cache: [1, H, Capacity, D]
    slot_mapping: [B*L]
    """
    #oops haha
    with torch.no_grad():
        BL, H, _, D = k_new.shape
        # Squeeze time dim: [B*L, H, 1, D] -> [B*L, H, D]
        k_src = k_new.squeeze(2)
        v_src = v_new.squeeze(2)
        
        # Expand slot_mapping for heads: [B*L] -> [B*L, H]
        # Then flatten to [B*L*H]
        slots_expanded = slot_mapping.unsqueeze(1).expand(-1, H).flatten()
        
        # Flatten k_src to [B*L*H, D]
        k_flat = k_src.reshape(-1, D)
        v_flat = v_src.reshape(-1, D)
        
        # Scatter write (heads are independent)
        for h in range(H):
            head_slots = slot_mapping  # Same slots for all heads
            k_cache[0, h, head_slots, :] = k_src[:, h, :].to(k_cache.dtype)
            v_cache[0, h, head_slots, :] = v_src[:, h, :].to(v_cache.dtype)