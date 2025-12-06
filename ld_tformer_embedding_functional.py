# ld_tformer_embedding_functional.py
import torch
import xxhash
import numpy as np
import math
from typing import List, Dict, Tuple, Any, Callable, Optional

# =========================================================
# 1. CONTENT IDENTITY (Hashing Policy)
# =========================================================

def generate_content_hash_stream(spans: List[Dict]) -> List[int]:
    """
    Transforms a list of Spans into a linear stream of Atomic Content IDs.
    These IDs are used by the BlockManager to detect identical content 
    (Prefix Caching).
    
    Args:
        spans: List of dicts. Expected structure:
               - 'type': 'text' | 'latent'
               - 'id': int | str (Unique ID for this specific content)
               - 'shape': tuple (e.g., (128,) or (16, 16))
               - 'data': (Optional) List[int] for text tokens.
               
    Returns:
        List[int]: A sequence of 64-bit integers representing the content.
    """
    stream = []
    
    for span in spans:
        span_type = span.get('type', 'latent')
        shape = span['shape']
        num_tokens = math.prod(shape)
        
        if span_type == 'text':
            # Text Identity = The Token ID itself
            # Data must be provided
            if 'data' not in span:
                raise ValueError("Text spans must provide 'data' (token IDs).")
            tokens = span['data']
            if len(tokens) != num_tokens:
                 raise ValueError(f"Text span shape {shape} != data len {len(tokens)}")
            stream.extend([int(t) for t in tokens])
            
        elif span_type == 'latent':
            # Latent Identity = Hash(Unique_Span_ID, Relative_Index)
            # This distinguishes "Pixel 0 of Image A" from "Pixel 0 of Image B"
            span_id = span['id']
            
            # Use xxhash to generate a deterministic stream based on the ID
            # We seed with span_id (hashed to int), and update with index.
            # Optimization: Generate standard hash for span_id, then XOR or mix with index?
            # Safer: Full hash per token.
            
            # If span_id is string, hash it first
            ### REVIEWER NOTE
            ### WTF IS THIS 'IF STRING HASH FIRST' NONSENSE?
            ### DISCRETE TOKEN EMBEDDINGS CAN ONLY BE HASHED BY THEIR TOKEN IDS
            if isinstance(span_id, str):
                seed = xxhash.xxh64(span_id).intdigest()
            else:
                seed = int(span_id)
            
            # Generate stream
            # (Note: In Python loop this is slow for huge latents. 
            #  Production code would vectorize this with numpy).
            
            # Vectorized approach using numpy for speed
            indices = np.arange(num_tokens, dtype=np.int64)
            # Simple mixing function to avoid calling xxhash 4096 times
            # mix(seed, idx) -> unique_content_id
            # A simple shift-XOR is usually sufficient for cache collisions 
            # combined with the causal block hash chain.
            
            # seed_high = seed >> 32
            # seed_low = seed & 0xFFFFFFFF
            # mixed = seed ^ indices 
            # (This assumes seed is good. Let's rely on xxhash for the block manager logic).
            
            # To be safe and compliant with 'Atomic Content ID' requirement:
            # We return a list of ints.
            # Let's assume span_id is effectively the "Content ID" base.
            
            # For the BlockManager, we need distinct integers.
            # Let's assume 64-bit space.
            # We create a deterministic stream: hash(seed, idx)
            
            base_hasher = xxhash.xxh64(seed=seed)
            for i in range(num_tokens):
                base_hasher.reset()
                base_hasher.update(i.to_bytes(8, 'little'))
                stream.append(base_hasher.intdigest())
                
    return stream

# =========================================================
# 2. GEOMETRY (Topology Policy)
# =========================================================

def render_topology_embeddings(
    spans: List[Dict],
    max_dims: int,
    device: torch.device,
    highway_offset: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Renders GLOBAL topology coordinates.
    
    Args:
        highway_offset: Starting highway value. 
                       0 for new sequences.
                       prev_max+1 for extending sequences.
    """
    highway_idx = []
    manifold_coords = []
    doc_ids = []
    
    current_highway = highway_offset
    
    # Dimension 0 is Highway. Remaining dimensions are Spatial/Manifold.
    spatial_dim_capacity = max_dims - 1
    
    for i, span in enumerate(spans):
        shape = span['shape']
        num_tokens = math.prod(shape)
        
        # 1. Highway (Global Linear Time)
        h_range = torch.arange(
            current_highway, 
            current_highway + num_tokens, 
            device=device
        )
        highway_idx.append(h_range)
        current_highway += num_tokens
        
        # 2. Manifold (Local Spatial Grid)
        # Uniform logic for 1D (Text), 2D (Images), or ND
        dims = [torch.arange(d, device=device) for d in shape]
        
        # meshgrid works for 1 arg (1D) or N args (ND)
        mesh = torch.meshgrid(*dims, indexing='ij')
        
        # Stack coordinates: 
        # 1D -> [L, 1], 2D -> [H*W, 2], etc.
        coords = torch.stack([m.flatten() for m in mesh], dim=-1)
        
        # Pad to fixed spatial capacity (R^k -> R^N)
        current_dim = coords.shape[-1]
        
        if current_dim < spatial_dim_capacity:
            pad_size = spatial_dim_capacity - current_dim
            # Pad with zeros in the extra dimensions
            padding = torch.zeros((num_tokens, pad_size), device=device)
            coords = torch.cat([coords, padding], dim=-1)
        elif current_dim > spatial_dim_capacity:
             raise ValueError(f"Span dimension {current_dim} exceeds model capacity {spatial_dim_capacity}")
             
        manifold_coords.append(coords)
        
        # 3. Doc IDs
        doc_ids.append(torch.full((num_tokens,), i, device=device, dtype=torch.int32))

    # Stack
    flat_highway = torch.cat(highway_idx).unsqueeze(-1).float()
    flat_manifold = torch.cat(manifold_coords).float()
    
    # [Total_L, 1 + Spatial_Cap]
    topo_embeds = torch.cat([flat_highway, flat_manifold], dim=-1)
    flat_doc_ids = torch.cat(doc_ids)
    
    return topo_embeds, flat_doc_ids

# =========================================================
# 3. CONNECTIVITY (Masking Policy)
# =========================================================

def get_block_causal_mod(doc_ids: torch.Tensor) -> Callable:
    """
    Returns a pure Python closure defining the Block-Causal connectivity rules.
    """
    def block_causal_mod(b, h, q_idx, kv_idx):
        q_doc = doc_ids[q_idx]
        k_doc = doc_ids[kv_idx]
        return (q_doc == k_doc) | (q_doc > k_doc)
        
    return block_causal_mod

def get_sliding_window_mod(
    topo_embeds: torch.Tensor, 
    window_size: float,
    doc_ids: Optional[torch.Tensor] = None
) -> Callable:
    """
    Returns a closure that enforces a Spatial Sliding Window in R^n.
    Optionally combines with Block-Causal logic if doc_ids are provided.
    
    Args:
        topo_embeds: [Total_L, 1 + N_Dims] (Col 0 is Highway, Cols 1..N are Space)
        window_size: Maximum Euclidean distance for connection.
        doc_ids: (Optional) If provided, enforces block-causal rules AND spatial window.
    """
    
    def swa_mod(b, h, q_idx, kv_idx):
        # 1. Spatial Rule (R^n)
        # We slice off the Highway dimension (Col 0) to get purely spatial coords
        q_pos = topo_embeds[q_idx, 1:]
        k_pos = topo_embeds[kv_idx, 1:]
        
        # Calculate Rn Distance
        dist = compute_rn_distance(q_pos, k_pos)
        spatial_mask = dist < window_size
        
        # 2. Block-Causal Rule (Optional Composition)
        if doc_ids is not None:
            q_doc = doc_ids[q_idx]
            k_doc = doc_ids[kv_idx]
            causal_mask = (q_doc == k_doc) | (q_doc > k_doc)
            return spatial_mask & causal_mask
            
        return spatial_mask

    return swa_mod