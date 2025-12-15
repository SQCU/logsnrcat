# src/bucket_manager.py
# leaf dependency, may never import other internal project dependencies
import math
import random
from typing import List, Tuple, Optional, Any, Union, Dict
from dataclasses import dataclass

@dataclass
class ResolutionBucket:
    resolution: int
    batch_size: int
    aspect_ratio: Tuple[int, int] = (1, 1)

class BucketManager:
    def __init__(self, base_resolution=32, base_batch_size=128, patch_stride=2):
        self.patch_stride = patch_stride
        # Calculate base cost: tokens * batch_size
        base_tokens = (base_resolution // patch_stride) ** 2
        self.base_cost = base_tokens * base_batch_size
        self.buckets: List[ResolutionBucket] = []

    def add_bucket(self, resolution: int, batch_size: Optional[int] = None):
        if batch_size is None:
            # Auto-scale batch size: B_new * T_new = Base_Cost
            tokens = (resolution // self.patch_stride) ** 2
            # Clamp minimum batch to 1
            batch_size = max(1, int(self.base_cost / (tokens + 1e-6)))
        
        self.buckets.append(ResolutionBucket(resolution, batch_size))
    
    def sample_bucket(self) -> ResolutionBucket:
        # Uniform sampling for now
        return random.choice(self.buckets)

def build_bucket_manager_from_config(cfg: Dict[str, Any], model_stride: Optional[int] = None) -> BucketManager:
    """
    Constructs BucketManager from a pure dictionary.
    Strictly requires cfg['training']['bucketing'] structure.
    """
    # 1. Determine Stride (Fail fast if missing)
    if model_stride is not None:
        patch_stride = model_stride
    else:
        # Strict dictionary access. If 'model' isn't there, we crash.
        patch_stride = cfg['model']['patch_embedder']['stride']

    # 2. Extract Config Section (Strict access)
    # We assume the sanitizer has ensured these keys exist via defaults if not provided
    bucketing_cfg = cfg['training']['bucketing']
    
    enabled = bucketing_cfg['enabled']
    base_res = bucketing_cfg['base_resolution']
    base_bs = bucketing_cfg['base_batch_size']
            
    # 3. Initialize Manager
    manager = BucketManager(
        base_resolution=base_res,
        base_batch_size=base_bs,
        patch_stride=patch_stride
    )
    
    # 4. Add Buckets
    if enabled:
        img_buckets = bucketing_cfg['image_buckets']
        for b in img_buckets:
            # b is a dict now, not an object
            manager.add_bucket(b['resolution'], batch_size=b['batch_size'])
    else:
        # Fallback Defaults (Explicitly logged if we were using a logger here)
        manager.add_bucket(32)
        if base_bs >= 16:
            manager.add_bucket(64)
            
    return manager