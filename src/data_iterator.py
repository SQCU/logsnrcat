# src/data_iterator.py
import torch
import random
from .model import ContextBlock
from .data_functional import (
    generate_checkerboard_query, render_checkerboard,
    generate_torus_query, render_torus,
    serialize_query
)
# Reuse existing noise functions
from .data_functional import get_logsnr_batch 
from typing import Tuple, List, Dict, Any, Optional, Union

class FunctionalIterator:
    """
    Stateful orchestrator for functional generators.
    """
    def __init__(self, device: torch.device, generator_func, renderer_func, config: Dict[str, Any]):
        self.device = device
        self.gen_func = generator_func
        self.render_func = renderer_func
        self.config = config
        self.seed = config.get('seed', 42)
        self.resolution_override = config.get('resolution', None)
        
        # Yassification: Configurable Text Position
        # options: "prefix", "suffix", "none", "random"
        self.text_pos = config.get('text_position', 'prefix')

    def generate_batch_list(self, batch_size: int, resolution: int = 32, **kwargs) -> List[ContextBlock]:
        start_group_id = kwargs.get('start_group_id', 0)
        res = self.resolution_override if self.resolution_override else resolution
        
        blocks = []
        for i in range(batch_size):
            # 1. Generate Params
            query = self.gen_func(self.seed, self.config)
            self.seed += 1
            curr_gid = start_group_id + i
            
            # 2. Determine Layout
            layout = self.text_pos
            if layout == 'random':
                layout = random.choice(['prefix', 'suffix', 'none'])
                
            text_block = ContextBlock(
                content=serialize_query(query).to(self.device),
                type='text', causal=True, group_id=curr_gid, id=f"txt_{curr_gid}"
            )
            img_block = ContextBlock(
                content=self.render_func(query, res, self.device),
                type='latent', causal=False, group_id=curr_gid, id=f"img_{curr_gid}"
            )
            
            # 3. Assemble
            if layout == 'prefix':
                blocks.extend([text_block, img_block])
            elif layout == 'suffix':
                blocks.extend([img_block, text_block])
            else: # none
                blocks.append(img_block)
                
        return blocks

class CompositeIterator:
    def __init__(self, device, config: Dict[str, Any], **kwargs):
        self.device = device
        self.splits = []
        
        for name, split_cfg in config.items():
            # Support both object and dict config due to loading nuances
            if hasattr(split_cfg, 'model_dump'):
                split_cfg = split_cfg.model_dump()
            
            sType = split_cfg.get('type')
            params = split_cfg.get('params', {})
            ratio = split_cfg.get('ratio', 1.0)
            
            iterator = None
            if sType == 'checkerboard':
                iterator = FunctionalIterator(device, generate_checkerboard_query, render_checkerboard, params)
            elif sType == 'torus':
                iterator = FunctionalIterator(device, generate_torus_query, render_torus, params)
            elif sType == 'video':
                # Video needs the complex logic from old data.py, presumably imported or re-implemented.
                # For this functional refactor, we focus on the generative ones.
                # Assuming VideoFolderIterator exists in .data (legacy) or ported here.
                # We will skip implementation for brevity unless requested, 
                # but the Composite structure supports it.
                from .data import VideoFolderIterator
                iterator = VideoFolderIterator(params['path'], device=device, caching_resolution=kwargs['caching_resolution'])
            
            if iterator:
                self.splits.append({
                    'name': name,
                    'iterator': iterator,
                    'ratio': ratio,
                    'type': sType,          # <--- Stored for dispatch
                    'params': params,       # <--- Stored for config lookup
                    'noise_mode': split_cfg.get('noise_mode', 'uniform'),
                    'noise_params': split_cfg.get('noise_params', {})
                })
        
        
        # Normalize ratios
        total = sum(s['ratio'] for s in self.splits)
        for s in self.splits: s['ratio'] /= total

    def generate_batch_list(self, batch_size: int, **kwargs) -> List[ContextBlock]:
        counts = [int(batch_size * s['ratio']) for s in self.splits]
        remainder = batch_size - sum(counts)
        if counts: counts[0] += remainder
        
        all_blocks = []
        global_gid = 0
        if 'start_group_id' in kwargs: global_gid = kwargs['start_group_id']
        
        for i, split in enumerate(self.splits):
            count = counts[i]
            if count == 0: continue
            
            # --- Dispatch Logic ---
            if split['type'] == 'video':
                # Video requires 'sequence_config' positional argument
                # 1. Retrieve base structure
                seq_conf = split['params'].get('sequence_structure', [{'res': 32}])
                
                # 2. Check for resolution override from kwargs
                if 'resolution' in kwargs:
                    res_target = kwargs['resolution']
                    # Apply relative scaling logic
                    overridden = []
                    for frame in seq_conf:
                        f = frame.copy()
                        rel = f.get('relative_res', 1.0)
                        f['res'] = int(res_target * rel)
                        # Ensure even
                        if f['res'] % 2 != 0: f['res'] += 1
                        overridden.append(f)
                    seq_conf = overridden
                
                blocks = split['iterator'].generate_batch_list(count, seq_conf, start_group_id=global_gid)
            else:
                # Functional iterators handle kwargs (resolution, etc.) directly
                blocks = split['iterator'].generate_batch_list(count, start_group_id=global_gid, **kwargs)

                # Assign LogSNR for functional latents (generated clean)
                latents = [b for b in blocks if b.type == 'latent']
                if latents:
                    H, W = latents[0].content.shape[-2:]
                    lsnrs = get_logsnr_batch(
                        split['noise_mode'], len(latents), H, W, self.device, split['noise_params']
                    )
                    for b, l in zip(latents, lsnrs):
                        b.logsnr = l
            
            for b in blocks: b.source = split['name']
            
            all_blocks.extend(blocks)
            
            if blocks:
                gids = [b.group_id for b in blocks]
                used = max(gids) - min(gids) + 1
                global_gid += used
                
        return all_blocks