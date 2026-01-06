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
        self.seed = config['seed']
        self.resolution_override = config['resolution']

        # Yassification: Configurable Text Position
        # options: "prefix", "suffix", "none", "random"
        self.text_pos = config['text_position']

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
 
            # 3. Create blocks with EXPLICIT shape_meta
            text_content = serialize_query(query).to(self.device)
            text_block = ContextBlock(
                content=text_content,
                type='text',
                causal=True,
                shape_meta=(text_content.shape[0],),  # Explicit: (seq_len,)
                group_id=curr_gid,
                id=f"txt_{curr_gid}"
            )
 
            img_content = self.render_func(query, res, self.device)
            img_block = ContextBlock(
                content=img_content,
                type='latent',
                causal=False,
                shape_meta=(res, res),  # Explicit: (H, W) for broadcast with (C, H, W)
                group_id=curr_gid,
                id=f"img_{curr_gid}"
            )
 
            # 4. Assemble
            if layout == 'prefix':
                blocks.extend([text_block, img_block])
            elif layout == 'suffix':
                blocks.extend([img_block, text_block])
            else:  # none
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

            sType = split_cfg['type']
            params = split_cfg['params']
            ratio = split_cfg['ratio']

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
            elif sType == 'fractal':
                from .fractal import FractalIterator
                iterator = FractalIterator(device, params)
            elif sType == 'sprite_atlas':
                from .sprite_atlas import SpriteAtlasIterator
                iterator = SpriteAtlasIterator(device, params)
            elif sType == 'procedural':
                from .procedural import ProceduralIterator
                iterator = ProceduralIterator(device, params)

            if iterator:
                self.splits.append({
                    'name': name,
                    'iterator': iterator,
                    'ratio': ratio,
                    'type': sType,          # <--- Stored for dispatch
                    'params': params,       # <--- Stored for config lookup
                    'noise_mode': split_cfg['noise_mode'],
                    'noise_params': split_cfg['noise_params']
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
                seq_conf = split['params']['sequence_structure']
                
                # 2. Check for resolution override from kwargs
                if 'resolution' in kwargs:
                    res_target = kwargs['resolution']
                    # Apply relative scaling logic
                    overridden = []
                    for frame in seq_conf:
                        f = frame.copy()
                        rel = f['relative_res']
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

 
    def get_split_names(self) -> List[str]:
        """Returns list of available split names."""
        return [s['name'] for s in self.splits]
 
    def generate_from_split(self, split_name: str, count: int, **kwargs) -> List[ContextBlock]:
        """
        Generate blocks from a specific split by name.
 
        This is the correct interface for eval code that needs homogeneous batches.
        Fails loudly if split doesn't exist - no silent defaults.
        """
        split = None
        for s in self.splits:
            if s['name'] == split_name:
                split = s
                break
 
        if split is None:
            available = self.get_split_names()
            raise KeyError(f"Split '{split_name}' not found. Available: {available}")
 
        global_gid = kwargs.pop('start_group_id', 0)
 
        # --- Dispatch by type (same logic as generate_batch_list, but single split) ---
        if split['type'] == 'video':
            # Video REQUIRES sequence_structure - no defaults
            seq_conf = split['params']['sequence_structure']
 
            if 'resolution' in kwargs:
                res_target = kwargs['resolution']
                overridden = []
                for frame in seq_conf:
                    f = frame.copy()
                    # relative_res is REQUIRED for video frames
                    rel = f['relative_res']
                    f['res'] = int(res_target * rel)
                    if f['res'] % 2 != 0: f['res'] += 1
                    overridden.append(f)
                seq_conf = overridden
 
            blocks = split['iterator'].generate_batch_list(count, seq_conf, start_group_id=global_gid)
        else:
            # Functional iterators
            blocks = split['iterator'].generate_batch_list(count, start_group_id=global_gid, **kwargs)
 
            # Assign LogSNR for functional latents
            latents = [b for b in blocks if b.type == 'latent']
            if latents:
                H, W = latents[0].content.shape[-2:]
                lsnrs = get_logsnr_batch(
                    split['noise_mode'], len(latents), H, W, self.device, split['noise_params']
                )
                for b, l in zip(latents, lsnrs):
                    b.logsnr = l
 
        for b in blocks:
            b.source = split['name']

        return blocks


class AsyncPrefetcher:
    """
    Async data prefetcher that breaks serial data generation dependency.

    Runs a background thread that proactively generates batches using its own
    PRNG sequence, filling a buffer queue. Training pulls from the buffer
    instead of waiting for generation.

    Usage:
        prefetcher = AsyncPrefetcher(
            iterator=composite_iterator,
            split_name='sprite_atlas',
            count=4,
            resolution=64,
            buffer_size=8,
            seed=42
        )

        # In training loop:
        blocks = prefetcher.get()  # Returns instantly if buffer populated

        # When done:
        prefetcher.stop()
    """

    def __init__(
        self,
        iterator: 'CompositeIterator',
        split_name: str,
        count: int,
        resolution: int = 64,
        buffer_size: int = 8,
        seed: int = 42,
        device: torch.device = None
    ):
        import threading
        import queue

        self.iterator = iterator
        self.split_name = split_name
        self.count = count
        self.resolution = resolution
        self.device = device or iterator.device

        # Thread-safe buffer queue
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.buffer_size = buffer_size

        # PRNG state - we advance our own sequence independently
        self.seed = seed
        self.rng = random.Random(seed)

        # Control flags
        self._stop_event = threading.Event()
        self._thread = None
        self._started = False

        # Stats
        self._generated_count = 0
        self._cache_hits = 0
        self._cache_misses = 0

    def start(self):
        """Start the background prefetch thread."""
        import threading

        if self._started:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._thread.start()
        self._started = True

    def stop(self):
        """Stop the background thread and drain the buffer."""
        if not self._started:
            return

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._started = False

        # Drain buffer to free memory
        while not self.buffer.empty():
            try:
                self.buffer.get_nowait()
            except:
                break

    def _prefetch_loop(self):
        """Background loop that generates batches proactively."""
        # Create a separate CUDA stream for async generation
        if self.device.type == 'cuda':
            stream = torch.cuda.Stream(device=self.device)
        else:
            stream = None

        while not self._stop_event.is_set():
            # Don't overfill - wait if buffer is full
            if self.buffer.full():
                self._stop_event.wait(timeout=0.01)  # Brief sleep
                continue

            try:
                # Generate in separate stream to avoid blocking training
                if stream is not None:
                    with torch.cuda.stream(stream):
                        blocks = self._generate_batch()
                        # Sync this stream before putting in queue
                        stream.synchronize()
                else:
                    blocks = self._generate_batch()

                self.buffer.put(blocks, timeout=1.0)
                self._generated_count += 1

            except Exception as e:
                # Log but don't crash the thread
                print(f"[AsyncPrefetcher] Error generating batch: {e}")
                self._stop_event.wait(timeout=0.1)

    def _generate_batch(self) -> List[ContextBlock]:
        """Generate one batch using our PRNG sequence."""
        # Advance our RNG and use it to seed the iterator's generation
        batch_seed = self.rng.randint(0, 2**31 - 1)

        # Generate from the specified split
        blocks = self.iterator.generate_from_split(
            self.split_name,
            count=self.count,
            resolution=self.resolution,
            start_group_id=batch_seed % 10000  # Vary group IDs
        )

        return blocks

    def get(self, timeout: float = 30.0) -> List[ContextBlock]:
        """
        Get a pre-generated batch from the buffer.

        If buffer is empty, blocks until data is available (up to timeout).
        This should be rare once the prefetcher is warmed up.
        """
        if not self._started:
            self.start()

        try:
            blocks = self.buffer.get(timeout=timeout)
            self._cache_hits += 1
            return blocks
        except:
            # Buffer was empty - generate synchronously as fallback
            self._cache_misses += 1
            return self._generate_batch()

    def get_nowait(self) -> Optional[List[ContextBlock]]:
        """Non-blocking get - returns None if buffer empty."""
        if not self._started:
            self.start()

        try:
            blocks = self.buffer.get_nowait()
            self._cache_hits += 1
            return blocks
        except:
            self._cache_misses += 1
            return None

    def warmup(self, min_items: int = None):
        """Block until buffer has at least min_items ready."""
        if min_items is None:
            min_items = self.buffer_size // 2

        if not self._started:
            self.start()

        # Wait for buffer to fill
        import time
        while self.buffer.qsize() < min_items:
            time.sleep(0.05)

    @property
    def stats(self) -> Dict[str, Any]:
        """Return prefetcher statistics."""
        return {
            'buffer_size': self.buffer_size,
            'buffer_fill': self.buffer.qsize(),
            'generated': self._generated_count,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': self._cache_hits / max(1, self._cache_hits + self._cache_misses),
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __del__(self):
        self.stop()


class MultiResolutionPrefetcher:
    """
    Async prefetcher for multi-resolution training with bucketing.

    Maintains separate prefetch buffers per resolution, allowing the training
    loop to request data at any resolution without blocking. Each resolution
    has its own background thread generating batches proactively.

    Usage:
        from src.bucket_manager import build_bucket_manager_from_config

        bucket_mgr = build_bucket_manager_from_config(config, model_stride=4)
        prefetcher = MultiResolutionPrefetcher(
            iterator=composite_iterator,
            bucket_manager=bucket_mgr,
            split_name='sprite_atlas',
            count=4,
            buffer_per_resolution=4,
            seed=42,
            device=device
        )

        # In training loop:
        bucket = bucket_mgr.sample_bucket()
        blocks = prefetcher.get(bucket.resolution)  # Near-instant

        # When done:
        prefetcher.stop()
    """

    def __init__(
        self,
        iterator: 'CompositeIterator',
        bucket_manager: 'BucketManager',
        split_name: str,
        count: int,
        buffer_per_resolution: int = 4,
        seed: int = 42,
        device: torch.device = None
    ):
        self.iterator = iterator
        self.bucket_manager = bucket_manager
        self.split_name = split_name
        self.count = count
        self.device = device or iterator.device
        self.buffer_per_resolution = buffer_per_resolution

        # Base seed - each resolution gets a deterministic offset
        self.base_seed = seed

        # Create a prefetcher per resolution bucket
        self._prefetchers: Dict[int, AsyncPrefetcher] = {}
        for i, bucket in enumerate(bucket_manager.buckets):
            res = bucket.resolution
            # Deterministic seed per resolution for reproducibility
            res_seed = seed + i * 10000
            self._prefetchers[res] = AsyncPrefetcher(
                iterator=iterator,
                split_name=split_name,
                count=count,
                resolution=res,
                buffer_size=buffer_per_resolution,
                seed=res_seed,
                device=self.device
            )

        self._started = False

    def start(self):
        """Start all background prefetch threads."""
        if self._started:
            return
        for pf in self._prefetchers.values():
            pf.start()
        self._started = True

    def stop(self):
        """Stop all background threads and drain buffers."""
        if not self._started:
            return
        for pf in self._prefetchers.values():
            pf.stop()
        self._started = False

    def warmup(self, min_items_per_resolution: int = None):
        """
        Block until all resolution buffers have at least min_items ready.

        Args:
            min_items_per_resolution: Minimum items per buffer (default: buffer_size // 2)
        """
        if min_items_per_resolution is None:
            min_items_per_resolution = max(1, self.buffer_per_resolution // 2)

        if not self._started:
            self.start()

        for pf in self._prefetchers.values():
            pf.warmup(min_items=min_items_per_resolution)

    def get(self, resolution: int, timeout: float = 30.0) -> List[ContextBlock]:
        """
        Get a pre-generated batch at the specified resolution.

        If buffer is empty, blocks until data is available (up to timeout).
        This should be rare once the prefetcher is warmed up.

        Args:
            resolution: The target resolution (must match a bucket)
            timeout: Maximum wait time in seconds

        Returns:
            List of ContextBlocks at the requested resolution
        """
        if not self._started:
            self.start()

        if resolution not in self._prefetchers:
            # Fallback: generate synchronously for unknown resolutions
            return self.iterator.generate_from_split(
                self.split_name, count=self.count, resolution=resolution
            )

        return self._prefetchers[resolution].get(timeout=timeout)

    def get_nowait(self, resolution: int) -> Optional[List[ContextBlock]]:
        """Non-blocking get - returns None if buffer empty."""
        if not self._started:
            self.start()

        if resolution not in self._prefetchers:
            return None

        return self._prefetchers[resolution].get_nowait()

    @property
    def stats(self) -> Dict[str, Any]:
        """Return aggregate statistics across all resolutions."""
        total_generated = 0
        total_hits = 0
        total_misses = 0
        per_resolution = {}

        for res, pf in self._prefetchers.items():
            s = pf.stats
            total_generated += s['generated']
            total_hits += s['cache_hits']
            total_misses += s['cache_misses']
            per_resolution[res] = s

        return {
            'total_generated': total_generated,
            'total_hits': total_hits,
            'total_misses': total_misses,
            'hit_rate': total_hits / max(1, total_hits + total_misses),
            'per_resolution': per_resolution,
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __del__(self):
        self.stop()