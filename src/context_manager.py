# src/context_manager.py - Context management for multimodal sequences
"""
Core context management infrastructure for handling multimodal sequences
(text + images) with proper attention masking, topology encoding, and
span tracking.

This module was previously named 'context_utils' but 'context_manager'
better reflects its purpose: managing the context window for attention.

Key concepts:
- ContextBlock: Atomic unit holding content + metadata (image tensor or text tokens)
- Span: Tracks a contiguous range of embedded tokens with type/position info
- SpanEmbedder: Wraps text/patch embedders, produces flat token sequence + span list
- SpanUnembedder: Decodes flat sequence back to per-span outputs
- Topology: Encodes position as (highway, spatial_coords) for RnRoPE
- Masks: Builds FlexAttention BlockMasks respecting document/causal boundaries
"""

import math
import xxhash
import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask, BlockMask
from typing import Tuple, List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field

from .paging import PageTable


# =========================================================
# 1. CORE DATA STRUCTURES
# =========================================================

@dataclass
class ContextBlock:
    """
    Canonical atomic unit of the dataset.
    Holds raw data and its topological metadata.

    shape_meta: For latents, this is (H, W) - the spatial dimensions of content.
                For text, this is (seq_len,) - the token count.
                Must broadcast correctly with content for logsnr map operations.
    """
    content: Union[torch.Tensor, str]  # [C, H, W] for latent or [L] for text tokens
    type: str = 'latent'
    causal: bool = True
    # Metadata
    shape_meta: Tuple[int, ...] = field(default_factory=tuple)
    logsnr: Optional[torch.Tensor] = None  # [1, H, W] for latents
    group_id: int = 0
    id: str = ""
    source: str = "unknown"

    def __post_init__(self):
        # Derive shape_meta from content if not explicitly set
        # This is a fallback - prefer explicit setting in iterators
        if not self.shape_meta and isinstance(self.content, torch.Tensor):
            if self.type == 'latent':
                print(f"type inference of pre-pooling image shape from an unknown tensor of unknown source is not possible.")
                raise TypeError("we are crashing this run... with no survivors.")
            elif self.type == 'text':
                # shape_meta = (seq_len,) for text tokens
                self.shape_meta = (self.content.shape[0],)


@dataclass
class Span:
    """
    Tracks a contiguous range of embedded tokens.

    Created by SpanEmbedder when embedding ContextBlocks.
    Used by masking functions to determine attention patterns.
    """
    type: str  # 'text' | 'latent'
    start_idx: int
    end_idx: int
    shape: Tuple[int, ...]
    causal: bool
    doc_id: int
    # Store original unpadded dimensions for cropping on decode
    original_shape: Optional[Tuple[int, ...]] = None


# =========================================================
# 2. EMBEDDING / UNEMBEDDING WRAPPERS
# =========================================================

class SpanEmbedder:
    """
    Wraps text and patch embedders to produce a flat token sequence with span tracking.

    Takes a list of ContextBlocks (mixed text and images) and produces:
    - z_flat: [L_total, D] concatenated embeddings
    - span_objects: List[Span] tracking each block's position
    - content_hashes: List[int] for prefix caching

    Supports efficient batched processing of images with the same grid_shape.
    When the patch_embedder has attention layers (n_attn_layers > 0), images
    are grouped by resolution and processed in batches with shared masks.
    """
    def __init__(
        self,
        text_embedder,
        patch_embedder,
        attn_config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            text_embedder: nn.Embedding or similar, maps token IDs to embeddings
            patch_embedder: Module with forward(x, logsnr, block_mask) -> (z, grid_shape)
            attn_config: Optional attention config for encoder masks:
                - 'mode': 'full', 'local', 'bigbird', etc.
                - 'window_size': Spatial window for local attention
                - 'n_global_tokens': Number of register tokens
        """
        self.text_emb = text_embedder
        self.patch_emb = patch_embedder
        self.attn_config = attn_config or {'mode': 'full'}
        self._mask_cache: Dict[Tuple[int, int], Optional[BlockMask]] = {}

    def _get_cached_mask(
        self,
        grid_shape: Tuple[int, int],
        device: torch.device
    ) -> Optional[BlockMask]:
        """Get or build cached mask for a given grid shape."""
        if grid_shape not in self._mask_cache:
            # Build mask using the encoder mask infrastructure
            # Provide defaults for optional keys (full mode doesn't use window_size/n_registers)
            mode = self.attn_config.get('mode', 'full')
            # Map config mode names to build_encoder_mask mode names
            if mode == 'sliding':
                mode = 'local'
            self._mask_cache[grid_shape] = build_encoder_mask(
                grid_shape=grid_shape,
                window_size=self.attn_config.get('window_size', 4.0),
                n_registers=self.attn_config.get('n_global_tokens', 0),
                mode=mode,
                device=device
            )
        return self._mask_cache[grid_shape]

    def clear_mask_cache(self):
        """Clear cached masks (call when moving to new device or changing config)."""
        self._mask_cache.clear()

    def embed(self, context_blocks: List[ContextBlock]) -> Tuple[torch.Tensor, List[Span], List[int]]:
        """
        Embed a list of ContextBlocks into a flat sequence.

        Groups images by grid_shape for efficient batched processing when
        the embedder has attention layers.

        Args:
            context_blocks: List of ContextBlock objects

        Returns:
            z_flat: [L_total, D] concatenated embeddings
            span_objects: List[Span] with position info
            content_hashes: List[int] for each token (for prefix caching)
        """
        # Check if embedder has attention (needs batching)
        has_attention = (
            hasattr(self.patch_emb, 'n_attn_layers') and
            self.patch_emb.n_attn_layers > 0
        )

        if has_attention:
            return self._embed_batched(context_blocks)
        else:
            return self._embed_sequential(context_blocks)

    def _embed_sequential(self, context_blocks: List[ContextBlock]) -> Tuple[torch.Tensor, List[Span], List[int]]:
        """Original sequential embedding (MLP-only, no attention)."""
        all_embeds = []
        span_objects = []
        cursor = 0
        hash_spans = []

        for block in context_blocks:
            original_shape = None

            if block.type == 'text':
                tokens = block.content
                if isinstance(tokens, str):
                    raise ValueError("SpanEmbedder expects tokenized text tensors, not strings.")

                emb = self.text_emb(tokens)
                span_len = tokens.shape[0]
                actual_shape = (span_len,)
                hash_spans.append({'type': 'text', 'shape': actual_shape, 'data': tokens.cpu().tolist()})

            elif block.type == 'latent':
                img = block.content
                logsnr = block.logsnr

                original_shape = img.shape[-2:]
                emb, grid_shape = self.patch_emb(img, logsnr)
                span_len = emb.shape[0]
                actual_shape = grid_shape
                hash_spans.append({'type': 'latent', 'shape': grid_shape, 'id': block.id})

            all_embeds.append(emb)
            span_objects.append(Span(
                type=block.type,
                start_idx=cursor,
                end_idx=cursor + span_len,
                shape=actual_shape,
                causal=block.causal,
                doc_id=block.group_id,
                original_shape=original_shape
            ))
            cursor += span_len

        content_hashes = generate_content_hash_stream(hash_spans)
        return torch.cat(all_embeds, dim=0), span_objects, content_hashes

    def _embed_batched(self, context_blocks: List[ContextBlock]) -> Tuple[torch.Tensor, List[Span], List[int]]:
        """Batched embedding with attention - groups images by grid_shape."""
        # Separate text and latent blocks, tracking original indices
        text_blocks = []  # (original_idx, block)
        latent_groups: Dict[Tuple[int, int], List[Tuple[int, ContextBlock]]] = {}

        for i, block in enumerate(context_blocks):
            if block.type == 'text':
                text_blocks.append((i, block))
            elif block.type == 'latent':
                # Compute grid_shape for grouping
                img = block.content
                # Use a dummy logsnr to get grid_shape via _pad_and_patch
                dummy_logsnr = torch.zeros(1, img.shape[-2], img.shape[-1], device=img.device)
                patches = self.patch_emb._pad_and_patch(img)
                if img.dim() == 4:  # batched
                    grid_shape = (patches.shape[2], patches.shape[3])
                else:  # single
                    grid_shape = (patches.shape[1], patches.shape[2])

                if grid_shape not in latent_groups:
                    latent_groups[grid_shape] = []
                latent_groups[grid_shape].append((i, block))

        # Build results dict: idx -> (emb, grid_shape, original_shape, hash_info)
        results: Dict[int, Tuple[torch.Tensor, Tuple[int, int], Optional[Tuple[int, int]], Dict]] = {}

        # Process text blocks (no batching needed)
        for orig_idx, block in text_blocks:
            tokens = block.content
            if isinstance(tokens, str):
                raise ValueError("SpanEmbedder expects tokenized text tensors, not strings.")
            emb = self.text_emb(tokens)
            span_len = tokens.shape[0]
            results[orig_idx] = (emb, (span_len,), None, {'type': 'text', 'shape': (span_len,), 'data': tokens.cpu().tolist()})

        # Process latent groups in batches
        for grid_shape, group in latent_groups.items():
            if len(group) == 1:
                # Single image, no batching benefit
                orig_idx, block = group[0]
                img = block.content
                original_shape = img.shape[-2:]
                mask = self._get_cached_mask(grid_shape, img.device)
                emb, actual_grid = self.patch_emb(img, block.logsnr, block_mask=mask)
                results[orig_idx] = (emb, actual_grid, original_shape, {'type': 'latent', 'shape': actual_grid, 'id': block.id})
            else:
                # Batch multiple images with same grid_shape
                indices = [orig_idx for orig_idx, _ in group]
                imgs = torch.stack([block.content for _, block in group], dim=0)  # [B, C, H, W]
                logsnrs = torch.stack([block.logsnr for _, block in group], dim=0)  # [B, 1, H, W]

                mask = self._get_cached_mask(grid_shape, imgs.device)
                embs_batched, actual_grid = self.patch_emb(imgs, logsnrs, block_mask=mask)  # [B, L, D]

                # Scatter results back
                for batch_idx, (orig_idx, block) in enumerate(group):
                    original_shape = block.content.shape[-2:]
                    emb = embs_batched[batch_idx]  # [L, D]
                    results[orig_idx] = (emb, actual_grid, original_shape, {'type': 'latent', 'shape': actual_grid, 'id': block.id})

        # Reassemble in original order
        all_embeds = []
        span_objects = []
        hash_spans = []
        cursor = 0

        for i, block in enumerate(context_blocks):
            emb, actual_shape, original_shape, hash_info = results[i]
            span_len = emb.shape[0]

            all_embeds.append(emb)
            hash_spans.append(hash_info)
            span_objects.append(Span(
                type=block.type,
                start_idx=cursor,
                end_idx=cursor + span_len,
                shape=actual_shape,
                causal=block.causal,
                doc_id=block.group_id,
                original_shape=original_shape
            ))
            cursor += span_len

        content_hashes = generate_content_hash_stream(hash_spans)
        return torch.cat(all_embeds, dim=0), span_objects, content_hashes


class SpanUnembedder:
    """
    Decodes a flat embedding sequence back to per-span outputs.

    Uses span position info to slice the sequence and decode each span
    with the appropriate head (text or image).

    Supports efficient batched processing of images with the same grid_shape
    when the patch_unembedder has attention layers.
    """
    def __init__(
        self,
        text_head,
        patch_unembedder,
        attn_config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            text_head: Linear layer mapping embeddings to vocab logits
            patch_unembedder: Module with forward(z, shape, block_mask) -> [C+1, H, W]
            attn_config: Optional attention config for encoder masks
        """
        self.text_head = text_head
        self.patch_unembed = patch_unembedder
        self.attn_config = attn_config or {'mode': 'full'}
        self._mask_cache: Dict[Tuple[int, int], Optional[BlockMask]] = {}

    def _get_cached_mask(
        self,
        grid_shape: Tuple[int, int],
        device: torch.device
    ) -> Optional[BlockMask]:
        """Get or build cached mask for a given grid shape."""
        if grid_shape not in self._mask_cache:
            # Provide defaults for optional keys (full mode doesn't use window_size/n_registers)
            mode = self.attn_config.get('mode', 'full')
            # Map config mode names to build_encoder_mask mode names
            if mode == 'sliding':
                mode = 'local'
            self._mask_cache[grid_shape] = build_encoder_mask(
                grid_shape=grid_shape,
                window_size=self.attn_config.get('window_size', 4.0),
                n_registers=self.attn_config.get('n_global_tokens', 0),
                mode=mode,
                device=device
            )
        return self._mask_cache[grid_shape]

    def clear_mask_cache(self):
        """Clear cached masks."""
        self._mask_cache.clear()

    def decode(self, z: torch.Tensor, spans: List[Span]) -> List[Dict[str, Any]]:
        """
        Decode embeddings to per-span outputs.

        Groups latent spans by grid_shape for efficient batched processing
        when the unembedder has attention layers.

        Args:
            z: [L_total, D] flat embedding sequence
            spans: List[Span] from SpanEmbedder

        Returns:
            List of dicts, one per span, containing:
            - 'text_logits': [L, vocab_size] for all spans
            - 'image_vpreds': [C, H, W] for latent spans (cropped to original)
            - 'image_logsnrs': [1, H, W] for latent spans (cropped to original)
        """
        # Check if unembedder has attention (needs batching)
        has_attention = (
            hasattr(self.patch_unembed, 'n_attn_layers') and
            self.patch_unembed.n_attn_layers > 0
        )

        if has_attention:
            return self._decode_batched(z, spans)
        else:
            return self._decode_sequential(z, spans)

    def _decode_sequential(self, z: torch.Tensor, spans: List[Span]) -> List[Dict[str, Any]]:
        """Original sequential decoding (MLP-only, no attention)."""
        outputs = []
        for span in spans:
            spandict = {}
            z_span = z[span.start_idx:span.end_idx]

            # Text Head (Always computable)
            spandict['text_logits'] = self.text_head(z_span)

            # Latent Head
            if span.type == 'latent':
                reconstruction = self.patch_unembed(z_span, span.shape)

                if span.original_shape is not None:
                    orig_h, orig_w = span.original_shape
                    reconstruction = reconstruction[:, :orig_h, :orig_w]

                spandict['image_vpreds'] = reconstruction[:-1]
                spandict['image_logsnrs'] = reconstruction[-1:]

            outputs.append(spandict)
        return outputs

    def _decode_batched(self, z: torch.Tensor, spans: List[Span]) -> List[Dict[str, Any]]:
        """Batched decoding with attention - groups latent spans by grid_shape."""
        device = z.device

        # Group latent spans by grid_shape
        latent_groups: Dict[Tuple[int, int], List[Tuple[int, Span]]] = {}
        for i, span in enumerate(spans):
            if span.type == 'latent':
                grid_shape = span.shape
                if grid_shape not in latent_groups:
                    latent_groups[grid_shape] = []
                latent_groups[grid_shape].append((i, span))

        # Pre-compute text logits and store results
        results: Dict[int, Dict[str, Any]] = {}

        for i, span in enumerate(spans):
            z_span = z[span.start_idx:span.end_idx]
            results[i] = {'text_logits': self.text_head(z_span)}

        # Process latent groups in batches
        for grid_shape, group in latent_groups.items():
            if len(group) == 1:
                # Single span, no batching
                span_idx, span = group[0]
                z_span = z[span.start_idx:span.end_idx]
                mask = self._get_cached_mask(grid_shape, device)
                reconstruction = self.patch_unembed(z_span, span.shape, block_mask=mask)

                if span.original_shape is not None:
                    orig_h, orig_w = span.original_shape
                    reconstruction = reconstruction[:, :orig_h, :orig_w]

                results[span_idx]['image_vpreds'] = reconstruction[:-1]
                results[span_idx]['image_logsnrs'] = reconstruction[-1:]
            else:
                # Batch multiple spans with same grid_shape
                span_embeddings = []
                for span_idx, span in group:
                    z_span = z[span.start_idx:span.end_idx]
                    span_embeddings.append(z_span)

                z_batch = torch.stack(span_embeddings, dim=0)  # [B, L, D]
                mask = self._get_cached_mask(grid_shape, device)
                reconstructions = self.patch_unembed(z_batch, grid_shape, block_mask=mask)  # [B, C+1, H, W]

                # Scatter results back
                for batch_idx, (span_idx, span) in enumerate(group):
                    reconstruction = reconstructions[batch_idx]  # [C+1, H, W]

                    if span.original_shape is not None:
                        orig_h, orig_w = span.original_shape
                        reconstruction = reconstruction[:, :orig_h, :orig_w]

                    results[span_idx]['image_vpreds'] = reconstruction[:-1]
                    results[span_idx]['image_logsnrs'] = reconstruction[-1:]

        # Return in original order
        return [results[i] for i in range(len(spans))]


# =========================================================
# 3. CONTENT IDENTITY (Hashing for Prefix Cache)
# =========================================================

def generate_content_hash_stream(spans: List[Any]) -> List[int]:
    """
    Transforms a list of Spans (or dicts) into a linear stream of Atomic Content IDs.
    These IDs are used by the BlockManager to detect identical content (Prefix Caching).

    For text: Each token's ID is used directly
    For latents: Hash(span_id, relative_index) creates unique per-patch IDs
    """
    stream = []

    for span in spans:
        # Support both Dataclass and Dict interfaces
        if isinstance(span, dict):
            span_type = span.get('type', 'latent')
            shape = span['shape']
            span_id = span.get('id', 0)
            data = span.get('data', None)
        else:
            span_type = getattr(span, 'type', 'latent')
            shape = getattr(span, 'shape', ())
            span_id = getattr(span, 'id', 0)
            data = getattr(span, 'data', None)

        num_tokens = math.prod(shape)

        if span_type == 'text':
            # Text Identity = The Token ID itself
            if data is None:
                raise ValueError("Text spans must provide 'data' (token IDs) for hashing.")

            # Ensure data is flat list
            if hasattr(data, 'tolist'):
                data = data.tolist()

            stream.extend([int(t) for t in data])

        elif span_type == 'latent':
            # Latent Identity = Hash(Unique_Span_ID, Relative_Index)
            if isinstance(span_id, str):
                seed = xxhash.xxh64(span_id).intdigest()
            else:
                seed = int(span_id)

            # Generate deterministic stream: Hash(Seed + Index)
            base_hasher = xxhash.xxh64(seed=seed)
            for i in range(num_tokens):
                base_hasher.reset()
                base_hasher.update(i.to_bytes(8, 'little'))
                stream.append(base_hasher.intdigest())

    return stream


# =========================================================
# 4. GEOMETRY (Topology Rendering)
# =========================================================

def render_topology_embeddings(
    spans: List[Span],
    max_dims: int,
    device: torch.device,
    highway_offset: int = 0,
    dtype: torch.dtype = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Renders Global Topology coordinates for RnRoPE.

    Creates a tensor where:
    - Column 0: "highway" - monotonic sequential position across all tokens
    - Columns 1+: Spatial coordinates (grid coords for images, zeros for text)

    Text tokens exist at spatial origin (0,0,...) but still have unique highway positions.
    Images get actual grid coordinates in the spatial columns.

    Args:
        spans: List of Span objects
        max_dims: Total topology dimensions (1 + spatial_dims)
        device: Target device
        highway_offset: Starting highway value (for sequence continuation)
        dtype: If None, defaults to torch.float32

    Returns:
        topo_embeds: [L_total, max_dims] topology coordinates
        doc_ids: [L_total] document IDs per token
    """
    if dtype is None:
        dtype = torch.float32

    highway_idx = []
    manifold_coords = []
    doc_ids = []

    current_highway = highway_offset
    spatial_dim_capacity = max_dims - 1

    for i, span in enumerate(spans):
        # Flattened length
        if span.type == 'text':
            num_tokens = span.shape[0]
        else:
            num_tokens = math.prod(span.shape)  # e.g. H*W

        # 1. Highway (Shared Global Time)
        h_range = torch.arange(current_highway, current_highway + num_tokens, device=device, dtype=dtype)
        highway_idx.append(h_range)
        current_highway += num_tokens

        # 2. Manifold (Spatial)
        if span.type == 'text':
            # Text exists at the "singularity" (0,0) of the spatial manifold
            coords = torch.zeros((num_tokens, spatial_dim_capacity), device=device, dtype=dtype)
        else:
            # Latents exist on a grid
            dims = [torch.arange(d, device=device, dtype=dtype) for d in span.shape]
            mesh = torch.meshgrid(*dims, indexing='ij')
            coords = torch.stack([m.flatten() for m in mesh], dim=-1)

            # Pad spatial dims if needed (e.g., 2D grid in 3D manifold)
            curr_dim = coords.shape[-1]
            if curr_dim < spatial_dim_capacity:
                padding = torch.zeros((num_tokens, spatial_dim_capacity - curr_dim), device=device, dtype=dtype)
                coords = torch.cat([coords, padding], dim=-1)

        manifold_coords.append(coords)
        doc_ids.append(torch.full((num_tokens,), span.doc_id, device=device, dtype=torch.int32))

    # Stack - dtype already correct
    flat_highway = torch.cat(highway_idx).unsqueeze(-1)
    flat_manifold = torch.cat(manifold_coords)
    topo_embeds = torch.cat([flat_highway, flat_manifold], dim=-1)
    flat_doc_ids = torch.cat(doc_ids)

    return topo_embeds, flat_doc_ids


def render_latent_topology_embeddings(
    n_patches: int,
    n_levels: int,
    grid_shape: Tuple[int, int],
    device: torch.device,
    highway_offset: int = 0,
    level_scale: float = 1.0,
    dtype: torch.dtype = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Renders topology coordinates for latent diffusion with multi-level codes.

    Creates 4D topology: [highway, spatial_x, spatial_y, level] where:
    - highway: monotonic position across all (patch, level) tokens
    - spatial_x, spatial_y: grid coordinates (same across levels for same patch)
    - level: residual quantization level (scaled by level_scale)

    Token ordering: patches are grouped by level (all patches at level 0, then level 1, etc.)
    This matches the natural output of hierarchical AE encoding.

    Args:
        n_patches: Number of spatial patches per level (H * W)
        n_levels: Number of residual quantization levels
        grid_shape: (H, W) spatial grid dimensions
        device: Target device
        highway_offset: Starting highway position
        level_scale: Scaling factor for level coordinates in RoPE
        dtype: Output dtype (default float32)

    Returns:
        topo_embeds: [n_patches * n_levels, 4] topology coordinates
        level_ids: [n_patches * n_levels] level index per token
        patch_ids: [n_patches * n_levels] patch index per token
    """
    if dtype is None:
        dtype = torch.float32

    H, W = grid_shape
    total_tokens = n_patches * n_levels

    # Build spatial grid coordinates (shared across levels)
    grid_y = torch.arange(H, device=device, dtype=dtype)
    grid_x = torch.arange(W, device=device, dtype=dtype)
    mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
    spatial_coords = torch.stack([mesh_x.flatten(), mesh_y.flatten()], dim=-1)  # [n_patches, 2]

    # Build per-token coordinates
    highway = torch.arange(highway_offset, highway_offset + total_tokens, device=device, dtype=dtype)

    # Tile spatial coords for each level
    spatial_tiled = spatial_coords.repeat(n_levels, 1)  # [n_patches * n_levels, 2]

    # Level coordinates: [0, 0, ..., 1, 1, ..., 2, 2, ...] each repeated n_patches times
    level_coords = torch.arange(n_levels, device=device, dtype=dtype).repeat_interleave(n_patches)
    level_coords = level_coords * level_scale

    # Concatenate: [highway, spatial_x, spatial_y, level]
    topo_embeds = torch.cat([
        highway.unsqueeze(-1),
        spatial_tiled,
        level_coords.unsqueeze(-1)
    ], dim=-1)  # [total_tokens, 4]

    # Auxiliary indices for reconstruction
    level_ids = torch.arange(n_levels, device=device).repeat_interleave(n_patches)
    patch_ids = torch.arange(n_patches, device=device).repeat(n_levels)

    return topo_embeds, level_ids, patch_ids


def compute_latent_distance_squared(
    q_spatial: Tuple[torch.Tensor, ...],
    k_spatial: Tuple[torch.Tensor, ...],
    q_level: torch.Tensor,
    k_level: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    level_lambda: float = 0.5,
    vertical_free: bool = True
) -> torch.Tensor:
    """
    Compute distance² for latent diffusion with level-aware metric.

    dist² = spatial_dist² + (level_lambda * level_dist)²

    With vertical_free=True, same-position cross-level attention has dist²=0
    (creates "vertical tubes" through the level stack).

    Args:
        q_spatial: Tuple of spatial coordinate tensors for queries
        k_spatial: Tuple of spatial coordinate tensors for keys
        q_level: Level coordinates for queries
        k_level: Level coordinates for keys
        q_idx: Query indices
        kv_idx: Key indices
        level_lambda: Scaling for level distance
        vertical_free: If True, same-position cross-level has zero distance

    Returns:
        dist_sq: Distance squared tensor
    """
    # Spatial distance
    spatial_dist_sq = 0.0
    for q_col, k_col in zip(q_spatial, k_spatial):
        d = q_col[q_idx] - k_col[kv_idx]
        spatial_dist_sq = spatial_dist_sq + (d * d)

    # Level distance (scaled)
    level_d = q_level[q_idx] - k_level[kv_idx]
    level_dist_sq = (level_lambda * level_d) ** 2

    if vertical_free:
        # Same spatial position across levels = zero distance (vertical tube)
        same_position = (spatial_dist_sq == 0.0)
        # Return spatial distance only for same-position (which is 0),
        # otherwise return full distance
        return torch.where(same_position, spatial_dist_sq, spatial_dist_sq + level_dist_sq)
    else:
        return spatial_dist_sq + level_dist_sq


# =========================================================
# 5. CONNECTIVITY (Mask Construction)
# =========================================================

def build_dual_masks(
    spans: List[Span],
    topo_active: torch.Tensor,
    topo_heap: torch.Tensor,
    page_table: Optional[PageTable] = None,
    flat_page_table: Optional[torch.Tensor] = None,
    inverse_page_table: Optional[torch.Tensor] = None,
    window_size: float = 10.0,
    return_mask_closures: bool = False
) -> Tuple[BlockMask, BlockMask]:
    """
    Build local (spatially-windowed) and global attention masks.

    The masks respect:
    - Document isolation: Tokens only attend within their document
    - Block-causal: Later spans can attend to earlier spans
    - Intra-span causality: Respects per-span causal flag
    - Spatial window (local only): Limits attention by spatial distance

    Args:
        spans: List of Span objects defining the sequence structure
        topo_active: [L_active, topo_dim] topology for query positions
        topo_heap: [L_heap, topo_dim] topology for key/value positions
        page_table: PageTable for physical->logical translation (inference)
        flat_page_table: [N_blocks] logical->physical mapping (training)
        inverse_page_table: [Capacity] physical->logical mapping (training)
        window_size: Spatial distance threshold for local attention
        return_mask_closures: If True, also return debug dict with mask_mod functions

    Returns:
        local_mask: BlockMask with spatial windowing
        global_mask: BlockMask without spatial constraints
        (optional) debug_dict: Contains mask_mod functions for analysis
    """
    device = topo_active.device
    L_active = topo_active.shape[0]
    L_heap = topo_heap.shape[0]
    block_size = page_table.block_size

    # 1. Build doc_ids for ACTIVE tokens
    doc_ids_active = []
    span_ids_active = []
    causal_modes_active = []
    for i, span in enumerate(spans):
        span_len = span.end_idx - span.start_idx
        doc_ids_active.extend([span.doc_id] * span_len)
        span_ids_active.extend([i] * span_len)
        causal_modes_active.extend([span.causal] * span_len)

    doc_ids_active_t = torch.tensor(doc_ids_active, dtype=torch.long, device=device)
    span_ids_active_t = torch.tensor(span_ids_active, dtype=torch.long, device=device)
    causal_modes_active_t = torch.tensor(causal_modes_active, dtype=torch.bool, device=device)

    # 2. Build doc_ids for HEAP
    L_heap = topo_heap.shape[0]
    doc_ids_heap_t = torch.full((L_heap,), -1, dtype=torch.long, device=device)
    span_ids_heap_t = torch.full((L_heap,), -1, dtype=torch.long, device=device)
    block_size = page_table.block_size
    cursor = 0

    for i, span in enumerate(spans):
        span_len = span.end_idx - span.start_idx

        # Trivial Case (Training/ZC)
        if flat_page_table is None:
            doc_ids_heap_t[cursor:cursor+span_len] = span.doc_id
            span_ids_heap_t[cursor:cursor+span_len] = i
        else:
            # Inference Case (Paged) - Iterate Logical Blocks
            start_block = cursor // block_size
            end_block = (cursor + span_len - 1) // block_size + 1

            for log_block_idx in range(start_block, end_block):
                if log_block_idx >= len(flat_page_table):
                    break

                phys_block = flat_page_table[log_block_idx].item()

                # Intersection of Span and Block
                block_start_global = log_block_idx * block_size
                block_end_global = (log_block_idx + 1) * block_size

                start_in_span = max(0, block_start_global - cursor)
                end_in_span = min(span_len, block_end_global - cursor)

                # Global offsets
                global_start = cursor + start_in_span
                global_end = cursor + end_in_span

                # Physical offsets
                offset_start = global_start % block_size
                offset_end = offset_start + (end_in_span - start_in_span)

                phys_start = phys_block * block_size + offset_start
                phys_end = phys_block * block_size + offset_end

                doc_ids_heap_t[phys_start:phys_end] = span.doc_id
                span_ids_heap_t[phys_start:phys_end] = i

        cursor += span_len

    # 3. Decompose Topology
    topo_active_cols = topo_active.unbind(dim=-1)
    highway_active = topo_active_cols[0]
    spatial_active = topo_active_cols[1:]

    topo_heap_cols = topo_heap.unbind(dim=-1)
    highway_heap = topo_heap_cols[0]
    spatial_heap = topo_heap_cols[1:]

    win_sq = torch.tensor(window_size * window_size, device=device, dtype=topo_active.dtype)

    # 4. Core Connectivity Logic
    def base_connectivity(q_idx, kv_idx):
        # 1. Document Separation
        q_doc = doc_ids_active_t[q_idx]
        k_doc = doc_ids_heap_t[kv_idx]
        same_doc = (q_doc == k_doc)

        # 2. Span Identification
        q_span = span_ids_active_t[q_idx]
        k_span = span_ids_heap_t[kv_idx]

        # 3. Block Causal Logic (Global Hierarchy)
        block_condition = (q_span > k_span)
        same_span = (q_span == k_span)

        # 4. Intra-Span Logic (Local Visibility)
        is_ar = causal_modes_active_t[q_idx]
        # If AR: Enforce Time. If BiDir: Allow All.
        internal_condition = (~is_ar) | (highway_active[q_idx] >= highway_heap[kv_idx])

        # 5. Composition
        # Visible if: (Same Doc) AND ((Strictly Past Span) OR (Same Span AND Internal Condition))
        valid_connection = block_condition | (same_span & internal_condition)

        return same_doc & valid_connection

    # 5. Local Mod (Spatial Window)
    def mask_mod_local(b, h, q_idx, kv_idx):
        base = base_connectivity(q_idx, kv_idx)

        dist_sq = 0.0
        for q_col, k_col in zip(spatial_active, spatial_heap):
            d = q_col[q_idx] - k_col[kv_idx]
            dist_sq = dist_sq + (d * d)

        spatial_ok = dist_sq < win_sq
        return base & spatial_ok

    # 6. Global Mod (Infinite Window)
    def mask_mod_global(b, h, q_idx, kv_idx):
        return base_connectivity(q_idx, kv_idx)

    # 7. Compile BlockMasks
    local_mask = create_block_mask(
        mask_mod_local, B=None, H=None, Q_LEN=L_active, KV_LEN=L_heap
    )
    global_mask = create_block_mask(
        mask_mod_global, B=None, H=None, Q_LEN=L_active, KV_LEN=L_heap
    )

    if return_mask_closures:
        debug_dict = {'mask_mod_local': mask_mod_local, 'mask_mod_global': mask_mod_global}
        return local_mask, global_mask, debug_dict
    else:
        return local_mask, global_mask


def materialize_mask_for_analysis(spans: List[Span], topo_active: torch.Tensor) -> torch.Tensor:
    """
    Materialize a full attention mask matrix for debugging/visualization.

    Warning: O(L^2) memory - only use for small sequences.
    """
    from .paging import PageTable

    # Create minimal page_table for the helper
    L = topo_active.shape[0]
    dummy_page_table = PageTable(
        num_blocks=max(1, (L + 127) // 128),
        block_size=128,
        max_batch_size=1,
        max_logical_blocks=max(1, (L + 127) // 128),
        device=topo_active.device
    )

    _, _, debug = build_dual_masks(spans, topo_active, topo_active, dummy_page_table, return_mask_closures=True)
    mod = debug['mask_mod_global']

    dev = topo_active.device
    q = torch.arange(L, device=dev).unsqueeze(1).expand(L, L)
    k = torch.arange(L, device=dev).unsqueeze(0).expand(L, L)
    return mod(0, 0, q, k)


def build_latent_diffusion_mask(
    n_patches: int,
    n_levels: int,
    grid_shape: Tuple[int, int],
    window_size: float = 4.0,
    level_lambda: float = 0.5,
    vertical_free: bool = True,
    mode: str = 'local',
    device: torch.device = None,
    dtype: torch.dtype = None
) -> Optional[BlockMask]:
    """
    Build attention mask for latent diffusion over multi-level AE codes.

    Creates masks for the flattened [n_patches * n_levels] token sequence where
    each token represents one (patch_position, residual_level) tuple.

    Distance metric: dist² = spatial_dist² + (level_lambda * level_dist)²
    With vertical_free=True, same-position tokens across levels have zero distance.

    Args:
        n_patches: Number of spatial patches per level
        n_levels: Number of residual quantization levels
        grid_shape: (H, W) spatial grid dimensions
        window_size: Spatial distance threshold for local attention
        level_lambda: Scaling factor for level distance in metric
        vertical_free: If True, same-position cross-level attention is always allowed
        mode: 'full', 'local', or 'bidirectional'
        device: Target device
        dtype: Coordinate dtype

    Returns:
        BlockMask for flex_attention, or None for full attention
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dtype is None:
        dtype = torch.float32

    if mode == 'full':
        return None

    H, W = grid_shape
    total_tokens = n_patches * n_levels

    # Build coordinate tensors (must be tensors for torch.compile compatibility)
    # Spatial coordinates: tiled for each level
    grid_y = torch.arange(H, device=device, dtype=dtype)
    grid_x = torch.arange(W, device=device, dtype=dtype)
    mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
    spatial_x = mesh_x.flatten().repeat(n_levels)  # [total_tokens]
    spatial_y = mesh_y.flatten().repeat(n_levels)  # [total_tokens]

    # Level coordinates: [0, 0, ..., 1, 1, ..., n_levels-1, ...]
    level_coords = torch.arange(n_levels, device=device, dtype=dtype).repeat_interleave(n_patches)

    # Convert scalars to tensors for closure capture
    win_sq = torch.tensor(window_size * window_size, device=device, dtype=dtype)
    level_lam = torch.tensor(level_lambda, device=device, dtype=dtype)
    vert_free = vertical_free  # Python bool is fine for branch

    def mask_mod_latent(b, h, q_idx, kv_idx):
        # Spatial distance squared
        dx = spatial_x[q_idx] - spatial_x[kv_idx]
        dy = spatial_y[q_idx] - spatial_y[kv_idx]
        spatial_dist_sq = dx * dx + dy * dy

        # Level distance (scaled)
        dl = level_coords[q_idx] - level_coords[kv_idx]
        level_dist_sq = (level_lam * dl) ** 2

        if vert_free:
            # Vertical tube: same spatial position = zero effective distance
            same_position = (spatial_dist_sq == 0.0)
            effective_dist_sq = torch.where(same_position, spatial_dist_sq, spatial_dist_sq + level_dist_sq)
        else:
            effective_dist_sq = spatial_dist_sq + level_dist_sq

        return effective_dist_sq < win_sq

    return create_block_mask(
        mask_mod_latent, B=None, H=None, Q_LEN=total_tokens, KV_LEN=total_tokens
    )


# Cache for latent diffusion masks (keyed by geometry parameters)
_latent_mask_cache: Dict[Tuple, Optional[BlockMask]] = {}


def get_cached_latent_mask(
    n_patches: int,
    n_levels: int,
    grid_shape: Tuple[int, int],
    window_size: float,
    level_lambda: float,
    vertical_free: bool,
    mode: str,
    device: torch.device
) -> Optional[BlockMask]:
    """
    Get or build cached latent diffusion mask.

    Caching is critical for training efficiency - mask construction is expensive
    and should happen once per unique geometry configuration.
    """
    key = (n_patches, n_levels, grid_shape, window_size, level_lambda, vertical_free, mode, str(device))

    if key not in _latent_mask_cache:
        _latent_mask_cache[key] = build_latent_diffusion_mask(
            n_patches=n_patches,
            n_levels=n_levels,
            grid_shape=grid_shape,
            window_size=window_size,
            level_lambda=level_lambda,
            vertical_free=vertical_free,
            mode=mode,
            device=device
        )

    return _latent_mask_cache[key]


def clear_latent_mask_cache():
    """Clear the latent diffusion mask cache."""
    _latent_mask_cache.clear()


# =========================================================
# 6. ENCODER MASKS (Bidirectional for Image Encoders/Decoders)
# =========================================================

def build_encoder_mask(
    grid_shape: Tuple[int, int],
    window_size: float = 4.0,
    n_registers: int = 0,
    mode: str = 'full',
    device: torch.device = None,
    dtype: torch.dtype = None
) -> Optional[BlockMask]:
    """
    Build attention mask for image encoder/decoder with bidirectional attention.

    This function creates masks compatible with flex_attention for processing
    image patches. All attention is symmetric/bidirectional (non-causal).

    CRITICAL: All values captured in mask_mod closures are tensors, not Python
    ints. This ensures torch.compile doesn't recompile for different sizes.

    Args:
        grid_shape: (H, W) grid dimensions of patches
        window_size: Spatial distance for local attention (Euclidean, not Manhattan)
        n_registers: Number of global register tokens prepended (for BigBird)
        mode: 'full', 'local', or 'bigbird'
            - 'full': Returns None (use default full attention)
            - 'local': Sliding window based on spatial distance
            - 'bigbird': Local attention + global registers
        device: Target device
        dtype: Coordinate dtype (default float32)

    Returns:
        BlockMask or None (for full attention)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dtype is None:
        dtype = torch.float32

    # Full attention needs no mask
    if mode == 'full':
        return None

    H, W = grid_shape
    n_patches = H * W
    L = n_registers + n_patches

    # Build spatial coordinates as TENSORS (critical for avoiding recompilation)
    # Registers are at spatial origin (0, 0)
    # Patches get grid coordinates
    # Build as SEPARATE 1D tensors to avoid double-indexing in closures
    # (double-indexing like spatial_coords[q_idx][0] causes inductor bounds analysis to use infinity)

    # Grid coordinates for patches
    h_range = torch.arange(H, device=device, dtype=dtype)
    w_range = torch.arange(W, device=device, dtype=dtype)
    mesh_h, mesh_w = torch.meshgrid(h_range, w_range, indexing='ij')
    patch_h = mesh_h.flatten()  # [n_patches]
    patch_w = mesh_w.flatten()  # [n_patches]

    # Prepend register coordinates (all at origin)
    if n_registers > 0:
        reg_h = torch.zeros(n_registers, device=device, dtype=dtype)
        reg_w = torch.zeros(n_registers, device=device, dtype=dtype)
        coords_h = torch.cat([reg_h, patch_h], dim=0)  # [L]
        coords_w = torch.cat([reg_w, patch_w], dim=0)  # [L]
    else:
        coords_h = patch_h  # [L]
        coords_w = patch_w  # [L]

    # Window threshold as tensor
    win_sq = torch.tensor(window_size * window_size, device=device, dtype=dtype)
    n_reg_tensor = torch.tensor(n_registers, device=device, dtype=torch.long)

    if mode == 'local':
        # Pure local/sliding window attention
        def mask_mod_local(b, h, q_idx, kv_idx):
            # Single-level indexing only - avoids double-indexing bounds issues
            diff_h = coords_h[q_idx] - coords_h[kv_idx]
            diff_w = coords_w[q_idx] - coords_w[kv_idx]
            dist_sq = diff_h * diff_h + diff_w * diff_w
            return dist_sq <= win_sq

        return create_block_mask(
            mask_mod_local, B=None, H=None, Q_LEN=L, KV_LEN=L
        )

    elif mode == 'bigbird':
        # BigBird: local attention + global register tokens
        def mask_mod_bigbird(b, h, q_idx, kv_idx):
            # Global tokens: registers can see and be seen by everyone
            q_is_reg = q_idx < n_reg_tensor
            k_is_reg = kv_idx < n_reg_tensor

            # Local: spatial distance for non-register tokens
            # Single-level indexing only - avoids double-indexing bounds issues
            diff_h = coords_h[q_idx] - coords_h[kv_idx]
            diff_w = coords_w[q_idx] - coords_w[kv_idx]
            dist_sq = diff_h * diff_h + diff_w * diff_w
            local_ok = dist_sq <= win_sq

            return q_is_reg | k_is_reg | local_ok

        return create_block_mask(
            mask_mod_bigbird, B=None, H=None, Q_LEN=L, KV_LEN=L
        )

    else:
        raise ValueError(f"Unknown encoder mask mode: {mode}")


def get_encoder_mask_for_layer(
    grid_shape: Tuple[int, int],
    layer_idx: int,
    attn_config: Dict[str, Any],
    device: torch.device = None
) -> Optional[BlockMask]:
    """
    Get the appropriate mask for a specific encoder layer based on config.

    Supports the same attention patterns as the main transformer:
        - 'full': All layers use full attention
        - 'sliding': All layers use sliding window
        - 'bigbird': All layers use BigBird pattern
        - 'gemma': Alternating local/global (every Nth layer is global)
        - 'gemma_bigbird': Alternating sliding/bigbird (configurable layout)

    Args:
        grid_shape: (H, W) patch grid dimensions
        layer_idx: Current layer index
        attn_config: Dict with keys:
            - 'mode': Overall attention strategy
            - 'window_size': Spatial window for local attention
            - 'n_global_tokens': Number of register tokens for bigbird modes
            - 'global_layer_interval': For gemma mode, every Nth layer is global
            - 'bigbird_layout': For gemma_bigbird, (n_local, n_bigbird) per cycle
        device: Target device

    Returns:
        BlockMask or None for this layer
    """
    mode = attn_config['mode']
    window_size = attn_config['window_size']
    n_registers = attn_config['n_global_tokens']
    global_interval = attn_config['global_layer_interval']
    bigbird_layout = tuple(attn_config['bigbird_layout'])

    # Determine layer-specific mode
    if mode == 'full':
        layer_mode = 'full'
    elif mode == 'sliding':
        layer_mode = 'local'
    elif mode == 'bigbird':
        layer_mode = 'bigbird'
    elif mode == 'gemma':
        # Every Nth layer is global, rest are local
        is_global = ((layer_idx + 1) % global_interval == 0)
        layer_mode = 'full' if is_global else 'local'
    elif mode == 'gemma_bigbird':
        # Cycle: n_local sliding layers, then n_bigbird bigbird layers
        n_local, n_bigbird = bigbird_layout
        cycle_len = n_local + n_bigbird
        cycle_pos = layer_idx % cycle_len
        # Note: layer_mode determines attention PATTERN, not register presence
        # Registers are always in the sequence if n_registers > 0
        layer_mode = 'local' if cycle_pos < n_local else 'bigbird'
    else:
        layer_mode = 'full'

    # CRITICAL: Registers are ALWAYS in the sequence if n_registers > 0,
    # regardless of layer_mode. The mask must account for them.
    # layer_mode only determines the attention PATTERN (local window vs global).

    return build_encoder_mask(
        grid_shape=grid_shape,
        window_size=window_size,
        n_registers=n_registers,  # Always pass, not conditional on layer_mode
        mode=layer_mode,
        device=device
    )
