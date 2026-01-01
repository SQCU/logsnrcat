"""
Sparse Sprite Atlas Iterator

Provides training data from sparse sprite atlas datasets.
Sprites are organized as spritesheets (texture atlases) where:
- Each sheet contains sprites with a specific "head" index
- Grid positions correspond to different "body" indices
- Naming: HEAD.BODY.png (e.g., 1.25.png = head=1, body=25)

Expected data layout:
    data/sprite_atlas/
        {split}/spritesheets/   # Split-organized sheets
            1.png, 1a.png, ...  # Head=1 sprites (with variants)
            2.png, ...          # Head=2 sprites
        metadata/
            sprite_credits.csv  # Artist credits and tags
            entries.json        # Text descriptions
"""
import os
import json
import csv
import zipfile
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator
from dataclasses import dataclass
import random

import torch
import torch.nn.functional as F

# Try to import torchvision for image loading
try:
    from torchvision.io import read_image, ImageReadMode
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


@dataclass
class SpriteMetadata:
    """Metadata for a single sprite in the atlas."""
    head_id: int
    body_id: int
    variant: str  # '', 'a', 'b', etc.
    artist: str
    sprite_type: str  # 'main', 'alt', 'temp'
    tags: List[str]
    description: Optional[str] = None


@dataclass
class SpritesheetInfo:
    """Info about a spritesheet file."""
    path: Path
    head_id: int
    variant: str
    width: int
    height: int
    sprite_size: int = 96  # Each sprite is 96x96
    cols: int = 10  # Sprites per row
    rows: int = 51  # Total rows


def get_png_dimensions(path: Path) -> Tuple[int, int]:
    """Read PNG dimensions without loading full image."""
    with open(path, 'rb') as f:
        data = f.read(24)
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError(f"Not a PNG file: {path}")
    w = struct.unpack('>I', data[16:20])[0]
    h = struct.unpack('>I', data[20:24])[0]
    return w, h


class SpriteAtlasDataset:
    """
    Dataset manager for sparse sprite atlas data.

    Handles extraction from zip, metadata parsing, and sprite loading.

    Spritesheet format:
    - Each sheet is for a specific "head" index (filename = head_id.png)
    - 96x96 pixel sprites arranged in a variable-column grid
    - Grid position (idx) maps to body_id: body_id = idx + 1
    - Example: 1.png row=2,col=4 (idx=24) = fusion 1.25
    """

    # Standard sprite size in the spritesheets
    SPRITE_SIZE = 96
    GRID_COLS = 10  # Sprites per row

    # Maximum head/body ID in the atlas
    MAX_SPRITE_ID = 565

    def __init__(
        self,
        data_dir: str = "data/sprite_atlas",
        zip_path: Optional[str] = None,
        sprite_size: int = 288,
        device: torch.device = None
    ):
        self.data_dir = Path(data_dir)
        self.zip_path = Path(zip_path) if zip_path else None
        self.sprite_size = sprite_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.spritesheets_dir = self.data_dir / "spritesheets"
        self.metadata_dir = self.data_dir / "metadata"

        # Loaded metadata
        self.sprite_credits: Dict[str, SpriteMetadata] = {}
        self.dex_entries: Dict[str, List[str]] = {}  # fusion_id -> list of entries
        self.sprite_names: Dict[int, str] = {}

        # Spritesheet index
        self.spritesheets: Dict[str, SpritesheetInfo] = {}

        # Check if data is extracted
        if not self.spritesheets_dir.exists():
            if self.zip_path and self.zip_path.exists():
                print(f"Extracting data from {self.zip_path}...")
                self._extract_from_zip()
            else:
                raise FileNotFoundError(
                    f"Data not found at {self.data_dir}. "
                    f"Provide zip_path to extract, or extract manually."
                )

        self._load_metadata()
        self._index_spritesheets()

    def _extract_from_zip(self):
        """Extract spritesheets and metadata from the sprite archive zip."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.spritesheets_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            # Extract spritesheets from CustomBattlers/spritesheets/
            print("  Extracting spritesheets...")
            sheet_paths = [
                n for n in zf.namelist()
                if 'CustomBattlers/spritesheets/' in n
                and n.endswith('.png')
                and '/spritesheets_custom/' not in n  # Skip nested custom
            ]

            # Also get custom sheets
            custom_paths = [
                n for n in zf.namelist()
                if '/spritesheets_custom/' in n and n.endswith('.png')
            ]

            # Extract base spritesheets
            for path in sheet_paths:
                name = path.split('/')[-1]
                out_path = self.spritesheets_dir / name
                if not out_path.exists():
                    data = zf.read(path)
                    out_path.write_bytes(data)

            # Extract custom sheets to subdirectories
            for path in custom_paths:
                parts = path.split('/spritesheets_custom/')[-1].split('/')
                if len(parts) == 2:
                    head_id, name = parts
                    subdir = self.spritesheets_dir / "custom" / head_id
                    subdir.mkdir(parents=True, exist_ok=True)
                    out_path = subdir / name
                    if not out_path.exists():
                        data = zf.read(path)
                        out_path.write_bytes(data)

            print(f"  Extracted {len(sheet_paths) + len(custom_paths)} spritesheets")

            # Extract metadata
            print("  Extracting metadata...")

            # Sprite Credits CSV
            credits_data = zf.read('Data/Sprite Credits.csv').decode('utf-8', errors='replace')
            (self.metadata_dir / 'sprite_credits.csv').write_text(credits_data)

            # Dex entries
            dex_data = zf.read('Data/dex.json').decode('utf-8')
            (self.metadata_dir / 'dex_entries.json').write_text(dex_data)

            # Also grab the larger dex if available
            try:
                full_dex = zf.read('Data/pokedex/dex.json').decode('utf-8')
                (self.metadata_dir / 'full_dex.json').write_text(full_dex)
            except KeyError:
                pass

            print("  Metadata extracted")

    def _load_metadata(self):
        """Load sprite credits and dex entries."""
        # Load sprite credits
        credits_path = self.metadata_dir / 'sprite_credits.csv'
        if credits_path.exists():
            with open(credits_path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3:
                        fusion_id = row[0]
                        artist = row[1]
                        sprite_type = row[2]
                        tags = row[3].split(';') if len(row) > 3 and row[3] else []

                        # Parse head.body from fusion_id
                        head_id, body_id, variant = self._parse_fusion_id(fusion_id)
                        if head_id is not None:
                            self.sprite_credits[fusion_id] = SpriteMetadata(
                                head_id=head_id,
                                body_id=body_id,
                                variant=variant,
                                artist=artist,
                                sprite_type=sprite_type,
                                tags=tags
                            )

        # Load dex entries
        dex_path = self.metadata_dir / 'dex_entries.json'
        if dex_path.exists():
            with open(dex_path, 'r', encoding='utf-8') as f:
                entries = json.load(f)
                for entry in entries:
                    sprite = entry.get('sprite', '').replace('.png', '')
                    text = entry.get('entry', '')
                    if sprite and text:
                        if sprite not in self.dex_entries:
                            self.dex_entries[sprite] = []
                        self.dex_entries[sprite].append(text)

        # Basic sprite names (can be expanded)
        self._init_sprite_names()

        print(f"  Loaded {len(self.sprite_credits)} sprite credits, "
              f"{len(self.dex_entries)} dex entries")

    def _parse_fusion_id(self, fusion_id: str) -> Tuple[Optional[int], int, str]:
        """Parse fusion ID like '1.25' or '1.25a' into (head, body, variant)."""
        # Remove variant suffix
        variant = ''
        base_id = fusion_id
        while base_id and base_id[-1].isalpha():
            variant = base_id[-1] + variant
            base_id = base_id[:-1]

        parts = base_id.split('.')
        if len(parts) == 2:
            try:
                head = int(parts[0])
                body = int(parts[1])
                return head, body, variant
            except ValueError:
                pass
        elif len(parts) == 1:
            # Base sprite (unfused)
            try:
                return int(parts[0]), int(parts[0]), variant
            except ValueError:
                pass

        return None, 0, variant

    def _init_sprite_names(self):
        """Initialize sprite name mapping (override in subclass if needed)."""
        # Placeholder - real names would come from metadata
        self.sprite_names = {}

    def _index_spritesheets(self):
        """Build index of available spritesheets."""
        if not self.spritesheets_dir.exists():
            return

        for path in self.spritesheets_dir.glob("*.png"):
            name = path.stem
            # Parse head ID and variant
            variant = ''
            base = name
            while base and base[-1].isalpha():
                variant = base[-1] + variant
                base = base[:-1]

            try:
                head_id = int(base)
                w, h = get_png_dimensions(path)
                cols = w // self.SPRITE_SIZE
                rows = h // self.SPRITE_SIZE
                self.spritesheets[name] = SpritesheetInfo(
                    path=path,
                    head_id=head_id,
                    variant=variant,
                    width=w,
                    height=h,
                    sprite_size=self.SPRITE_SIZE,
                    cols=cols,
                    rows=rows
                )
            except (ValueError, Exception):
                continue

        # Also index custom sheets
        custom_dir = self.spritesheets_dir / "custom"
        if custom_dir.exists():
            for head_dir in custom_dir.iterdir():
                if head_dir.is_dir():
                    for path in head_dir.glob("*.png"):
                        name = f"custom/{head_dir.name}/{path.stem}"
                        try:
                            head_id = int(head_dir.name)
                            w, h = get_png_dimensions(path)
                            cols = w // self.SPRITE_SIZE
                            rows = h // self.SPRITE_SIZE
                            self.spritesheets[name] = SpritesheetInfo(
                                path=path,
                                head_id=head_id,
                                variant=path.stem,
                                width=w,
                                height=h,
                                sprite_size=self.SPRITE_SIZE,
                                cols=cols,
                                rows=rows
                            )
                        except (ValueError, Exception):
                            continue

        print(f"  Indexed {len(self.spritesheets)} spritesheets")

    def get_fusion_count(self) -> int:
        """Estimate total number of available fusions."""
        return len(self.sprite_credits)

    def get_random_fusion_id(self) -> str:
        """Get a random fusion ID from the credits."""
        if not self.sprite_credits:
            return "1.1"
        return random.choice(list(self.sprite_credits.keys()))

    def get_sprite_from_sheet(
        self,
        sheet_info: SpritesheetInfo,
        body_id: int
    ) -> Optional[torch.Tensor]:
        """
        Extract a single sprite from a spritesheet.

        Args:
            sheet_info: Spritesheet metadata
            body_id: Body ID (determines grid position)

        Returns:
            Tensor of shape [3, 96, 96] or None
        """
        if not HAS_TORCHVISION:
            raise ImportError("torchvision required for image loading")

        # Load full sheet
        sheet = read_image(str(sheet_info.path), mode=ImageReadMode.RGB)

        # Calculate grid dimensions from actual image size
        cols = sheet_info.width // self.SPRITE_SIZE
        rows = sheet_info.height // self.SPRITE_SIZE
        total_sprites = cols * rows

        # Body ID maps to grid position: idx = body_id - 1
        idx = body_id - 1
        if idx < 0 or idx >= total_sprites:
            return None

        row = idx // cols
        col = idx % cols

        # Extract sprite
        y = row * self.SPRITE_SIZE
        x = col * self.SPRITE_SIZE
        sprite = sheet[:, y:y+self.SPRITE_SIZE, x:x+self.SPRITE_SIZE]

        # Check if sprite has content (not empty/black)
        if sprite.float().std() < 5:
            return None

        # Normalize to [0, 1]
        sprite = sprite.float() / 255.0

        return sprite

    def get_fusion_sprite(
        self,
        fusion_id: str,
        target_size: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """
        Load a fusion sprite by ID.

        Args:
            fusion_id: Fusion ID like '1.25' or '1.25a'
            target_size: Resize to this size if specified

        Returns:
            Tensor of shape [3, H, W] or None
        """
        head_id, body_id, variant = self._parse_fusion_id(fusion_id)
        if head_id is None:
            return None

        # Find the right spritesheet
        sheet_name = f"{head_id}{variant}" if variant else str(head_id)
        if sheet_name not in self.spritesheets:
            # Try without variant
            sheet_name = str(head_id)
            if sheet_name not in self.spritesheets:
                return None

        sheet_info = self.spritesheets[sheet_name]
        sprite = self.get_sprite_from_sheet(sheet_info, body_id)

        if sprite is None:
            return None

        # Resize if requested
        if target_size and target_size != self.SPRITE_SIZE:
            sprite = F.interpolate(
                sprite.unsqueeze(0),
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        return sprite

    def iter_fusions(
        self,
        shuffle: bool = True,
        filter_type: Optional[str] = None  # 'main', 'alt', 'temp'
    ) -> Iterator[Tuple[str, SpriteMetadata]]:
        """Iterate over fusion IDs and metadata."""
        items = list(self.sprite_credits.items())

        if filter_type:
            items = [(k, v) for k, v in items if v.sprite_type == filter_type]

        if shuffle:
            random.shuffle(items)

        yield from items

    def get_dex_entry(self, fusion_id: str) -> Optional[str]:
        """Get a random dex entry for a fusion."""
        entries = self.dex_entries.get(fusion_id, [])
        if entries:
            return random.choice(entries)
        return None


@dataclass
class RenderConfig:
    """Configuration for how sprites are rendered into training tensors.

    These are pixel-art sprites with meaningful detail at every pixel.
    Unlike photographic data, they should not be arbitrarily rescaled.

    Attributes:
        res_scaling: How to handle resolution mismatches
            - "do_not": Place sprite at native resolution with random jitter
            - "crop_down_int_nn_up": Crop smaller regions, or NN upscale to
              integer multiples (2x, 3x) that fit in requested resolution
        background_mode: How to fill transparent regions
            - "noise": Perlin/simplex noise texture
            - "solid_random": Random solid color per sample
            - "solid_gray": Fixed mid-gray (#808080)
            - "checkerboard": Classic transparency checkerboard
            - "gradient": Random gradient backgrounds
        jitter: Randomly offset sprite within aperture (default True)
        native_size: Native sprite resolution (default 96)
    """
    res_scaling: str = "do_not"
    background_mode: str = "noise"
    jitter: bool = True
    native_size: int = 96

    @classmethod
    def from_dict(cls, d: dict) -> 'RenderConfig':
        return cls(
            res_scaling=d.get('res_scaling', 'do_not'),
            background_mode=d.get('background_mode', 'noise'),
            jitter=d.get('jitter', True),
            native_size=d.get('native_size', 96)
        )


class SpriteAtlasIterator:
    """
    Iterator that yields ContextBlocks for the training pipeline.

    Uses validated priors (built by sprite_validator.py) to only sample
    from sprite positions that have actual pixel content.

    Config structure:
        params:
            data_dir: str = "data/sprite_atlas"

            sampling_config:  # Passed to validated iterator
                split: "custom" | "base" | "procedural" | "all"
                mode: "uniform_sprites" | "uniform_types" | "logit_weighted"
                type_key: "head" | "body" | "both"
                adjustments: {pattern: logit_delta}
                adjustment_mode: "additive" | "multiplicative"
                temperature: float = 1.0
                seed: int = 42

            render_config:  # How to render into tensors
                res_scaling: "do_not" | "crop_down_int_nn_up"
                background_mode: "noise" | "solid_random" | "checkerboard" | ...
                jitter: bool = True
    """

    def __init__(
        self,
        device: torch.device,
        config: Dict,
    ):
        self.device = device

        # Import the validated iterator from the gitignored data directory
        import sys
        import importlib
        from pathlib import Path
        data_dir = config.get('data_dir', 'data/sprite_atlas')

        # Add data dir to path for import
        data_path = Path(data_dir)
        if data_path.exists():
            sys.path.insert(0, str(data_path.parent.parent))

        # Dynamic import from configured data directory
        # Module path: data/{dataset_name}/iterator
        dataset_name = data_path.name  # e.g., "sprite_atlas" or actual dir name
        module_path = f"data.{dataset_name}.iterator"
        try:
            iterator_module = importlib.import_module(module_path)
            # Look for iterator class (try common names for backwards compatibility)
            for class_name in ['SpriteAtlasIterator', 'ValidatedSpriteIterator',
                               'SpriteIterator', 'InfiniteFusionIterator']:
                ValidatedIterator = getattr(iterator_module, class_name, None)
                if ValidatedIterator is not None:
                    break
            SamplingConfig = iterator_module.SamplingConfig
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import iterator from {module_path}: {e}")

        # Parse nested sampling_config
        sampling_dict = config.get('sampling_config', {})
        sampling_config = SamplingConfig(
            split=sampling_dict.get('split', 'custom'),
            mode=sampling_dict.get('mode', 'uniform_sprites'),
            type_key=sampling_dict.get('type_key', 'head'),
            adjustments=sampling_dict.get('adjustments', {}),
            adjustment_mode=sampling_dict.get('adjustment_mode', 'additive'),
            temperature=sampling_dict.get('temperature', 1.0),
            seed=sampling_dict.get('seed', 42)
        )

        # Parse nested render_config
        render_dict = config.get('render_config', {})
        self.render_config = RenderConfig.from_dict(render_dict)

        # Create validated iterator
        self._iterator = ValidatedIterator(
            data_dir=data_dir,
            config=sampling_config,
            use_validated_prior=True
        )

        # RNG for rendering
        self._rng = random.Random(sampling_dict.get('seed', 42))

        # GPU spritesheet cache: {sheet_path: tensor[4, H, W]}
        # Avoids repeated disk I/O and H2D transfers
        self._sheet_cache: Dict[str, torch.Tensor] = {}
        self._cache_max_sheets = 64  # Limit GPU memory usage

        print(f"  SpriteAtlasIterator: {len(self._iterator.sprites):,} validated sprites")
        print(f"  Split: {sampling_config.split}, Mode: {sampling_config.mode}")
        print(f"  Render: res_scaling={self.render_config.res_scaling}, bg={self.render_config.background_mode}")
        if sampling_config.adjustments:
            print(f"  Adjustments: {sampling_config.adjustments}")

    def _get_cached_sheet(self, sheet_path: str) -> torch.Tensor:
        """Get spritesheet from GPU cache, loading if needed."""
        if sheet_path not in self._sheet_cache:
            # Evict oldest if at capacity (simple FIFO)
            if len(self._sheet_cache) >= self._cache_max_sheets:
                oldest_key = next(iter(self._sheet_cache))
                del self._sheet_cache[oldest_key]

            # Load sheet to GPU once
            from PIL import Image
            img = Image.open(sheet_path).convert('RGBA')
            # Direct to tensor without numpy intermediate
            import numpy as np
            arr = np.array(img)
            sheet_tensor = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)
            self._sheet_cache[sheet_path] = sheet_tensor.to(self.device)

        return self._sheet_cache[sheet_path]

    def _generate_background(self, width: int, height: int) -> torch.Tensor:
        """Generate background texture based on render_config.background_mode.

        All operations are GPU-native to avoid CPU-GPU sync.
        """
        mode = self.render_config.background_mode

        if mode == "solid_gray":
            bg = torch.full((3, height, width), 0.5, device=self.device)

        elif mode == "solid_random":
            # Generate random color directly on GPU
            color = torch.rand(3, 1, 1, device=self.device)
            bg = color.expand(3, height, width).contiguous()

        elif mode == "checkerboard":
            # Classic transparency checkerboard (8x8 tiles)
            tile = 8
            y_idx = torch.arange(height, device=self.device) // tile
            x_idx = torch.arange(width, device=self.device) // tile
            checker = ((y_idx.unsqueeze(1) + x_idx.unsqueeze(0)) % 2).float()
            # Light gray / dark gray
            bg = checker * 0.3 + 0.5
            bg = bg.unsqueeze(0).expand(3, -1, -1).contiguous()

        elif mode == "gradient":
            # Random gradient - all GPU ops
            angle = torch.rand(1, device=self.device).item() * 6.28318  # 2*pi
            c1 = torch.rand(3, device=self.device)
            c2 = torch.rand(3, device=self.device)
            y = torch.linspace(0, 1, height, device=self.device)
            x = torch.linspace(0, 1, width, device=self.device)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            # Use torch ops for trig
            t = (torch.cos(torch.tensor(angle, device=self.device)) * xx +
                 torch.sin(torch.tensor(angle, device=self.device)) * yy).clamp(0, 1)
            bg = c1.view(3, 1, 1) * (1 - t) + c2.view(3, 1, 1) * t

        else:  # "noise" - default
            # Simple colored noise - already GPU native
            bg = torch.rand(3, height, width, device=self.device) * 0.3 + 0.35

        return bg

    def _render_sprite(
        self,
        sprite_rgba: torch.Tensor,
        target_h: int,
        target_w: int
    ) -> torch.Tensor:
        """
        Render RGBA sprite onto background at target resolution.

        Respects res_scaling mode for pixel-perfect rendering.
        """
        native = self.render_config.native_size
        _, sh, sw = sprite_rgba.shape  # Should be 4, 96, 96

        # Generate background
        bg = self._generate_background(target_w, target_h)

        if self.render_config.res_scaling == "do_not":
            # Place at native resolution with jitter
            render_h, render_w = sh, sw

            if target_h < render_h or target_w < render_w:
                # Target smaller than sprite - crop sprite
                crop_h = min(render_h, target_h)
                crop_w = min(render_w, target_w)
                if self.render_config.jitter:
                    y0 = self._rng.randint(0, render_h - crop_h) if render_h > crop_h else 0
                    x0 = self._rng.randint(0, render_w - crop_w) if render_w > crop_w else 0
                else:
                    y0, x0 = 0, 0
                sprite_rgba = sprite_rgba[:, y0:y0+crop_h, x0:x0+crop_w]
                render_h, render_w = crop_h, crop_w

            # Calculate placement position
            if self.render_config.jitter:
                place_y = self._rng.randint(0, max(0, target_h - render_h))
                place_x = self._rng.randint(0, max(0, target_w - render_w))
            else:
                place_y = (target_h - render_h) // 2
                place_x = (target_w - render_w) // 2

        elif self.render_config.res_scaling == "crop_down_int_nn_up":
            # Calculate largest integer scale that fits
            scale = min(target_h // native, target_w // native)
            scale = max(1, scale)  # At least 1x

            if scale == 1:
                # No upscale possible, might need to crop
                render_h = min(native, target_h)
                render_w = min(native, target_w)
                if render_h < native or render_w < native:
                    # Crop from sprite
                    if self.render_config.jitter:
                        y0 = self._rng.randint(0, native - render_h) if native > render_h else 0
                        x0 = self._rng.randint(0, native - render_w) if native > render_w else 0
                    else:
                        y0, x0 = (native - render_h) // 2, (native - render_w) // 2
                    sprite_rgba = sprite_rgba[:, y0:y0+render_h, x0:x0+render_w]
            else:
                # Integer NN upscale
                render_h = native * scale
                render_w = native * scale
                sprite_rgba = F.interpolate(
                    sprite_rgba.unsqueeze(0),
                    size=(render_h, render_w),
                    mode='nearest'
                ).squeeze(0)

            # Calculate placement
            if self.render_config.jitter:
                place_y = self._rng.randint(0, max(0, target_h - render_h))
                place_x = self._rng.randint(0, max(0, target_w - render_w))
            else:
                place_y = (target_h - render_h) // 2
                place_x = (target_w - render_w) // 2

        else:
            raise ValueError(f"Unknown res_scaling: {self.render_config.res_scaling}")

        # Alpha composite onto background
        rgb = sprite_rgba[:3]
        alpha = sprite_rgba[3:4]

        # Place sprite region
        y1, y2 = place_y, place_y + sprite_rgba.shape[1]
        x1, x2 = place_x, place_x + sprite_rgba.shape[2]

        # Composite: out = fg * alpha + bg * (1 - alpha)
        bg_region = bg[:, y1:y2, x1:x2]
        composited = rgb * alpha + bg_region * (1 - alpha)
        bg[:, y1:y2, x1:x2] = composited

        return bg

    def generate_batch_list(
        self,
        batch_size: int,
        resolution: Optional[int] = None,
        **kwargs
    ) -> List:
        """
        Generate a batch of ContextBlocks with fusion sprites.

        Optimized: Uses GPU-cached spritesheets and GPU-side cropping.
        Single H2D transfer per unique sheet, not per sprite.

        Returns list of ContextBlock with type='latent'.
        """
        from .model import ContextBlock

        # Resolution can come from kwargs or default to native
        res = resolution or self.render_config.native_size

        if len(self._iterator.sprites) == 0:
            raise RuntimeError("No validated fusion sprites found")

        # Collect all sprite refs first
        sprite_refs = list(self._iterator.iter_samples(batch_size))

        # Group by sheet path for efficient batching
        from collections import defaultdict
        by_sheet: Dict[str, List] = defaultdict(list)
        for ref in sprite_refs:
            by_sheet[str(ref.sheet_path)].append(ref)

        blocks = []

        # Process each sheet's sprites together
        for sheet_path, refs in by_sheet.items():
            try:
                # Get sheet from GPU cache (single H2D per unique sheet)
                sheet = self._get_cached_sheet(sheet_path)

                # GPU-side batch crop: extract all sprites from this sheet
                for sprite_ref in refs:
                    x0 = sprite_ref.grid_x * 96
                    y0 = sprite_ref.grid_y * 96

                    # GPU tensor slicing (no H2D transfer)
                    sprite_rgba = sheet[:, y0:y0+96, x0:x0+96]

                    # Check if sprite has content (not empty/black)
                    if sprite_rgba[:3].std() < 0.02:
                        continue

                    # Render with background compositing and resolution handling
                    rendered = self._render_sprite(sprite_rgba, res, res)

                    # Create block
                    H, W = rendered.shape[-2:]
                    block = ContextBlock(
                        content=rendered,
                        type='latent',
                        causal=True,
                        source='sprite_atlas',
                        id=sprite_ref.sprite_id,
                        shape_meta=(H, W)
                    )

                    # Attach metadata
                    block.fusion_meta = {
                        'head_id': sprite_ref.head_id,
                        'body_id': sprite_ref.body_id,
                        'variant': sprite_ref.variant,
                        'split': sprite_ref.split_name
                    }

                    blocks.append(block)

            except Exception as e:
                # Skip problematic sheets
                continue

        return blocks


def extract_sprite_atlas(
    zip_path: str,
    output_dir: str = "data/sprite_atlas"
):
    """
    Convenience function to extract a sprite atlas dataset from zip.

    Usage:
        python -c "from src.sprite_atlas import extract_sprite_atlas; extract_sprite_atlas('path/to/sprites.zip')"
    """
    print(f"Extracting sprite atlas dataset...")
    print(f"  Source: {zip_path}")
    print(f"  Destination: {output_dir}")

    dataset = SpriteAtlasDataset(
        data_dir=output_dir,
        zip_path=zip_path
    )

    print(f"\nDataset ready!")
    print(f"  Spritesheets: {len(dataset.spritesheets)}")
    print(f"  Sprite credits: {len(dataset.sprite_credits)}")
    print(f"  Dex entries: {len(dataset.dex_entries)}")

    return dataset


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.sprite_atlas <path_to_zip>")
        sys.exit(1)

    # Test extraction and loading
    dataset = extract_sprite_atlas(sys.argv[1])

    # Test loading a sprite
    if HAS_TORCHVISION:
        sprite = dataset.get_fusion_sprite("1.25")
        if sprite is not None:
            print(f"\nLoaded sprite 1.25: {sprite.shape}")
        else:
            print("\nCould not load sprite 1.25")
