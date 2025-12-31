"""
Pokémon Infinite Fusion Dataset Iterator

Provides training data from the Infinite Fusion sprite dataset.
Sprites are organized as spritesheets where:
- Each sheet contains fusions with a specific "head" Pokémon
- Grid positions correspond to different "body" Pokémon
- Naming: HEAD.BODY.png (e.g., 1.25.png = Bulbasaur head + Pikachu body)

Expected data layout:
    data/infinite_fusion/
        spritesheets/           # Extracted from zip
            1.png, 1a.png, ...  # Head=1 (Bulbasaur) fusions
            2.png, ...          # Head=2 (Ivysaur) fusions
        metadata/
            sprite_credits.csv  # Artist credits and tags
            dex_entries.json    # Pokédex descriptions
            pokemon_names.json  # ID -> name mapping
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
class FusionSprite:
    """Metadata for a single fusion sprite."""
    head_id: int
    body_id: int
    variant: str  # '', 'a', 'b', etc.
    artist: str
    sprite_type: str  # 'main', 'alt', 'temp'
    tags: List[str]
    dex_entry: Optional[str] = None


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


class InfiniteFusionDataset:
    """
    Dataset manager for Infinite Fusion sprites.

    Handles extraction from zip, metadata parsing, and sprite loading.

    Spritesheet format:
    - Each sheet is for a specific "head" Pokémon (filename = head_id.png)
    - 96x96 pixel sprites arranged in a 10-column grid
    - Grid position (idx) maps to body_id: body_id = idx + 1
    - Example: 1.png row=2,col=4 (idx=24) = fusion 1.25 (Bulbasaur + Pikachu)
    """

    # Standard sprite size in the spritesheets
    SPRITE_SIZE = 96
    GRID_COLS = 10  # Sprites per row

    # National Dex number ranges
    # Gen 1: 1-151, Gen 2: 152-251, etc.
    MAX_POKEMON_ID = 565  # Approximate max in Infinite Fusion

    def __init__(
        self,
        data_dir: str = "data/infinite_fusion",
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
        self.sprite_credits: Dict[str, FusionSprite] = {}
        self.dex_entries: Dict[str, List[str]] = {}  # fusion_id -> list of entries
        self.pokemon_names: Dict[int, str] = {}

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
        """Extract spritesheets and metadata from the Infinite Fusion zip."""
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
                            self.sprite_credits[fusion_id] = FusionSprite(
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

        # Basic pokemon names (can be expanded)
        self._init_pokemon_names()

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
            # Base pokemon (unfused)
            try:
                return int(parts[0]), int(parts[0]), variant
            except ValueError:
                pass

        return None, 0, variant

    def _init_pokemon_names(self):
        """Initialize basic Pokemon name mapping."""
        # Gen 1 starters and common pokemon
        names = {
            1: "Bulbasaur", 2: "Ivysaur", 3: "Venusaur",
            4: "Charmander", 5: "Charmeleon", 6: "Charizard",
            7: "Squirtle", 8: "Wartortle", 9: "Blastoise",
            25: "Pikachu", 26: "Raichu",
            # Add more as needed
        }
        self.pokemon_names = names

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
            body_id: Body pokemon ID (determines grid position)

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
    ) -> Iterator[Tuple[str, FusionSprite]]:
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


class InfiniteFusionIterator:
    """
    Iterator that yields ContextBlocks for the training pipeline.

    Integrates with CompositeIterator via the same interface as
    FunctionalIterator, TorusIterator, etc.
    """

    def __init__(
        self,
        device: torch.device,
        config: Dict,
        dataset: Optional[InfiniteFusionDataset] = None
    ):
        self.device = device
        self.seed = config.get('seed', 42)
        self.resolution = config.get('resolution', 256)
        self.text_position = config.get('text_position', 'none')

        # Dataset paths
        data_dir = config.get('data_dir', 'data/infinite_fusion')
        zip_path = config.get('zip_path', None)

        # Initialize or use provided dataset
        if dataset is not None:
            self.dataset = dataset
        else:
            self.dataset = InfiniteFusionDataset(
                data_dir=data_dir,
                zip_path=zip_path,
                device=device
            )

        # RNG
        self.rng = random.Random(self.seed)

        # Cache of valid fusion IDs (ones we can actually load)
        self._valid_ids: Optional[List[str]] = None

    def _get_valid_ids(self) -> List[str]:
        """Get list of fusion IDs that we can actually load sprites for."""
        if self._valid_ids is None:
            valid = []
            for fusion_id in self.dataset.sprite_credits.keys():
                head_id, body_id, variant = self.dataset._parse_fusion_id(fusion_id)
                if head_id is not None:
                    sheet_name = f"{head_id}{variant}" if variant else str(head_id)
                    if sheet_name in self.dataset.spritesheets:
                        valid.append(fusion_id)
            self._valid_ids = valid
            print(f"  Found {len(valid)} loadable fusions")
        return self._valid_ids

    def generate_batch_list(
        self,
        batch_size: int,
        resolution: Optional[int] = None,
        **kwargs
    ) -> List:
        """
        Generate a batch of ContextBlocks with fusion sprites.

        Returns list of ContextBlock with type='latent'.
        """
        from .model import ContextBlock

        res = resolution or self.resolution
        valid_ids = self._get_valid_ids()

        if not valid_ids:
            raise RuntimeError("No valid fusion sprites found")

        blocks = []
        for _ in range(batch_size):
            # Sample random fusion
            fusion_id = self.rng.choice(valid_ids)

            # Load sprite
            sprite = self.dataset.get_fusion_sprite(fusion_id, target_size=res)
            if sprite is None:
                continue

            sprite = sprite.to(self.device)

            # Create logsnr map (uniform for now)
            logsnr_val = self.rng.uniform(-4.0, 5.0)
            logsnr_map = torch.full(
                (1, res, res),
                logsnr_val,
                device=self.device,
                dtype=sprite.dtype
            )

            # Get metadata for potential text conditioning
            meta = self.dataset.sprite_credits.get(fusion_id)
            dex_entry = self.dataset.get_dex_entry(fusion_id)

            # Create block - shape_meta is required for latent type
            H, W = sprite.shape[-2:]
            block = ContextBlock(
                content=sprite,
                logsnr=logsnr_map,
                type='latent',
                causal=True,
                source='infinite_fusion',
                id=fusion_id,
                shape_meta=(H, W)
            )

            # Attach extra metadata (not used in training but useful for logging)
            block.fusion_meta = {
                'head_id': meta.head_id if meta else None,
                'body_id': meta.body_id if meta else None,
                'artist': meta.artist if meta else None,
                'dex_entry': dex_entry
            }

            blocks.append(block)

        return blocks


def extract_infinite_fusion(
    zip_path: str = "/mnt/f/dox/repos/InfiniteFusion.zip",
    output_dir: str = "data/infinite_fusion"
):
    """
    Convenience function to extract the Infinite Fusion dataset.

    Usage:
        python -c "from src.infinite_fusion import extract_infinite_fusion; extract_infinite_fusion()"
    """
    print(f"Extracting Infinite Fusion dataset...")
    print(f"  Source: {zip_path}")
    print(f"  Destination: {output_dir}")

    dataset = InfiniteFusionDataset(
        data_dir=output_dir,
        zip_path=zip_path
    )

    print(f"\nDataset ready!")
    print(f"  Spritesheets: {len(dataset.spritesheets)}")
    print(f"  Sprite credits: {len(dataset.sprite_credits)}")
    print(f"  Dex entries: {len(dataset.dex_entries)}")

    return dataset


if __name__ == "__main__":
    # Test extraction and loading
    dataset = extract_infinite_fusion()

    # Test loading a sprite
    if HAS_TORCHVISION:
        sprite = dataset.get_fusion_sprite("1.25")  # Bulbasaur + Pikachu
        if sprite is not None:
            print(f"\nLoaded sprite 1.25: {sprite.shape}")
        else:
            print("\nCould not load sprite 1.25")
