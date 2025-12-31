"""
Infinite Fusion Sprite Synchronization Service

Downloads custom spritesheets to the canonical game directory structure,
producing a complete reference copy suitable for backup/repackaging.

Target directory structure (matching game expectations):
    Graphics/CustomBattlers/spritesheets/
        spritesheets_base/
            1.png, 2.png, ...
        spritesheets_custom/
            1/
                1.png, 1a.png, 1b.png, ...
            2/
                2.png, 2a.png, ...

Usage:
    python -m src.infinite_fusion_sync --game-dir /path/to/InfiniteFusion

    # Or as a module:
    from src.infinite_fusion_sync import InfiniteFusionSync
    sync = InfiniteFusionSync("/path/to/InfiniteFusion")
    sync.run()
"""

import os
import time
import json
import hashlib
import urllib.request
import urllib.error
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Iterator
from collections import defaultdict
from datetime import datetime
import argparse


# =============================================================================
# Configuration (mirrors DownloadedSettings.rb)
# =============================================================================

@dataclass
class SyncConfig:
    """Configuration matching the game's Settings module."""

    # Spritesheet URLs (true size, not resized)
    base_spritesheet_url: str = "https://infinitefusion.net/customsprites/spritesheets/spritesheets_base/"
    custom_spritesheet_url: str = "https://infinitefusion.net/customsprites/spritesheets/spritesheets_custom/"

    # Index/metadata URLs
    credits_url: str = "https://infinitefusion.net/customsprites/Sprite_Credits.csv"
    custom_sprites_list_url: str = "https://raw.githubusercontent.com/infinitefusion/pif-downloadables/refs/heads/master/CUSTOM_SPRITES"
    base_sprites_list_url: str = "https://raw.githubusercontent.com/infinitefusion/pif-downloadables/refs/heads/master/BASE_SPRITES"

    # Rate limiting (conservative - game uses 15/60s)
    max_requests_per_window: int = 12  # Slightly under game's 15
    rate_window_seconds: int = 60
    retry_delay_seconds: int = 30  # Wait when rate limited
    max_retries: int = 3

    # Paths relative to game root
    spritesheets_base_path: str = "Graphics/CustomBattlers/spritesheets/spritesheets_base"
    spritesheets_custom_path: str = "Graphics/CustomBattlers/spritesheets/spritesheets_custom"
    sprites_data_path: str = "Data/sprites"

    # State tracking
    state_file: str = "sync_state.json"

    # Pokemon count
    max_pokemon_id: int = 565


@dataclass
class SyncState:
    """Tracks synchronization progress for resumability."""

    # Downloaded spritesheets (set of relative paths)
    downloaded_base: Set[str] = field(default_factory=set)
    downloaded_custom: Set[str] = field(default_factory=set)

    # Failed downloads (for retry)
    failed: Dict[str, int] = field(default_factory=dict)  # path -> retry count

    # Index file hashes (to detect updates)
    index_hashes: Dict[str, str] = field(default_factory=dict)

    # Stats
    total_downloaded: int = 0
    total_bytes: int = 0
    last_sync: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'downloaded_base': list(self.downloaded_base),
            'downloaded_custom': list(self.downloaded_custom),
            'failed': self.failed,
            'index_hashes': self.index_hashes,
            'total_downloaded': self.total_downloaded,
            'total_bytes': self.total_bytes,
            'last_sync': self.last_sync
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'SyncState':
        return cls(
            downloaded_base=set(d.get('downloaded_base', [])),
            downloaded_custom=set(d.get('downloaded_custom', [])),
            failed=d.get('failed', {}),
            index_hashes=d.get('index_hashes', {}),
            total_downloaded=d.get('total_downloaded', 0),
            total_bytes=d.get('total_bytes', 0),
            last_sync=d.get('last_sync')
        )


@dataclass
class SpritesheetRef:
    """Reference to a spritesheet to download."""
    head_id: int
    variant: str  # '', 'a', 'b', etc.
    is_base: bool  # True for base pokemon, False for custom fusions

    @property
    def filename(self) -> str:
        if self.is_base:
            return f"{self.head_id}.png"
        else:
            return f"{self.head_id}{self.variant}.png"

    @property
    def relative_path(self) -> str:
        if self.is_base:
            return self.filename
        else:
            return f"{self.head_id}/{self.filename}"

    def __hash__(self):
        return hash((self.head_id, self.variant, self.is_base))

    def __eq__(self, other):
        return (self.head_id, self.variant, self.is_base) == (other.head_id, other.variant, other.is_base)


class RateLimiter:
    """Tracks request rate to stay under limits."""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_times: List[float] = []

    def _cleanup_old(self):
        """Remove requests outside the window."""
        cutoff = time.time() - self.window_seconds
        self.request_times = [t for t in self.request_times if t > cutoff]

    def can_request(self) -> bool:
        """Check if we can make a request without exceeding limit."""
        self._cleanup_old()
        return len(self.request_times) < self.max_requests

    def record_request(self):
        """Record that a request was made."""
        self.request_times.append(time.time())

    def time_until_available(self) -> float:
        """Seconds until next request is allowed."""
        self._cleanup_old()
        if len(self.request_times) < self.max_requests:
            return 0.0
        oldest = min(self.request_times)
        return max(0.0, oldest + self.window_seconds - time.time())

    @property
    def current_rate(self) -> Tuple[int, int]:
        """Returns (current_count, max_count)."""
        self._cleanup_old()
        return len(self.request_times), self.max_requests


class InfiniteFusionSync:
    """
    Synchronizes custom sprites from infinitefusion.net to the game directory.

    Produces a complete reference copy in the format the game expects,
    suitable for backup, offline play, or repackaging.
    """

    def __init__(
        self,
        game_dir: str,
        config: Optional[SyncConfig] = None,
        verbose: bool = True
    ):
        self.game_dir = Path(game_dir)
        self.config = config or SyncConfig()
        self.verbose = verbose

        # Validate game directory
        if not (self.game_dir / "Game.exe").exists() and not (self.game_dir / "Graphics").exists():
            raise ValueError(f"Does not appear to be an Infinite Fusion directory: {game_dir}")

        # Set up paths
        self.base_dir = self.game_dir / self.config.spritesheets_base_path
        self.custom_dir = self.game_dir / self.config.spritesheets_custom_path
        self.data_dir = self.game_dir / self.config.sprites_data_path
        self.state_path = self.game_dir / self.config.state_file

        # Ensure directories exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.custom_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load state
        self.state = self._load_state()

        # Rate limiter
        self.rate_limiter = RateLimiter(
            self.config.max_requests_per_window,
            self.config.rate_window_seconds
        )

        # Index data
        self.custom_sprites_index: Set[str] = set()  # e.g., "1.25a.png"
        self.base_sprites_index: Set[str] = set()
        self.credits: Dict[str, dict] = {}  # fusion_id -> {artist, type, tags}

        # Stats for current run
        self._run_stats = {
            'downloaded': 0,
            'skipped': 0,
            'failed': 0,
            'bytes': 0,
            'rate_limited_waits': 0
        }

    def _load_state(self) -> SyncState:
        """Load sync state from disk."""
        if self.state_path.exists():
            try:
                with open(self.state_path, 'r') as f:
                    return SyncState.from_dict(json.load(f))
            except Exception as e:
                self._log(f"Warning: Could not load state: {e}")
        return SyncState()

    def _save_state(self):
        """Save sync state to disk."""
        self.state.last_sync = datetime.now().isoformat()
        with open(self.state_path, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def _log(self, msg: str, level: str = "INFO"):
        """Log a message."""
        if self.verbose or level in ("ERROR", "WARN", "RATE_LIMITED"):
            timestamp = datetime.now().strftime("%H:%M:%S")
            prefix = {
                "INFO": "   ",
                "WARN": "⚠  ",
                "ERROR": "✗  ",
                "OK": "✓  ",
                "RATE_LIMITED": "⏳ ",
                "DOWNLOAD": "↓  "
            }.get(level, "   ")
            print(f"[{timestamp}] {prefix}{msg}")

    def _fetch_url(self, url: str, dest_path: Optional[Path] = None) -> Optional[bytes]:
        """
        Fetch a URL, respecting rate limits.

        Returns bytes if successful, None if failed.
        Saves to dest_path if provided.
        """
        # Wait for rate limit
        while not self.rate_limiter.can_request():
            wait_time = self.rate_limiter.time_until_available()
            current, max_req = self.rate_limiter.current_rate
            self._log(
                f"Rate limit reached ({current}/{max_req}), waiting {wait_time:.1f}s...",
                "RATE_LIMITED"
            )
            self._run_stats['rate_limited_waits'] += 1
            time.sleep(min(wait_time + 1, self.config.retry_delay_seconds))

        self.rate_limiter.record_request()

        try:
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'InfiniteFusion-Sync/1.0 (backup tool)',
                    'Accept': '*/*'
                }
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()

                if dest_path:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(dest_path, 'wb') as f:
                        f.write(data)

                return data

        except urllib.error.HTTPError as e:
            if e.code == 429:  # Rate limited by server
                self._log(f"Server rate limit hit (429), backing off...", "RATE_LIMITED")
                time.sleep(self.config.retry_delay_seconds)
                return None
            elif e.code == 404:
                # Not found - this is expected for some sprites
                return None
            else:
                self._log(f"HTTP error {e.code} for {url}", "ERROR")
                return None
        except Exception as e:
            self._log(f"Fetch error for {url}: {e}", "ERROR")
            return None

    def _hash_content(self, content: bytes) -> str:
        """Get SHA256 hash of content."""
        return hashlib.sha256(content).hexdigest()[:16]

    def fetch_indices(self) -> bool:
        """
        Fetch the sprite index files from GitHub/infinitefusion.net.

        Returns True if indices were updated.
        """
        self._log("Fetching sprite indices...")
        updated = False

        # Fetch CUSTOM_SPRITES list
        custom_data = self._fetch_url(self.config.custom_sprites_list_url)
        if custom_data:
            new_hash = self._hash_content(custom_data)
            if new_hash != self.state.index_hashes.get('custom_sprites'):
                self.state.index_hashes['custom_sprites'] = new_hash
                updated = True

            # Parse: one filename per line
            self.custom_sprites_index = set(
                line.strip() for line in custom_data.decode('utf-8').splitlines()
                if line.strip() and line.strip().endswith('.png')
            )
            self._log(f"  Custom sprites index: {len(self.custom_sprites_index)} entries", "OK")

            # Save locally
            (self.data_dir / "CUSTOM_SPRITES").write_bytes(custom_data)

        # Fetch BASE_SPRITES list
        base_data = self._fetch_url(self.config.base_sprites_list_url)
        if base_data:
            new_hash = self._hash_content(base_data)
            if new_hash != self.state.index_hashes.get('base_sprites'):
                self.state.index_hashes['base_sprites'] = new_hash
                updated = True

            self.base_sprites_index = set(
                line.strip() for line in base_data.decode('utf-8').splitlines()
                if line.strip() and line.strip().endswith('.png')
            )
            self._log(f"  Base sprites index: {len(self.base_sprites_index)} entries", "OK")

            (self.data_dir / "BASE_SPRITES").write_bytes(base_data)

        # Fetch Sprite_Credits.csv
        credits_data = self._fetch_url(self.config.credits_url)
        if credits_data:
            new_hash = self._hash_content(credits_data)
            if new_hash != self.state.index_hashes.get('credits'):
                self.state.index_hashes['credits'] = new_hash
                updated = True

            # Parse credits
            lines = credits_data.decode('utf-8', errors='replace').splitlines()
            for line in lines:
                parts = line.split(',')
                if len(parts) >= 3:
                    fusion_id = parts[0]
                    self.credits[fusion_id] = {
                        'artist': parts[1],
                        'type': parts[2],
                        'tags': parts[3].split(';') if len(parts) > 3 else []
                    }

            self._log(f"  Credits: {len(self.credits)} entries", "OK")
            (self.data_dir / "Sprite_Credits.csv").write_bytes(credits_data)

        if updated:
            self._log("Index files updated", "OK")
        else:
            self._log("Index files unchanged")

        return updated

    def _parse_sprite_to_sheet(self, sprite_name: str, is_base: bool) -> Optional[SpritesheetRef]:
        """
        Parse a sprite filename to determine which spritesheet it belongs to.

        e.g., "1.25a.png" -> SpritesheetRef(head_id=1, variant='a', is_base=False)
        """
        name = sprite_name.replace('.png', '')

        if is_base:
            # Base sprites: "25.png" or "25a.png" - all map to single sheet "{head_id}.png"
            # Strip any variant letters to get just the head_id
            base = name
            while base and base[-1].isalpha():
                base = base[:-1]
            try:
                # Base sheets don't have variants - one sheet per pokemon
                return SpritesheetRef(int(base), '', is_base=True)
            except ValueError:
                return None
        else:
            # Custom fusions: "1.25.png" or "1.25a.png"
            variant = ''
            base = name
            while base and base[-1].isalpha():
                variant = base[-1] + variant
                base = base[:-1]

            parts = base.split('.')
            if len(parts) >= 1:
                try:
                    head_id = int(parts[0])
                    return SpritesheetRef(head_id, variant, is_base=False)
                except ValueError:
                    return None

        return None

    def get_required_spritesheets(self) -> Tuple[Set[SpritesheetRef], Set[SpritesheetRef]]:
        """
        Determine which spritesheets are needed based on the index.

        Returns (base_sheets, custom_sheets).
        """
        base_sheets: Set[SpritesheetRef] = set()
        custom_sheets: Set[SpritesheetRef] = set()

        # Parse custom sprites index to find required sheets
        for sprite_name in self.custom_sprites_index:
            ref = self._parse_sprite_to_sheet(sprite_name, is_base=False)
            if ref:
                custom_sheets.add(ref)

        # Parse base sprites index
        for sprite_name in self.base_sprites_index:
            ref = self._parse_sprite_to_sheet(sprite_name, is_base=True)
            if ref:
                base_sheets.add(ref)

        # Also add base sheets for all pokemon (1 to max_pokemon_id)
        # since autogen sheets exist for all
        for i in range(1, self.config.max_pokemon_id + 1):
            base_sheets.add(SpritesheetRef(i, '', is_base=True))

        return base_sheets, custom_sheets

    def _get_missing_sheets(
        self,
        required: Set[SpritesheetRef],
        downloaded: Set[str],
        target_dir: Path
    ) -> List[SpritesheetRef]:
        """Find sheets that need to be downloaded."""
        missing = []
        for sheet in required:
            rel_path = sheet.relative_path
            if rel_path in downloaded:
                continue
            full_path = target_dir / rel_path
            if full_path.exists():
                # Already on disk, update state
                downloaded.add(rel_path)
            else:
                missing.append(sheet)
        return missing

    def download_spritesheet(self, sheet: SpritesheetRef) -> bool:
        """
        Download a single spritesheet.

        Returns True if successful.
        """
        if sheet.is_base:
            url = f"{self.config.base_spritesheet_url}{sheet.filename}"
            dest = self.base_dir / sheet.filename
            downloaded_set = self.state.downloaded_base
        else:
            url = f"{self.config.custom_spritesheet_url}{sheet.head_id}/{sheet.filename}"
            dest = self.custom_dir / str(sheet.head_id) / sheet.filename
            downloaded_set = self.state.downloaded_custom

        rel_path = sheet.relative_path

        # Check retry count
        retry_count = self.state.failed.get(rel_path, 0)
        if retry_count >= self.config.max_retries:
            return False

        data = self._fetch_url(url, dest)

        if data:
            downloaded_set.add(rel_path)
            self.state.total_downloaded += 1
            self.state.total_bytes += len(data)
            self._run_stats['downloaded'] += 1
            self._run_stats['bytes'] += len(data)

            # Clear from failed if it was there
            self.state.failed.pop(rel_path, None)

            sheet_type = "base" if sheet.is_base else "custom"
            self._log(f"Downloaded {sheet_type}/{rel_path} ({len(data)//1024}KB)", "DOWNLOAD")
            return True
        else:
            # Record failure
            self.state.failed[rel_path] = retry_count + 1
            self._run_stats['failed'] += 1
            return False

    def sync_base_spritesheets(self, required: Set[SpritesheetRef]) -> int:
        """
        Sync base pokemon spritesheets.

        Returns count of newly downloaded sheets.
        """
        self._log(f"\nSyncing base spritesheets ({len(required)} total)...")

        missing = self._get_missing_sheets(required, self.state.downloaded_base, self.base_dir)
        self._log(f"  {len(missing)} missing, {len(self.state.downloaded_base)} already synced")

        downloaded = 0
        for i, sheet in enumerate(sorted(missing, key=lambda s: (s.head_id, s.variant))):
            self.download_spritesheet(sheet)
            downloaded += 1

            # Progress update every 10
            if (i + 1) % 10 == 0:
                current, max_req = self.rate_limiter.current_rate
                self._log(f"  Progress: {i+1}/{len(missing)} (rate: {current}/{max_req})")

            # Save state periodically
            if (i + 1) % 50 == 0:
                self._save_state()

        return downloaded

    def sync_custom_spritesheets(self, required: Set[SpritesheetRef]) -> int:
        """
        Sync custom fusion spritesheets.

        Returns count of newly downloaded sheets.
        """
        self._log(f"\nSyncing custom spritesheets ({len(required)} total)...")

        missing = self._get_missing_sheets(required, self.state.downloaded_custom, self.custom_dir)
        self._log(f"  {len(missing)} missing, {len(self.state.downloaded_custom)} already synced")

        # Group by head_id for organized downloading
        by_head: Dict[int, List[SpritesheetRef]] = defaultdict(list)
        for sheet in missing:
            by_head[sheet.head_id].append(sheet)

        downloaded = 0
        total = len(missing)

        for head_id in sorted(by_head.keys()):
            sheets = sorted(by_head[head_id], key=lambda s: s.variant)
            for sheet in sheets:
                self.download_spritesheet(sheet)
                downloaded += 1

                # Progress update
                if downloaded % 25 == 0:
                    current, max_req = self.rate_limiter.current_rate
                    pct = downloaded / total * 100
                    self._log(f"  Progress: {downloaded}/{total} ({pct:.1f}%) (rate: {current}/{max_req})")

                # Save state periodically
                if downloaded % 100 == 0:
                    self._save_state()

        return downloaded

    def run(self, skip_indices: bool = False) -> dict:
        """
        Run the full synchronization.

        Returns stats dict.
        """
        start_time = time.time()
        self._log("=" * 60)
        self._log("Infinite Fusion Sprite Sync")
        self._log(f"Game directory: {self.game_dir}")
        self._log("=" * 60)

        # Reset run stats
        self._run_stats = {
            'downloaded': 0,
            'skipped': 0,
            'failed': 0,
            'bytes': 0,
            'rate_limited_waits': 0
        }

        # Step 1: Fetch indices
        if not skip_indices:
            self.fetch_indices()
        else:
            # Load from disk
            custom_path = self.data_dir / "CUSTOM_SPRITES"
            base_path = self.data_dir / "BASE_SPRITES"
            if custom_path.exists():
                self.custom_sprites_index = set(
                    line.strip() for line in custom_path.read_text().splitlines()
                    if line.strip()
                )
            if base_path.exists():
                self.base_sprites_index = set(
                    line.strip() for line in base_path.read_text().splitlines()
                    if line.strip()
                )

        if not self.custom_sprites_index:
            self._log("No custom sprites index found. Run without --skip-indices first.", "ERROR")
            return self._run_stats

        # Step 2: Determine required spritesheets
        base_required, custom_required = self.get_required_spritesheets()
        self._log(f"\nRequired: {len(base_required)} base sheets, {len(custom_required)} custom sheets")

        # Step 3: Sync base sheets
        self.sync_base_spritesheets(base_required)

        # Step 4: Sync custom sheets
        self.sync_custom_spritesheets(custom_required)

        # Final save
        self._save_state()

        # Summary
        elapsed = time.time() - start_time
        self._log("\n" + "=" * 60)
        self._log("Sync Complete", "OK")
        self._log(f"  Downloaded: {self._run_stats['downloaded']} files")
        self._log(f"  Bytes: {self._run_stats['bytes'] / 1024 / 1024:.1f} MB")
        self._log(f"  Failed: {self._run_stats['failed']}")
        self._log(f"  Rate limit waits: {self._run_stats['rate_limited_waits']}")
        self._log(f"  Time: {elapsed:.1f}s")
        self._log(f"\nTotal synced (all time):")
        self._log(f"  Base sheets: {len(self.state.downloaded_base)}")
        self._log(f"  Custom sheets: {len(self.state.downloaded_custom)}")
        self._log(f"  Total bytes: {self.state.total_bytes / 1024 / 1024:.1f} MB")
        self._log("=" * 60)

        return self._run_stats

    def iter_sync(self, batch_size: int = 10) -> Iterator[dict]:
        """
        Iterator-style sync that yields progress after each batch.

        Useful for integration with training loops or progress UIs.

        Yields dicts with:
            - downloaded: count this batch
            - total_done: total downloaded so far
            - total_needed: total sheets needed
            - rate_limited: bool, True if currently throttled
            - bytes: bytes downloaded this batch
        """
        # Fetch indices first
        self.fetch_indices()
        base_required, custom_required = self.get_required_spritesheets()

        all_required = [
            (sheet, True) for sheet in base_required
        ] + [
            (sheet, False) for sheet in custom_required
        ]

        total_needed = len(all_required)

        # Filter to missing only
        missing = []
        for sheet, is_base in all_required:
            downloaded_set = self.state.downloaded_base if is_base else self.state.downloaded_custom
            target_dir = self.base_dir if is_base else self.custom_dir
            rel_path = sheet.relative_path
            if rel_path not in downloaded_set and not (target_dir / rel_path).exists():
                missing.append(sheet)

        total_done = total_needed - len(missing)

        # Process in batches
        batch_downloaded = 0
        batch_bytes = 0

        for i, sheet in enumerate(missing):
            success = self.download_spritesheet(sheet)
            if success:
                batch_downloaded += 1
                batch_bytes += self._run_stats['bytes']

            # Yield after each batch
            if (i + 1) % batch_size == 0 or i == len(missing) - 1:
                current, max_req = self.rate_limiter.current_rate
                yield {
                    'downloaded': batch_downloaded,
                    'total_done': total_done + i + 1,
                    'total_needed': total_needed,
                    'rate_limited': current >= max_req - 1,
                    'bytes': batch_bytes,
                    'current_rate': f"{current}/{max_req}"
                }
                batch_downloaded = 0
                batch_bytes = 0

                # Save state
                self._save_state()


def estimate_sync_time(missing_count: int, rate_per_minute: int = 12) -> str:
    """Estimate time to complete sync."""
    minutes = missing_count / rate_per_minute
    if minutes < 60:
        return f"{minutes:.0f} minutes"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f} hours"
    days = hours / 24
    return f"{days:.1f} days"


def main():
    parser = argparse.ArgumentParser(
        description="Sync Infinite Fusion custom sprites to game directory"
    )
    parser.add_argument(
        '--game-dir',
        default='/mnt/f/dox/repos/infinitefusion',
        help='Path to Infinite Fusion game directory'
    )
    parser.add_argument(
        '--skip-indices',
        action='store_true',
        help='Skip fetching index files (use cached)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be downloaded without actually downloading'
    )
    parser.add_argument(
        '--estimate',
        action='store_true',
        help='Just show estimate of work needed'
    )

    args = parser.parse_args()

    sync = InfiniteFusionSync(
        game_dir=args.game_dir,
        verbose=not args.quiet
    )

    if args.estimate or args.dry_run:
        # Just fetch indices and show estimate
        sync.fetch_indices()
        base_req, custom_req = sync.get_required_spritesheets()

        # Count missing
        base_missing = sync._get_missing_sheets(base_req, sync.state.downloaded_base, sync.base_dir)
        custom_missing = sync._get_missing_sheets(custom_req, sync.state.downloaded_custom, sync.custom_dir)

        print("\n" + "=" * 60)
        print("SYNC ESTIMATE")
        print("=" * 60)
        print(f"Base spritesheets:   {len(base_req):>6} total, {len(base_missing):>6} missing")
        print(f"Custom spritesheets: {len(custom_req):>6} total, {len(custom_missing):>6} missing")

        total_missing = len(base_missing) + len(custom_missing)
        est_mb = total_missing * 0.5  # ~500KB average
        est_time = estimate_sync_time(total_missing)

        print(f"\nTotal to download:   {total_missing} files")
        print(f"Estimated size:      ~{est_mb:.0f} MB")
        print(f"Estimated time:      ~{est_time} (at {sync.config.max_requests_per_window}/min)")
        print("=" * 60)

        if args.dry_run:
            print("\nFirst 20 custom sheets that would be downloaded:")
            for sheet in sorted(custom_missing, key=lambda s: (s.head_id, s.variant))[:20]:
                url = f"{sync.config.custom_spritesheet_url}{sheet.head_id}/{sheet.filename}"
                print(f"  {url}")

        return

    sync.run(skip_indices=args.skip_indices)


if __name__ == "__main__":
    main()
