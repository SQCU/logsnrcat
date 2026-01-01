# Sparse Texturemap Iterator Design Notes

## Overview

This document describes the design of a configurable iterator for sampling from sparse texture atlases (spritesheets) with support for logit-weighted sampling biases.

## Problem Statement

Texture atlases (spritesheets) aggregate multiple sprites into grid-based image files. When the atlas is **sparse** - meaning not all grid positions contain actual content - naive sampling approaches fail:

1. **Empty samples**: Random `(x, y)` grid positions may reference transparent/empty cells
2. **Dimension assumptions**: Hardcoded column counts cause position calculation errors
3. **Metadata unreliability**: External indices may not reflect actual pixel content

Observed failure rate with naive sampling: 43-55% empty samples.

## Solution Architecture

### Three-Layer Design

```
┌─────────────────────────────────────────────────────────────────┐
│ Training Pipeline Integration (src/sprite_atlas.py)            │
│ - SpriteAtlasIterator: yields ContextBlocks for training        │
│ - RenderConfig: pixel-art-aware rendering (background, scaling) │
│ - Compositing: RGBA → RGB with background generation            │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│ Validated Sampling Layer (data/*/iterator.py)                   │
│ - SamplingConfig: split, mode, adjustments, temperature         │
│ - Logit-weighted categorical sampling                           │
│ - Pattern matching for adjustment rules                         │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│ Ground Truth Layer (data/*/validated_prior.json)                │
│ - Pixel-scanned index of valid positions                        │
│ - Built by sprite_validator.py                                  │
│ - Cached to disk for fast loading                               │
└─────────────────────────────────────────────────────────────────┘
```

### Ground Truth: Validated Priors

Instead of trusting metadata to define what exists, we scan actual pixel content:

```python
# sprite_validator.py pseudocode
MIN_VISIBLE_PIXELS = 100
ALPHA_THRESHOLD = 128

for sheet_path in glob("**/*.png"):
    img = load_rgba(sheet_path)
    cols = img.width // SPRITE_SIZE
    rows = img.height // SPRITE_SIZE

    for grid_y in range(rows):
        for grid_x in range(cols):
            crop = extract_cell(img, grid_x, grid_y)
            visible = count(crop.alpha >= ALPHA_THRESHOLD)
            if visible >= MIN_VISIBLE_PIXELS:
                record_valid_position(sheet_path, grid_x, grid_y)
```

Output format (`validated_prior.json`):

```json
{
  "splits": {
    "custom": {
      "valid_positions": {
        "custom/spritesheets/134/134.png": [[0, 0, 1], [1, 0, 2], ...],
        ...
      },
      "total_valid_positions": 206963
    },
    "base": { ... },
    "procedural": { ... }
  }
}
```

### Sampling Configuration

```python
@dataclass
class SamplingConfig:
    split: str = "all"           # "custom" | "base" | "procedural" | "all"
    mode: str = "uniform_sprites" # "uniform_sprites" | "uniform_types" | "logit_weighted"
    type_key: str = "head"       # "head" | "body" | "both"
    adjustments: Dict[str, float] = {}  # Pattern -> logit delta
    adjustment_mode: str = "additive"   # "additive" | "multiplicative"
    temperature: float = 1.0
    seed: int = 42
```

### Pattern Language for Logit Adjustments

The `adjustments` dict supports a pattern language for targeting specific sprites:

| Pattern | Matches | Example |
|---------|---------|---------|
| `"134"` | head_id == 134 | Boost head type 134 |
| `"*.134"` | body_id == 134 | Boost body type 134 |
| `"134.6"` | head=134, body=6 | Specific fusion |
| `"procedural:*"` | split == "procedural" | All procedural sprites |
| `"custom:134"` | split=custom, head=134 | Custom sprites with head 134 |

Implementation:

```python
def _match_pattern(self, pattern: str, sprite: SpriteRef) -> bool:
    # Handle split prefix
    split_filter = None
    if ':' in pattern:
        split_part, pattern = pattern.split(':', 1)
        split_filter = split_part

    if split_filter and split_filter != '*':
        if sprite.split_name != split_filter:
            return False

    if pattern == '*':
        return True

    # Handle body wildcard: *.N
    if pattern.startswith('*.'):
        target_body = int(pattern[2:])
        return sprite.body_id == target_body

    # Head ID match
    target_head = int(pattern)
    return sprite.head_id == target_head
```

### Logit-Weighted Sampling

```python
def _compute_sampling_weights(self) -> torch.Tensor:
    base_logits = torch.zeros(len(self.sprites))

    for pattern, delta in self.config.adjustments.items():
        for i, sprite in enumerate(self.sprites):
            if self._match_pattern(pattern, sprite):
                if self.config.adjustment_mode == "additive":
                    base_logits[i] += delta
                else:  # multiplicative
                    base_logits[i] *= delta

    # Apply temperature
    scaled = base_logits / self.config.temperature
    return F.softmax(scaled, dim=0)
```

### Rendering Configuration

For pixel art, standard bilinear scaling destroys detail. We use pixel-art-aware rendering:

```python
@dataclass
class RenderConfig:
    res_scaling: str = "do_not"      # "do_not" | "crop_down_int_nn_up"
    background_mode: str = "noise"   # "noise" | "solid_random" | "checkerboard" | "gradient"
    jitter: bool = True              # Random placement within aperture
    native_size: int = 96            # Native sprite resolution
```

**Resolution scaling modes:**

- `"do_not"`: Keep native resolution, place with jitter, crop if target is smaller
- `"crop_down_int_nn_up"`: Integer nearest-neighbor upscaling (2x, 3x, etc.) or crop

**Background compositing:** RGBA sprites are composited onto generated backgrounds to avoid transparency during training.

## Configuration Integration

TOML config structure:

```toml
[dataset_mix.sprite_atlas]
type = "sprite_atlas"
ratio = 0.4
noise_mode = "uniform"

[dataset_mix.sprite_atlas.params]
data_dir = "data/sprite_atlas"

[dataset_mix.sprite_atlas.params.sampling_config]
split = "all"
mode = "uniform_sprites"
adjustment_mode = "additive"
temperature = 1.0
seed = 42

[dataset_mix.sprite_atlas.params.sampling_config.adjustments]
"134" = 0.1           # Head ID 134 +0.1 logits
"*.134" = 0.1         # Body ID 134 +0.1 logits
"procedural:*" = -0.1 # Procedural sprites -0.1 (prefer custom/base)

[dataset_mix.sprite_atlas.params.render_config]
res_scaling = "do_not"
background_mode = "noise"
jitter = true
```

## Key Design Decisions

### 1. Validated Prior as Ground Truth

**Decision**: Scan pixels to build sampling index rather than trusting metadata.

**Rationale**: Metadata describes intent/relationships; only pixel data describes what actually exists. This is especially important for compiled assets where the source-of-truth is the binary asset, not the generating process.

### 2. Nested Config Structure

**Decision**: Use `sampling_config` and `render_config` sub-dicts rather than flat params.

**Rationale**:
- Reduces parameter pollution at top level
- Clear separation of concerns (what to sample vs. how to render)
- Easier to extend either subsystem independently

### 3. Pattern Language over Separate Entries

**Decision**: Single dataset entry with pattern-based adjustments rather than multiple dataset_mix entries.

**Rationale**:
- Avoids duplicating boilerplate config across entries
- Keeps sampling logic in the iterator (where it belongs)
- More expressive (can combine split, head, body filters)

### 4. Logit-Space Adjustments

**Decision**: Work in logit space (pre-softmax) rather than probability space.

**Rationale**:
- Additive adjustments are more intuitive (+1 logit ≈ e^1 ≈ 2.7x more likely)
- Avoids denormalization issues when modifying probabilities directly
- Temperature scaling works naturally in logit space

### 5. Split as Metadata Field

**Decision**: Store `split_name` in `SpriteRef` rather than maintaining separate collections.

**Rationale**:
- Simpler data structure (single flat list of sprites)
- Split filtering happens at pattern-match time
- Enables cross-split patterns (e.g., "all custom with head 134")

## Validation

The `validate_logits.py` script samples with different adjustment configurations and analyzes resulting color distributions to verify that biasing works as expected.

```bash
python data/sprite_atlas/validate_logits.py
# Outputs:
#   validate_logits_figure.png  - Visual comparison
#   validate_logits_results.json - Quantitative metrics
```

## Future Extensions

1. **Per-sprite caching**: Cache individual sprite tensors for faster iteration
2. **Streaming validation**: Validate spritesheets on-demand rather than upfront
3. **Hierarchical patterns**: Support nested patterns like `"custom:134.*.fire"` (custom, head 134, any body, fire-tagged)
4. **Dynamic reweighting**: Adjust weights during training based on loss per-category
