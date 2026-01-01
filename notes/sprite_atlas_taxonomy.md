# Sprite Atlas Taxonomy: Data Structures and Iterator Challenges

## Overview

"Sprite atlas" is an umbrella term covering fundamentally different data structures with distinct organizational principles, sampling semantics, and rendering requirements. This document surveys three major categories and their implications for training data iteration.

---

## Category 1: Foreground Composition Sprites

**Examples:** Character fusion sprites, paper doll systems, equipment overlays

**Current Implementation:** `src/sprite_atlas.py`, `data/*/iterator.py`

### Organizational Principle

Sprites exist at intersections of categorical indices - a 2D (or higher) combinatorial space where each axis represents a component category. In the fusion sprite case: `head_id × body_id × variant`. The space is sparse; not all combinations have corresponding pixel content.

### Validation Problem

*"Does pixel content exist at grid position (x, y)?"*

Solved via validated priors: scan actual pixel alpha channels, cache the set of valid positions. Sampling operates over this validated subset.

### Rendering Pipeline

Sprites are RGBA with meaningful transparency boundaries. Rendering = alpha-composite onto synthetic background. The sprite *is* the foreground; background is generated (noise, gradient, solid color).

### Sampling Semantics

Sample individual sprites. Logit biases operate over the index space:
- `"134"` → boost head_id 134
- `"*.25"` → boost body_id 25
- `"procedural:*"` → adjust entire split

Unit of training data: single composited sprite image.

---

## Category 2: Animation Sequence Sprites

**Examples:** Doom/Quake WAD sprites, fighting game frame data, skeletal animation spritesheets

### Organizational Principle

Sprites encode **temporal sequences** with **viewpoint variants**. Naming conventions embed semantic structure:

```
TROOA1    = monster TROO, state A, frame 1
TROOA2A8  = state A, frame 2, angles 2 and 8 (mirrored)
SARGA1-H1 = SARG walking, 8 rotation angles
```

The organizational axes are: entity → state (walk/attack/death/pain) → frame index → rotation angle.

### Validation Problem

*"Is this animation sequence complete and consistent?"*

Validation must check:
- All frames of a state exist
- All rotation angles are present (or mirrorable)
- Anchor offsets are consistent across frames (no jitter)
- Palette/colorkey transparency is correctly specified

A missing frame breaks the entire sequence.

### Rendering Pipeline

Sprites use **palette-indexed color** with colorkey transparency (cyan `#00FFFF`, magenta `#FF00FF`). Rendering requires:
1. Colorkey extraction → alpha mask
2. Offset metadata application (sprites have anchor points)
3. Consistent framing across animation sequences

Unlike foreground sprites, the *offset* matters - death animations where the sprite "falls" require consistent coordinate systems.

### Sampling Semantics

Sample **sequences**, not individual frames:
- "Complete walk cycle of TROO from angle 3"
- "Frame 4 of every monster's death animation"
- "All attack wind-up frames across all entities"

Pattern language would need hierarchical queries:
```
"TROO:walk:*"       # All walk frames
"*:death:1-5"       # First 5 death frames, all entities
"*:*:*:angle=3"     # Everything from viewing angle 3
```

Unit of training data: animation sequence (possibly as video) or frame-in-context with sequence metadata.

---

## Category 3: Background Tiling Sprites

**Examples:** RPG Maker tilesets, platformer terrain, city builder assets, autotile systems

### Organizational Principle

Tiles are **compositional primitives** that combine to form coherent regions. The individual tile is semantically meaningless; meaning emerges from **adjacency relationships**.

Tileset organization:
- **Autotiles:** 16-tile (basic) or 47-tile (blob) sets encoding all neighbor configurations
- **Wang tiles:** Edge-colored tiles where matching edges must connect
- **Layered systems:** A-tiles (terrain), B-tiles (structures), C-tiles (decorations) with z-ordering

### Validation Problem

*"Does this tileset provide complete adjacency coverage?"*

A 47-tile blob autotile missing one corner configuration will produce glitches when that specific 8-neighbor pattern occurs. Validation requires:
- Checking combinatorial completeness of transition variants
- Verifying seamless tiling (edge pixels match across tile boundaries)
- Confirming layer compatibility (overlay tiles have correct transparency)

### Rendering Pipeline

**The rendering problem inverts.** These sprites ARE the background. There is no "composite onto background" step - instead:

1. Generate valid tilemap via constraint propagation
2. Render tilemap layers in z-order
3. Foreground sprites (Category 1) composite on top

The "background mode" concept (noise, gradient, checkerboard) becomes irrelevant. The equivalent is "tilemap generation mode" - itself a complex sampling problem.

### Sampling Semantics

Sample **valid tilemaps** or **tile-in-context**:
- "3×3 patches containing grass-water transitions"
- "Cliff edges with shadow overlays"
- "Valid 16×16 dungeon room layouts"

Sampling requires **constraint satisfaction**. You cannot shuffle tiles randomly - a water tile cannot appear adjacent to a cliff-top tile without an intervening transition. Options:

1. **Wave Function Collapse:** Propagate constraints to generate valid tilemaps
2. **Template sampling:** Hand-authored valid regions, sample from those
3. **Contextual windows:** Sample a center tile conditioned on valid 8-neighborhood

Pattern language operates over **spatial predicates**:
```
"transition:grass→water"           # Coastline tiles
"corner:3-way"                      # Triple junctions
"interior:uniform:forest"           # Boring interior regions (downweight)
```

Unit of training data: tilemap region, not individual tile.

---

## The Meta-Observation: Tileset Rendering Is Generative Modeling

Correct application of background tilesets is not a simple rendering operation - it is itself a **generative modeling problem** requiring:

1. **Training data:** Valid tilemap examples demonstrating correct adjacency patterns, layer composition, and aesthetic coherence

2. **Score functions:** Measures of tilemap validity (constraint satisfaction), aesthetic quality (variety, spatial coherence), and semantic correctness (water flows downhill, shadows face consistent direction)

3. **Model-based samplers:** Wave Function Collapse, constraint propagation, learned tilemap generators - systems that produce valid outputs from the exponentially large space of possible tile arrangements

This creates a **recursive structure**: to generate training data for an image model using tileset backgrounds, you first need a tileset composition model to generate valid tilemaps. The iterator for Category 3 sprites is not a data loader but a **generative pipeline**.

### Implications for Iterator Design

The current `SpriteAtlasIterator` architecture assumes:
- Sampling is selection from a pre-existing validated set
- Rendering is composition of selected sprite onto generated background
- No inter-sample dependencies (each sample is independent)

Tileset iteration would require:
- Sampling is **generation** under constraints
- Rendering is **multi-layer composition** with z-ordering
- Samples may share context (sequential frames of exploring a generated dungeon)

This suggests that rather than extending `SpriteAtlasIterator` to handle tilesets, the correct architecture is:

```
TilesetIterator
├── tilemap_generator: WFCSampler | TemplateSampler | LearnedGenerator
├── layer_compositor: handles A/B/C tile z-ordering
├── foreground_injector: places Category 1 sprites into scene
└── camera_sampler: extracts training crops from large tilemaps
```

The iterator becomes an **orchestrator of generative subsystems** rather than a selector from static assets.

---

## Summary Table

| Aspect | Foreground Sprites | Animation Sprites | Background Tiles |
|--------|-------------------|-------------------|------------------|
| **Unit** | Single sprite | Animation sequence | Tilemap region |
| **Indexing** | Categorical (head×body) | Hierarchical (entity→state→frame→angle) | Spatial (x,y,layer) |
| **Validation** | Pixel content exists | Sequence complete | Adjacency coverage |
| **Transparency** | Alpha channel | Colorkey palette | Layer-dependent |
| **Sampling** | Select from valid set | Select sequences | Generate under constraints |
| **Rendering** | Composite onto background | Apply offsets, extract colorkey | Multi-layer composition |
| **Dependencies** | Independent | Temporal (frame order) | Spatial (neighbor constraints) |

---

## Future Directions

1. **Animation iterator:** Extend pattern language for temporal queries, implement sequence-aware sampling, handle offset metadata

2. **Tileset generator:** Implement WFC or learned tilemap generation as a preprocessing stage, then sample crops from generated maps

3. **Unified scene iterator:** Compose all three categories - generate tilemap background, place animated NPCs, composite foreground effects - producing coherent "game screenshot" training data

4. **Constraint specification language:** Formalize adjacency rules, layer ordering, and animation timing as declarative constraints that different iterator implementations can consume
