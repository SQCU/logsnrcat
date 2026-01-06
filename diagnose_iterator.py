"""
Diagnose data iterator resolution distribution.
"""
import torch
import time
from collections import Counter
from pathlib import Path

def main():
    from src.config import load_config
    from src.data_iterator import CompositeIterator

    cfg = load_config(Path("configs/sparse_ae_swiglu_shared.toml"))
    device = torch.device('cuda')

    val_iterator = CompositeIterator(
        device,
        config=cfg['dataset_mix'],
        caching_resolution=cfg['training']['bucketing']['caching_resolution']
    )

    print("=" * 60)
    print("ITERATOR RESOLUTION DIAGNOSTIC")
    print("=" * 60)

    # Generate a batch and check resolutions
    print("\nGenerating 100 blocks...")
    t0 = time.perf_counter()
    blocks = val_iterator.generate_batch_list(100)
    t1 = time.perf_counter()
    print(f"  Time: {(t1-t0)*1000:.1f}ms for 100 blocks = {(t1-t0)*10:.1f}ms/block")

    # Count resolutions
    res_counts = Counter()
    source_counts = Counter()
    source_res = {}

    for b in blocks:
        if b.type == 'latent':
            h, w = b.content.shape[-2:]
            res_counts[(h, w)] += 1
            source_counts[b.source] += 1
            if b.source not in source_res:
                source_res[b.source] = set()
            source_res[b.source].add((h, w))

    print("\n" + "-" * 40)
    print("Resolution distribution:")
    for res, count in sorted(res_counts.items()):
        pct = count / len(blocks) * 100
        print(f"  {res[0]}x{res[1]}: {count} ({pct:.1f}%)")

    print("\n" + "-" * 40)
    print("Source distribution:")
    for source, count in sorted(source_counts.items()):
        pct = count / len(blocks) * 100
        resolutions = source_res.get(source, set())
        res_str = ", ".join(f"{r[0]}x{r[1]}" for r in sorted(resolutions))
        print(f"  {source}: {count} ({pct:.1f}%) - outputs: {res_str}")

    # Test filtering efficiency for 64px
    print("\n" + "-" * 40)
    print("64px filtering efficiency:")
    matching_64 = [b for b in blocks if b.content.shape[-1] == 64]
    print(f"  Generated: 100 blocks")
    print(f"  64px matches: {len(matching_64)} ({len(matching_64)}%)")
    print(f"  Waste ratio: {(100 - len(matching_64))/max(1,len(matching_64)):.1f}x")

    # Time breakdown by source
    print("\n" + "-" * 40)
    print("Per-source timing (10 blocks each):")

    for split in val_iterator.splits:
        name = split['name']
        t0 = time.perf_counter()
        try:
            blocks = val_iterator.generate_from_split(name, count=10, resolution=64)
            t1 = time.perf_counter()
            avg_ms = (t1 - t0) / 10 * 1000
            res_set = set(b.content.shape[-1] for b in blocks if b.type == 'latent')
            print(f"  {name}: {avg_ms:.2f}ms/block, outputs: {res_set}")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    # Recommend fix
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print("""
The sprite_atlas uses native 96px resolution with res_scaling='do_not'.
When filtering for 64px, most sprite_atlas outputs are discarded.

Options:
1. Use generate_from_split() for specific sources instead of filtering
2. Change sprite_atlas res_scaling to 'crop_down_int_nn_up'
3. Change profiling resolution to 96px to match sprite_atlas
4. Reduce sprite_atlas ratio in dataset mix for testing
""")


if __name__ == "__main__":
    main()
