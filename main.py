#!/usr/bin/env python3
"""
Config-driven training script for field diffusion.

Usage:
    python main.py configs/multisnr_default.toml
    python main.py configs/multisnr_default.toml --mode factorized
    python main.py configs/multisnr_default.toml --steps 1000
"""
import argparse
import torch
from src.config import load_config
from src.model import coolerLDTformerZC, SpanEmbedder, SpanUnembedder, PageTable
from src.data_iterator import CompositeIterator
from src.data_functional import get_tokenizer
from src.train import train_autoembed, train_denoise
from src.plotting import plot_multimetric_analysis, ExperimentLogger#, plot_dset_reconstruction
import src.sample as sampler


def build_model(cfg, device: torch.device):
    """Instantiate model from raw config dictionary."""
    m_cfg = cfg['model']
    p_cfg = m_cfg['patch_embedder']
    
    model = coolerLDTformerZC(
        dim=m_cfg['dim'],
        depth=m_cfg['depth'],
        num_heads=m_cfg['num_heads'],
        topo_dim=m_cfg['topo_dim'],
        mlp_depth=m_cfg['mlp_depth'],
        vocab_size=m_cfg['vocab_size'],
        global_layer_interval=m_cfg['global_layer_interval'],
        num_experts=m_cfg['num_experts'],
        num_active=m_cfg['num_active'],
        rope_base=m_cfg['rope_base'],
        mlp_ratio=m_cfg['mlp_ratio'],
        jitter_noise=m_cfg['jitter_noise'],
        context_size=p_cfg['context_size'],
        stride=p_cfg['stride'],
        fourier_dim=p_cfg['fourier_dim'],
        window_size=m_cfg['window_size']
    ).to(device)
    return model

def build_components(cfg, device):
    """Build full component tuple."""
    dtype_str = cfg['training']['precision']
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    dtype = cfg['dtype']

    model = build_model(cfg, device).to(dtype=dtype)
    if cfg['training']['compile']:
        model = torch.compile(model, dynamic=cfg['training']['compile_dynamic'])

    # Check if sparse AE is enabled
    sparse_ae_cfg = cfg['training']['sparse_ae']
    if sparse_ae_cfg['enabled']:
        from kmaze_ae.model_sparse_dim import (
            SparsePerDimFSQAutoencoder,
            SparseAEPatchEmbedder,
            SparseAEPatchUnembedder
        )

        # Build sparse AE with config
        attn_cfg = sparse_ae_cfg['attention']
        sparse_ae = SparsePerDimFSQAutoencoder(
            n_levels=sparse_ae_cfg['n_levels'],
            patch_size=sparse_ae_cfg['patch_size'],
            image_size=256,  # Will be dynamic per batch
            hidden_dim=sparse_ae_cfg['hidden_dim'],
            code_dim=sparse_ae_cfg['code_dim'],
            k_per_patch=sparse_ae_cfg['k_per_patch'],
            residual_scale=sparse_ae_cfg['residual_scale'],
            fourier_dim=sparse_ae_cfg['fourier_dim'],
            n_layers=sparse_ae_cfg['n_layers'],
            attn_config=attn_cfg
        ).to(device, dtype=dtype)

        if cfg['training']['compile']:
            sparse_ae = torch.compile(sparse_ae, dynamic=cfg['training']['compile_dynamic'])

        # Create interface wrappers (move to device/dtype)
        ae_embedder = SparseAEPatchEmbedder(sparse_ae, embed_dim=cfg['model']['dim']).to(device, dtype=dtype)
        ae_unembedder = SparseAEPatchUnembedder(sparse_ae, ae_embedder).to(device, dtype=dtype)

        # Use sparse AE wrappers instead of model's patch embedder/unembedder
        span_emb = SpanEmbedder(model.text_embed, ae_embedder)
        span_unemb = SpanUnembedder(model.text_head, ae_unembedder)

        print(f"[SparseAE] Enabled: {sparse_ae_cfg['n_levels']} levels, "
              f"code_dim={sparse_ae_cfg['code_dim']}, k={sparse_ae_cfg['k_per_patch']}, "
              f"attn={attn_cfg['mode']}")
    else:
        # Standard embedder/unembedder from model
        span_emb = SpanEmbedder(model.text_embed, model.patch_embedder)
        span_unemb = SpanUnembedder(model.text_head, model.patch_unembedder)

    # Page Table
    pt_cfg = cfg['page_table']
    page_table = PageTable(
        num_blocks=pt_cfg['num_blocks'],
        block_size=pt_cfg['block_size'],
        max_batch_size=pt_cfg['max_batch_size'],
        max_logical_blocks=pt_cfg['max_logical_blocks'],
        device=device
    )

    # Always return 4-tuple - sparse AE is integrated via span_emb/span_unemb
    return (model, span_emb, span_unemb, page_table)


def main():
    parser = argparse.ArgumentParser(description="Train field diffusion model")
    parser.add_argument("config", nargs="?", default=None, help="Path to TOML config")
    parser.add_argument("--mode", choices=["naive", "factorized"], help="Override training mode")
    parser.add_argument("--steps", type=int, help="Override training steps")
    parser.add_argument("--ae-steps", type=int, help="Override AE training steps")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    args = parser.parse_args()
    
    # 1. Load & Sanitize (Merges modular configs internally)
    cfg = load_config(args.config)
    # 2. Apply CLI Overrides
    if args.mode: cfg['training']['mode'] = args.mode
    if args.steps: cfg['training']['steps'] = args.steps
    if args.ae_steps: cfg['training']['ae_steps'] = args.ae_steps
    if args.no_compile: cfg['training']['compile'] = False
    
    dtype_str = cfg['training']['precision']
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    dtype = dtype_map[dtype_str]
    cfg['dtype'] = dtype
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    cfg['device'] = device
    
    # 3. Print Config Summary (Using Dict Access)
    print("=" * 60)
    print("Field Diffusion Training")
    print("=" * 60)
    print(f"Model: {cfg['model']['dim']}d, {cfg['model']['depth']}L, {cfg['model']['num_heads']}H")
    print(f"MoE: {cfg['model']['num_experts']} experts, {cfg['model']['num_active']} active")
    print(f"Training: {cfg['training']['steps']} steps, mode={cfg['training']['mode']}")
    print(f"Schedule: {cfg['training']['schedule_bounds']}")
    print(f"Dataset splits: {list(cfg['dataset_mix'].keys())}")
    print("=" * 60)
    
    print("\nBuilding model...")
    components = build_components(cfg, device)
    
    print("\nInitializing eval data tooling...")
    # Initialize tokenizer wrapper early
    tokenizer = get_tokenizer()
    # 4. Setup Training
    val_iterator = CompositeIterator(device, config=cfg['dataset_mix'], 
        caching_resolution=cfg['training']['bucketing']['caching_resolution'])
    logger = ExperimentLogger(output_dir=str(cfg['logging']['output_dir']))
    
    print(f"\nTraining: {cfg['training']['mode'].upper()} mode")
    # Use components[0] for model access (param_init)
    components[0].param_init()

    
    # 5. Run Training
    df_ae = train_autoembed(components, cfg, val_iterator, logger)
    df_train = train_denoise(components, cfg, val_iterator, logger)

    print("\nPlotting Metrics...")
    plot_multimetric_analysis(df_train, logger, f"multimetric_{cfg['training']['mode']}")
    

    # --- Sampling & Evaluation ---
    if cfg['logging']['sample_after_training']:
        print("Sampling...")
        samp_cfg = cfg['sampling']

        use_amp = (dtype == torch.bfloat16) or (dtype == torch.float16)
        # FIX: Use new torch.amp API
        scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))
        with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
            # 0. AE vs Diffusion Diagnostic (NEW)
            # Helps identify whether issues stem from AE compression or diffusion
            print("Running AE vs Diffusion Diagnostic...")
            for res in samp_cfg['resolutions'][:2]:  # Run for first 2 resolutions
                s_dict = samp_cfg.copy()
                s_dict['mode'] = cfg['training']['mode']
                s_dict['res'] = res
                sampler.diagnostic_ae_vs_diffusion(components, val_iterator, s_dict, logger)

            # 1. Dataset Reconstruction (Latent Refinement)
            for res in samp_cfg['resolutions']:
                s_dict = samp_cfg.copy()
                s_dict['mode'] = cfg['training']['mode']
                s_dict['res'] = res
                sampler.sample_viz_dset(components, val_iterator, s_dict, logger)
                
            # 2. Causal Sweep (Video Gen) - Now with Resolution Sweep
            if samp_cfg['enable_sweep']:
                print("Running Causal Sweep...")
                for res in samp_cfg['resolutions']:
                    s_dict = samp_cfg.copy()
                    s_dict['mode'] = cfg['training']['mode']
                    s_dict['res'] = res # Config injection for iterator
                    sampler.sample_viz_causal_sweep(components, val_iterator, s_dict, logger)
                    
                # 3. Custom Queries (Text / Mixed)
            if samp_cfg['queries']:
                print(f"Running {len(samp_cfg['queries'])} custom eval sessions...")
    
                # Seed Context: Use FIRST split explicitly (avoids mixed resolution/type issues)
                # For text generation, we want consistent context from a single source
                split_names = val_iterator.get_split_names()
                if not split_names:
                    print("    No splits available for seed context, skipping queries")
                else:
                    # Prefer a functional split (checkerboard/torus) for consistent text+latent pairs
                    # Fall back to first available split
                    seed_split = split_names[0]
                    for name in split_names:
                        if 'checker' in name or 'torus' in name:
                            seed_split = name
                            break
    
                    print(f"    Seeding context from split: {seed_split}")
                    try:
                        seed_batch = val_iterator.generate_from_split(seed_split, count=4, resolution=32)
                        seed_ctx = sampler.MultiTurnContext(seed_batch)
    
                        # Execute
                        results = sampler.execute_multiturn_session(components, seed_ctx, samp_cfg['queries'])
    
                        # Text Decoding & Logging
                        print("\n--- Eval Session Outputs ---")
                        for i, b in enumerate(results):
                            if b.type == 'text':
                                try:
                                    text = tokenizer.decode(b.content)
                                    print_msg = f"Block {i} (Text): {text[:200]}... (Len: {len(b.content)})"
                                    print(print_msg)
                                    log_msg = f"{print_msg}\n{text}\n"
                                    logger.log_text("eval_outputs.txt", log_msg)
                                except Exception as e:
                                    print(f"Failed to decode text block {i}: {e}")
                            elif b.type == 'latent':
                                print(f"Block {i} (Latent): shape={b.content.shape}")
                    except Exception as e:
                        print(f"    Error running queries: {e}")

    print(f"\nDone! Results in {logger.run_dir}")

if __name__ == "__main__":
    main()