#!/usr/bin/env python3
"""
Config-driven training script for field diffusion.

Usage:
    python main.py configs/multisnr_default.toml
    python main.py configs/multisnr_default.toml --mode factorized
    python main.py configs/multisnr_default.toml --steps 1000
"""
# Force non-interactive matplotlib backend BEFORE any imports that might use pyplot
# This prevents Tcl/Tk threading crashes when figures are GC'd from non-main thread
import matplotlib
matplotlib.use('Agg')

import argparse
import torch

# Apply patches for upstream bugs BEFORE any torch.compile() calls
from src.patches import apply_all as apply_patches
apply_patches()

# Enable caching for flex attention backward - prevents recompilation overhead
torch._inductor.config.unsafe_marked_cacheable_functions['torch.ops.higher_order.flex_attention_backward'] = True

from src.config import load_config
from src.model import coolerLDTformerZC, SpanEmbedder, SpanUnembedder, PageTable
from src.data_iterator import CompositeIterator
from src.data_functional import get_tokenizer
from src.train import train_autoembed, train_denoise, train_latent_diffusion
from src.plotting import (
    plot_multimetric_analysis, plot_ae_roundtrip, plot_loss_schedule_analysis,
    plot_subspace_sensitivity, plot_subspace_sensitivity_heatmap,
    plot_subspace_sensitivity_exemplars, ExperimentLogger
)
import src.sample as sampler


def build_model(cfg, device: torch.device):
    """Instantiate model from raw config dictionary."""
    m_cfg = cfg['model']
    p_cfg = m_cfg['patch_embedder']

    # Pass sparse AE config to model - it will create the AE as a submodule
    sparse_ae_cfg = cfg['training']['sparse_ae'] if 'training' in cfg else None

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
        window_size=m_cfg['window_size'],
        sparse_ae_config=sparse_ae_cfg
    ).to(device)
    return model

def build_components(cfg, device):
    """Build full component tuple."""
    dtype = cfg['dtype']

    model = build_model(cfg, device).to(dtype=dtype)

    # Save references to submodules BEFORE compile (compiled model still exposes these,
    # but explicitly saving avoids any potential issues with attribute access)
    text_embed = model.text_embed
    text_head = model.text_head
    patch_embedder = model.patch_embedder
    patch_unembedder = model.patch_unembedder

    # Compile transformer layers if requested
    # NOTE: torch.compile + GraphRunner conflict for flex_attention:
    # - GraphRunner pre-creates BlockMasks at init with CONCRETE dimensions (L=max_ctx)
    # - torch.compile(dynamic=True) traces with SYMBOLIC dimensions (s27, s58...)
    # - When compiled model calls flex_attention with pre-created mask, Inductor fails
    #   ("unbacked_bindings" - mask's concrete L doesn't match trace's symbolic shapes)
    # - Sparse AE works fine because it creates masks DURING forward (inside trace context)
    graph_capture_enabled = cfg['training'].get('graph_capture', {}).get('enabled', False)

    if cfg['training']['compile']:
        # Sparse AE transformers: always safe to compile (masks created inline during forward)
        if model.uses_sparse_ae:
            for enc in model.sparse_ae.encoders:
                enc.transformer = torch.compile(enc.transformer)
            for dec in model.sparse_ae.decoders:
                dec.transformer = torch.compile(dec.transformer)

        # Main model: only compile if NOT using GraphRunner (which pre-creates masks)
        if not graph_capture_enabled:
            model = torch.compile(model, dynamic=cfg['training']['compile_dynamic'])
        else:
            print("[INFO] graph_capture enabled - skipping torch.compile for main model")
            print("       (GraphRunner pre-creates masks; torch.compile traces with symbolic shapes)")
            print("       (CUDA graph capture provides equivalent optimization)")

    # Wrap embedders in SpanEmbedder/SpanUnembedder
    # For latent diffusion (SwiGLU AE), the embedder has internal attention that needs
    # proper mask config for batched processing during sampling/diagnostics
    sparse_ae_cfg = cfg['training'].get('sparse_ae', {})
    if sparse_ae_cfg.get('enabled', False):
        # Use sparse AE's attention config for embedder mask building
        emb_attn_config = sparse_ae_cfg.get('attention', {
            'mode': 'sliding',
            'window_size': 2,
            'n_global_tokens': 0
        })
    else:
        # Pixel diffusion: embedder has no attention, use full (won't be used)
        emb_attn_config = {'mode': 'full', 'window_size': 0, 'n_global_tokens': 0}

    span_emb = SpanEmbedder(text_embed, patch_embedder, attn_config=emb_attn_config)
    span_unemb = SpanUnembedder(text_head, patch_unembedder, attn_config=emb_attn_config)

    # Page Table
    pt_cfg = cfg['page_table']
    page_table = PageTable(
        num_blocks=pt_cfg['num_blocks'],
        block_size=pt_cfg['block_size'],
        max_batch_size=pt_cfg['max_batch_size'],
        max_logical_blocks=pt_cfg['max_logical_blocks'],
        device=device
    )

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
    # Save config immediately for reproducibility (before any training)
    logger.save_config(cfg, "config.json")

    print(f"\nTraining: {cfg['training']['mode'].upper()} mode")
    # param_init is called in model's __init__, no need to call again

    # 5. Run Training
    df_ae = train_autoembed(components, cfg, val_iterator, logger)
    # Plot AE warmup metrics if AE training was run
    if cfg['training']['ae_steps'] > 0 and not df_ae.empty:
        # Save dataframe BEFORE plotting (crash safety)
        logger.save_dataframe(df_ae, f"history_ae_{cfg['training']['mode']}")
        plot_multimetric_analysis(df_ae, logger, f"multimetric_ae_{cfg['training']['mode']}")

        # Loss schedule analysis (MSE/BCE compatibility) if scheduled loss was used
        loss_schedule_cfg = cfg['training']['sparse_ae'].get('loss_schedule', {})
        if isinstance(loss_schedule_cfg, dict) and loss_schedule_cfg.get('enabled', False):
            print("Plotting MSE/BCE loss schedule analysis...")
            plot_loss_schedule_analysis(df_ae, logger, f"loss_schedule_ae_{cfg['training']['mode']}")

        # AE reconstruction quality with round-trip analysis
        print("Plotting AE round-trip reconstruction...")
        for res in cfg['sampling']['resolutions'][:2]:
            plot_ae_roundtrip(components, val_iterator, logger,
                              name=f"ae_roundtrip_{res}", n_samples=8, resolution=res)

        # Subspace sensitivity sweep (for wavelet-gating FSQ AE)
        sens_cfg = cfg['sampling']['subspace_sensitivity']
        if sens_cfg['enabled'] and cfg['training']['sparse_ae'].get('wavelet_gating', False):
            print("\nRunning subspace sensitivity sweep...")
            model = components[0]  # coolerLDTformerZC
            if hasattr(model, 'sparse_ae') and model.sparse_ae is not None:
                results_by_res = {}
                for res in sens_cfg['resolutions']:
                    print(f"  Sensitivity sweep at resolution {res}...")
                    # Get batch of images for evaluation
                    # ContextBlocks are heterogeneous by design - filter to target resolution
                    blocks = val_iterator.generate_batch_list(
                        batch_size=sens_cfg['n_samples'] * 4,  # Over-generate to filter
                        resolution=res
                    )
                    # Filter to blocks matching target resolution and stack
                    matching = [b.content for b in blocks
                                if b.content.shape[-1] == res and b.content.shape[-2] == res]
                    if len(matching) < sens_cfg['n_samples']:
                        print(f"    Warning: only {len(matching)} blocks at {res}px (wanted {sens_cfg['n_samples']})")
                    images = torch.stack(matching[:sens_cfg['n_samples']]).to(cfg['device'])

                    # Run sensitivity sweep - must use same autocast as training
                    use_amp = (dtype == torch.bfloat16) or (dtype == torch.float16)
                    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
                        sweep_results = model.sparse_ae.subspace_sensitivity_sweep(
                            images=images,
                            ablation_rates=sens_cfg['ablation_rates'],
                            n_trials=sens_cfg['n_trials']
                        )
                    results_by_res[res] = sweep_results

                    # Plot individual resolution results
                    plot_subspace_sensitivity(sweep_results, logger,
                                              name=f"subspace_sensitivity_{res}")

                    # Plot visual exemplars showing what ablation looks like
                    plot_subspace_sensitivity_exemplars(
                        model.sparse_ae, images, logger,
                        ablation_rates=[0.0, 0.25, 0.5, 0.75, 1.0],
                        name=f"subspace_exemplars_{res}",
                        dtype=dtype
                    )

                # Plot cross-resolution heatmap if multiple resolutions
                if len(results_by_res) > 1:
                    plot_subspace_sensitivity_heatmap(results_by_res, logger,
                                                      name="subspace_sensitivity_heatmap")
                print("  Sensitivity sweep complete.")

    # Use latent diffusion if sparse AE is enabled with latent_diffusion=true
    sparse_ae_cfg = cfg['training'].get('sparse_ae', {})
    use_latent_diffusion = (
        sparse_ae_cfg.get('enabled', False) and
        sparse_ae_cfg.get('latent_diffusion', False)
    )

    if use_latent_diffusion:
        print("\n[Main] Using LATENT diffusion (noise in code space)")
        df_train = train_latent_diffusion(components, cfg, val_iterator, logger)
    else:
        print("\n[Main] Using PIXEL diffusion (noise in pixel space)")
        df_train = train_denoise(components, cfg, val_iterator, logger)

    print("\nPlotting Metrics...")
    # Save dataframe BEFORE plotting (crash safety)
    if not df_train.empty:
        logger.save_dataframe(df_train, f"history_denoise_{cfg['training']['mode']}")
    # Plot diffusion training metrics
    plot_multimetric_analysis(df_train, logger, f"multimetric_{cfg['training']['mode']}")

    # Loss schedule analysis for latent diffusion (v-field MSE/BCE compatibility)
    if use_latent_diffusion and not df_train.empty:
        diffusion_loss_schedule = sparse_ae_cfg.get('diffusion_loss_schedule', {})
        if isinstance(diffusion_loss_schedule, dict) and diffusion_loss_schedule.get('enabled', False):
            print("Plotting diffusion loss schedule analysis...")
            plot_loss_schedule_analysis(df_train, logger, f"loss_schedule_diffusion_{cfg['training']['mode']}")

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

    # --- Eval Server Integration ---
    eval_server_cfg = cfg['logging']['eval_server']
    if eval_server_cfg['enabled']:
        from src.eval_server import yeet_to_server, query_health

        print(f"\nYeeting weights to eval server at {eval_server_cfg['url']}...")
        model = components[0]  # coolerLDTformerZC
        yeet_success = yeet_to_server(model, eval_server_cfg['url'])

        if yeet_success and eval_server_cfg['health_check']:
            health = query_health(eval_server_cfg['url'])
            print(f"\n[Eval Server Health Check]")
            print(f"  Status: {health.get('status', 'unknown')}")
            print(f"  Weights loaded: {health.get('weights_loaded', False)}")
            print(f"  Params: {health.get('params', 0):,}")
            if not health.get('weights_loaded'):
                print(f"  WARNING: Weights not confirmed on server!")
        elif not yeet_success:
            print(f"  WARNING: Failed to yeet weights to eval server")

    print(f"\nDone! Results in {logger.run_dir}")

if __name__ == "__main__":
    main()