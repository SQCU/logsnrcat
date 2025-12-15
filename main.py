#!/usr/bin/env python3
"""
Config-driven training script for field diffusion.

Usage:
    python main.py configs/multisnr_default.toml
    python main.py configs/multisnr_default.toml --mode factorized
    python main.py configs/multisnr_default.toml --steps 1000
"""
import argparse
import sys
from pathlib import Path

import torch

from src.config import load_config, ExperimentConfig
from src.model import coolerLDTformerZC, SpanEmbedder, SpanUnembedder, PageTable
from src.utils import ExperimentLogger, plot_multimetric_analysis, plot_dset_reconstruction
from src.data import CompositeIterator
from src.train import train_autoembed, train_denoise
from src.sample import (
    sample_viz_dset, 
    sample_viz_split_topology, 
    sample_viz_causal_sweep  # <--- Add this
)



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
    
    if cfg['training']['compile']:
        model = torch.compile(model, dynamic=cfg['training']['compile_dynamic'])
    
    return model


def build_components(cfg, device: torch.device):
    """Build full component tuple using dictionary lookups."""
    # 1. Model
    dtype_str = cfg['training']['precision']
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    dtype = dtype_map[dtype_str]
    
    model = build_model(cfg, device).to(dtype=dtype)
    
    # 2. Helpers
    span_emb = SpanEmbedder(model.text_embed, model.patch_embedder)
    span_unemb = SpanUnembedder(model.text_head, model.patch_unembedder)
    
    # 3. Page Table
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
    
    # 1. Load & Sanitize (Returns Dict)
    cfg = load_config(args.config)
    dtype_str = cfg['training']['precision']
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    dtype = dtype_map[dtype_str]
    cfg['dtype'] = dtype

    # 2. Apply CLI Overrides to Dict
    if args.mode:
        cfg['training']['mode'] = args.mode
    if args.steps:
        cfg['training']['steps'] = args.steps
    if args.ae_steps:
        cfg['training']['ae_steps'] = args.ae_steps
    if args.no_compile:
        cfg['training']['compile'] = False
    
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    
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
    model = components[0]
    
    # 4. Setup Training
    val_iterator = CompositeIterator(device, config=cfg['dataset_mix'])
    logger = ExperimentLogger(output_dir=str(cfg['logging']['output_dir']))
    
    print(f"\nTraining: {cfg['training']['mode'].upper()} mode")
    
    model.param_init()
    
    # 5. Run Training (Pass full Dict)
    df_ae = train_autoembed(components, cfg, logger)
    df_train = train_denoise(components, cfg, logger)


    print("\nPlotting Metrics...")
    plot_multimetric_analysis(df_train, logger, f"multimetric_{cfg['training']['mode']}")
    
    use_amp = (dtype_str != "fp32")
    # 6. Sampling & Plotting
    if cfg['logging']['sample_after_training']:
        with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
            print("\nGenerating samples...")
            samp_cfg = cfg['sampling']
            
            for res in samp_cfg['resolutions']:
                # Create a localized config dict for the sampler
                local_samp_config = {
                    "mode": cfg['training']['mode'],
                    "res": res,
                    "num_samples": samp_cfg['num_samples'],
                    "sampling_steps": samp_cfg['steps'],
                    "target_logsnr": samp_cfg['target_logsnr'],
                    "schedule_bounds": cfg['training']['schedule_bounds'],
                }
                
                # Stratified
                res_strat = sample_viz_dset(components, val_iterator, local_samp_config)
                plot_dset_reconstruction(res_strat, logger, f"{cfg['training']['mode']}_stratified_{res}")
                
                # Split Topology
                res_split = sample_viz_split_topology(components, val_iterator, local_samp_config)
                plot_dset_reconstruction(res_split, logger, f"{cfg['training']['mode']}_split_{res}", show_map=True)
                
                # Causal Sweep (Video)
                # --- CAUSAL SWEEP LOGIC ---
                if samp_cfg.get('enable_sweep', False):
                    print(f"Generating Causal Information Sweeps (Res: {res})...")
                    
                    # 1. Identify Target Video Sources
                    # Look at dataset_mix to find all inputs of type 'video'
                    video_sources = []
                    
                    # Check for an explicit override in config
                    explicit_source = samp_cfg.get('sweep_video_source')
                    
                    if explicit_source:
                        # If user forced one, use only that (and fail downstream if it doesn't exist)
                        video_sources.append(explicit_source)
                    else:
                        # Otherwise, AUTO-DISCOVER all video splits
                        for name, split_cfg in cfg['dataset_mix'].items():
                            if split_cfg.get('type') == 'video':
                                video_sources.append(name)
                    
                    if not video_sources:
                        print("ℹ️ No video sources found for causal sweep.")
                    
                    # 2. Iterate and Generate
                    M = samp_cfg.get('sweep_length', 4)
                    
                    # Define structure for the Iterator (how to format the batch)
                    video_seq_structure = [{
                        'res': res, 
                        'noise_mode': 'uniform',
                        'noise_params': {'min_snr': -4.0, 'max_snr': 1.0} 
                    } for _ in range(M)]
                    
                    for source_name in video_sources:
                        # Construct precise config for THIS source
                        sweep_cfg = {
                            "mode": cfg['training']['mode'],
                            "target_logsnr": samp_cfg['target_logsnr'],
                            "sampling_steps": samp_cfg['steps'],
                            "num_sweep_sequences": samp_cfg.get('sweep_count', 4),
                            "sequence_length": M,
                            "prefix_snr_range": samp_cfg.get('sweep_range', (2.0, -4.0)),
                            "video_sequence_structure": video_seq_structure,
                            # THE KEY FIX: Explicitly pass the name we want
                            "video_source_name": source_name, 
                        }
                        
                        fig_sweep = sample_viz_causal_sweep(components, val_iterator, sweep_cfg)
                        if fig_sweep:
                            logger.save_figure(fig_sweep, f"{cfg['training']['mode']}_sweep_{source_name}_{res}")

    print(f"\nDone! Results in {logger.run_dir}")


if __name__ == "__main__":
    main()
