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
from src.model import coolerLDTformerZC, SpanEmbedder, SpanUnembedder
from src.utils import PageTable, ExperimentLogger, plot_multimetric_analysis, plot_dset_reconstruction
from src.data import CompositeIterator
from src.train import train_autoembed, train_denoise
from src.sample import (
    sample_viz_dset, 
    sample_viz_split_topology, 
    sample_viz_causal_sweep  # <--- Add this
)


def build_model(cfg: ExperimentConfig, device: torch.device):
    """Instantiate model from config."""
    model = coolerLDTformerZC(
        dim=cfg.model.dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        topo_dim=cfg.model.topo_dim,
        mlp_depth=cfg.model.mlp_depth,
        vocab_size=cfg.model.vocab_size,
        global_layer_interval=cfg.model.global_layer_interval,
        num_experts=cfg.model.num_experts,
        num_active=cfg.model.num_active,
        rope_base=cfg.model.rope_base,
        mlp_ratio=cfg.model.mlp_ratio,
        jitter_noise=cfg.model.jitter_noise,
        context_size=cfg.model.patch_embedder.context_size,
        stride=cfg.model.patch_embedder.stride,
        fourier_dim=cfg.model.patch_embedder.fourier_dim,
        window_size=cfg.model.window_size
    ).to(device)
    
    if cfg.training.compile:
        model = torch.compile(model, dynamic=cfg.training.compile_dynamic)
    
    return model


def build_components(cfg: ExperimentConfig, device: torch.device):
    """Build full component tuple expected by training functions."""
    if cfg.training.precision == "bf16":
        dtype = torch.bfloat16
    if cfg.training.precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    model = build_model(cfg, device).to(dtype=dtype)
    
    span_emb = SpanEmbedder(model.text_embed, model.patch_embedder)
    span_unemb = SpanUnembedder(model.text_head, model.patch_unembedder)
    
    page_table = PageTable(
        num_blocks=cfg.page_table.num_blocks,
        block_size=cfg.page_table.block_size,
        max_batch_size=cfg.page_table.max_batch_size,
        max_logical_blocks=cfg.page_table.max_logical_blocks,
        device=device
    )
    
    return (model, span_emb, span_unemb, page_table)


def build_train_config(cfg: ExperimentConfig) -> dict:
    """Convert ExperimentConfig to dict format expected by training functions."""
    if cfg.training.precision == "bf16":
        dtype = torch.bfloat16
    if cfg.training.precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    return {
        "ae_steps": cfg.training.ae_steps,
        "steps": cfg.training.steps,
        "lambda_coeff": cfg.training.lambda_coeff,
        "mode": cfg.training.mode,
        "dataset_mix": cfg.get_dataset_mix_dict(),
        "schedule_bounds": cfg.training.schedule_bounds,
        "batch_size": cfg.training.batch_size,
        "lr": cfg.training.optimizer.lr,
        "weight_decay": cfg.training.optimizer.weight_decay,
        "max_lr": cfg.training.optimizer.max_lr,
        "pct_start": cfg.training.optimizer.pct_start,
        "ae_lr": cfg.training.ae_optimizer.lr,
        "ae_weight_decay": cfg.training.ae_optimizer.weight_decay,
        "ae_max_lr": cfg.training.ae_optimizer.max_lr,
        "ae_pct_start": cfg.training.ae_optimizer.pct_start,
        "log_interval": cfg.logging.log_interval,
        "dtype":dtype
    }


def main():
    parser = argparse.ArgumentParser(description="Train field diffusion model")
    parser.add_argument("config", nargs="?", default=None, help="Path to TOML config")
    parser.add_argument("--mode", choices=["naive", "factorized"], help="Override training mode")
    parser.add_argument("--steps", type=int, help="Override training steps")
    parser.add_argument("--ae-steps", type=int, help="Override AE training steps")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    if args.mode:
        cfg.training.mode = args.mode
    if args.steps:
        cfg.training.steps = args.steps
    if args.ae_steps:
        cfg.training.ae_steps = args.ae_steps
    if args.no_compile:
        cfg.training.compile = False
    
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    
    print("=" * 60)
    print("Field Diffusion Training")
    print("=" * 60)
    print(f"Model: {cfg.model.dim}d, {cfg.model.depth}L, {cfg.model.num_heads}H")
    print(f"MoE: {cfg.model.num_experts} experts, {cfg.model.num_active} active")
    print(f"Training: {cfg.training.steps} steps, mode={cfg.training.mode}")
    print(f"Schedule: {cfg.training.schedule_bounds}")
    print(f"Dataset splits: {list(cfg.dataset_mix.keys())}")
    print("=" * 60)
    
    print("\nBuilding model...")
    components = build_components(cfg, device)
    model = components[0]
    
    val_iterator = CompositeIterator(device, config=cfg.get_dataset_mix_dict())
    logger = ExperimentLogger(output_dir=str(cfg.logging.output_dir))
    train_cfg = build_train_config(cfg)
    
    print(f"\nTraining: {cfg.training.mode.upper()} mode")
    
    model.param_init()
    df_ae = train_autoembed(components, train_cfg, logger)
    df_train = train_denoise(components, train_cfg, logger)
    
    if cfg.logging.sample_after_training:
        print("\nGenerating samples...")
        for res in cfg.sampling.resolutions:
            sample_cfg = {
                "mode": cfg.training.mode,
                "res": res,
                "num_samples": cfg.sampling.num_samples,
                "sampling_steps": cfg.sampling.steps,
                "target_logsnr": cfg.sampling.target_logsnr,
                "schedule_bounds": cfg.training.schedule_bounds,
            }
            
            res_strat = sample_viz_dset(components, val_iterator, sample_cfg)
            plot_dset_reconstruction(res_strat, logger, f"{cfg.training.mode}_stratified_{res}")
            
            res_split = sample_viz_split_topology(components, val_iterator, sample_cfg)
            plot_dset_reconstruction(res_split, logger, f"{cfg.training.mode}_split_{res}", show_map=True)
            # --- NEW: Causal Sweep Visualization ---
            # We check if 'enable_sweep' is set in config to avoid running it if not desired

            # --- NEW/UPDATED: Causal Sweep Visualization ---
            if getattr(cfg.sampling, "enable_sweep", False):
                print("Generating Causal Information Sweep from Video Data...")
                M = getattr(cfg.sampling, "sweep_length", 4)
                # Construct a sequence structure for the video iterator
                # This dictates the resolution and noise properties of each frame in the sequence
                video_seq_structure = [{
                    'res': res, 
                    'noise_mode': 'uniform', # Or 'split' if desired for video context
                    'noise_params': {'min_snr': -4.0, 'max_snr': 1.0} 
                } for _ in range(M)] # Ensure sequence length M
                sweep_cfg = {
                    "mode": cfg.training.mode,
                    "target_logsnr": cfg.sampling.target_logsnr,
                    "sampling_steps": cfg.sampling.steps,
                    "num_sweep_sequences": getattr(cfg.sampling, "sweep_count", 4),
                    "sequence_length": M,
                    "prefix_snr_range": getattr(cfg.sampling, "sweep_range", (2.0, -4.0)),
                    "video_sequence_structure": video_seq_structure, # Pass this
                    "video_source_name": getattr(cfg.sampling, "sweep_video_source", None), # Optional: specify a particular video split
                }
                
                fig_sweep = sample_viz_causal_sweep(components, val_iterator, sweep_cfg)
                logger.save_figure(fig_sweep, f"{cfg.training.mode}_causal_sweep")

    print("\nPlotting...")
    plot_multimetric_analysis(df_train, logger, f"multimetric_{cfg.training.mode}")
    
    print(f"\nDone! Results in {logger.run_dir}")
    print("\nPlotting...")
    plot_multimetric_analysis(df_train, logger, f"multimetric_{cfg.training.mode}")
    
    print(f"\nDone! Results in {logger.run_dir}")


if __name__ == "__main__":
    main()
