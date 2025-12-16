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
from src.train import train_autoembed, train_denoise
from src.plotting import plot_multimetric_analysis, ExperimentLogger, plot_dset_reconstruction
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
    
    # Sample
    if cfg['logging']['sample_after_training']:
        print("Sampling...")
        samp_cfg = cfg['sampling']
        
        for res in samp_cfg['resolutions']:
            # Construct dict for sampler
            s_dict = samp_cfg.copy()
            s_dict['mode'] = cfg['training']['mode']
            s_dict['res'] = res
            
            sampler.sample_viz_dset(components, val_iterator, s_dict, logger)
            
            if samp_cfg.get('enable_sweep'):
                 sampler.sample_viz_causal_sweep(components, val_iterator, s_dict, logger)


    print(f"\nDone! Results in {logger.run_dir}")


if __name__ == "__main__":
    main()
