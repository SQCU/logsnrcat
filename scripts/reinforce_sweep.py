#!/usr/bin/env python3
"""
Comprehensive REINFORCE sweep for histogram fidelity.

Runs multiple configurations and reports efficiency metrics:
- Improvement per step
- Final absolute performance
- Best configuration selection

Then runs extended training on the best configs.

Usage:
    python scripts/reinforce_sweep.py --vaporeon --phase grid    # Initial grid search
    python scripts/reinforce_sweep.py --vaporeon --phase extend  # Extend best configs
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import requests

DEFAULT_HOST = "172.26.160.1"
DEFAULT_PORT = 8421
DEFAULT_OUTPUT = "experiments_swiglu_ae/main_run_091"


def eval_code(code: str, host: str, port: int, timeout: int = 300) -> dict:
    url = f"http://{host}:{port}/eval"
    resp = requests.post(url, json={"code": code}, timeout=timeout)
    return resp.json()


def format_error(error) -> str:
    """Format error from eval server (handles both string and dict formats)."""
    if isinstance(error, str):
        return error
    if isinstance(error, dict):
        msg = f"{error.get('type', 'Error')}: {error.get('message', 'Unknown error')}"
        if error.get('traceback'):
            msg += f"\n{error['traceback']}"
        return msg
    return str(error)


@dataclass
class Config:
    name: str
    mse_weight: float
    kl_weight: float
    double_ae: bool
    double_weight: float
    ablation_robust: bool
    ablation_weight: float


# Configuration grid
CONFIGS = [
    # Histogram-focused configs
    Config("hist_only", mse_weight=0.0, kl_weight=0.0, double_ae=False, double_weight=0.0, ablation_robust=False, ablation_weight=0.0),
    Config("hist_mse", mse_weight=1.0, kl_weight=0.0, double_ae=False, double_weight=0.0, ablation_robust=False, ablation_weight=0.0),
    Config("hist_mse_strong", mse_weight=2.0, kl_weight=0.0, double_ae=False, double_weight=0.0, ablation_robust=False, ablation_weight=0.0),

    # Trust region configs
    Config("hist_kl", mse_weight=0.0, kl_weight=0.01, double_ae=False, double_weight=0.0, ablation_robust=False, ablation_weight=0.0),
    Config("hist_mse_kl", mse_weight=1.0, kl_weight=0.001, double_ae=False, double_weight=0.0, ablation_robust=False, ablation_weight=0.0),

    # Super-REINFORCE (double AE)
    Config("super_light", mse_weight=1.0, kl_weight=0.0, double_ae=True, double_weight=0.25, ablation_robust=False, ablation_weight=0.0),
    Config("super_medium", mse_weight=1.0, kl_weight=0.0, double_ae=True, double_weight=0.5, ablation_robust=False, ablation_weight=0.0),
    Config("super_strong", mse_weight=1.0, kl_weight=0.0, double_ae=True, double_weight=1.0, ablation_robust=False, ablation_weight=0.0),

    # Ablation robustness
    Config("ablation_light", mse_weight=1.0, kl_weight=0.0, double_ae=False, double_weight=0.0, ablation_robust=True, ablation_weight=0.25),
    Config("ablation_medium", mse_weight=1.0, kl_weight=0.0, double_ae=False, double_weight=0.0, ablation_robust=True, ablation_weight=0.5),

    # Combined
    Config("combined", mse_weight=1.0, kl_weight=0.001, double_ae=True, double_weight=0.5, ablation_robust=True, ablation_weight=0.25),
]


def setup_lora(host: str, port: int, lora_rank: int, lr: float, vaporeon: bool) -> bool:
    """Setup LoRA layers and optimizer."""
    setup_code = f'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=8):
        super().__init__()
        self.base = base_layer
        self.rank = rank
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        for p in self.base.parameters():
            p.requires_grad = False
        self.enabled = True

    def forward(self, x):
        base_out = self.base(x)
        if self.enabled:
            return base_out + F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base_out

target_patterns = [
    "encoders.0.amplitude_proj", "encoders.0.wavelet_proj",
    "encoders.0.transformer.layers.0.attn.out_proj",
    "encoders.0.transformer.layers.1.attn.out_proj",
    "decoders.0.wav_embed", "decoders.0.amp_embed",
    "decoders.0.transformer.layers.0.attn.out_proj",
    "decoders.0.transformer.layers.1.attn.out_proj",
]

ae = model.sparse_ae
ctx._lora_layers = {{}}

for name, module in ae.named_modules():
    if name in target_patterns:
        if hasattr(module, 'lora_A'):
            module.lora_A.data.normal_(0, 0.01)
            module.lora_B.data.zero_()
            module.enabled = True
            ctx._lora_layers[name] = module
        elif isinstance(module, nn.Linear):
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = ae.get_submodule(parts[0])
                attr_name = parts[1]
            else:
                parent = ae
                attr_name = name
            lora_layer = LoRALinear(module, rank={lora_rank}).to(ctx.device)
            setattr(parent, attr_name, lora_layer)
            ctx._lora_layers[name] = lora_layer

lora_params = []
for layer in ctx._lora_layers.values():
    lora_params.extend([layer.lora_A, layer.lora_B])
ctx._lora_optimizer = torch.optim.Adam(lora_params, lr={lr})

{"" if not vaporeon else """
from src.sprite_atlas import SpriteAtlasIterator
vaporeon_config = {
    "data_dir": "data/infinite_fusion",
    "sampling_config": {
        "split": "all", "mode": "uniform_sprites",
        "adjustment_mode": "additive", "temperature": 1.0, "seed": 42,
        "adjustments": {"134": 1.0, "*.134": 1.0}
    },
    "render_config": {"res_scaling": "do_not", "background_mode": "solid_random", "jitter": True}
}
ctx._rl_iterator = SpriteAtlasIterator(ctx.device, vaporeon_config)
"""}
{"ctx._rl_iterator = ctx.iterator" if not vaporeon else ""}

ctx._rl_history = {{"step": [], "js_div": [], "mse": [], "reward": []}}
ctx._reward_baseline = None
"LoRA ready"
'''
    result = eval_code(setup_code, host, port)
    return result['success']


def train_step(host: str, port: int, cfg: Config, batch_size: int, resolution: int) -> Optional[dict]:
    """Run one training step with given config."""
    train_code = f'''
import torch
import torch.nn.functional as F
import numpy as np

batch_size = {batch_size}
resolution = {resolution}
mse_weight = {cfg.mse_weight}
kl_weight = {cfg.kl_weight}
double_ae = {cfg.double_ae}
double_weight = {cfg.double_weight}
ablation_robust = {cfg.ablation_robust}
ablation_weight = {cfg.ablation_weight}

# Get batch
blocks = ctx._rl_iterator.generate_batch_list(batch_size=batch_size * 4, resolution=resolution)
matching = [b.content for b in blocks if b.content.shape[-1] == resolution][:batch_size]
images = torch.stack(matching).to(ctx.device)

ae = model.sparse_ae
p = ae.patch_size
grid_shape = (resolution // p, resolution // p)
encoder_masks, decoder_masks = ae.build_masks(grid_shape, images.device)

for layer in ctx._lora_layers.values():
    layer.enabled = True

# Forward pass
with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    codes = ae.encode(images, grid_shape=grid_shape,
                      encoder_masks=encoder_masks, decoder_masks=decoder_masks)
    recon = ae.decode(codes, grid_shape, decoder_masks).float()

    # Double AE
    recon_double = None
    if double_ae:
        codes2 = ae.encode(recon.to(torch.bfloat16), grid_shape=grid_shape,
                           encoder_masks=encoder_masks, decoder_masks=decoder_masks)
        recon_double = ae.decode(codes2, grid_shape, decoder_masks).float()

# JS divergence (reward)
def compute_js(imgs, recs, n_bins=32):
    imgs_np = imgs.detach().cpu().numpy()
    recs_np = recs.detach().cpu().numpy()
    js = 0.0
    for c in range(3):
        h_i, _ = np.histogram(imgs_np[:, c].flatten(), bins=n_bins, range=(0, 1), density=True)
        h_r, _ = np.histogram(recs_np[:, c].flatten(), bins=n_bins, range=(0, 1), density=True)
        h_i = h_i / (h_i.sum() + 1e-10)
        h_r = h_r / (h_r.sum() + 1e-10)
        m = 0.5 * (h_i + h_r)
        js += 0.5 * np.sum(h_i * np.log((h_i + 1e-10) / (m + 1e-10)))
        js += 0.5 * np.sum(h_r * np.log((h_r + 1e-10) / (m + 1e-10)))
    return js / 3.0

with torch.no_grad():
    js_div = compute_js(images, recon)
    mse = F.mse_loss(recon, images).item()

reward = -js_div
if ctx._reward_baseline is None:
    ctx._reward_baseline = reward
else:
    ctx._reward_baseline = 0.9 * ctx._reward_baseline + 0.1 * reward
advantage = reward - ctx._reward_baseline

# Loss computation
ctx._lora_optimizer.zero_grad()

with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    codes = ae.encode(images, grid_shape=grid_shape,
                      encoder_masks=encoder_masks, decoder_masks=decoder_masks)
    recon = ae.decode(codes, grid_shape, decoder_masks).float()

# Soft histogram loss
def soft_hist_loss(img, ref, n_bins=32, sigma=0.05):
    bins = torch.linspace(0, 1, n_bins, device=img.device, dtype=img.dtype)
    loss = 0.0
    for c in range(3):
        i_flat = img[:, c].reshape(-1)
        r_flat = ref[:, c].reshape(-1)
        i_d = (i_flat.unsqueeze(1) - bins.unsqueeze(0)) / sigma
        r_d = (r_flat.unsqueeze(1) - bins.unsqueeze(0)) / sigma
        i_h = torch.exp(-0.5 * i_d ** 2).sum(0)
        r_h = torch.exp(-0.5 * r_d ** 2).sum(0)
        i_h = i_h / (i_h.sum() + 1e-8)
        r_h = r_h / (r_h.sum() + 1e-8)
        m = 0.5 * (i_h + r_h)
        loss += 0.5 * (r_h * torch.log((r_h + 1e-8) / (m + 1e-8))).sum()
        loss += 0.5 * (i_h * torch.log((i_h + 1e-8) / (m + 1e-8))).sum()
    return loss / 3

hist_loss = soft_hist_loss(recon, images)
base_mse = F.mse_loss(recon, images)

# Trust region
kl_loss = torch.tensor(0.0, device=images.device)
if kl_weight > 0:
    for layer in ctx._lora_layers.values():
        layer.enabled = False
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        codes_ref = ae.encode(images, grid_shape=grid_shape,
                              encoder_masks=encoder_masks, decoder_masks=decoder_masks)
        recon_ref = ae.decode(codes_ref, grid_shape, decoder_masks).float()
    for layer in ctx._lora_layers.values():
        layer.enabled = True
    codes_flat = torch.cat([c.view(c.shape[0], -1) for c in codes], dim=1).float()
    codes_ref_flat = torch.cat([c.view(c.shape[0], -1) for c in codes_ref], dim=1).float()
    kl_loss = F.mse_loss(codes_flat, codes_ref_flat.detach())

# Double AE loss
double_loss = torch.tensor(0.0, device=images.device)
if double_ae:
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        codes2 = ae.encode(recon.detach().to(torch.bfloat16), grid_shape=grid_shape,
                           encoder_masks=encoder_masks, decoder_masks=decoder_masks)
        recon2 = ae.decode(codes2, grid_shape, decoder_masks).float()
    double_loss = F.mse_loss(recon2, images) + soft_hist_loss(recon2, images)

# Ablation robustness loss
ablation_loss = torch.tensor(0.0, device=images.device)
if ablation_robust:
    code_dim = ae.code_dim
    n_wav = getattr(ae, 'n_wavelet_dims', None) or code_dim // 2
    n_amp = code_dim - n_wav

    def ablate(codes_list, subspace, rate):
        result = []
        for c in codes_list:
            c = c.clone()
            D = c.shape[-1]
            if subspace == "wav":
                n = int(n_wav * rate)
                if n > 0:
                    mask = torch.ones(D, device=c.device, dtype=c.dtype)
                    mask[torch.randperm(n_wav)[:n]] = 0
                    c = c * mask
            else:
                n = int(n_amp * rate)
                if n > 0:
                    mask = torch.ones(D, device=c.device, dtype=c.dtype)
                    mask[torch.randperm(n_amp)[:n] + n_wav] = 0
                    c = c * mask
            result.append(c)
        return result

    for rate in [0.25, 0.5]:
        codes_w = ablate(codes, "wav", rate)
        codes_a = ablate(codes, "amp", rate)
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            recon_w = ae.decode(codes_w, grid_shape, decoder_masks).float()
            recon_a = ae.decode(codes_a, grid_shape, decoder_masks).float()
        ablation_loss = ablation_loss + F.mse_loss(recon_w, images) + F.mse_loss(recon_a, images)
    ablation_loss = ablation_loss / 4

# Combined loss
reward_w = 1.0 + max(0, advantage * 2.0)
loss = reward_w * hist_loss + mse_weight * base_mse + kl_weight * kl_loss
loss = loss + double_weight * double_loss + ablation_weight * ablation_loss

loss.backward()
ctx._lora_optimizer.step()

# Log
step = len(ctx._rl_history["step"])
ctx._rl_history["step"].append(step)
ctx._rl_history["js_div"].append(js_div)
ctx._rl_history["mse"].append(mse)
ctx._rl_history["reward"].append(reward)

ctx._step_result = {{"step": step, "js_div": js_div, "mse": mse, "reward": reward, "loss": loss.item()}}
'''
    result = eval_code(train_code, host, port)
    if not result['success']:
        return None
    fetch = eval_code("ctx._step_result", host, port)
    if not fetch['success']:
        return None
    return fetch['result']


def run_config(host: str, port: int, cfg: Config, steps: int, batch_size: int,
               resolution: int, lora_rank: int, lr: float, vaporeon: bool) -> dict:
    """Run full training for a config."""
    # Reset LoRA
    if not setup_lora(host, port, lora_rank, lr, vaporeon):
        return {"error": "Setup failed"}

    history = {"js_div": [], "mse": [], "reward": []}

    for step in range(steps):
        result = train_step(host, port, cfg, batch_size, resolution)
        if result is None:
            return {"error": f"Step {step} failed"}
        history["js_div"].append(result["js_div"])
        history["mse"].append(result["mse"])
        history["reward"].append(result["reward"])

    # Compute metrics
    start_js = np.mean(history["js_div"][:5])
    end_js = np.mean(history["js_div"][-10:])
    improvement = start_js - end_js
    efficiency = improvement / steps * 1000  # Improvement per 1000 steps

    return {
        "config": cfg.name,
        "start_js": start_js,
        "end_js": end_js,
        "improvement": improvement,
        "efficiency": efficiency,
        "end_mse": np.mean(history["mse"][-10:]),
        "history": history
    }


def main():
    parser = argparse.ArgumentParser(description="REINFORCE sweep")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--phase", choices=["grid", "extend"], default="grid")
    parser.add_argument("--grid-steps", type=int, default=100, help="Steps for grid search")
    parser.add_argument("--extend-steps", type=int, default=500, help="Steps for extended training")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--vaporeon", action="store_true")
    parser.add_argument("--top-k", type=int, default=3, help="Top K configs to extend")
    args = parser.parse_args()

    print(f"Connecting to eval server at http://{args.host}:{args.port}...")
    health = requests.get(f"http://{args.host}:{args.port}/health").json()
    print(f"  Status: {health['status']}, Weights: {health['weights_loaded']}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "reinforce_sweep_results.json"

    if args.phase == "grid":
        print(f"\n=== GRID SEARCH ({len(CONFIGS)} configs x {args.grid_steps} steps) ===\n")

        results = []
        for i, cfg in enumerate(CONFIGS):
            print(f"[{i+1}/{len(CONFIGS)}] Running {cfg.name}...")
            result = run_config(args.host, args.port, cfg, args.grid_steps,
                                args.batch_size, args.resolution, args.lora_rank,
                                args.lr, args.vaporeon)

            if "error" in result:
                print(f"  ERROR: {result['error']}")
                continue

            print(f"  JS: {result['start_js']:.4f} -> {result['end_js']:.4f} "
                  f"(Δ={result['improvement']:+.4f}, eff={result['efficiency']:.4f})")
            results.append(result)

        # Sort by efficiency
        results.sort(key=lambda x: x["efficiency"], reverse=True)

        print("\n=== GRID SEARCH RESULTS (sorted by efficiency) ===\n")
        print(f"{'Config':<20} {'Start JS':>10} {'End JS':>10} {'Improve':>10} {'Eff/1k':>10} {'MSE':>10}")
        print("-" * 70)
        for r in results:
            print(f"{r['config']:<20} {r['start_js']:>10.4f} {r['end_js']:>10.4f} "
                  f"{r['improvement']:>+10.4f} {r['efficiency']:>10.4f} {r['end_mse']:>10.6f}")

        # Save results
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: x if not isinstance(x, np.floating) else float(x))
        print(f"\nSaved results to: {results_file}")

        # Plot comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        names = [r['config'] for r in results]
        improvements = [r['improvement'] for r in results]
        efficiencies = [r['efficiency'] for r in results]
        end_js = [r['end_js'] for r in results]

        axes[0].barh(names, improvements, color=['green' if x > 0 else 'red' for x in improvements])
        axes[0].set_xlabel('JS Divergence Improvement')
        axes[0].set_title('Total Improvement')
        axes[0].axvline(x=0, color='black', linewidth=0.5)

        axes[1].barh(names, efficiencies, color='blue')
        axes[1].set_xlabel('Improvement per 1000 steps')
        axes[1].set_title('Training Efficiency')

        axes[2].barh(names, end_js, color='purple')
        axes[2].set_xlabel('Final JS Divergence')
        axes[2].set_title('Final Performance (lower=better)')

        plt.tight_layout()
        plt.savefig(output_dir / "reinforce_sweep_grid.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to: {output_dir / 'reinforce_sweep_grid.png'}")

        # Identify top configs
        top_configs = [r['config'] for r in results[:args.top_k]]
        print(f"\nTop {args.top_k} configs for extended training: {top_configs}")

    else:  # extend phase
        print(f"\n=== EXTENDED TRAINING ===\n")

        # Load previous results
        if not results_file.exists():
            print("ERROR: Run grid phase first")
            return

        with open(results_file) as f:
            grid_results = json.load(f)

        top_configs = [r['config'] for r in grid_results[:args.top_k]]
        print(f"Extending top {args.top_k} configs: {top_configs}")

        extended_results = []
        for cfg_name in top_configs:
            cfg = next(c for c in CONFIGS if c.name == cfg_name)
            print(f"\nExtended training: {cfg_name} ({args.extend_steps} steps)...")

            result = run_config(args.host, args.port, cfg, args.extend_steps,
                                args.batch_size, args.resolution, args.lora_rank,
                                args.lr, args.vaporeon)

            if "error" in result:
                print(f"  ERROR: {format_error(result['error'])}")
                continue

            print(f"  JS: {result['start_js']:.4f} -> {result['end_js']:.4f} "
                  f"(Δ={result['improvement']:+.4f})")
            extended_results.append(result)

            # Plot learning curve
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(result['history']['js_div'], label='JS Divergence')
            ax.set_xlabel('Step')
            ax.set_ylabel('JS Divergence')
            ax.set_title(f'Extended Training: {cfg_name}')
            ax.axhline(y=result['end_js'], color='r', linestyle='--', alpha=0.5, label=f'Final: {result["end_js"]:.4f}')
            ax.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f"reinforce_extended_{cfg_name}.png", dpi=150, bbox_inches='tight')
            plt.close()

        # Summary
        print("\n=== EXTENDED TRAINING RESULTS ===\n")
        print(f"{'Config':<20} {'Start JS':>10} {'End JS':>10} {'Improve':>10}")
        print("-" * 50)
        for r in extended_results:
            print(f"{r['config']:<20} {r['start_js']:>10.4f} {r['end_js']:>10.4f} {r['improvement']:>+10.4f}")

        # Save extended results
        with open(output_dir / "reinforce_sweep_extended.json", 'w') as f:
            json.dump(extended_results, f, indent=2, default=lambda x: x if not isinstance(x, np.floating) else float(x))


if __name__ == "__main__":
    main()
