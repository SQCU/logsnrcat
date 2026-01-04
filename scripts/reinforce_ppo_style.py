#!/usr/bin/env python3
"""
PPO-style REINFORCE trainer for histogram fidelity.

Key differences from naive REINFORCE:
    - LoRA on actual linear projection layers (not code perturbation)
    - Trust region via KL divergence from reference model (no LoRA)
    - Low-weight penalty for deviating from reference codes/outputs
    - REINFORCE on histogram divergence as verified reward

This mirrors PPO's approach: the KL penalty acts as a soft trust region,
preventing the policy (LoRA-augmented model) from drifting too far from
the reference (frozen model without LoRA).

Usage:
    python scripts/reinforce_ppo_style.py --vaporeon --steps 100
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests

DEFAULT_HOST = "172.26.160.1"
DEFAULT_PORT = 8421
DEFAULT_OUTPUT = "experiments_swiglu_ae/main_run_091"


def eval_code(code: str, host: str, port: int, timeout: int = 180) -> dict:
    """Execute Python code on eval server."""
    url = f"http://{host}:{port}/eval"
    resp = requests.post(url, json={"code": code}, timeout=timeout)
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="PPO-style REINFORCE histogram trainer")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--steps", type=int, default=100, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA adapter rank")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adapter learning rate")
    parser.add_argument("--kl-weight", type=float, default=0.0001, help="KL penalty weight (trust region)")
    parser.add_argument("--mse-weight", type=float, default=0.0, help="Reconstruction MSE weight (alternative to trust region)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--vaporeon", action="store_true")
    args = parser.parse_args()

    print(f"Connecting to eval server at http://{args.host}:{args.port}...")
    health = requests.get(f"http://{args.host}:{args.port}/health").json()
    print(f"  Status: {health['status']}, Weights: {health['weights_loaded']}")

    # Setup: create LoRA wrappers for projection layers
    setup_code = f'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# LoRA wrapper for nn.Linear - can be toggled on/off
class LoRALinear(nn.Module):
    """Wraps a frozen linear layer with trainable LoRA adapter."""
    def __init__(self, base_layer, rank=8):
        super().__init__()
        self.base = base_layer
        self.rank = rank
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # LoRA decomposition: W' = W + B @ A
        # A: [rank, in_features], B: [out_features, rank]
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Freeze base layer
        for p in self.base.parameters():
            p.requires_grad = False

        self.enabled = True  # Toggle for reference comparison

    def forward(self, x):
        base_out = self.base(x)
        if self.enabled:
            # LoRA contribution: x @ A.T @ B.T
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
            return base_out + lora_out
        return base_out

# Target layers for LoRA (key projection layers in encoder/decoder)
target_patterns = [
    "encoders.0.amplitude_proj",
    "encoders.0.wavelet_proj",
    "encoders.0.transformer.layers.0.attn.out_proj",
    "encoders.0.transformer.layers.1.attn.out_proj",
    "decoders.0.wav_embed",
    "decoders.0.amp_embed",
    "decoders.0.transformer.layers.0.attn.out_proj",
    "decoders.0.transformer.layers.1.attn.out_proj",
]

ae = model.sparse_ae

# Find or create LoRA layers
ctx._lora_layers = {{}}

# First check if layers are already wrapped (from previous run)
for name, module in ae.named_modules():
    if name in target_patterns:
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # Already a LoRA layer - reset weights and reuse
            module.lora_A.data.normal_(0, 0.01)
            module.lora_B.data.zero_()
            module.enabled = True
            ctx._lora_layers[name] = module
        elif isinstance(module, nn.Linear):
            # Fresh Linear - wrap with LoRA
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = ae.get_submodule(parent_name)
            else:
                parent = ae
                attr_name = name

            lora_layer = LoRALinear(module, rank={args.lora_rank}).to(ctx.device)
            setattr(parent, attr_name, lora_layer)
            ctx._lora_layers[name] = lora_layer

# Optimizer for LoRA parameters only
lora_params = []
for layer in ctx._lora_layers.values():
    lora_params.extend([layer.lora_A, layer.lora_B])
ctx._lora_optimizer = torch.optim.Adam(lora_params, lr={args.lr})

# Setup Vaporeon iterator
{"" if not args.vaporeon else """
from src.sprite_atlas import SpriteAtlasIterator
vaporeon_config = {
    "data_dir": "data/infinite_fusion",
    "sampling_config": {
        "split": "all", "mode": "uniform_sprites",
        "adjustment_mode": "additive", "temperature": 1.0, "seed": 42,
        "adjustments": {"134": 10.0, "*.134": 10.0}
    },
    "render_config": {"res_scaling": "do_not", "background_mode": "solid_random", "jitter": True}
}
ctx._rl_iterator = SpriteAtlasIterator(ctx.device, vaporeon_config)
"""}
{"ctx._rl_iterator = ctx.iterator" if not args.vaporeon else ""}

# Training history
ctx._rl_history = {{"step": [], "reward": [], "mse": [], "js_div": [], "loss": [],
                   "kl_codes": [], "mse_ref": [], "hist_loss": []}}
ctx._reward_baseline = None

n_lora_params = sum(p.numel() for p in lora_params)
f"LoRA layers wrapped: {{len(ctx._lora_layers)}} layers, {{n_lora_params}} params (rank={args.lora_rank})"
'''

    print(f"\nSetting up LoRA on projection layers (rank={args.lora_rank})...")
    result = eval_code(setup_code, args.host, args.port)
    if not result['success']:
        print(f"ERROR: {result['error']}")
        return
    print(f"  {result['result']}")

    # Training loop with PPO-style trust region
    train_step_code = f'''
import torch
import torch.nn.functional as F
import numpy as np

batch_size = {args.batch_size}
resolution = {args.resolution}
kl_weight = {args.kl_weight}
mse_weight = {args.mse_weight}

# Get batch
blocks = ctx._rl_iterator.generate_batch_list(batch_size=batch_size * 4, resolution=resolution)
matching = [b.content for b in blocks if b.content.shape[-1] == resolution][:batch_size]
images = torch.stack(matching).to(ctx.device)

ae = model.sparse_ae
p = ae.patch_size
H, W = images.shape[2], images.shape[3]
grid_shape = (H // p, W // p)
encoder_masks, decoder_masks = ae.build_masks(grid_shape, images.device)

# === Reference forward (LoRA disabled) ===
for layer in ctx._lora_layers.values():
    layer.enabled = False

with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    codes_ref = ae.encode(images, grid_shape=grid_shape,
                          encoder_masks=encoder_masks, decoder_masks=decoder_masks)
    recon_ref = ae.decode(codes_ref, grid_shape, decoder_masks)
    # Flatten codes for KL computation
    codes_ref_flat = torch.cat([c.view(c.shape[0], -1) for c in codes_ref], dim=1).float()
    recon_ref = recon_ref.float()

# === Policy forward (LoRA enabled) ===
for layer in ctx._lora_layers.values():
    layer.enabled = True

with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    codes_lora = ae.encode(images, grid_shape=grid_shape,
                           encoder_masks=encoder_masks, decoder_masks=decoder_masks)
    recon_lora = ae.decode(codes_lora, grid_shape, decoder_masks)
    codes_lora_flat = torch.cat([c.view(c.shape[0], -1) for c in codes_lora], dim=1).float()
    recon_lora = recon_lora.float()

# === Compute reward (histogram divergence) ===
def compute_js_divergence(imgs, recons, n_bins=32):
    imgs_np = imgs.detach().float().cpu().numpy()
    recons_np = recons.detach().float().cpu().numpy()
    js_total = 0.0
    for c in range(3):
        hist_in, _ = np.histogram(imgs_np[:, c].flatten(), bins=n_bins, range=(0, 1), density=True)
        hist_re, _ = np.histogram(recons_np[:, c].flatten(), bins=n_bins, range=(0, 1), density=True)
        hist_in = hist_in / (hist_in.sum() + 1e-10)
        hist_re = hist_re / (hist_re.sum() + 1e-10)
        m = 0.5 * (hist_in + hist_re)
        kl_pm = np.sum(hist_in * np.log((hist_in + 1e-10) / (m + 1e-10)))
        kl_qm = np.sum(hist_re * np.log((hist_re + 1e-10) / (m + 1e-10)))
        js_total += 0.5 * kl_pm + 0.5 * kl_qm
    return js_total / 3.0

with torch.no_grad():
    js_div = compute_js_divergence(images, recon_lora)
    mse = F.mse_loss(recon_lora, images).item()

# REINFORCE baseline
reward = -js_div
if ctx._reward_baseline is None:
    ctx._reward_baseline = reward
else:
    ctx._reward_baseline = 0.9 * ctx._reward_baseline + 0.1 * reward
advantage = reward - ctx._reward_baseline

# === Compute losses ===
ctx._lora_optimizer.zero_grad()

# Trust region: KL divergence on codes (approximate via MSE since codes are continuous)
# For continuous representations, MSE serves as a proxy for KL
kl_codes = F.mse_loss(codes_lora_flat, codes_ref_flat.detach())

# Trust region: MSE between policy and reference outputs
mse_ref = F.mse_loss(recon_lora, recon_ref.detach())

# Differentiable soft histogram loss
def soft_histogram_loss(img, ref, n_bins=32, sigma=0.05):
    B, C, H, W = img.shape
    bins = torch.linspace(0, 1, n_bins, device=img.device, dtype=img.dtype)
    total_loss = 0.0
    for c in range(C):
        img_flat = img[:, c].reshape(-1)
        ref_flat = ref[:, c].reshape(-1)
        img_dists = (img_flat.unsqueeze(1) - bins.unsqueeze(0)) / sigma
        ref_dists = (ref_flat.unsqueeze(1) - bins.unsqueeze(0)) / sigma
        img_hist = torch.exp(-0.5 * img_dists ** 2).sum(0)
        ref_hist = torch.exp(-0.5 * ref_dists ** 2).sum(0)
        img_hist = img_hist / (img_hist.sum() + 1e-8)
        ref_hist = ref_hist / (ref_hist.sum() + 1e-8)
        m = 0.5 * (img_hist + ref_hist)
        kl_pm = (ref_hist * torch.log((ref_hist + 1e-8) / (m + 1e-8))).sum()
        kl_qm = (img_hist * torch.log((img_hist + 1e-8) / (m + 1e-8))).sum()
        total_loss += 0.5 * kl_pm + 0.5 * kl_qm
    return total_loss / C

hist_loss = soft_histogram_loss(recon_lora, images)

# Reconstruction MSE (differentiable) - for mse_weight constraint
recon_mse = F.mse_loss(recon_lora, images)

# PPO-style loss: reward objective + trust region penalty
# When advantage > 0, emphasize histogram improvement
# KL penalty keeps policy close to reference
# MSE weight penalizes reconstruction error (alternative to trust region)
reward_weight = 1.0 + max(0, advantage * 2.0)
loss = reward_weight * hist_loss + kl_weight * (kl_codes + mse_ref) + mse_weight * recon_mse

loss.backward()
ctx._lora_optimizer.step()

# Log
step_num = len(ctx._rl_history["step"])
ctx._rl_history["step"].append(step_num)
ctx._rl_history["reward"].append(reward)
ctx._rl_history["mse"].append(mse)
ctx._rl_history["js_div"].append(js_div)
ctx._rl_history["loss"].append(loss.item())
ctx._rl_history["kl_codes"].append(kl_codes.item())
ctx._rl_history["mse_ref"].append(mse_ref.item())
ctx._rl_history["hist_loss"].append(hist_loss.item())

ctx._last_step_result = {{
    "step": step_num, "reward": reward, "mse": mse, "js_div": js_div,
    "loss": loss.item(), "kl_codes": kl_codes.item(), "mse_ref": mse_ref.item(),
    "hist_loss": hist_loss.item(), "advantage": advantage
}}
'''

    print(f"\nTraining for {args.steps} steps (KL weight={args.kl_weight})...")
    print("-" * 85)
    print(f"{'Step':>6} {'Reward':>10} {'JS Div':>10} {'MSE':>10} {'KL Codes':>10} {'MSE Ref':>10} {'Adv':>8}")
    print("-" * 85)

    for step in range(args.steps):
        result = eval_code(train_step_code, args.host, args.port)
        if not result['success']:
            print(f"ERROR at step {step}: {result['error']}")
            break

        fetch = eval_code("ctx._last_step_result", args.host, args.port)
        if not fetch['success']:
            print(f"ERROR fetching step result: {fetch['error']}")
            break

        metrics = fetch['result']
        if step % 10 == 0 or step == args.steps - 1:
            print(f"{metrics['step']:>6} {metrics['reward']:>10.4f} {metrics['js_div']:>10.4f} "
                  f"{metrics['mse']:>10.6f} {metrics['kl_codes']:>10.6f} {metrics['mse_ref']:>10.6f} "
                  f"{metrics['advantage']:>+8.4f}")

    # Generate before/after comparison images
    print("\nGenerating before/after comparison...")
    compare_code = '''
import torch

# Get a batch for visualization
blocks = ctx._rl_iterator.generate_batch_list(batch_size=16, resolution=64)
matching = [b.content for b in blocks if b.content.shape[-1] == 64][:4]
images = torch.stack(matching).to(ctx.device)

ae = model.sparse_ae
p = ae.patch_size
H, W = images.shape[2], images.shape[3]
grid_shape = (H // p, W // p)
encoder_masks, decoder_masks = ae.build_masks(grid_shape, images.device)

# Reconstruct WITHOUT LoRA (baseline)
for layer in ctx._lora_layers.values():
    layer.enabled = False

with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    codes_base = ae.encode(images, grid_shape=grid_shape,
                           encoder_masks=encoder_masks, decoder_masks=decoder_masks)
    recon_base = ae.decode(codes_base, grid_shape, decoder_masks)

# Reconstruct WITH LoRA (policy)
for layer in ctx._lora_layers.values():
    layer.enabled = True

with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    codes_lora = ae.encode(images, grid_shape=grid_shape,
                           encoder_masks=encoder_masks, decoder_masks=decoder_masks)
    recon_lora = ae.decode(codes_lora, grid_shape, decoder_masks)

ctx._compare_images = {
    "original": images.float().cpu().numpy().tolist(),
    "baseline": recon_base.float().cpu().numpy().tolist(),
    "policy": recon_lora.float().cpu().numpy().tolist()
}
"Comparison images ready"
'''
    result = eval_code(compare_code, args.host, args.port)
    if result['success']:
        fetch_result = eval_code("ctx._compare_images", args.host, args.port)
        if fetch_result['success'] and isinstance(fetch_result['result'], dict):
            compare_data = fetch_result['result']
        else:
            print(f"Warning: Comparison data not in expected format")
            compare_data = None
    else:
        print(f"Warning: Could not generate comparison: {result['error']}")
        compare_data = None

    # Fetch training history and plot
    print("\nFetching training history...")
    history = eval_code("ctx._rl_history", args.host, args.port)['result']

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    axes[0, 0].plot(history['step'], history['reward'])
    axes[0, 0].set_title('Reward (-JS Divergence)')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].axhline(y=np.mean(history['reward'][-10:]), color='r', linestyle='--', alpha=0.5)

    axes[0, 1].plot(history['step'], history['js_div'])
    axes[0, 1].set_title('JS Divergence (eval)')
    axes[0, 1].set_xlabel('Step')

    axes[0, 2].plot(history['step'], history['hist_loss'])
    axes[0, 2].set_title('Soft Histogram Loss (train)')
    axes[0, 2].set_xlabel('Step')

    axes[0, 3].plot(history['step'], history['mse'])
    axes[0, 3].set_title('MSE (policy)')
    axes[0, 3].set_xlabel('Step')

    # Trust region metrics
    axes[1, 0].plot(history['step'], history['kl_codes'])
    axes[1, 0].set_title('KL Codes (trust region)')
    axes[1, 0].set_xlabel('Step')

    axes[1, 1].plot(history['step'], history['mse_ref'])
    axes[1, 1].set_title('MSE from Reference')
    axes[1, 1].set_xlabel('Step')

    axes[1, 2].plot(history['step'], history['loss'])
    axes[1, 2].set_title('Total Loss')
    axes[1, 2].set_xlabel('Step')

    # Improvement over time
    if len(history['reward']) > 1:
        baseline = history['reward'][0]
        improvement = [(r - baseline) for r in history['reward']]
        axes[1, 3].plot(history['step'], improvement)
        axes[1, 3].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 3].set_title('Reward vs Start')
        axes[1, 3].set_xlabel('Step')
        axes[1, 3].fill_between(history['step'], 0, improvement,
                                 where=[i > 0 for i in improvement], alpha=0.3, color='green')
        axes[1, 3].fill_between(history['step'], 0, improvement,
                                 where=[i < 0 for i in improvement], alpha=0.3, color='red')

    plt.suptitle(f'PPO-style REINFORCE ({"Vaporeon" if args.vaporeon else "Mixed"}, '
                 f'rank={args.lora_rank}, lr={args.lr}, KL={args.kl_weight})', fontsize=12)
    plt.tight_layout()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_vaporeon" if args.vaporeon else ""
    output_path = output_dir / f"reinforce_ppo_style{suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved training plot to: {output_path}")

    # Save before/after comparison image
    if compare_data is not None:
        fig2, axes2 = plt.subplots(4, 3, figsize=(9, 12))

        original = np.array(compare_data['original'])
        baseline = np.array(compare_data['baseline'])
        policy = np.array(compare_data['policy'])

        for i in range(4):
            # Original
            img_orig = np.transpose(original[i], (1, 2, 0))
            axes2[i, 0].imshow(np.clip(img_orig, 0, 1))
            axes2[i, 0].set_title('Original' if i == 0 else '')
            axes2[i, 0].axis('off')

            # Baseline (no LoRA)
            img_base = np.transpose(baseline[i], (1, 2, 0))
            axes2[i, 1].imshow(np.clip(img_base, 0, 1))
            axes2[i, 1].set_title('Baseline (no LoRA)' if i == 0 else '')
            axes2[i, 1].axis('off')

            # Policy (with LoRA)
            img_policy = np.transpose(policy[i], (1, 2, 0))
            axes2[i, 2].imshow(np.clip(img_policy, 0, 1))
            axes2[i, 2].set_title('Policy (with LoRA)' if i == 0 else '')
            axes2[i, 2].axis('off')

        plt.suptitle('Before/After LoRA Policy Optimization', fontsize=12)
        plt.tight_layout()

        compare_path = output_dir / f"reinforce_ppo_compare{suffix}.png"
        plt.savefig(compare_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison to: {compare_path}")

    print(f"\nFinal metrics (last 10 steps):")
    print(f"  Reward: {np.mean(history['reward'][-10:]):.4f}")
    print(f"  JS Div: {np.mean(history['js_div'][-10:]):.4f}")
    print(f"  MSE: {np.mean(history['mse'][-10:]):.6f}")
    print(f"  KL Codes: {np.mean(history['kl_codes'][-10:]):.6f}")
    print(f"  MSE Ref: {np.mean(history['mse_ref'][-10:]):.6f}")

    if len(history['reward']) > 1:
        start_reward = history['reward'][0]
        end_reward = np.mean(history['reward'][-10:])
        print(f"\n  Improvement: {end_reward - start_reward:+.4f} (start: {start_reward:.4f})")


if __name__ == "__main__":
    main()
