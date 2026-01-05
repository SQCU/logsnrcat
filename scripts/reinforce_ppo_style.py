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

Now uses shared src.lora module for:
    - torch.compile kernel fusion on LoRA forward
    - CUDA stream parallelism for ref/policy comparison
    - Clean context managers for LoRA toggling

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
DEFAULT_OUTPUT = "experiments_swiglu_ae/main_run_096"

# LoRA applied to CODES (after encode, before decode) - same architecture as histogram
# but with separate encoder/decoder adapters for asymmetric learning
# This avoids backprop through compiled transformers


def eval_code(code: str, host: str, port: int, timeout: int = 180) -> dict:
    """Execute Python code on eval server."""
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
        if error.get('locals'):
            msg += f"\nLocals: {error['locals']}"
        return msg
    return str(error)


def main():
    parser = argparse.ArgumentParser(description="PPO-style REINFORCE histogram trainer")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--steps", type=int, default=200, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA adapter rank")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adapter learning rate")
    parser.add_argument("--kl-weight", type=float, default=0.0001, help="KL penalty weight (trust region)")
    parser.add_argument("--mse-weight", type=float, default=0.0, help="Reconstruction MSE weight (alternative to trust region)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--vaporeon", action="store_true")
    parser.add_argument("--run-id", type=str, default=None, help="Run identifier for output files")
    parser.add_argument("--double-ae", action="store_true", help="Super-REINFORCE: double autoencoding for rollout-style optimization")
    parser.add_argument("--double-weight", type=float, default=1.0, help="Weight for double-AE loss")
    args = parser.parse_args()

    # Auto-generate run ID if not provided
    if args.run_id is None:
        import datetime
        args.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Connecting to eval server at http://{args.host}:{args.port}...")
    health = requests.get(f"http://{args.host}:{args.port}/health").json()
    print(f"  Status: {health['status']}, Weights: {health['weights_loaded']}")

    # Setup: use code adapters (like histogram) but with PPO-style trust region
    setup_code = f'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import LoRACodeAdapter from shared module
from src.lora import LoRACodeAdapter

ae = model.sparse_ae
code_dim = ae.code_dim  # 128

# Create encoder and decoder code adapters
# These perturb codes AFTER encode and BEFORE decode
# Gradient flows through adapters, not through compiled transformers
ctx._lora_enc = LoRACodeAdapter(code_dim, rank={args.lora_rank}).to(ctx.device).to(torch.bfloat16)
ctx._lora_dec = LoRACodeAdapter(code_dim, rank={args.lora_rank}).to(ctx.device).to(torch.bfloat16)

# Optimizer for adapter params
ctx._lora_optimizer = torch.optim.Adam(
    list(ctx._lora_enc.parameters()) + list(ctx._lora_dec.parameters()),
    lr={args.lr}
)

# Setup Vaporeon iterator
{"" if not args.vaporeon else """
from src.sprite_atlas import SpriteAtlasIterator
vaporeon_config = {{
    "data_dir": "data/infinite_fusion",
    "sampling_config": {{
        "split": "all", "mode": "uniform_sprites",
        "adjustment_mode": "additive", "temperature": 1.0, "seed": 42,
        "adjustments": {{"134": 1.0, "*.134": 1.0}}
    }},
    "render_config": {{"res_scaling": "do_not", "background_mode": "solid_random", "jitter": True}}
}}
ctx._rl_iterator = SpriteAtlasIterator(ctx.device, vaporeon_config)
"""}
{"ctx._rl_iterator = ctx.iterator" if not args.vaporeon else ""}

# Training history
ctx._rl_history = {{"step": [], "reward": [], "mse": [], "js_div": [], "loss": [],
                   "kl_codes": [], "mse_ref": [], "hist_loss": []}}
ctx._reward_baseline = None

n_enc_params = sum(p.numel() for p in ctx._lora_enc.parameters())
n_dec_params = sum(p.numel() for p in ctx._lora_dec.parameters())
f"Code adapters: enc={{n_enc_params}} + dec={{n_dec_params}} = {{n_enc_params + n_dec_params}} params (rank={args.lora_rank})"
'''

    print(f"\nRun ID: {args.run_id}")
    print(f"Setting up LoRA on projection layers (rank={args.lora_rank})...")
    result = eval_code(setup_code, args.host, args.port)
    if not result['success']:
        print(f"ERROR: {format_error(result['error'])}")
        return
    print(f"  {result['result']}")

    # Training loop runs ENTIRELY on server - single eval() call
    # Uses code adapter pattern (like histogram) but with PPO-style trust region
    training_loop_code = f'''
import torch
import torch.nn.functional as F
import numpy as np

# Training config
n_steps = {args.steps}
batch_size = {args.batch_size}
resolution = {args.resolution}
kl_weight = {args.kl_weight}
mse_weight = {args.mse_weight}
double_ae = {args.double_ae}
double_weight = {args.double_weight}
log_interval = 10

ae = model.sparse_ae
p = ae.patch_size

# Helper functions
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

# === TRAINING LOOP ===
for step in range(n_steps):
    # Get batch
    blocks = ctx._rl_iterator.generate_batch_list(batch_size=batch_size * 4, resolution=resolution)
    matching = [b.content for b in blocks if b.content.shape[-1] == resolution][:batch_size]
    images = torch.stack(matching).to(ctx.device)

    H, W = images.shape[2], images.shape[3]
    grid_shape = (H // p, W // p)
    encoder_masks, decoder_masks = ae.build_masks(grid_shape, images.device)

    # === Reference forward (no adapters) - completely frozen ===
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        codes_ref = ae.encode(images, grid_shape=grid_shape,
                              encoder_masks=encoder_masks, decoder_masks=decoder_masks)
        recon_ref = ae.decode(codes_ref, grid_shape, decoder_masks)
        codes_ref_flat = torch.cat([c.view(c.shape[0], -1) for c in codes_ref], dim=1).float()
        recon_ref = recon_ref.float()

    # === Policy forward (with adapters) ===
    # Encode with no_grad (avoid backprop through compiled transformer)
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        codes_list = ae.encode(images, grid_shape=grid_shape,
                               encoder_masks=encoder_masks, decoder_masks=decoder_masks)
        codes_list = [c.detach().float() for c in codes_list]

    # Apply adapters (these have gradients) - keep in bf16 for adapter
    codes_adapted = []
    for codes in codes_list:
        codes_bf16 = codes.to(torch.bfloat16)
        adapted = codes_bf16 + ctx._lora_enc(codes_bf16)
        adapted = adapted + ctx._lora_dec(adapted)
        codes_adapted.append(adapted.float())

    codes_lora_flat = torch.cat([c.view(c.shape[0], -1) for c in codes_adapted], dim=1)

    # Decode in fp32 (matches histogram pattern - avoids flex_attention autograd issues)
    with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
        recon_lora = ae.decode([c.to(torch.bfloat16) for c in codes_adapted], grid_shape, decoder_masks)
        recon_lora = recon_lora.float()

    # Super-REINFORCE: Double autoencoding (if enabled)
    recon_double = None
    if double_ae:
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            codes_double = ae.encode(recon_lora.to(torch.bfloat16), grid_shape=grid_shape,
                                      encoder_masks=encoder_masks, decoder_masks=decoder_masks)
            codes_double = [c.detach().float() for c in codes_double]
        codes_double_adapted = []
        for codes in codes_double:
            codes_bf16 = codes.to(torch.bfloat16)
            adapted = codes_bf16 + ctx._lora_enc(codes_bf16)
            adapted = adapted + ctx._lora_dec(adapted)
            codes_double_adapted.append(adapted.float())
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            recon_double = ae.decode([c.to(torch.bfloat16) for c in codes_double_adapted], grid_shape, decoder_masks).float()

    # Compute reward (JS divergence)
    with torch.no_grad():
        js_div = compute_js_divergence(images, recon_lora)
        mse = F.mse_loss(recon_lora, images).item()

    reward = -js_div
    if ctx._reward_baseline is None:
        ctx._reward_baseline = reward
    else:
        ctx._reward_baseline = 0.9 * ctx._reward_baseline + 0.1 * reward
    advantage = reward - ctx._reward_baseline

    # === PPO-style losses ===
    ctx._lora_optimizer.zero_grad()

    # Trust region: KL on codes (policy vs reference)
    kl_codes = F.mse_loss(codes_lora_flat, codes_ref_flat.detach())

    # Trust region: MSE between policy and reference reconstructions
    mse_ref = F.mse_loss(recon_lora, recon_ref.detach())

    # Differentiable histogram loss
    hist_loss = soft_histogram_loss(recon_lora, images)

    # Reconstruction MSE
    recon_mse = F.mse_loss(recon_lora, images)

    # PPO-style combined loss
    reward_weight = 1.0 + max(0, advantage * 2.0)
    loss = reward_weight * hist_loss + kl_weight * (kl_codes + mse_ref) + mse_weight * recon_mse

    # Double-AE loss (if enabled)
    double_mse_val = 0.0
    double_hist_val = 0.0
    if double_ae and recon_double is not None:
        double_mse = F.mse_loss(recon_double, images)
        double_hist = soft_histogram_loss(recon_double, images)
        loss = loss + double_weight * (double_mse + double_hist)
        double_mse_val = double_mse.item()
        double_hist_val = double_hist.item()

    loss.backward()
    ctx._lora_optimizer.step()

    # Log
    ctx._rl_history["step"].append(step)
    ctx._rl_history["reward"].append(reward)
    ctx._rl_history["mse"].append(mse)
    ctx._rl_history["js_div"].append(js_div)
    ctx._rl_history["loss"].append(loss.item())
    ctx._rl_history["kl_codes"].append(kl_codes.item())
    ctx._rl_history["mse_ref"].append(mse_ref.item())
    ctx._rl_history["hist_loss"].append(hist_loss.item())
    if "double_mse" not in ctx._rl_history:
        ctx._rl_history["double_mse"] = []
        ctx._rl_history["double_hist"] = []
    ctx._rl_history["double_mse"].append(double_mse_val)
    ctx._rl_history["double_hist"].append(double_hist_val)

    if step % log_interval == 0 or step == n_steps - 1:
        print(f"Step {{step:>4}}: reward={{reward:>8.4f}}, js_div={{js_div:>8.4f}}, mse={{mse:>8.6f}}, kl={{kl_codes.item():>8.6f}}, adv={{advantage:>+7.4f}}")

f"Training complete: {{n_steps}} steps, final reward={{ctx._rl_history['reward'][-1]:.4f}}"
'''

    mode = "Super-REINFORCE (double-AE)" if args.double_ae else "PPO-style"
    print(f"\nTraining for {args.steps} steps ({mode}, KL={args.kl_weight}, MSE={args.mse_weight})...")
    if args.double_ae:
        print(f"  Double-AE weight: {args.double_weight}")

    # Run in batches to fit within server timeout (default 30s)
    # ~5 steps per batch (PPO is slower than histogram due to trust region)
    batch_size_steps = 5
    total_batches = (args.steps + batch_size_steps - 1) // batch_size_steps
    print(f"(Running in {total_batches} batches of ~{batch_size_steps} steps)")
    print("-" * 60)

    steps_completed = 0
    for batch_idx in range(total_batches):
        steps_this_batch = min(batch_size_steps, args.steps - steps_completed)

        # Override n_steps for this batch
        batch_code = training_loop_code.replace(
            f"n_steps = {args.steps}",
            f"n_steps = {steps_this_batch}"
        )

        result = eval_code(batch_code, args.host, args.port, timeout=120)
        if not result['success']:
            print(f"ERROR at batch {batch_idx}: {format_error(result['error'])}")
            break

        steps_completed += steps_this_batch
        print(f"Batch {batch_idx + 1}/{total_batches} complete ({steps_completed}/{args.steps} steps)")

    print(f"\nTraining complete: {steps_completed} steps")

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

with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    # Baseline (no adapters)
    codes_base = ae.encode(images, grid_shape=grid_shape,
                           encoder_masks=encoder_masks, decoder_masks=decoder_masks)
    recon_base = ae.decode(codes_base, grid_shape, decoder_masks)

    # Policy (with adapters) - adapters are bf16
    codes_adapted = []
    for codes in codes_base:
        adapted = codes + ctx._lora_enc(codes)
        adapted = adapted + ctx._lora_dec(adapted)
        codes_adapted.append(adapted)
    recon_lora = ae.decode(codes_adapted, grid_shape, decoder_masks)

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
        print(f"Warning: Could not generate comparison: {format_error(result['error'])}")
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
    output_path = output_dir / f"reinforce_ppo_{args.run_id}{suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved training plot to: {output_path}")

    # Save before/after comparison image with histogram ribbons
    if compare_data is not None:
        def make_color_ribbon(img, height=64, width=16, n_bins=32):
            """Create a color ribbon showing histogram distribution.

            Colors that appear more frequently occupy more vertical space.
            """
            # img is [C, H, W] in range [0, 1]
            ribbon = np.zeros((height, width, 3))

            # Compute color histogram by binning each pixel's RGB
            img_hwc = np.transpose(img, (1, 2, 0))  # [H, W, C]
            pixels = img_hwc.reshape(-1, 3)  # [N, 3]

            # Bin each channel to reduce colors
            binned = (np.clip(pixels, 0, 1) * (n_bins - 1)).astype(int)

            # Count unique colors
            color_counts = {}
            for p in binned:
                key = tuple(p)
                color_counts[key] = color_counts.get(key, 0) + 1

            # Sort by frequency
            sorted_colors = sorted(color_counts.items(), key=lambda x: -x[1])

            # Fill ribbon proportionally
            total = sum(c for _, c in sorted_colors)
            y = 0
            for (r, g, b), count in sorted_colors:
                h = max(1, int(height * count / total))
                if y + h > height:
                    h = height - y
                # Convert bin back to color
                color = np.array([r, g, b]) / (n_bins - 1)
                ribbon[y:y+h, :, :] = color
                y += h
                if y >= height:
                    break

            return ribbon

        # 4 rows, 6 columns: [orig, ribbon, baseline, ribbon, policy, ribbon]
        fig2, axes2 = plt.subplots(4, 6, figsize=(12, 12),
                                    gridspec_kw={'width_ratios': [4, 1, 4, 1, 4, 1]})

        original = np.array(compare_data['original'])
        baseline = np.array(compare_data['baseline'])
        policy = np.array(compare_data['policy'])

        for i in range(4):
            # Original + ribbon
            img_orig = np.transpose(original[i], (1, 2, 0))
            axes2[i, 0].imshow(np.clip(img_orig, 0, 1))
            axes2[i, 0].set_title('Original' if i == 0 else '')
            axes2[i, 0].axis('off')

            ribbon_orig = make_color_ribbon(original[i])
            axes2[i, 1].imshow(ribbon_orig)
            axes2[i, 1].axis('off')

            # Baseline + ribbon
            img_base = np.transpose(baseline[i], (1, 2, 0))
            axes2[i, 2].imshow(np.clip(img_base, 0, 1))
            axes2[i, 2].set_title('Baseline' if i == 0 else '')
            axes2[i, 2].axis('off')

            ribbon_base = make_color_ribbon(baseline[i])
            axes2[i, 3].imshow(ribbon_base)
            axes2[i, 3].axis('off')

            # Policy + ribbon
            img_policy = np.transpose(policy[i], (1, 2, 0))
            axes2[i, 4].imshow(np.clip(img_policy, 0, 1))
            axes2[i, 4].set_title('Policy (LoRA)' if i == 0 else '')
            axes2[i, 4].axis('off')

            ribbon_policy = make_color_ribbon(policy[i])
            axes2[i, 5].imshow(ribbon_policy)
            axes2[i, 5].axis('off')

        plt.suptitle('Before/After LoRA Policy Optimization (with color histograms)', fontsize=12)
        plt.tight_layout()

        compare_path = output_dir / f"reinforce_ppo_{args.run_id}_compare{suffix}.png"
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
