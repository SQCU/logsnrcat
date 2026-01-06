#!/usr/bin/env python3
"""
REINFORCE for ablation robustness.

RL target: policies that reconstruct well even when wavelet or amplitude
subspaces are partially ablated. Encourages robust/redundant representations.

Ablation rates: [0.0, 0.25, 0.5, 0.75, 1.0] - same as probe_subspace_ablation.py

Reward signals:
    - MSE under amplitude ablation (various rates)
    - MSE under wavelet ablation (various rates)
    - Graceful degradation curve (area under ablation-MSE curve)

Usage:
    python scripts/reinforce_ablation_robustness.py --vaporeon --steps 150
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests

DEFAULT_HOST = "172.26.160.1"
DEFAULT_PORT = 8421
DEFAULT_OUTPUT = "experiments_swiglu_ae/main_run_096"


def eval_code(code: str, host: str, port: int, timeout: int = 180) -> dict:
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


def save_ablation_grid(host: str, port: int, step: int, n_samples: int,
                       output_dir: Path, run_id: str, vaporeon: bool):
    """Generate and save ablation comparison grid at current policy state."""
    viz_code = f'''
import torch
import numpy as np

# Get fresh test batch - use generate_from_split if available
if hasattr(ctx._rl_iterator, 'generate_from_split'):
    blocks = ctx._rl_iterator.generate_from_split('sprite_atlas', count={n_samples}, resolution=64)
else:
    blocks = ctx._rl_iterator.generate_batch_list({n_samples}, resolution=64)
images = torch.stack([b.content for b in blocks[:{n_samples}]]).to(ctx.device)

ae = model.sparse_ae
p = ae.patch_size
grid_shape = (64 // p, 64 // p)
encoder_masks, decoder_masks = ae.build_masks(grid_shape, images.device)
code_dim = ae.code_dim
n_wavelet_dims = getattr(ae, 'n_wavelet_dims', None) or code_dim // 2
n_amp_dims = code_dim - n_wavelet_dims

def ablate_codes(codes_list, subspace, rate):
    ablated = []
    for codes in codes_list:
        c = codes.clone()
        B, N, D = c.shape
        if subspace == "wavelet":
            n_zero = int(n_wavelet_dims * rate)
            if n_zero > 0:
                mask = torch.ones(D, device=c.device, dtype=c.dtype)
                zero_idx = torch.randperm(n_wavelet_dims)[:n_zero]
                mask[zero_idx] = 0
                c = c * mask
        elif subspace == "amplitude":
            n_zero = int(n_amp_dims * rate)
            if n_zero > 0:
                mask = torch.ones(D, device=c.device, dtype=c.dtype)
                zero_idx = torch.randperm(n_amp_dims)[:n_zero] + n_wavelet_dims
                mask[zero_idx] = 0
                c = c * mask
        ablated.append(c)
    return ablated

ablation_rates = [0.0, 0.25, 0.5, 0.75, 1.0]

with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    codes = ae.encode(images, grid_shape=grid_shape,
                      encoder_masks=encoder_masks, decoder_masks=decoder_masks)

    # Collect reconstructions at each ablation rate
    viz_data = {{"images": [], "recons_base": [], "wav_ablations": {{}}, "amp_ablations": {{}}}}

    recon_base = ae.decode(codes, grid_shape, decoder_masks).float()

    for i in range({n_samples}):
        viz_data["images"].append(images[i].float().permute(1,2,0).cpu().numpy().clip(0,1).tolist())
        viz_data["recons_base"].append(recon_base[i].permute(1,2,0).cpu().numpy().clip(0,1).tolist())

    for rate in ablation_rates:
        viz_data["wav_ablations"][rate] = []
        viz_data["amp_ablations"][rate] = []

        codes_w = ablate_codes(codes, "wavelet", rate)
        codes_a = ablate_codes(codes, "amplitude", rate)
        recon_w = ae.decode(codes_w, grid_shape, decoder_masks).float()
        recon_a = ae.decode(codes_a, grid_shape, decoder_masks).float()

        for i in range({n_samples}):
            viz_data["wav_ablations"][rate].append(recon_w[i].permute(1,2,0).cpu().numpy().clip(0,1).tolist())
            viz_data["amp_ablations"][rate].append(recon_a[i].permute(1,2,0).cpu().numpy().clip(0,1).tolist())

ctx._periodic_viz = viz_data
'''
    result = eval_code(viz_code, host, port)
    if not result['success']:
        print(f"    Warning: Could not generate viz at step {step}: {format_error(result['error'])}")
        return

    fetch = eval_code("ctx._periodic_viz", host, port)
    if not fetch['success']:
        return

    viz_data = fetch['result']
    ablation_rates = [0.0, 0.25, 0.5, 0.75, 1.0]

    # Create grid: rows = samples, cols = Original | Base | Wav@rates... | Amp@rates...
    n_cols = 2 + 2 * len(ablation_rates)
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(2 * n_cols, 2 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for row in range(n_samples):
        col = 0

        # Original
        axes[row, col].imshow(np.array(viz_data["images"][row]))
        axes[row, col].set_title("Original" if row == 0 else "")
        axes[row, col].axis('off')
        col += 1

        # Base reconstruction
        axes[row, col].imshow(np.array(viz_data["recons_base"][row]))
        axes[row, col].set_title("Recon" if row == 0 else "")
        axes[row, col].axis('off')
        col += 1

        # Wavelet ablations
        for rate in ablation_rates:
            img = np.array(viz_data["wav_ablations"][str(rate)][row])
            axes[row, col].imshow(img)
            axes[row, col].set_title(f"Wav{int(rate*100)}%" if row == 0 else "")
            axes[row, col].axis('off')
            col += 1

        # Amplitude ablations
        for rate in ablation_rates:
            img = np.array(viz_data["amp_ablations"][str(rate)][row])
            axes[row, col].imshow(img)
            axes[row, col].set_title(f"Amp{int(rate*100)}%" if row == 0 else "")
            axes[row, col].axis('off')
            col += 1

    suffix = "_vaporeon" if vaporeon else ""
    plt.suptitle(f'Ablation Robustness @ Step {step} ({run_id})', fontsize=12)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ablation_step{step:04d}_{run_id}{suffix}.png"
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Ablation robustness REINFORCE")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--steps", type=int, default=150, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mse-weight", type=float, default=1.0, help="Base reconstruction MSE weight")
    parser.add_argument("--ablation-weight", type=float, default=0.5, help="Ablation robustness weight")
    parser.add_argument("--target-subspace", choices=["wavelet", "amplitude", "both"], default="both",
                        help="Which subspace to train robustness for")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--vaporeon", action="store_true")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--save-interval", type=int, default=25,
                        help="Save ablation image grid every N steps (0 to disable)")
    parser.add_argument("--n-viz-samples", type=int, default=4,
                        help="Number of samples in periodic visualization grids")
    args = parser.parse_args()

    if args.run_id is None:
        import datetime
        args.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Connecting to eval server at http://{args.host}:{args.port}...")
    health = requests.get(f"http://{args.host}:{args.port}/health").json()
    print(f"  Status: {health['status']}, Weights: {health['weights_loaded']}")

    # Setup code - LoRA wrappers + ablation functions
    setup_code = f'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# LoRA wrapper
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
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
            return base_out + lora_out
        return base_out

# Target layers
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
            lora_layer = LoRALinear(module, rank={args.lora_rank}).to(ctx.device)
            setattr(parent, attr_name, lora_layer)
            ctx._lora_layers[name] = lora_layer

# Optimizer
lora_params = []
for layer in ctx._lora_layers.values():
    lora_params.extend([layer.lora_A, layer.lora_B])
ctx._lora_optimizer = torch.optim.Adam(lora_params, lr={args.lr})

# Vaporeon iterator
{"" if not args.vaporeon else """
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
{"ctx._rl_iterator = ctx.iterator" if not args.vaporeon else ""}

# Training history
ctx._rl_history = {{
    "step": [], "mse_base": [], "mse_wav_ablated": [], "mse_amp_ablated": [],
    "robustness_score": [], "loss": [], "reward": [], "advantage": []
}}
ctx._reward_baseline = None

# Ablation rates to test
ctx._ablation_rates = [0.25, 0.5, 0.75]

n_params = sum(p.numel() for p in lora_params)
f"LoRA ready: {{len(ctx._lora_layers)}} layers, {{n_params}} params"
'''

    print(f"\nRun ID: {args.run_id}")
    print(f"Setting up LoRA (rank={args.lora_rank})...")
    result = eval_code(setup_code, args.host, args.port)
    if not result['success']:
        print(f"ERROR: {format_error(result['error'])}")
        return
    print(f"  {result['result']}")

    # Training loop
    train_code = f'''
import torch
import torch.nn.functional as F
import numpy as np

batch_size = {args.batch_size}
resolution = {args.resolution}
mse_weight = {args.mse_weight}
ablation_weight = {args.ablation_weight}
target_subspace = "{args.target_subspace}"

# Get batch - use generate_from_split if available (CompositeIterator), else direct (SpriteAtlasIterator)
if hasattr(ctx._rl_iterator, 'generate_from_split'):
    blocks = ctx._rl_iterator.generate_from_split('sprite_atlas', count=batch_size, resolution=resolution)
else:
    blocks = ctx._rl_iterator.generate_batch_list(batch_size, resolution=resolution)
images = torch.stack([b.content for b in blocks[:batch_size]]).to(ctx.device)

ae = model.sparse_ae
p = ae.patch_size
H, W = images.shape[2], images.shape[3]
grid_shape = (H // p, W // p)
encoder_masks, decoder_masks = ae.build_masks(grid_shape, images.device)

# Enable LoRA
for layer in ctx._lora_layers.values():
    layer.enabled = True

# === Encode once ===
with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    codes_list = ae.encode(images, grid_shape=grid_shape,
                           encoder_masks=encoder_masks, decoder_masks=decoder_masks)

# Get subspace dimensions
code_dim = ae.code_dim
n_wavelet_dims = getattr(ae, 'n_wavelet_dims', None) or code_dim // 2
n_amp_dims = code_dim - n_wavelet_dims

# === Ablation function ===
def ablate_codes(codes_list, subspace, rate):
    """Ablate wavelet or amplitude dimensions at given rate."""
    ablated = []
    for codes in codes_list:
        c = codes.clone()
        B, N, D = c.shape

        if subspace == "wavelet":
            # Zero out random wavelet dimensions
            n_zero = int(n_wavelet_dims * rate)
            if n_zero > 0:
                mask = torch.ones(D, device=c.device, dtype=c.dtype)
                zero_idx = torch.randperm(n_wavelet_dims)[:n_zero]
                mask[zero_idx] = 0
                c = c * mask
        elif subspace == "amplitude":
            # Zero out random amplitude dimensions
            n_zero = int(n_amp_dims * rate)
            if n_zero > 0:
                mask = torch.ones(D, device=c.device, dtype=c.dtype)
                zero_idx = torch.randperm(n_amp_dims)[:n_zero] + n_wavelet_dims
                mask[zero_idx] = 0
                c = c * mask
        ablated.append(c)
    return ablated

# === Compute MSE at different ablation rates ===
with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    # Base reconstruction (no ablation)
    recon_base = ae.decode(codes_list, grid_shape, decoder_masks).float()
    mse_base = F.mse_loss(recon_base, images).item()

    # Ablated reconstructions
    mse_wav_ablated = []
    mse_amp_ablated = []

    for rate in ctx._ablation_rates:
        # Wavelet ablation
        codes_wav = ablate_codes(codes_list, "wavelet", rate)
        recon_wav = ae.decode(codes_wav, grid_shape, decoder_masks).float()
        mse_wav_ablated.append(F.mse_loss(recon_wav, images).item())

        # Amplitude ablation
        codes_amp = ablate_codes(codes_list, "amplitude", rate)
        recon_amp = ae.decode(codes_amp, grid_shape, decoder_masks).float()
        mse_amp_ablated.append(F.mse_loss(recon_amp, images).item())

# === Compute robustness score (lower is more robust) ===
# Area under degradation curve - smaller = more graceful degradation
if target_subspace == "wavelet":
    degradation = np.mean(mse_wav_ablated) - mse_base
elif target_subspace == "amplitude":
    degradation = np.mean(mse_amp_ablated) - mse_base
else:  # both
    degradation = 0.5 * (np.mean(mse_wav_ablated) + np.mean(mse_amp_ablated)) - mse_base

robustness_score = degradation  # Lower = better

# === REINFORCE reward ===
# Reward = -robustness_score (want to minimize degradation)
# Also reward low base MSE
reward = -robustness_score - 0.1 * mse_base

if ctx._reward_baseline is None:
    ctx._reward_baseline = reward
else:
    ctx._reward_baseline = 0.9 * ctx._reward_baseline + 0.1 * reward
advantage = reward - ctx._reward_baseline

# === Compute differentiable loss ===
ctx._lora_optimizer.zero_grad()

with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    # Re-encode for gradients
    codes_list = ae.encode(images, grid_shape=grid_shape,
                           encoder_masks=encoder_masks, decoder_masks=decoder_masks)

    # Base MSE loss
    recon = ae.decode(codes_list, grid_shape, decoder_masks).float()
    base_loss = F.mse_loss(recon, images)

    # Ablation robustness loss (average over rates)
    ablation_loss = torch.tensor(0.0, device=images.device)
    n_ablations = 0

    for rate in ctx._ablation_rates:
        if target_subspace in ["wavelet", "both"]:
            codes_wav = ablate_codes(codes_list, "wavelet", rate)
            recon_wav = ae.decode(codes_wav, grid_shape, decoder_masks).float()
            ablation_loss = ablation_loss + F.mse_loss(recon_wav, images)
            n_ablations += 1

        if target_subspace in ["amplitude", "both"]:
            codes_amp = ablate_codes(codes_list, "amplitude", rate)
            recon_amp = ae.decode(codes_amp, grid_shape, decoder_masks).float()
            ablation_loss = ablation_loss + F.mse_loss(recon_amp, images)
            n_ablations += 1

    ablation_loss = ablation_loss / max(n_ablations, 1)

# Combined loss with advantage weighting
reward_weight = 1.0 + max(0, advantage * 2.0)
loss = mse_weight * base_loss + ablation_weight * reward_weight * ablation_loss

loss.backward()
ctx._lora_optimizer.step()

# Log
step_num = len(ctx._rl_history["step"])
ctx._rl_history["step"].append(step_num)
ctx._rl_history["mse_base"].append(mse_base)
ctx._rl_history["mse_wav_ablated"].append(np.mean(mse_wav_ablated))
ctx._rl_history["mse_amp_ablated"].append(np.mean(mse_amp_ablated))
ctx._rl_history["robustness_score"].append(robustness_score)
ctx._rl_history["loss"].append(loss.item())
ctx._rl_history["reward"].append(reward)
ctx._rl_history["advantage"].append(advantage)

ctx._last_step = {{
    "step": step_num,
    "mse_base": mse_base,
    "mse_wav": np.mean(mse_wav_ablated),
    "mse_amp": np.mean(mse_amp_ablated),
    "robustness": robustness_score,
    "reward": reward,
    "advantage": advantage
}}
'''

    print(f"\nTraining for {args.steps} steps (target: {args.target_subspace})...")
    print(f"  MSE weight: {args.mse_weight}, Ablation weight: {args.ablation_weight}")
    print("-" * 95)
    print(f"{'Step':>6} {'MSE Base':>10} {'MSE Wav↓':>10} {'MSE Amp↓':>10} {'Robust':>10} {'Reward':>10} {'Adv':>8}")
    print("-" * 95)

    for step in range(args.steps):
        result = eval_code(train_code, args.host, args.port)
        if not result['success']:
            print(f"ERROR at step {step}: {format_error(result['error'])}")
            break

        fetch = eval_code("ctx._last_step", args.host, args.port)
        if not fetch['success']:
            print(f"ERROR fetching: {format_error(fetch['error'])}")
            break

        m = fetch['result']
        if step % 10 == 0 or step == args.steps - 1:
            print(f"{m['step']:>6} {m['mse_base']:>10.6f} {m['mse_wav']:>10.6f} "
                  f"{m['mse_amp']:>10.6f} {m['robustness']:>10.6f} "
                  f"{m['reward']:>10.4f} {m['advantage']:>+8.4f}")

        # Periodic image saving
        if args.save_interval > 0 and (step % args.save_interval == 0 or step == args.steps - 1):
            save_ablation_grid(args.host, args.port, step, args.n_viz_samples,
                               Path(args.output_dir), args.run_id, args.vaporeon)

    # Generate comparison with ablation curves
    print("\nGenerating ablation curve comparison...")
    compare_code = '''
import torch
import numpy as np

# Get test batch - use generate_from_split if available
if hasattr(ctx._rl_iterator, 'generate_from_split'):
    blocks = ctx._rl_iterator.generate_from_split('sprite_atlas', count=4, resolution=64)
else:
    blocks = ctx._rl_iterator.generate_batch_list(4, resolution=64)
images = torch.stack([b.content for b in blocks[:4]]).to(ctx.device)

ae = model.sparse_ae
p = ae.patch_size
grid_shape = (64 // p, 64 // p)
encoder_masks, decoder_masks = ae.build_masks(grid_shape, images.device)
code_dim = ae.code_dim
n_wavelet_dims = getattr(ae, 'n_wavelet_dims', None) or code_dim // 2
n_amp_dims = code_dim - n_wavelet_dims

ablation_rates = [0.0, 0.25, 0.5, 0.75, 1.0]

def ablate_codes(codes_list, subspace, rate):
    ablated = []
    for codes in codes_list:
        c = codes.clone()
        B, N, D = c.shape
        if subspace == "wavelet":
            n_zero = int(n_wavelet_dims * rate)
            if n_zero > 0:
                mask = torch.ones(D, device=c.device, dtype=c.dtype)
                zero_idx = torch.randperm(n_wavelet_dims)[:n_zero]
                mask[zero_idx] = 0
                c = c * mask
        elif subspace == "amplitude":
            n_zero = int(n_amp_dims * rate)
            if n_zero > 0:
                mask = torch.ones(D, device=c.device, dtype=c.dtype)
                zero_idx = torch.randperm(n_amp_dims)[:n_zero] + n_wavelet_dims
                mask[zero_idx] = 0
                c = c * mask
        ablated.append(c)
    return ablated

# Test WITH LoRA (policy)
for layer in ctx._lora_layers.values():
    layer.enabled = True

with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    codes_lora = ae.encode(images, grid_shape=grid_shape,
                           encoder_masks=encoder_masks, decoder_masks=decoder_masks)

    policy_wav_mse = []
    policy_amp_mse = []
    for rate in ablation_rates:
        codes_w = ablate_codes(codes_lora, "wavelet", rate)
        codes_a = ablate_codes(codes_lora, "amplitude", rate)
        recon_w = ae.decode(codes_w, grid_shape, decoder_masks).float()
        recon_a = ae.decode(codes_a, grid_shape, decoder_masks).float()
        policy_wav_mse.append(F.mse_loss(recon_w, images).item())
        policy_amp_mse.append(F.mse_loss(recon_a, images).item())

    # Store sample reconstructions at 50% ablation
    codes_w50 = ablate_codes(codes_lora, "wavelet", 0.5)
    codes_a50 = ablate_codes(codes_lora, "amplitude", 0.5)
    recon_policy_base = ae.decode(codes_lora, grid_shape, decoder_masks).float()
    recon_policy_wav50 = ae.decode(codes_w50, grid_shape, decoder_masks).float()
    recon_policy_amp50 = ae.decode(codes_a50, grid_shape, decoder_masks).float()

# Test WITHOUT LoRA (baseline)
for layer in ctx._lora_layers.values():
    layer.enabled = False

with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    codes_base = ae.encode(images, grid_shape=grid_shape,
                           encoder_masks=encoder_masks, decoder_masks=decoder_masks)

    baseline_wav_mse = []
    baseline_amp_mse = []
    for rate in ablation_rates:
        codes_w = ablate_codes(codes_base, "wavelet", rate)
        codes_a = ablate_codes(codes_base, "amplitude", rate)
        recon_w = ae.decode(codes_w, grid_shape, decoder_masks).float()
        recon_a = ae.decode(codes_a, grid_shape, decoder_masks).float()
        baseline_wav_mse.append(F.mse_loss(recon_w, images).item())
        baseline_amp_mse.append(F.mse_loss(recon_a, images).item())

    recon_base_base = ae.decode(codes_base, grid_shape, decoder_masks).float()
    codes_w50 = ablate_codes(codes_base, "wavelet", 0.5)
    codes_a50 = ablate_codes(codes_base, "amplitude", 0.5)
    recon_base_wav50 = ae.decode(codes_w50, grid_shape, decoder_masks).float()
    recon_base_amp50 = ae.decode(codes_a50, grid_shape, decoder_masks).float()

ctx._ablation_compare = {
    "rates": ablation_rates,
    "baseline_wav": baseline_wav_mse,
    "baseline_amp": baseline_amp_mse,
    "policy_wav": policy_wav_mse,
    "policy_amp": policy_amp_mse,
    "images": images[:2].float().cpu().numpy().tolist(),
    "recon_base_base": recon_base_base[:2].float().cpu().numpy().tolist(),
    "recon_base_wav50": recon_base_wav50[:2].float().cpu().numpy().tolist(),
    "recon_base_amp50": recon_base_amp50[:2].float().cpu().numpy().tolist(),
    "recon_policy_base": recon_policy_base[:2].float().cpu().numpy().tolist(),
    "recon_policy_wav50": recon_policy_wav50[:2].float().cpu().numpy().tolist(),
    "recon_policy_amp50": recon_policy_amp50[:2].float().cpu().numpy().tolist(),
}
"Ablation comparison ready"
'''

    result = eval_code(compare_code, args.host, args.port)
    if not result['success']:
        print(f"Warning: Could not generate comparison: {format_error(result['error'])}")
        compare_data = None
    else:
        compare_data = eval_code("ctx._ablation_compare", args.host, args.port)['result']

    # Fetch history
    print("\nFetching training history...")
    history = eval_code("ctx._rl_history", args.host, args.port)['result']

    # Plot training curves
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    axes[0, 0].plot(history['step'], history['mse_base'], label='Base MSE')
    axes[0, 0].set_title('Base Reconstruction MSE')
    axes[0, 0].set_xlabel('Step')

    axes[0, 1].plot(history['step'], history['mse_wav_ablated'], label='Wavelet ablated', color='blue')
    axes[0, 1].plot(history['step'], history['mse_amp_ablated'], label='Amplitude ablated', color='red')
    axes[0, 1].set_title('MSE Under Ablation')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].legend()

    axes[0, 2].plot(history['step'], history['robustness_score'])
    axes[0, 2].set_title('Robustness Score (lower=better)')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    axes[1, 0].plot(history['step'], history['reward'])
    axes[1, 0].set_title('Reward')
    axes[1, 0].set_xlabel('Step')

    axes[1, 1].plot(history['step'], history['loss'])
    axes[1, 1].set_title('Total Loss')
    axes[1, 1].set_xlabel('Step')

    # Ablation curves comparison
    if compare_data:
        rates = compare_data['rates']
        axes[1, 2].plot(rates, compare_data['baseline_wav'], 'b--', label='Baseline Wav', alpha=0.7)
        axes[1, 2].plot(rates, compare_data['baseline_amp'], 'r--', label='Baseline Amp', alpha=0.7)
        axes[1, 2].plot(rates, compare_data['policy_wav'], 'b-', label='Policy Wav', linewidth=2)
        axes[1, 2].plot(rates, compare_data['policy_amp'], 'r-', label='Policy Amp', linewidth=2)
        axes[1, 2].set_title('Ablation Curves')
        axes[1, 2].set_xlabel('Ablation Rate')
        axes[1, 2].set_ylabel('MSE')
        axes[1, 2].legend(fontsize=8)
    else:
        axes[1, 2].axis('off')

    plt.suptitle(f'Ablation Robustness REINFORCE (target={args.target_subspace}, {args.run_id})')
    plt.tight_layout()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_vaporeon" if args.vaporeon else ""
    output_path = output_dir / f"reinforce_ablation_{args.run_id}{suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved training plot to: {output_path}")

    # Visual comparison grid
    if compare_data:
        fig2, axes2 = plt.subplots(2, 6, figsize=(15, 5))

        images_np = np.array(compare_data['images'])

        for row in range(2):
            img = np.transpose(images_np[row], (1, 2, 0))
            axes2[row, 0].imshow(np.clip(img, 0, 1))
            axes2[row, 0].set_title('Original' if row == 0 else '')
            axes2[row, 0].axis('off')

            # Baseline reconstructions
            base = np.transpose(np.array(compare_data['recon_base_base'])[row], (1, 2, 0))
            axes2[row, 1].imshow(np.clip(base, 0, 1))
            axes2[row, 1].set_title('Base (no LoRA)' if row == 0 else '')
            axes2[row, 1].axis('off')

            base_w = np.transpose(np.array(compare_data['recon_base_wav50'])[row], (1, 2, 0))
            axes2[row, 2].imshow(np.clip(base_w, 0, 1))
            axes2[row, 2].set_title('Base Wav@50%' if row == 0 else '')
            axes2[row, 2].axis('off')

            # Policy reconstructions
            pol = np.transpose(np.array(compare_data['recon_policy_base'])[row], (1, 2, 0))
            axes2[row, 3].imshow(np.clip(pol, 0, 1))
            axes2[row, 3].set_title('Policy (LoRA)' if row == 0 else '')
            axes2[row, 3].axis('off')

            pol_w = np.transpose(np.array(compare_data['recon_policy_wav50'])[row], (1, 2, 0))
            axes2[row, 4].imshow(np.clip(pol_w, 0, 1))
            axes2[row, 4].set_title('Policy Wav@50%' if row == 0 else '')
            axes2[row, 4].axis('off')

            pol_a = np.transpose(np.array(compare_data['recon_policy_amp50'])[row], (1, 2, 0))
            axes2[row, 5].imshow(np.clip(pol_a, 0, 1))
            axes2[row, 5].set_title('Policy Amp@50%' if row == 0 else '')
            axes2[row, 5].axis('off')

        plt.suptitle('Ablation Robustness: Baseline vs Policy at 50% ablation')
        plt.tight_layout()

        compare_path = output_dir / f"reinforce_ablation_{args.run_id}_compare{suffix}.png"
        plt.savefig(compare_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison to: {compare_path}")

    # Print summary
    print(f"\nFinal metrics (last 10 steps):")
    print(f"  Base MSE: {np.mean(history['mse_base'][-10:]):.6f}")
    print(f"  Wav Ablated MSE: {np.mean(history['mse_wav_ablated'][-10:]):.6f}")
    print(f"  Amp Ablated MSE: {np.mean(history['mse_amp_ablated'][-10:]):.6f}")
    print(f"  Robustness Score: {np.mean(history['robustness_score'][-10:]):.6f}")

    if len(history['robustness_score']) > 1:
        start = history['robustness_score'][0]
        end = np.mean(history['robustness_score'][-10:])
        print(f"\n  Robustness improvement: {start - end:+.6f} (lower=better)")


if __name__ == "__main__":
    main()
