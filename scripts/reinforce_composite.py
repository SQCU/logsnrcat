#!/usr/bin/env python3
"""
Composite REINFORCE trainer with multi-group rewards:
  - Histogram JS divergence (color distribution matching)
  - Subspace ablation MSE (joint, wavelet-only, amplitude-only)
  - Reconstruction MSE trust region (constrains overall quality)

Each reward group has independent advantage computation, expanding
the dynamic range of learning signals vs single-task REINFORCE.
"""

import argparse
import datetime
from pathlib import Path
import requests

DEFAULT_HOST = "172.26.160.1"
DEFAULT_PORT = 8421
DEFAULT_OUTPUT = "experiments_swiglu_ae/main_run_097"


def eval_code(code: str, host: str, port: int, timeout: int = 600) -> dict:
    url = f"http://{host}:{port}/eval"
    resp = requests.post(url, json={"code": code}, timeout=timeout)
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Composite multi-reward REINFORCE")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--steps", type=int, default=210, help="Training steps")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--save-interval", type=int, default=30)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--ablation-rate", type=float, default=1.0)
    parser.add_argument("--trust-weight", type=float, default=2.0,
                        help="Weight for MSE trust region penalty")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (default uses server batch)")
    args = parser.parse_args()

    if args.run_id is None:
        args.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(args.output_dir)

    print(f"Connecting to eval server at http://{args.host}:{args.port}...")
    health = requests.get(f"http://{args.host}:{args.port}/health", timeout=10).json()
    print(f"  Status: {health['status']}, Weights: {health['weights_loaded']}")

    training_code = f'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import traceback

# === Logging Tee ===
class LogTee:
    """Tee stdout to both console and logfile for remote monitoring."""
    def __init__(self, logfile_path):
        self.logfile = open(logfile_path, "w", buffering=1)  # line-buffered
        self.stdout = sys.stdout
    def write(self, msg):
        self.stdout.write(msg)
        self.logfile.write(msg)
        self.logfile.flush()
    def flush(self):
        self.stdout.flush()
        self.logfile.flush()
    def close(self):
        self.logfile.close()

log_path = "/tmp/reinforce_{run_id}.log"
_log_tee = LogTee(log_path)
sys.stdout = _log_tee

def _cleanup_tee():
    global _log_tee
    sys.stdout = _log_tee.stdout
    _log_tee.close()

# Wrap everything in try/finally to ensure cleanup
try:

# === LoRA Implementation ===

class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear.

    Computes: output = base_linear(x) + scale * (x @ A.T @ B.T)
    Where A is [rank, in_features] and B is [out_features, rank]
    """
    def __init__(self, base_linear: nn.Linear, rank: int = 8, scale: float = 1.0):
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.scale = scale

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # A: down-projection, B: up-projection
        # Initialize A with small random, B with zeros (so initial output = base output)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * (1.0 / in_features ** 0.5))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Freeze base weights
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        base_out = self.base(x)
        # LoRA path: x @ A.T @ B.T = (x @ A.T) @ B.T
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base_out + self.scale * lora_out


def inject_lora(model, target_patterns, rank=8, scale=1.0):
    """Inject LoRA adapters into modules matching target patterns.

    Args:
        model: The model to modify
        target_patterns: List of substrings to match in module names
        rank: LoRA rank
        scale: LoRA output scaling factor

    Returns:
        List of injected LoRA modules (for optimizer param groups)
    """
    lora_modules = []

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        # Check if name matches any target pattern
        if not any(pattern in name for pattern in target_patterns):
            continue

        # Get parent module and attribute name
        parts = name.rsplit('.', 1)
        if len(parts) == 1:
            parent = model
            attr_name = parts[0]
        else:
            parent_name, attr_name = parts
            parent = model.get_submodule(parent_name)

        # Create LoRA wrapper
        lora_linear = LoRALinear(module, rank=rank, scale=scale)
        lora_linear = lora_linear.to(module.weight.device).to(module.weight.dtype)

        # Replace module
        setattr(parent, attr_name, lora_linear)
        lora_modules.append(lora_linear)
        print(f"  LoRA injected: {{name}} [{{module.in_features}}x{{module.out_features}}] rank={{rank}}")

    return lora_modules


def get_lora_params(lora_modules):
    """Get all trainable LoRA parameters."""
    params = []
    for m in lora_modules:
        params.extend([m.lora_A, m.lora_B])
    return params


class lora_disabled:
    """Context manager to temporarily disable LoRA (set scale=0)."""
    def __init__(self, lora_modules):
        self.modules = lora_modules
        self.saved_scales = []

    def __enter__(self):
        for m in self.modules:
            self.saved_scales.append(m.scale)
            m.scale = 0.0
        return self

    def __exit__(self, *args):
        for m, scale in zip(self.modules, self.saved_scales):
            m.scale = scale


# === Reward Modules ===

def compute_js_divergence(imgs, recons, n_bins=32):
    """Histogram JS divergence - lower = better color matching."""
    imgs_np = imgs.detach().float().cpu().numpy()
    recons_np = recons.detach().float().cpu().numpy()
    js = 0.0
    for c in range(3):
        h_in, _ = np.histogram(imgs_np[:, c].flatten(), bins=n_bins, range=(0, 1), density=True)
        h_re, _ = np.histogram(recons_np[:, c].flatten(), bins=n_bins, range=(0, 1), density=True)
        h_in = h_in / (h_in.sum() + 1e-10)
        h_re = h_re / (h_re.sum() + 1e-10)
        m = 0.5 * (h_in + h_re)
        js += 0.5 * np.sum(h_in * np.log((h_in + 1e-10) / (m + 1e-10)))
        js += 0.5 * np.sum(h_re * np.log((h_re + 1e-10) / (m + 1e-10)))
    return js / 3.0

def soft_histogram_loss(img, ref, n_bins=32, sigma=0.05):
    """Differentiable soft histogram JS divergence."""
    bins = torch.linspace(0, 1, n_bins, device=img.device, dtype=img.dtype)
    loss = 0.0
    for c in range(3):
        img_flat = img[:, c].reshape(-1)
        ref_flat = ref[:, c].reshape(-1)
        img_d = (img_flat.unsqueeze(1) - bins) / sigma
        ref_d = (ref_flat.unsqueeze(1) - bins) / sigma
        img_h = torch.exp(-0.5 * img_d ** 2).sum(0)
        ref_h = torch.exp(-0.5 * ref_d ** 2).sum(0)
        img_h = img_h / (img_h.sum() + 1e-8)
        ref_h = ref_h / (ref_h.sum() + 1e-8)
        m = 0.5 * (img_h + ref_h)
        loss += 0.5 * (ref_h * torch.log((ref_h + 1e-8) / (m + 1e-8))).sum()
        loss += 0.5 * (img_h * torch.log((img_h + 1e-8) / (m + 1e-8))).sum()
    return loss / 3

def compute_subspace_mse(ae, codes_adapted, images, grid_shape, decoder_masks, ablation_rate):
    """Compute MSE for joint, wavelet-only, amplitude-only reconstructions."""
    images_f = images.float()
    with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
        recon_joint = ae.decode(codes_adapted, grid_shape, decoder_masks).float()
        recon_wav = ae.decode_with_ablation(
            codes_adapted, grid_shape,
            ablate_wavelet=0.0, ablate_amplitude=ablation_rate,
            decoder_masks=decoder_masks, deterministic=True
        ).float()
        recon_amp = ae.decode_with_ablation(
            codes_adapted, grid_shape,
            ablate_wavelet=ablation_rate, ablate_amplitude=0.0,
            decoder_masks=decoder_masks, deterministic=True
        ).float()
    return {{
        "joint": F.mse_loss(recon_joint, images_f),
        "wav": F.mse_loss(recon_wav, images_f),
        "amp": F.mse_loss(recon_amp, images_f),
        "recon_joint": recon_joint,
        "recon_wav": recon_wav,
        "recon_amp": recon_amp
    }}

# === Setup ===
ae = model.sparse_ae
code_dim = ae.code_dim
device = ctx.device
run_id = "{args.run_id}"

print(f"=== Composite REINFORCE run_id={{run_id}} ===")
print(f"Log file: {{log_path}}")

# Target modules for LoRA injection (decoder-only for reconstruction task)
# Targeting: attention projections, MLP layers, and code embedding layers
lora_targets = [
    # Decoder attention
    "decoders.0.transformer.layers",  # catches q_proj, k_proj, v_proj, out_proj
    "decoders.1.transformer.layers",
    # Decoder code embeddings (directly affect reconstruction)
    "decoders.0.wav_embed",
    "decoders.0.amp_embed",
    "decoders.1.wav_embed",
    "decoders.1.amp_embed",
]

print(f"Injecting LoRA (rank={args.lora_rank}) into decoder modules...")
lora_modules = inject_lora(ae, lora_targets, rank={args.lora_rank}, scale=1.0)
lora_params = get_lora_params(lora_modules)
print(f"Total LoRA modules: {{len(lora_modules)}}, params: {{sum(p.numel() for p in lora_params)}}")

optimizer = torch.optim.Adam(lora_params, lr={args.lr})

# History for all reward groups
history = {{
    "step": [],
    # Histogram group
    "js_div": [], "reward_js": [], "adv_js": [],
    # Subspace groups
    "mse_joint": [], "mse_wav": [], "mse_amp": [],
    "reward_joint": [], "reward_wav": [], "reward_amp": [],
    "adv_joint": [], "adv_wav": [], "adv_amp": [],
    # Trust region
    "mse_recon": [], "trust_penalty": [],
    # Total
    "loss": []
}}

# Running baselines for each reward group (5 groups total)
baselines = {{"js": None, "joint": None, "wav": None, "amp": None, "recon": None}}
ablation_rate = {args.ablation_rate}
trust_weight = {args.trust_weight}

# === Setup ===
batch_size = {args.batch_size}
if batch_size > 4:
    # Generate larger batch from iterator
    blocks = ctx.iterator.generate_batch_list(batch_size=batch_size * 2, resolution=64)
    matching = [b.content for b in blocks if b.content.shape[-1] == 64][:batch_size]
    images = torch.stack(matching).to(device)
    print(f"Generated batch of {{images.shape[0]}} images at 64px")
else:
    images = batch
p = ae.patch_size
grid_shape = (images.shape[2] // p, images.shape[3] // p)
encoder_masks, decoder_masks = ae.build_masks(grid_shape, images.device)

# Get baseline reconstruction MSE (trust region anchor)
with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    base_codes = ae.encode(images, grid_shape=grid_shape, encoder_masks=encoder_masks, decoder_masks=decoder_masks)
    base_recon = ae.decode(base_codes, grid_shape, decoder_masks)
baseline_recon_mse = F.mse_loss(base_recon.float(), images.float()).item()
print(f"Baseline reconstruction MSE: {{baseline_recon_mse:.6f}} (trust region anchor)")

# Sanity check - LoRA B initialized to zero, so initial output should match base
# After first gradient step, output will differ
print(f"Sanity check: LoRA initialized with B=0 (output matches base until first update)")

n_wav = ae.n_wavelet_dims if ae.n_wavelet_dims is not None else ae.code_dim // 2
print(f"Subspace dims: wavelet={{n_wav}}, amplitude={{ae.code_dim - n_wav}}")
print(f"Reward groups: JS divergence, Joint MSE, Wavelet-only MSE, Amplitude-only MSE")
print(f"Trust region weight: {{trust_weight}}")

# === Training loop ===
n_steps = {args.steps}
save_interval = {args.save_interval}

for step in range(n_steps):
    # Encode (no grad - encoder is frozen)
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        codes_list = ae.encode(images, grid_shape=grid_shape,
                               encoder_masks=encoder_masks, decoder_masks=decoder_masks)
        codes_list = [c.detach() for c in codes_list]

    # Decode WITH gradients through LoRA-adapted decoder
    # (LoRA modules are inside ae.decode, gradients flow through them)

    # === Compute all rewards ===
    images_f = images.float()

    # Subspace MSE (also gives us recon_joint for JS and trust region)
    # LoRA is inside decoder, so gradients flow through automatically
    subspace = compute_subspace_mse(ae, codes_list, images, grid_shape, decoder_masks, ablation_rate)
    mse_joint = subspace["joint"].item()
    mse_wav = subspace["wav"].item()
    mse_amp = subspace["amp"].item()
    recon_joint = subspace["recon_joint"]

    # JS divergence (histogram matching)
    js_div = compute_js_divergence(images, recon_joint)

    # Trust region: reconstruction MSE relative to baseline
    mse_recon = mse_joint  # Joint reconstruction is main output
    trust_violation = max(0, mse_recon - baseline_recon_mse)

    # === Rewards (negative losses = higher is better) ===
    reward_js = -js_div
    reward_joint = -mse_joint
    reward_wav = -mse_wav
    reward_amp = -mse_amp
    reward_recon = -mse_recon

    # === Update baselines with EMA ===
    for key, reward in [("js", reward_js), ("joint", reward_joint), ("wav", reward_wav),
                        ("amp", reward_amp), ("recon", reward_recon)]:
        if baselines[key] is None:
            baselines[key] = reward
        else:
            baselines[key] = 0.9 * baselines[key] + 0.1 * reward

    # === Compute advantages ===
    adv_js = reward_js - baselines["js"]
    adv_joint = reward_joint - baselines["joint"]
    adv_wav = reward_wav - baselines["wav"]
    adv_amp = reward_amp - baselines["amp"]

    # === Optimization ===
    optimizer.zero_grad()

    # Differentiable losses
    hist_loss = soft_histogram_loss(recon_joint, images_f)
    mse_loss_joint = subspace["joint"]
    mse_loss_wav = subspace["wav"]
    mse_loss_amp = subspace["amp"]

    # L2 regularization on LoRA params
    l2_reg = sum((p ** 2).sum() for p in lora_params)

    # Advantage-weighted losses (positive advantage = emphasize that group)
    weight_js = 1.0 + max(0.0, adv_js * 10.0)
    weight_joint = 1.0 + max(0.0, adv_joint * 10.0)
    weight_wav = 0.5 + max(0.0, adv_wav * 10.0)
    weight_amp = 0.5 + max(0.0, adv_amp * 10.0)

    # Trust region penalty: penalize if reconstruction degrades
    trust_penalty = trust_weight * trust_violation

    loss = (weight_js * hist_loss +
            weight_joint * mse_loss_joint +
            weight_wav * mse_loss_wav +
            weight_amp * mse_loss_amp +
            trust_penalty +
            0.005 * l2_reg)

    loss.backward()
    optimizer.step()

    # === Log ===
    history["step"].append(step)
    history["js_div"].append(js_div)
    history["reward_js"].append(reward_js)
    history["adv_js"].append(adv_js)
    history["mse_joint"].append(mse_joint)
    history["mse_wav"].append(mse_wav)
    history["mse_amp"].append(mse_amp)
    history["reward_joint"].append(reward_joint)
    history["reward_wav"].append(reward_wav)
    history["reward_amp"].append(reward_amp)
    history["adv_joint"].append(adv_joint)
    history["adv_wav"].append(adv_wav)
    history["adv_amp"].append(adv_amp)
    history["mse_recon"].append(mse_recon)
    history["trust_penalty"].append(trust_penalty)
    history["loss"].append(loss.item())

    if step % 30 == 0:
        print(f"Step {{step:3d}}: JS={{js_div:.4f}} joint={{mse_joint:.5f}} wav={{mse_wav:.5f}} amp={{mse_amp:.5f}} | adv: js={{adv_js:+.3f}} j={{adv_joint:+.4f}} w={{adv_wav:+.4f}} a={{adv_amp:+.4f}}")

    # === Save comparison plots ===
    if step % save_interval == 0:
        # Baseline: decode with LoRA disabled
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16), lora_disabled(lora_modules):
            base_codes = ae.encode(images, grid_shape=grid_shape, encoder_masks=encoder_masks, decoder_masks=decoder_masks)
            base_joint = ae.decode(base_codes, grid_shape, decoder_masks)

        js_base = compute_js_divergence(images, base_joint)
        mse_base = F.mse_loss(base_joint.float(), images_f).item()

        print(f"  [Step {{step}}] vs baseline: JS={{js_div:.4f}} (base={{js_base:.4f}}, Δ={{js_div-js_base:+.4f}}) | MSE={{mse_joint:.5f}} (base={{mse_base:.5f}}, Δ={{mse_joint-mse_base:+.5f}})")

        # Visual grid: 4 rows x 4 cols (orig, base, policy, diff)
        n = min(4, len(images))
        fig, axes = plt.subplots(n, 4, figsize=(10, 10))

        orig_np = images[:n].float().cpu().numpy().transpose(0, 2, 3, 1)
        base_np = base_joint[:n].float().cpu().numpy().transpose(0, 2, 3, 1)
        policy_np = recon_joint[:n].detach().cpu().numpy().transpose(0, 2, 3, 1)

        for i in range(n):
            axes[i, 0].imshow(orig_np[i].clip(0, 1))
            axes[i, 0].axis("off")
            if i == 0: axes[i, 0].set_title("Original", fontsize=9)

            axes[i, 1].imshow(base_np[i].clip(0, 1))
            axes[i, 1].axis("off")
            if i == 0: axes[i, 1].set_title(f"Baseline (JS={{js_base:.3f}})", fontsize=9)

            axes[i, 2].imshow(policy_np[i].clip(0, 1))
            axes[i, 2].axis("off")
            if i == 0: axes[i, 2].set_title(f"Policy (JS={{js_div:.3f}})", fontsize=9)

            # Difference map (amplified)
            diff = np.abs(policy_np[i] - base_np[i])
            axes[i, 3].imshow((diff * 5).clip(0, 1))
            axes[i, 3].axis("off")
            if i == 0: axes[i, 3].set_title("Diff (5x)", fontsize=9)

        plt.suptitle(f"Step {{step}} | JS Δ={{js_div-js_base:+.4f}} | MSE Δ={{mse_joint-mse_base:+.5f}}", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/composite_{{run_id}}_step{{step:03d}}.png", dpi=150, bbox_inches="tight")
        plt.close()

# === Final results ===
# Baseline metrics (LoRA disabled)
with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16), lora_disabled(lora_modules):
    base_codes = ae.encode(images, grid_shape=grid_shape, encoder_masks=encoder_masks, decoder_masks=decoder_masks)
    base_joint = ae.decode(base_codes, grid_shape, decoder_masks)
    base_wav = ae.decode_with_ablation(base_codes, grid_shape, ablate_wavelet=0.0, ablate_amplitude=ablation_rate, decoder_masks=decoder_masks, deterministic=True)
    base_amp = ae.decode_with_ablation(base_codes, grid_shape, ablate_wavelet=ablation_rate, ablate_amplitude=0.0, decoder_masks=decoder_masks, deterministic=True)

final_base_js = compute_js_divergence(images, base_joint)
final_base_mse_joint = F.mse_loss(base_joint.float(), images_f).item()
final_base_mse_wav = F.mse_loss(base_wav.float(), images_f).item()
final_base_mse_amp = F.mse_loss(base_amp.float(), images_f).item()

print(f"\\n=== Final Results ===")
print(f"JS Divergence: start={{history['js_div'][0]:.4f}} end={{history['js_div'][-1]:.4f}} base={{final_base_js:.4f}} Δ={{history['js_div'][-1]-final_base_js:+.4f}}")
print(f"MSE Joint:     start={{history['mse_joint'][0]:.5f}} end={{history['mse_joint'][-1]:.5f}} base={{final_base_mse_joint:.5f}} Δ={{history['mse_joint'][-1]-final_base_mse_joint:+.5f}}")
print(f"MSE Wavelet:   start={{history['mse_wav'][0]:.5f}} end={{history['mse_wav'][-1]:.5f}} base={{final_base_mse_wav:.5f}} Δ={{history['mse_wav'][-1]-final_base_mse_wav:+.5f}}")
print(f"MSE Amplitude: start={{history['mse_amp'][0]:.5f}} end={{history['mse_amp'][-1]:.5f}} base={{final_base_mse_amp:.5f}} Δ={{history['mse_amp'][-1]-final_base_mse_amp:+.5f}}")

# === Training curves ===
fig, axes = plt.subplots(3, 3, figsize=(14, 12))

# Row 1: JS and MSE trajectories
axes[0, 0].plot(history["step"], history["js_div"], "b-", lw=0.8)
axes[0, 0].axhline(y=final_base_js, color="b", ls="--", alpha=0.3, label="baseline")
axes[0, 0].set_title("JS Divergence (↓)")
axes[0, 0].legend(fontsize=7)

axes[0, 1].plot(history["step"], history["mse_joint"], "b-", label="Joint", lw=0.8)
axes[0, 1].plot(history["step"], history["mse_wav"], "g-", label="Wav-only", lw=0.8)
axes[0, 1].plot(history["step"], history["mse_amp"], "r-", label="Amp-only", lw=0.8)
axes[0, 1].axhline(y=final_base_mse_joint, color="b", ls="--", alpha=0.3)
axes[0, 1].set_title("Subspace MSE (↓)")
axes[0, 1].legend(fontsize=7)

axes[0, 2].plot(history["step"], history["trust_penalty"], "purple", lw=0.8)
axes[0, 2].set_title("Trust Region Penalty")

# Row 2: Advantages (expanded dynamic range)
axes[1, 0].plot(history["step"], history["adv_js"], "b-", lw=0.8)
axes[1, 0].axhline(y=0, color="gray", ls="--")
axes[1, 0].set_title("JS Advantage")

axes[1, 1].plot(history["step"], history["adv_joint"], "b-", label="Joint", lw=0.8)
axes[1, 1].plot(history["step"], history["adv_wav"], "g-", label="Wav", lw=0.8)
axes[1, 1].plot(history["step"], history["adv_amp"], "r-", label="Amp", lw=0.8)
axes[1, 1].axhline(y=0, color="gray", ls="--")
axes[1, 1].set_title("Subspace Advantages")
axes[1, 1].legend(fontsize=7)

# Combined advantage magnitude
total_adv = [abs(history["adv_js"][i]) + abs(history["adv_joint"][i]) + abs(history["adv_wav"][i]) + abs(history["adv_amp"][i]) for i in range(len(history["step"]))]
axes[1, 2].plot(history["step"], total_adv, "orange", lw=0.8)
axes[1, 2].set_title("Total |Advantage| (dynamic range)")

# Row 3: Improvements and loss
imp_js = [history["js_div"][0] - x for x in history["js_div"]]
imp_joint = [history["mse_joint"][0] - x for x in history["mse_joint"]]
axes[2, 0].plot(history["step"], imp_js, "b-", lw=0.8)
axes[2, 0].axhline(y=0, color="gray", ls="--")
axes[2, 0].set_title("JS Improvement vs Start (↑)")

axes[2, 1].plot(history["step"], imp_joint, "b-", label="Joint", lw=0.8)
imp_wav = [history["mse_wav"][0] - x for x in history["mse_wav"]]
imp_amp = [history["mse_amp"][0] - x for x in history["mse_amp"]]
axes[2, 1].plot(history["step"], imp_wav, "g-", label="Wav", lw=0.8)
axes[2, 1].plot(history["step"], imp_amp, "r-", label="Amp", lw=0.8)
axes[2, 1].axhline(y=0, color="gray", ls="--")
axes[2, 1].set_title("MSE Improvement vs Start (↑)")
axes[2, 1].legend(fontsize=7)

axes[2, 2].plot(history["step"], history["loss"], "purple", lw=0.8)
axes[2, 2].set_title("Total Loss")

plt.suptitle(f"Composite REINFORCE ({{run_id}}) - {{n_steps}} steps @ lr={args.lr}", fontsize=12)
plt.tight_layout()
plt.savefig(f"{args.output_dir}/composite_{{run_id}}_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# Store results
ctx._composite_result = {{
    "run_id": run_id,
    "n_steps": n_steps,
    "js": {{"start": history["js_div"][0], "end": history["js_div"][-1], "base": final_base_js, "improvement": history["js_div"][0] - history["js_div"][-1]}},
    "mse_joint": {{"start": history["mse_joint"][0], "end": history["mse_joint"][-1], "base": final_base_mse_joint, "improvement": history["mse_joint"][0] - history["mse_joint"][-1]}},
    "mse_wav": {{"start": history["mse_wav"][0], "end": history["mse_wav"][-1], "base": final_base_mse_wav, "improvement": history["mse_wav"][0] - history["mse_wav"][-1]}},
    "mse_amp": {{"start": history["mse_amp"][0], "end": history["mse_amp"][-1], "base": final_base_mse_amp, "improvement": history["mse_amp"][0] - history["mse_amp"][-1]}},
    "final_advantage_range": total_adv[-1] if total_adv else 0
}}
ctx._lora_modules = lora_modules
ctx._composite_history = history

except Exception as e:
    print(f"\\n!!! EXCEPTION !!!")
    print(traceback.format_exc())
    ctx._composite_error = str(e)
    raise
finally:
    print(f"\\n=== Run complete. Log saved to: {{log_path}} ===")
    _cleanup_tee()
'''

    print(f"\nSubmitting {args.steps}-step composite REINFORCE to server...")
    print(f"Run ID: {args.run_id}")
    print(f"Reward groups: JS divergence + Joint/Wavelet/Amplitude MSE")
    print(f"Trust region weight: {args.trust_weight}")
    print("(All computation happens server-side)")

    result = eval_code(training_code, args.host, args.port, timeout=1200)

    if not result['success']:
        print(f"ERROR: {result['error']}")
        print("Training may still be running server-side. Poll ctx._composite_result later.")
        return

    # Fetch results
    result = eval_code("ctx._composite_result", args.host, args.port)
    if result['success']:
        r = result['result']
        print(f"\n{'='*60}")
        print(f"Results (run_id={r['run_id']})")
        print(f"{'='*60}")
        print(f"JS Divergence: {r['js']['start']:.4f} → {r['js']['end']:.4f} (Δ {r['js']['improvement']:+.4f}, base={r['js']['base']:.4f})")
        print(f"MSE Joint:     {r['mse_joint']['start']:.5f} → {r['mse_joint']['end']:.5f} (Δ {r['mse_joint']['improvement']:+.5f}, base={r['mse_joint']['base']:.5f})")
        print(f"MSE Wavelet:   {r['mse_wav']['start']:.5f} → {r['mse_wav']['end']:.5f} (Δ {r['mse_wav']['improvement']:+.5f}, base={r['mse_wav']['base']:.5f})")
        print(f"MSE Amplitude: {r['mse_amp']['start']:.5f} → {r['mse_amp']['end']:.5f} (Δ {r['mse_amp']['improvement']:+.5f}, base={r['mse_amp']['base']:.5f})")
        print(f"\nSaved:")
        print(f"  {output_dir}/composite_{args.run_id}_curves.png")
        print(f"  {output_dir}/composite_{args.run_id}_step*.png")


if __name__ == "__main__":
    main()
