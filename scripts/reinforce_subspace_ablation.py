#!/usr/bin/env python3
"""
Group policy REINFORCE with subspace ablation rewards.

Trains LoRA adapters to improve MSE reconstruction across:
  - Wavelet-only (amplitude ablated)
  - Amplitude-only (wavelet ablated)
  - Joint (no ablation)

Uses relative advantage across groups to weight gradients.
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
    parser = argparse.ArgumentParser(description="Subspace ablation REINFORCE")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--steps", type=int, default=120, help="Training steps")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--save-interval", type=int, default=30)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--ablation-rate", type=float, default=1.0,
                        help="Ablation rate for subspace tests (0-1)")
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

# === Setup ===
ae = model.sparse_ae
code_dim = ae.code_dim
device = ctx.device
run_id = "{args.run_id}"

class LoRACodeAdapter(nn.Module):
    def __init__(self, dim, rank=8):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, dim) * (1.0 / dim ** 0.5))
        self.B = nn.Parameter(torch.randn(dim, rank) * 0.01)
    def forward(self, x):
        return F.linear(F.linear(x, self.A), self.B)

lora_enc = LoRACodeAdapter(code_dim, rank={args.lora_rank}).to(device).to(torch.bfloat16)
lora_dec = LoRACodeAdapter(code_dim, rank={args.lora_rank}).to(device).to(torch.bfloat16)
optimizer = torch.optim.Adam(
    list(lora_enc.parameters()) + list(lora_dec.parameters()),
    lr={args.lr}
)

# History tracks all three groups
history = {{
    "step": [],
    "mse_joint": [], "mse_wav": [], "mse_amp": [],
    "reward_joint": [], "reward_wav": [], "reward_amp": [],
    "loss": [], "advantage_joint": [], "advantage_wav": [], "advantage_amp": []
}}

# Running baselines for each group
baselines = {{"joint": None, "wav": None, "amp": None}}
ablation_rate = {args.ablation_rate}

# === Precompute masks ===
images = batch
p = ae.patch_size
grid_shape = (images.shape[2] // p, images.shape[3] // p)
encoder_masks, decoder_masks = ae.build_masks(grid_shape, images.device)

# === Sanity check ===
with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    test_codes = ae.encode(images, grid_shape=grid_shape, encoder_masks=encoder_masks, decoder_masks=decoder_masks)
    recon_base = ae.decode(test_codes, grid_shape, decoder_masks)
    test_adapted = [c + lora_enc(c) + lora_dec(c + lora_enc(c)) for c in test_codes]
    recon_adapted = ae.decode(test_adapted, grid_shape, decoder_masks)
    sanity_mse = F.mse_loss(recon_base.float(), recon_adapted.float()).item()
print(f"Sanity check: adapter MSE diff = {{sanity_mse:.8f}}")
if sanity_mse < 1e-10:
    raise ValueError("Adapter produces no change!")

# Get subspace info
n_wav = ae.n_wavelet_dims if ae.n_wavelet_dims is not None else ae.code_dim // 2
n_amp = ae.code_dim - n_wav
print(f"Subspace dims: wavelet={{n_wav}}, amplitude={{n_amp}}, total={{ae.code_dim}}")

# === Training loop ===
n_steps = {args.steps}
save_interval = {args.save_interval}

for step in range(n_steps):
    # Encode with frozen AE
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        codes_list = ae.encode(images, grid_shape=grid_shape,
                               encoder_masks=encoder_masks, decoder_masks=decoder_masks)
        codes_list = [c.detach() for c in codes_list]

    # Apply adapters (with gradients) - keep in bf16
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        codes_adapted = []
        for codes in codes_list:
            adapted = codes + lora_enc(codes)
            adapted = adapted + lora_dec(adapted)
            codes_adapted.append(adapted)

    # Decode under three conditions
    with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
        # Joint (no ablation)
        recon_joint = ae.decode(codes_adapted, grid_shape, decoder_masks).float()

        # Wavelet-only (ablate amplitude)
        recon_wav = ae.decode_with_ablation(
            codes_adapted, grid_shape,
            ablate_wavelet=0.0, ablate_amplitude=ablation_rate,
            decoder_masks=decoder_masks, deterministic=True
        ).float()

        # Amplitude-only (ablate wavelet)
        recon_amp = ae.decode_with_ablation(
            codes_adapted, grid_shape,
            ablate_wavelet=ablation_rate, ablate_amplitude=0.0,
            decoder_masks=decoder_masks, deterministic=True
        ).float()

    # Compute MSE for each group
    images_f = images.float()
    mse_joint = F.mse_loss(recon_joint, images_f).item()
    mse_wav = F.mse_loss(recon_wav, images_f).item()
    mse_amp = F.mse_loss(recon_amp, images_f).item()

    # Rewards (negative MSE - lower MSE = higher reward)
    reward_joint = -mse_joint
    reward_wav = -mse_wav
    reward_amp = -mse_amp

    # Update baselines with EMA
    for key, reward in [("joint", reward_joint), ("wav", reward_wav), ("amp", reward_amp)]:
        if baselines[key] is None:
            baselines[key] = reward
        else:
            baselines[key] = 0.9 * baselines[key] + 0.1 * reward

    # Compute advantages
    adv_joint = reward_joint - baselines["joint"]
    adv_wav = reward_wav - baselines["wav"]
    adv_amp = reward_amp - baselines["amp"]

    # === Optimization ===
    optimizer.zero_grad()

    # Recompute with gradients for loss
    mse_loss_joint = F.mse_loss(recon_joint, images_f)
    mse_loss_wav = F.mse_loss(recon_wav, images_f)
    mse_loss_amp = F.mse_loss(recon_amp, images_f)

    # L2 regularization
    l2_reg = sum((p ** 2).sum() for p in lora_enc.parameters())
    l2_reg += sum((p ** 2).sum() for p in lora_dec.parameters())

    # Weighted loss: joint gets 2x weight, subspaces get advantage-weighted
    # When advantage > 0, we're doing better than baseline -> emphasize that group
    weight_joint = 2.0 + max(0.0, adv_joint * 10.0)
    weight_wav = 1.0 + max(0.0, adv_wav * 10.0)
    weight_amp = 1.0 + max(0.0, adv_amp * 10.0)

    loss = (weight_joint * mse_loss_joint +
            weight_wav * mse_loss_wav +
            weight_amp * mse_loss_amp +
            0.01 * l2_reg)

    loss.backward()
    optimizer.step()

    # Log
    history["step"].append(step)
    history["mse_joint"].append(mse_joint)
    history["mse_wav"].append(mse_wav)
    history["mse_amp"].append(mse_amp)
    history["reward_joint"].append(reward_joint)
    history["reward_wav"].append(reward_wav)
    history["reward_amp"].append(reward_amp)
    history["loss"].append(loss.item())
    history["advantage_joint"].append(adv_joint)
    history["advantage_wav"].append(adv_wav)
    history["advantage_amp"].append(adv_amp)

    if step % 20 == 0:
        print(f"Step {{step:3d}}: joint={{mse_joint:.5f}} wav={{mse_wav:.5f}} amp={{mse_amp:.5f}} | adv: j={{adv_joint:+.4f}} w={{adv_wav:+.4f}} a={{adv_amp:+.4f}}")

    # Save comparison plots
    if step % save_interval == 0:
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Baseline (no adapter)
            base_codes = ae.encode(images, grid_shape=grid_shape, encoder_masks=encoder_masks, decoder_masks=decoder_masks)
            base_joint = ae.decode(base_codes, grid_shape, decoder_masks)
            base_wav = ae.decode_with_ablation(base_codes, grid_shape, ablate_wavelet=0.0, ablate_amplitude=ablation_rate, decoder_masks=decoder_masks, deterministic=True)
            base_amp = ae.decode_with_ablation(base_codes, grid_shape, ablate_wavelet=ablation_rate, ablate_amplitude=0.0, decoder_masks=decoder_masks, deterministic=True)

        # Compute MSE deltas (policy - baseline)
        delta_joint = mse_joint - F.mse_loss(base_joint.float(), images_f).item()
        delta_wav = mse_wav - F.mse_loss(base_wav.float(), images_f).item()
        delta_amp = mse_amp - F.mse_loss(base_amp.float(), images_f).item()

        print(f"  [Step {{step}}] MSE delta vs baseline: joint={{delta_joint:+.6f}} wav={{delta_wav:+.6f}} amp={{delta_amp:+.6f}}")

        # Visual comparison grid: 4 rows (samples) x 7 cols (orig, base_j, base_w, base_a, pol_j, pol_w, pol_a)
        n = min(4, len(images))
        fig, axes = plt.subplots(n, 7, figsize=(14, 8))

        orig_np = images[:n].float().cpu().numpy().transpose(0, 2, 3, 1)
        base_j_np = base_joint[:n].float().cpu().numpy().transpose(0, 2, 3, 1)
        base_w_np = base_wav[:n].float().cpu().numpy().transpose(0, 2, 3, 1)
        base_a_np = base_amp[:n].float().cpu().numpy().transpose(0, 2, 3, 1)
        pol_j_np = recon_joint[:n].detach().cpu().numpy().transpose(0, 2, 3, 1)
        pol_w_np = recon_wav[:n].detach().cpu().numpy().transpose(0, 2, 3, 1)
        pol_a_np = recon_amp[:n].detach().cpu().numpy().transpose(0, 2, 3, 1)

        titles = ["Original", "Base Joint", "Base Wav", "Base Amp", "Policy Joint", "Policy Wav", "Policy Amp"]
        for i in range(n):
            for j, img in enumerate([orig_np[i], base_j_np[i], base_w_np[i], base_a_np[i], pol_j_np[i], pol_w_np[i], pol_a_np[i]]):
                axes[i, j].imshow(img.clip(0, 1))
                axes[i, j].axis("off")
                if i == 0:
                    axes[i, j].set_title(titles[j], fontsize=8)

        plt.suptitle(f"Step {{step}} | Joint Δ={{delta_joint:+.5f}} Wav Δ={{delta_wav:+.5f}} Amp Δ={{delta_amp:+.5f}}", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/subspace_{{run_id}}_step{{step:03d}}.png", dpi=150, bbox_inches="tight")
        plt.close()

# === Final comparison ===
with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    base_codes = ae.encode(images, grid_shape=grid_shape, encoder_masks=encoder_masks, decoder_masks=decoder_masks)
    base_joint = ae.decode(base_codes, grid_shape, decoder_masks)
    base_wav = ae.decode_with_ablation(base_codes, grid_shape, ablate_wavelet=0.0, ablate_amplitude=ablation_rate, decoder_masks=decoder_masks, deterministic=True)
    base_amp = ae.decode_with_ablation(base_codes, grid_shape, ablate_wavelet=ablation_rate, ablate_amplitude=0.0, decoder_masks=decoder_masks, deterministic=True)

final_base_mse_joint = F.mse_loss(base_joint.float(), images_f).item()
final_base_mse_wav = F.mse_loss(base_wav.float(), images_f).item()
final_base_mse_amp = F.mse_loss(base_amp.float(), images_f).item()

print(f"\\nFinal Results:")
print(f"  Baseline MSE: joint={{final_base_mse_joint:.6f}} wav={{final_base_mse_wav:.6f}} amp={{final_base_mse_amp:.6f}}")
print(f"  Policy MSE:   joint={{history['mse_joint'][-1]:.6f}} wav={{history['mse_wav'][-1]:.6f}} amp={{history['mse_amp'][-1]:.6f}}")
print(f"  Delta:        joint={{history['mse_joint'][-1] - final_base_mse_joint:+.6f}} wav={{history['mse_wav'][-1] - final_base_mse_wav:+.6f}} amp={{history['mse_amp'][-1] - final_base_mse_amp:+.6f}}")

# === Training curves ===
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# MSE curves
axes[0, 0].plot(history["step"], history["mse_joint"], "b-", label="Joint", lw=0.8)
axes[0, 0].plot(history["step"], history["mse_wav"], "g-", label="Wavelet-only", lw=0.8)
axes[0, 0].plot(history["step"], history["mse_amp"], "r-", label="Amplitude-only", lw=0.8)
axes[0, 0].axhline(y=final_base_mse_joint, color="b", ls="--", alpha=0.3)
axes[0, 0].axhline(y=final_base_mse_wav, color="g", ls="--", alpha=0.3)
axes[0, 0].axhline(y=final_base_mse_amp, color="r", ls="--", alpha=0.3)
axes[0, 0].set_title("MSE by Subspace")
axes[0, 0].legend(fontsize=8)

# Rewards
axes[0, 1].plot(history["step"], history["reward_joint"], "b-", label="Joint", lw=0.8)
axes[0, 1].plot(history["step"], history["reward_wav"], "g-", label="Wavelet", lw=0.8)
axes[0, 1].plot(history["step"], history["reward_amp"], "r-", label="Amplitude", lw=0.8)
axes[0, 1].set_title("Reward (-MSE)")
axes[0, 1].legend(fontsize=8)

# Advantages
axes[0, 2].plot(history["step"], history["advantage_joint"], "b-", label="Joint", lw=0.8)
axes[0, 2].plot(history["step"], history["advantage_wav"], "g-", label="Wavelet", lw=0.8)
axes[0, 2].plot(history["step"], history["advantage_amp"], "r-", label="Amplitude", lw=0.8)
axes[0, 2].axhline(y=0, color="gray", ls="--")
axes[0, 2].set_title("Advantage")
axes[0, 2].legend(fontsize=8)

# Loss
axes[1, 0].plot(history["step"], history["loss"], "purple", lw=0.8)
axes[1, 0].set_title("Total Loss")

# Improvement vs start
imp_joint = [history["mse_joint"][0] - m for m in history["mse_joint"]]
imp_wav = [history["mse_wav"][0] - m for m in history["mse_wav"]]
imp_amp = [history["mse_amp"][0] - m for m in history["mse_amp"]]
axes[1, 1].plot(history["step"], imp_joint, "b-", label="Joint", lw=0.8)
axes[1, 1].plot(history["step"], imp_wav, "g-", label="Wavelet", lw=0.8)
axes[1, 1].plot(history["step"], imp_amp, "r-", label="Amplitude", lw=0.8)
axes[1, 1].axhline(y=0, color="gray", ls="--")
axes[1, 1].set_title("MSE Reduction vs Start (↑ = better)")
axes[1, 1].legend(fontsize=8)

# Summary text
axes[1, 2].axis("off")
summary = f"Run: {{run_id}}\\nSteps: {{n_steps}}\\nLR: {args.lr}\\nRank: {args.lora_rank}\\nAblation: {{ablation_rate:.0%}}\\n\\n"
summary += f"Start MSE:\\n  Joint: {{history['mse_joint'][0]:.5f}}\\n  Wav: {{history['mse_wav'][0]:.5f}}\\n  Amp: {{history['mse_amp'][0]:.5f}}\\n\\n"
summary += f"End MSE:\\n  Joint: {{history['mse_joint'][-1]:.5f}}\\n  Wav: {{history['mse_wav'][-1]:.5f}}\\n  Amp: {{history['mse_amp'][-1]:.5f}}\\n\\n"
summary += f"Improvement:\\n  Joint: {{imp_joint[-1]:+.5f}}\\n  Wav: {{imp_wav[-1]:+.5f}}\\n  Amp: {{imp_amp[-1]:+.5f}}"
axes[1, 2].text(0.1, 0.9, summary, transform=axes[1, 2].transAxes, fontsize=9, verticalalignment="top", family="monospace")

plt.suptitle(f"Subspace Ablation REINFORCE ({{run_id}})", fontsize=12)
plt.tight_layout()
plt.savefig(f"{args.output_dir}/subspace_{{run_id}}_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# Store results
ctx._subspace_result = {{
    "run_id": run_id,
    "n_steps": n_steps,
    "start_mse": {{"joint": history["mse_joint"][0], "wav": history["mse_wav"][0], "amp": history["mse_amp"][0]}},
    "end_mse": {{"joint": history["mse_joint"][-1], "wav": history["mse_wav"][-1], "amp": history["mse_amp"][-1]}},
    "baseline_mse": {{"joint": final_base_mse_joint, "wav": final_base_mse_wav, "amp": final_base_mse_amp}},
    "improvement": {{"joint": imp_joint[-1], "wav": imp_wav[-1], "amp": imp_amp[-1]}}
}}
ctx._lora_enc = lora_enc
ctx._lora_dec = lora_dec
ctx._subspace_history = history
'''

    print(f"\nSubmitting {args.steps}-step subspace ablation REINFORCE to server...")
    print(f"Run ID: {args.run_id}")
    print("(All computation happens server-side)")

    result = eval_code(training_code, args.host, args.port, timeout=1200)

    if not result['success']:
        print(f"ERROR: {result['error']}")
        print("Training may still be running server-side. Poll ctx._subspace_result later.")
        return

    # Fetch results
    result = eval_code("ctx._subspace_result", args.host, args.port)
    if result['success']:
        stats = result['result']
        print(f"\nResults (run_id={stats['run_id']}):")
        print(f"  Start MSE:    joint={stats['start_mse']['joint']:.5f}  wav={stats['start_mse']['wav']:.5f}  amp={stats['start_mse']['amp']:.5f}")
        print(f"  End MSE:      joint={stats['end_mse']['joint']:.5f}  wav={stats['end_mse']['wav']:.5f}  amp={stats['end_mse']['amp']:.5f}")
        print(f"  Baseline MSE: joint={stats['baseline_mse']['joint']:.5f}  wav={stats['baseline_mse']['wav']:.5f}  amp={stats['baseline_mse']['amp']:.5f}")
        print(f"  Improvement:  joint={stats['improvement']['joint']:+.5f}  wav={stats['improvement']['wav']:+.5f}  amp={stats['improvement']['amp']:+.5f}")
        print(f"\nSaved:")
        print(f"  {output_dir}/subspace_{args.run_id}_curves.png")
        print(f"  {output_dir}/subspace_{args.run_id}_step*.png")


if __name__ == "__main__":
    main()
