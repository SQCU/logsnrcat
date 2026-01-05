#!/usr/bin/env python3
"""
PPO-style REINFORCE trainer with ribbon plot visualization.

Submits entire training loop to eval server - all computation happens server-side.
"""

import argparse
import datetime
from pathlib import Path
import requests

DEFAULT_HOST = "172.26.160.1"
DEFAULT_PORT = 8421
DEFAULT_OUTPUT = "experiments_swiglu_ae/main_run_097"


def eval_code(code: str, host: str, port: int, timeout: int = 600) -> dict:
    """Execute Python code on eval server."""
    url = f"http://{host}:{port}/eval"
    resp = requests.post(url, json={"code": code}, timeout=timeout)
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="PPO-style REINFORCE with ribbon plots")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--steps", type=int, default=250, help="Training steps")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--run-id", type=str, default=None, help="Run identifier for output files")
    args = parser.parse_args()

    if args.run_id is None:
        args.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(args.output_dir)

    print(f"Connecting to eval server at http://{args.host}:{args.port}...")
    health = requests.get(f"http://{args.host}:{args.port}/health", timeout=10).json()
    print(f"  Status: {health['status']}, Weights: {health['weights_loaded']}")

    # Submit ENTIRE training loop as one code block - all computation server-side
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

class LoRACodeAdapter(nn.Module):
    def __init__(self, dim, rank=8, scale=1.0):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, dim) * (1.0 / dim ** 0.5))
        self.B = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.scale = scale
    def forward(self, x):
        return self.scale * F.linear(F.linear(x, self.A), self.B)

lora_enc = LoRACodeAdapter(code_dim, rank={args.lora_rank}).to(device).to(torch.bfloat16)
lora_dec = LoRACodeAdapter(code_dim, rank={args.lora_rank}).to(device).to(torch.bfloat16)
optimizer = torch.optim.Adam(
    list(lora_enc.parameters()) + list(lora_dec.parameters()),
    lr={args.lr}
)

history = {{"step": [], "reward": [], "mse": [], "js_div": [], "loss": [], "hist_loss": []}}
reward_baseline = None

def compute_js(imgs, recons, n_bins=32):
    imgs_np = imgs.float().cpu().numpy()
    recons_np = recons.float().cpu().numpy()
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

def soft_hist_loss(img, ref, n_bins=32, sigma=0.05):
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

def sample_colors(img_np, n=64):
    pixels = img_np.reshape(-1, 3)
    idx = np.random.choice(len(pixels), size=min(n, len(pixels)), replace=False)
    colors = pixels[idx]
    lum = 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]
    return colors[np.argsort(lum)]

def make_ribbon(colors, h=8, w=16):
    n = len(colors)
    ribbon = np.zeros((h, w, 3))
    for i, c in enumerate(colors):
        x0, x1 = int(i * w / n), int((i + 1) * w / n)
        ribbon[:, x0:x1] = c
    return ribbon.clip(0, 1)

run_id = "{args.run_id}"

def save_ribbon(step, images, recon_base, recon_policy, js_val, mse_base, mse_policy):
    orig_np = images.float().cpu().numpy().transpose(0, 2, 3, 1)
    base_np = recon_base.float().cpu().numpy().transpose(0, 2, 3, 1)
    policy_np = recon_policy.float().cpu().numpy().transpose(0, 2, 3, 1)

    n = min(4, len(images))
    fig, axes = plt.subplots(n, 6, figsize=(12, 6), gridspec_kw={{"width_ratios": [4, 1, 4, 1, 4, 1]}})
    for i in range(n):
        axes[i, 0].imshow(orig_np[i].clip(0, 1)); axes[i, 0].axis("off")
        axes[i, 1].imshow(make_ribbon(sample_colors(orig_np[i].clip(0, 1)))); axes[i, 1].axis("off")
        axes[i, 2].imshow(base_np[i].clip(0, 1)); axes[i, 2].axis("off")
        axes[i, 3].imshow(make_ribbon(sample_colors(base_np[i].clip(0, 1)))); axes[i, 3].axis("off")
        axes[i, 4].imshow(policy_np[i].clip(0, 1)); axes[i, 4].axis("off")
        axes[i, 5].imshow(make_ribbon(sample_colors(policy_np[i].clip(0, 1)))); axes[i, 5].axis("off")
        if i == 0:
            axes[i, 0].set_title("Original", fontsize=9)
            axes[i, 2].set_title(f"Baseline (MSE={{mse_base:.4f}})", fontsize=8)
            axes[i, 4].set_title(f"Policy (MSE={{mse_policy:.4f}})", fontsize=8)
    plt.suptitle(f"Step {{step}}: JS={{js_val:.4f}} | MSE diff={{abs(mse_policy-mse_base):.6f}}", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/ppo_ribbon_{{run_id}}_step{{step:03d}}.png", dpi=150, bbox_inches="tight")
    plt.close()

# === Precompute masks once ===
images = batch
p = ae.patch_size
grid_shape = (images.shape[2] // p, images.shape[3] // p)
encoder_masks, decoder_masks = ae.build_masks(grid_shape, images.device)

# === Sanity check: verify adapter changes output ===
with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    test_codes = ae.encode(images, grid_shape=grid_shape, encoder_masks=encoder_masks, decoder_masks=decoder_masks)
    recon_no_adapter = ae.decode(test_codes, grid_shape, decoder_masks)
    test_adapted = [c + lora_enc(c) + lora_dec(c + lora_enc(c)) for c in test_codes]
    recon_with_adapter = ae.decode(test_adapted, grid_shape, decoder_masks)
    sanity_mse = F.mse_loss(recon_no_adapter.float(), recon_with_adapter.float()).item()
print(f"Sanity check: adapter MSE diff = {{sanity_mse:.8f}} (should be > 0)")
if sanity_mse < 1e-10:
    raise ValueError("FATAL: Adapter produces no change in output!")

# === Training loop (all server-side) ===
n_steps = {args.steps}
save_interval = {args.save_interval}

for step in range(n_steps):
    # Encode (no grad through AE)
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        codes_list = ae.encode(images, grid_shape=grid_shape,
                               encoder_masks=encoder_masks, decoder_masks=decoder_masks)
        codes_list = [c.detach() for c in codes_list]

        # Baseline reconstruction (no adapters)
        if step % save_interval == 0:
            recon_base = ae.decode(codes_list, grid_shape, decoder_masks)

    # Apply adapters (these have gradients)
    codes_adapted = []
    for codes in codes_list:
        codes_bf16 = codes.to(torch.bfloat16)
        adapted = codes_bf16 + lora_enc(codes_bf16)
        adapted = adapted + lora_dec(adapted)
        codes_adapted.append(adapted)

    # Decode
    with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
        recon = ae.decode(codes_adapted, grid_shape, decoder_masks).float()

    # Compute metrics
    with torch.no_grad():
        js_div = compute_js(images, recon)
        mse = F.mse_loss(recon, images.float()).item()

    reward = -js_div
    if reward_baseline is None:
        reward_baseline = reward
    else:
        reward_baseline = 0.9 * reward_baseline + 0.1 * reward
    advantage = reward - reward_baseline

    # Optimize with advantage weighting and L2 regularization
    optimizer.zero_grad()
    hist_loss = soft_hist_loss(recon, images.float())
    mse_loss = F.mse_loss(recon, images.float())

    # L2 regularization on adapter weights
    l2_reg = sum((p ** 2).sum() for p in lora_enc.parameters())
    l2_reg += sum((p ** 2).sum() for p in lora_dec.parameters())

    # Advantage-weighted: emphasize histogram when doing better
    hist_weight = 1.0 + max(0.0, advantage * 5.0)
    loss = mse_loss + hist_weight * hist_loss + 0.01 * l2_reg
    loss.backward()
    optimizer.step()

    # Log
    history["step"].append(step)
    history["reward"].append(reward)
    history["mse"].append(mse)
    history["js_div"].append(js_div)
    history["loss"].append(loss.item())
    history["hist_loss"].append(hist_loss.item())

    # Print progress
    if step % 50 == 0:
        print(f"Step {{step}}: reward={{reward:.4f}}, js={{js_div:.4f}}, mse={{mse:.4f}}")

    # Save ribbon plot with MSE verification
    if step % save_interval == 0:
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            recon_policy = ae.decode(codes_adapted, grid_shape, decoder_masks)
            mse_base = F.mse_loss(recon_base.float(), images.float()).item()
            mse_policy = F.mse_loss(recon_policy.float(), images.float()).item()
            mse_diff = F.mse_loss(recon_base.float(), recon_policy.float()).item()
        print(f"  [Step {{step}}] MSE base={{mse_base:.6f}}, policy={{mse_policy:.6f}}, diff={{mse_diff:.6f}}")
        save_ribbon(step, images, recon_base, recon_policy, js_div, mse_base, mse_policy)

# === Final ribbon ===
with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    codes_list = ae.encode(images, grid_shape=grid_shape,
                           encoder_masks=encoder_masks, decoder_masks=decoder_masks)
    recon_base = ae.decode(codes_list, grid_shape, decoder_masks)
codes_adapted = []
for codes in codes_list:
    codes_bf16 = codes.to(torch.bfloat16)
    adapted = codes_bf16 + lora_enc(codes_bf16)
    adapted = adapted + lora_dec(adapted)
    codes_adapted.append(adapted)
with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float32):
    recon_policy = ae.decode(codes_adapted, grid_shape, decoder_masks)
    final_mse_base = F.mse_loss(recon_base.float(), images.float()).item()
    final_mse_policy = F.mse_loss(recon_policy.float(), images.float()).item()
    final_mse_diff = F.mse_loss(recon_base.float(), recon_policy.float()).item()
print(f"Final: MSE base={{final_mse_base:.6f}}, policy={{final_mse_policy:.6f}}, diff={{final_mse_diff:.6f}}")
save_ribbon(n_steps, images, recon_base, recon_policy, history["js_div"][-1], final_mse_base, final_mse_policy)

# === Training curves ===
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes[0, 0].plot(history["step"], history["reward"], "b-", lw=0.8)
axes[0, 0].axhline(y=history["reward"][0], color="r", ls="--", alpha=0.5)
axes[0, 0].set_title("Reward (-JS Div)")
axes[0, 1].plot(history["step"], history["js_div"], "r-", lw=0.8)
axes[0, 1].set_title("JS Divergence")
axes[0, 2].plot(history["step"], history["hist_loss"], "g-", lw=0.8)
axes[0, 2].set_title("Soft Histogram Loss")
axes[1, 0].plot(history["step"], history["mse"], "purple", lw=0.8)
axes[1, 0].set_title("MSE")
axes[1, 1].plot(history["step"], history["loss"], "orange", lw=0.8)
axes[1, 1].set_title("Total Loss")
improvement = [r - history["reward"][0] for r in history["reward"]]
axes[1, 2].plot(history["step"], improvement, "b-", lw=0.8)
axes[1, 2].fill_between(history["step"], 0, improvement,
                        where=[x > 0 for x in improvement], alpha=0.3, color="green")
axes[1, 2].fill_between(history["step"], 0, improvement,
                        where=[x <= 0 for x in improvement], alpha=0.3, color="red")
axes[1, 2].axhline(y=0, color="gray", ls="--")
axes[1, 2].set_title("Reward vs Start")
best_js = min(history["js_div"])
plt.suptitle(f"PPO {{n_steps}}-step ({{run_id}}) - Final JS: {{history['js_div'][-1]:.4f}}, Best: {{best_js:.4f}}", fontsize=12)
plt.tight_layout()
plt.savefig(f"{args.output_dir}/ppo_{{run_id}}_{{n_steps}}_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# Store results for retrieval
ctx._ppo_result = {{
    "start_js": history["js_div"][0],
    "end_js": history["js_div"][-1],
    "best_js": best_js,
    "improvement": history["reward"][-1] - history["reward"][0],
    "n_steps": n_steps
}}
ctx._lora_enc = lora_enc
ctx._lora_dec = lora_dec
ctx._rl_history = history
'''

    print(f"\nSubmitting {args.steps}-step training loop to server...")
    print("(All computation happens server-side - no per-step network calls)")

    result = eval_code(training_code, args.host, args.port, timeout=1200)

    if not result['success']:
        print(f"ERROR: {result['error']}")
        return

    # Fetch results
    result = eval_code("ctx._ppo_result", args.host, args.port)
    if result['success']:
        stats = result['result']
        print(f"\nResults (run_id={args.run_id}):")
        print(f"  Start JS:    {stats['start_js']:.4f}")
        print(f"  End JS:      {stats['end_js']:.4f}")
        print(f"  Best JS:     {stats['best_js']:.4f}")
        print(f"  Improvement: {stats['improvement']:+.4f}")
        print(f"\nSaved:")
        print(f"  {output_dir}/ppo_{args.run_id}_{args.steps}_curves.png")
        print(f"  {output_dir}/ppo_ribbon_{args.run_id}_step*.png")


if __name__ == "__main__":
    main()
