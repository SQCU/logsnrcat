import torch
import math

def get_schedule(t):
    return 20.0 - 40.0 * t

def get_alpha_sigma(logsnr):
    alpha = torch.sqrt(torch.sigmoid(logsnr))
    sigma = torch.sqrt(torch.sigmoid(-logsnr))
    return alpha, sigma

class FourierFeatures(torch.nn.Module):
    def __init__(self, num_bands=4, max_range=40.0):
        super().__init__()
        base_freq = 2 * math.pi / max_range
        freqs = base_freq * (2.0 ** torch.arange(num_bands))
        self.register_buffer('freqs', freqs)

    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(-1)
        args = x * self.freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class BucketManager:
    """
    Manages alternating between different resolution buckets.
    """
    def __init__(self, buckets):
        # buckets: list of (resolution, batch_size) tuples
        # e.g. [(16, 1024), (32, 256)]
        self.buckets = buckets
        self.current_idx = 0
        
    def next_bucket(self):
        bucket = self.buckets[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.buckets)
        return bucket


def get_schedule(t, schedule_bounds: tuple = (5, -4)):
    """Linear LogSNR schedule."""
    return schedule_bounds[0] - t * (schedule_bounds[1] - schedule_bounds[0])

def logsnr_to_alpha_sigma(logsnr):
    """
    Returns alpha, sigma for a given logsnr.
    Handles broadcasting if logsnr is [B, 1, H, W].
    """
    # Ensure numerical stability
    sigmoid_lsnr = torch.sigmoid(logsnr)
    sigmoid_neg_lsnr = torch.sigmoid(-logsnr)
    alpha = torch.sqrt(sigmoid_lsnr)
    sigma = torch.sqrt(sigmoid_neg_lsnr)
    return alpha, sigma

def euler_forward_step(x0, logsnr, noise=None):
    """
    Diffuses x0 -> z_t. Returns z_t and the target velocity v_true.
    """
    if noise is None:
        noise = torch.randn_like(x0)
    
    alpha, sigma = logsnr_to_alpha_sigma(logsnr)
    
    # Broadcast check: logsnr might be [B, 1, H, W] or [B]
    if alpha.ndim == 1:
        alpha = alpha.view(-1, 1, 1, 1)
        sigma = sigma.view(-1, 1, 1, 1)
        
    z_t = x0 * alpha + noise * sigma
    v_true = alpha * noise - sigma * x0
    return z_t, v_true, noise

def euler_reverse_step(z_t, v_pred, logsnr_from, logsnr_to):
    """
    Denoises z_t -> z_{t-1}.
    """
    alpha_from, sigma_from = logsnr_to_alpha_sigma(logsnr_from)
    alpha_to, sigma_to = logsnr_to_alpha_sigma(logsnr_to)
    
    if alpha_from.ndim == 1:
        alpha_from = alpha_from.view(-1, 1, 1, 1)
        sigma_from = sigma_from.view(-1, 1, 1, 1)
        alpha_to = alpha_to.view(-1, 1, 1, 1)
        sigma_to = sigma_to.view(-1, 1, 1, 1)

    # Reconstruct x0 (prediction)
    x0_pred = alpha_from * z_t - sigma_from * v_pred
    # Reconstruct eps (prediction)
    eps_pred = sigma_from * z_t + alpha_from * v_pred
    
    # Step to next level
    z_next = alpha_to * x0_pred + sigma_to * eps_pred
    return z_next

def get_image_spans(resolution):
    latent_res = resolution // 2
    length = latent_res * latent_res
    return [{'type': 'latent', 'len': length, 'shape': (latent_res, latent_res), 'causal': False}]