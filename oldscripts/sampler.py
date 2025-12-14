#sampler.py
import torch
from diffusion_utils import get_schedule, get_alpha_sigma

@torch.no_grad()
def sample_euler_v(model, steps=50, device='cuda'):
    model.eval()
    z = torch.randn(8, 3, 16, 16, device=device)
    ts = torch.linspace(1.0, 0.001, steps, device=device)
    
    for i in range(len(ts) - 1):
        t_cur = ts[i]
        logsnr = get_schedule(torch.full((8,), t_cur, device=device))
        
        raw, l_pred = model(z, logsnr)
        
        # Reconstruction Logic
        if model.mode == 'factorized':
            sigma_p = torch.sqrt(torch.sigmoid(-l_pred)).view(-1, 1, 1, 1)
            v_pred = raw * sigma_p
        else:
            v_pred = raw
            
        # Physics
        alpha, sigma = get_alpha_sigma(logsnr)
        alpha, sigma = alpha.view(-1,1,1,1), sigma.view(-1,1,1,1)
        
        x0_pred = alpha * z - sigma * v_pred
        eps_pred = sigma * z + alpha * v_pred
        
        # Step
        logsnr_next = get_schedule(torch.full((8,), ts[i+1], device=device))
        alpha_next, sigma_next = get_alpha_sigma(logsnr_next)
        alpha_next, sigma_next = alpha_next.view(-1,1,1,1), sigma_next.view(-1,1,1,1)
        
        z = alpha_next * x0_pred + sigma_next * eps_pred
        
    return z.clamp(0, 1).cpu()