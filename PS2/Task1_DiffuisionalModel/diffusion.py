import torch    # for tensor operations
T = 1000         # Total number of diffusion steps

betas = torch.linspace(1e-4, 0.02, T)      # Linear schedule for beta values
alphas = 1 - betas                         # Compute alphas from betas
alpha_bar = torch.cumprod(alphas, dim=0)   # Cumulative product of alphas

sqrt_alpha_bar = torch.sqrt(alpha_bar)
sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)


def forward_diffusion(x0, t):         # Forward diffusion process to add noise
    noise = torch.randn_like(x0)      # Generate random Gaussian noise

    sqrt_ab = sqrt_alpha_bar[t].view(-1, 1, 1, 1)
    sqrt_1mab = sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)

    xt = sqrt_ab * x0 + sqrt_1mab * noise
    return xt, noise