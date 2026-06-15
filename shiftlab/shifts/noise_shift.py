import torch

def apply_noise_shift(x, sigma: float):
    noise = torch.randn_like(x) * sigma
    return x + noise
