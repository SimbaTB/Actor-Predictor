import torch
import numpy as np
import gymnasium as gym
import random

def get_params_num(model):
    with torch.no_grad():
        return sum(p.numel() for p in model.parameters())

def get_grad_norm(model):
	with torch.no_grad():
		grads = [p.grad for p in model.parameters() if p.grad is not None]
		if len(grads) == 0:
			return 0.0
		total_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in grads]), 2)
		return total_norm.item()

@torch.compile
def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))

@torch.compile
def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

def is_image(space):
    return len(space.shape)==3

def is_discrete(space):
    if isinstance(space, gym.spaces.Discrete):
        return True
    elif isinstance(space, gym.spaces.Box):
        return False
    else:
        raise NotImplementedError("Only support Box and Discrete space")

def grad_scale(x, scale=0.1):
    return scale * x + (1 - scale) * x.detach()

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False