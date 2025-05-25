'''
Sampling method from Lou et al
Correction comes from Zheng et al
'''
import torch
import torch.nn.functional as F


def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        # change to float64 according to the paper: 
        # masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs, 
                                               device=categorical_probs.device, 
                                               dtype=torch.float64) + 1e-10).log()
        sample = (categorical_probs / gumbel_norm).argmax(dim=-1)

        sample_probs = categorical_probs.gather(-1, sample.unsqueeze(-1)).squeeze(-1)
        # check if there is zero in the sample_probs
        if (sample_probs == 0).any():
            raise ValueError("Zero probability in the sample probs.")
        return sample
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")


def safe_log(x: torch.Tensor, eps: float = 1e-10):
    """Safe log.

    if element of x is smaller than 0, then take 0, otherwise take log, only for calculating entropy
    """
    mask = 1 - (x <= 0).float()
    return (x + eps).log() * mask


def safe_division(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-10):
    """Safe division.

    if denominator is 0, then take a large number, otherwise take division
    """
    return numerator / (denominator + eps)
