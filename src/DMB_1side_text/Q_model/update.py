'''
Update functions:
    - update lambda
    - update m_permute
'''
import torch
eps = 1e-15

def update_Lambda(self, 
                  h_sigma_T: torch.Tensor,
                  mu_data: torch.Tensor = None, 
                  phi_data: torch.Tensor = None, 
                  mu: torch.Tensor = None, 
                  phi: torch.Tensor = None,):
    """Analytic Update.

    Leverage permutation for the update of Lambda.
    Calculate according to the analytical formula.

    Args:
        mu_data [batch, seqlen]
        phi_data [batch, seqlen]
        mu [seqlen, vocab]
        phi [seqlen, vocab]
        h_sigma_T [1, 1]: int_0^T h_sigma(t) dt

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get mu distribution
    # [seqlen, vocab_size]
    if mu is None and mu_data is not None:
        mu_count = self.stat_count(mu_data)
        mu_dist = mu_count/ mu_count.sum()
    elif mu is not None and mu_data is None:
        mu_dist = mu
    else:
        raise ValueError

    # [seqlen, vocab_size]
    if phi is None and phi_data is not None:
        phi_count = self.stat_count(phi_data)
        phi_dist = phi_count/ phi_count.sum()
    elif phi is not None and phi_data is None:
        phi_dist = phi
    else:
        raise ValueError

    mu_dist = mu_dist.to(device)
    phi_dist = phi_dist.to(device)
    
    # permute: change vocabulary order
    # [seqlen, vocab_size]
    mu_permute = mu_dist.gather(-1, self.m_permute_inverse)
    # [seqlen, vocab_size]
    phi_permute = phi_dist.gather(-1, self.m_permute_inverse)

    # [seqlen, vocab_size]
    mu_cumsum = mu_permute.cumsum(dim=-1)
    phi_cumsum = phi_permute.cumsum(dim=-1)
    
    # [seqlen, vocab_size]
    ratio = (phi_cumsum) / (mu_cumsum + eps)

    log_ratio = torch.log(ratio + eps).to(device)
    Lambda = -(log_ratio[:, 1:] - log_ratio[:,:-1])/(h_sigma_T + eps)

    return Lambda



def update_permute(self, 
                   mu: torch.Tensor = None, 
                   phi: torch.Tensor = None):
    """Update permutation matrix.

    Update the permutation matrix according to data/prob

    Args:
        mu [seqlen, vocab]
        phi [seqlen, vocab]
    """
    self.m_permute, self.m_permute_inverse = self.stat_m_permute(mu=mu, 
                                                                 phi=phi)
