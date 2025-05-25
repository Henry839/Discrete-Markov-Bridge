import torch
from operator import xor
def grab_mu_phi_ratio(self, 
                      mu_data: torch.Tensor = None, 
                      phi_data: torch.Tensor = None,
                      mu: torch.Tensor = None, 
                      phi: torch.Tensor = None):
    """phi/mu.

    Return the ratio :phi/mu
    four modes: phi_data/mu_data, phi_data/mu, phi/mu_data, phi/mu

    Args:
        mu_data [batch, seqlen]: (p_0 ~ mu) batch of data from distribution mu
        phi_data [batch, seqlen]: (p_T ~ phi) batch of data from distribution phi
        mu [seqlen, vocab]: discrete probability of each elements
        phi [seqlen, vocab]: discrete probability of each elements

    Returns:
        ratio [`torch.Tensor [seqlen, vocab]`]: phi/mu
    """

    epsilon = 1e-15
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

    ratio = (phi_dist + epsilon) / (mu_dist + epsilon)
    return ratio


def stat_m_permute(self, 
                   mu_data: torch.Tensor = None, 
                   phi_data: torch.Tensor = None, 
                   mu: torch.Tensor = None, 
                   phi: torch.Tensor = None):
    """Permutation Matrix.

    Return the permutation matrix which satisfies that the 
    ratios of the entries of phi vs mu (i.e. phi/mu) 
    after permute are in ascending order 

    Args:
        mu_data [batch, seqlen]: (p_0 ~ mu) batch of data from distribution mu
        phi_data [batch, seqlen]: (p_T ~ phi) batch of data from distribution phi
        mu [seqlen, vocab]: discrete probability of each elements
        phi [seqlen, vocab]: discrete probability of each elements

    Returns:
        m_permute [seqlen, vocab]: permutation matrix
        m_permute_inverse [seqlen, vocab]
    """

    ratio = self.grab_mu_phi_ratio(
            mu_data=mu_data, 
            phi_data=phi_data, 
            mu=mu, 
            phi=phi)

    # sort the ratio in ascending order
    # [seqlen, vocab]
    m_permute_inverse = torch.argsort(ratio, descending=False, dim=-1)
    m_permute_inverse.requires_grad = False
    # [seqlen, vocab]
    m_permute = torch.argsort(m_permute_inverse, descending=False, dim=-1)
    m_permute.requires_grad = False

    return m_permute, m_permute_inverse



def stat_count(self, data: torch.Tensor):
    """Bincount.
    Return the bincount of the data

    Args:
        data [batch, seqlen]: batch of data

    Returns:
        dist [seqlen, vocab]: histogram
    """

    # [seqlen, batch]
    data = data.type(torch.int64).T
    seqlen = data.shape[0]

    # batched bincount
    # [seqlen, vocab]
    dist = torch.zeros(seqlen, self.vocab_size, dtype=data.dtype).scatter_add_(1, data, torch.ones_like(data))
    
    assert dist.shape == torch.Size([seqlen, self.vocab_size]), f"Expected shape {torch.Size([self.n])} but got {dist.shape}"
    return dist
