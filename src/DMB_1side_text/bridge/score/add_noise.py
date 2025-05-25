"""Noise adding.

Add noise to clean data
"""
from ..utils import sample_categorical

def add_noise(self, 
              x_0, 
              h_sigma,
              local_rank=-1):
    """Nosing adding.

    Add noise to the data, i.e. sample from 
                p(x_t|x_0) = exp(Q_t)[x_0,:] = exp(h_sigma * (Q  + weigth/n*J))[x_0,:]

    Args:
        x_0: clean data , [batch, seqlen]
        h_sigma: int_0^t sigma(s)ds, [batch, 1]

    Returns:
        perturbed data, [batch, seqlen]
    """
    h_sigma = h_sigma.view(-1)

    # [batch, seqlen, vocab]
    if local_rank != -1:
        cond_prob = self.Q_model.module.get_exp_row(x_0,
                                                    h_sigma)
    else:
        cond_prob = self.Q_model.get_exp_row(x_0,
                                             h_sigma)

    # sample per batch
    #perturbed_data = torch.multinomial(F.relu(cond_prob), 1, replacement=True)

    # [batch, seqlen]
    perturbed_data = sample_categorical(cond_prob)

    # [batch, seqlen]
    perturbed_data = perturbed_data.view(x_0.shape[0], x_0.shape[1])
    return perturbed_data 


