"""
Q model for the Q matrix
"""


import torch
import torch.nn.functional as F
from datasets import Dataset

class Q_model(torch.nn.Module):

    def __init__(self,
                 vocab_size,
                 seqlen,
                 m_permute=None,
                 m_permute_inverse=None,
                 mu_data=None,
                 phi=None,
                 h_sigma_T=None,
                 initialization="analytic",
                 negative_func="relu") -> None:
        """Initialization.

        Q' = U Lambda U^{-1}
            where U is upper triangular matrix with all ones
                U = 1 1 1 1
                    0 1 1 1
                    0 0 1 1
                    0 0 0 1
            and Lambda is a diagonal matrix with learnable parameters which are negative
            and U^{-1} is:
                U^{-1} = 1 -1  0  0
                         0  1 -1  0
                         0  0  1 -1
                         0  0  0  1
        Q = m_permute Q' m_permute^{-1}

        Args:
            vocab_size [`int`]: number of types of the data
            seqlen [`int`]: length of each sequence
            m_permute [`torch.Tensor [seqlen, vocab_size]` or `None`]: permutation matrix
            m_permute_inverse [`torch.Tensor [seqlen, vocab_size]` or `None`]: inverse of permutation matrix
            mu_data [`torch.Tensor [batch, seqlen]` or `None`]: mu data (use for generating m_permute)
            phi [`torch.Tensor [seqlen, vocab_size]` or `None`]: phi
            h_sigma_T [`torch.Tensor [1]` or `None`]: int_0^T h_sigma(t) dt
            initialization [`str`]: initialization method
            negative_func [`str`]: ensure that the diagonal matrix is negative
        """
        super(Q_model, self).__init__()

        self.vocab_size = vocab_size
        self.seqlen = seqlen

        # ReLU and softmax
        self.relu = torch.nn.ReLU()

        # diagonal matrix
        # vocab_size-1 for the reason that the last element is 0
        self.Lambda = torch.nn.Parameter(-torch.rand(seqlen, vocab_size-1))

        # Permutation matrix
        if m_permute is None or m_permute_inverse is None:
            if isinstance(mu_data, Dataset):
                mu_data = torch.stack([mu_data[i]['input_ids'] for i in range(len(mu_data))])
            m_permute, m_permute_inverse = self.stat_m_permute(
                    mu_data=mu_data, 
                    phi=phi)
        
        # check shape of m_permute and m_permute_inverse
        assert m_permute.shape == torch.Size([seqlen, vocab_size])
        assert m_permute_inverse.shape == torch.Size([seqlen, vocab_size])

        self.register_buffer('m_permute', m_permute)
        self.register_buffer('m_permute_inverse', m_permute_inverse)

        # initialize Lambda with analytical solution
        if initialization == "analytic":
            self.Lambda.data = self.update_Lambda(mu_data=mu_data, 
                                                  phi=phi, 
                                                  h_sigma_T=h_sigma_T)
        elif initialization == "gather":
            # all mass init gather to the last place
            self.Lambda.data = torch.zeros(seqlen, vocab_size-1)
            self.Lambda.data[:, -1] = 1
        elif initialization == "random":
            self.Lambda.data = torch.rand(seqlen, vocab_size-1)
        elif initialization == "zeros":
            self.Lambda.data = torch.zeros(seqlen, vocab_size-1)
        elif initialization == "smallunif":
            self.Lambda.data = torch.zeros(seqlen, vocab_size-1) + 1e-1
        elif initialization == "ones":
            self.Lambda.data = torch.ones(seqlen, vocab_size-1)
        else:
            raise ValueError(f"Invalid initialization method: {initialization}")

        if negative_func == "relu":
            self.negative_func = lambda x: -F.relu(x)
            self.Lambda.data += 1e-10
        elif negative_func == "square":
            self.negative_func = lambda x: -1 * (x**2)
        else:
            raise ValueError(f"Invalid negative_method: {negative_func}")

    
    from .cal import cal_p_T
    from .get import get_row, get_col, get_exp_row, get_lambda
    from .m_permute import stat_m_permute, stat_count, grab_mu_phi_ratio
    from .update import update_Lambda, update_permute
