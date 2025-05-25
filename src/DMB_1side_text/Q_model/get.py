'''
Get methods for the Q matrix and its related matrices
'''

import torch
import torch.nn.functional as F
from torch.testing import assert_close
def get_row(self, idx: torch.Tensor):
    """Get row.

    Get the idx-th row of the Q matrix is equivalent to 
    using one hot vectors to times Q, such as:
        [0, 0, ..., 1, 0, ...] @ Q

    Args:
        idx [batch, seqlen]: the index of the row

    Returns:
        row [batch, seqlen, vocab]: the idx-th row of the Q matrix
    """
    device = idx.device
    # off load idx for one hot
    idx = idx.to("cpu")
    row = F.one_hot(idx, num_classes=self.vocab_size)
    # back to device
    row = row.to(device)

    batch = row.shape[0]

    # m_permute
    row = row.gather(-1, 
                     self.m_permute_inverse[None, ...].repeat(batch, 1, 1))

    # U
    # [batch, seqlen, vocab]
    row = row.cumsum(dim=-1)

    # times Lambda
    Lambda = self.get_lambda()
    # [batch, seqlen, vocab]
    row = row * Lambda[None, ...]

    # U_inverse
    # [batch, seqlen, vocab]
    row = torch.cat([row[:, :, 0][..., None], 
                     (row[:, :, 1:] - row[:, :, :-1])], 
                    dim=-1)

    # m_permute_inverse
    # [batch, seqlen, vocab]
    row = row.gather(-1, 
                     self.m_permute[None, ...].repeat(batch, 1, 1))
    return row



def get_col(self, idx: torch.Tensor):
    """Get col.

    Get the idx-th column of the Q matrix is equivalent to 
    using Q to times one hot vector, such as:
            Q @ [0, 0, ..., 1, 0, ...]^T

    Args:
        idx [batch, seqlen]: the index of the column, can be vector

    Returns:
        col [batch, seqlen, vocab]: the idx-th column of the Q matrix
    """
    device = idx.device
    # off load idx for one hot
    idx = idx.to("cpu")
    col = F.one_hot(idx, num_classes=self.vocab_size)
    # back to device
    col = col.to(device)


    batch = col.shape[0]

    # m_permute_inverse
    # [batch, seqlen, vocab]
    col = col.gather(-1, 
                     self.m_permute_inverse[None, ...].repeat(batch, 1, 1))

    # U_inverse
    # [batch, seqlen, vocab]
    col = torch.cat([(col[:, :, :-1] - col[:, :, 1:]), 
                     col[:, :, -1][..., None]], 
                    dim=-1)

    # Lambda
    # [batch, seqlen, vocab]
    Lambda = self.get_lambda()
    col = col * Lambda[None, ...]

    # U: equivalent to cumsum from the back
    # [batch, seqlen, vocab]
    col = torch.flip(torch.flip(col, 
                                dims=[-1]).cumsum(dim=-1), 
                     dims=[-1])

    # m_permute
    # [batch, seqlen, vocab]
    col = col.gather(-1, self.m_permute[None, ...].repeat(batch, 1, 1))
    return col



def get_exp_row(self, 
                idx: torch.Tensor, 
                h_sigma: torch.Tensor):
    """exp(Q)

    Get the i-th row of the exp(Q) matrix
        Q = m_permute @ U @ diag(Lambda) @ U_inverse @ m_permute_inverse
        exp(Q) = m_permute @ U @ diag(exp(Lambda)) @ U_inverse @ m_permute_inverse
        Q' = U @ diag(Lambda) @ U_inverse (henry matrix)
        exp(Q') = U @ diag(exp(Lambda)) @ U_inverse

    Args:
        idx [batch, seqlen]: the indexes of the row to grab
        h_sigma [batch]: int_0^t sigma(s) ds

    Returns:
        exp_Q_row [batch, seqlen, vocab]: the i-th row of the exp(Q) matrix
    """
    batch = idx.shape[0]

    device = idx.device
    # off load idx for one hot
    idx = idx.to("cpu")
    exp_row = F.one_hot(idx, num_classes=self.vocab_size)
    # back to device
    exp_row = exp_row.to(device)


    # m_permute
    # [batch, seqlen, vocab]
    exp_row = exp_row.gather(-1, 
                             self.m_permute_inverse[None, ...].repeat(batch, 1, 1))

    # U
    # [batch, seqlen, vocab]
    exp_row = exp_row.cumsum(dim=-1)*1.0

    # times Lambda
    # [seqlen, vocab]
    Lambda = self.get_lambda()
    # exp
    # [batch, seqlen, vocab]
    exp_row = exp_row * torch.exp(Lambda[None, ...] * h_sigma.view(-1,1,1))

    # U_inverse
    # [batch, seqlen, vocab]
    exp_row = torch.cat([exp_row[:, :, 0][..., None], 
                         F.relu(exp_row[:, :, 1:] - exp_row[:, :, :-1])], dim=-1)
   
    # m_permute_inverse
    # [batch, seqlen, vocab]
    exp_row = exp_row.gather(-1, 
                             self.m_permute[None, ...].repeat(batch, 1, 1))

    # row sum is 1
    assert_close(exp_row.sum(dim=-1), 
                 torch.ones_like(exp_row.sum(dim=-1)))
    return exp_row 



def get_lambda(self):
    """Diagonal.

    Get the lambda matrix (the diagonal matrix)

    Returns:
        Lambda [`torch.Tensor [seqlen, vocab]`] : the diagonal
    """
    seqlen = self.Lambda.shape[0]

    # the last entry of Lambda is 0
    # [seqlen, vocab]
    Lambda = torch.cat([self.Lambda, 
                          torch.zeros((seqlen, 1), device=self.Lambda.device)], dim=-1)

    # entries of Lambda are negative
    Lambda = self.negative_func(Lambda)
    
    # set to float64
    Lambda = Lambda.to(torch.float64)

    # calculate reverse cumulative sum
    Lambda = Lambda.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
    
    return Lambda
