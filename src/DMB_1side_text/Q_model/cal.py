"""P_T

P_T = P_0 @ exp(h_sigma_t * Q)
"""
import torch
import torch.nn.functional as F


def cal_p_T(self, 
            p_0: torch.Tensor, 
            h_sigma: torch.Tensor):
    """P_T.
    Return the estimated p_T
        p_T = p_0 @ exp(h_sigma_t * Q)
        p_T = p_0 @ m_permute @ exp(h_sigma_t * Q') @ m_permute_inverse
        p_T = p_0 @ m_permute @  U @ exp(h_sigma_t * diag(Lambda)) @ U_inverse @ m_permute_inverse

    U: 1, 1, 1, 1, 1
       0, 1, 1, 1, 1
       0, 0, 1, 1, 1
       0, 0, 0, 1, 1
       0, 0, 0, 0, 1
       
    U_inverse: 1, -1,  0,  0,  0
               0,  1, -1,  0,  0
               0,  0,  1, -1,  0
               0,  0,  0,  1, -1
               0,  0,  0,  0,  1

    Args:
        p_0 [seqlen, vocab]: p_0 ~ mu
        h_sigma [batch] or [] or [1]: int_0^t sigma(s) ds

    Returns:
        p_T [batch, seqlen, vocab]: probability at time t
    """
    h_sigma = h_sigma.view(-1)
    # can be 1 if p_0 is a vector
    batch = h_sigma.shape[0]

    # m_permute
    # [seqlen, vocab]
    p_T = p_0.gather(-1, self.m_permute_inverse)

    #  U
    # [seqlen, vocab]
    p_T = p_T.cumsum(dim=-1)

    # exp(h_sigma_t * diag(Lambda))
    # [batch, seqlen, vocab]
    Lambda = self.get_lambda()
    p_T = p_T[None,...] * torch.exp((Lambda[None, ...] 
                                     * h_sigma[..., None, None]))

    # U_inverse
    # [batch, seqlen, vocab]
    p_T = torch.cat([p_T[:, :, 0][..., None], 
                     F.relu(p_T[:, :, 1:] - p_T[:, :, :-1])], 
                    dim=-1)

    # m_permute_inverse
    # [batch, seqlen, vocab]
    p_T = p_T.gather(-1, self.m_permute[None, ...].repeat(batch, 1, 1))

    # [batch, seqlen, vocab]
    p_T = p_T.squeeze(0)

    # check is prob
    assert torch.allclose(p_T.sum(dim=-1), 
                          torch.ones_like(p_T.sum(dim=-1))), f"Mistake in p_T: {p_T.sum(dim=-1)}"
    assert torch.all(p_T >= 0)

    return p_T

