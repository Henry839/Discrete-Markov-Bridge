"""Loss Func.
Training loss files for the score model

PS: as the Q_model will not be updated, we will detach all of its outputs
"""
import torch.nn.functional as F
import torch
from ..utils import safe_log



def score_entropy_loss(self, 
                       x_t, 
                       x_0,
                       h_sigma_t, 
                       sigma_t, 
                       logscore, 
                       local_rank=-1):
    """Score Entropy.

    Calculate the score entropy loss
        please refer to the paper for the formula

    Args:
        x_t [batch, seq len]: sampled noised data 
        x_0 [batch, seq len]: clean data 
        h_sigma_t [batch]: int_0^t sigma(s)ds 
        sigma_t [batch]: sigma(t) 
        logscore [batch, seq len, vocab size]: log score row s_{\theta}(x_t, t,y)

    Returns:
        loss: score entropy loss
    
    PS: As the Q_model will not be updated, we will detach all of its outputs
    """

    """
    Calculate weight
    """
    with torch.no_grad():
        # get x_t-th column Q_t(y, x_t) = sigma_t * Q
        # get the x_t-th column of Q
        # [batch, seqlen, vocab size]
        if local_rank != -1:
            weight = self.Q_model.module.get_col(x_t).detach()
        else:
            weight = self.Q_model.get_col(x_t).detach()

        # summation is on y != x_t
        # set the x_t-th column of weight to 0,
        # therefore when calculating the score entropy,
        # 0 * (...) = 0
        # [batch, seqlen, vocab size]
        weight = weight.scatter(dim=-1,
                                index=x_t[..., None],
                                src=torch.zeros_like(weight))

        # times sigma_t
        # [batch, seqlen, vocab size]
        weight = weight * sigma_t[..., None, None]

    """
    Calculate those in bracket
    """
    # ratio
    # [batch, seqlen, vocab size]
    # p(y|x_0): the x_0-th row of Q_t
    with torch.no_grad():
        if local_rank != -1:
            cond_prob = self.Q_model.module.get_exp_row(idx=x_0, 
                                                h_sigma=h_sigma_t).detach()
        else:
            cond_prob = self.Q_model.get_exp_row(idx=x_0, 
                                                h_sigma=h_sigma_t).detach()
        # p(y|x_0)/p(x_t|x_0)
        # [batch, seqlen, vocab_size]
        eps = self.eps
        ratio = cond_prob/(cond_prob.gather(dim=-1, index=x_t[..., None]) + eps) + eps
#        ratio = cond_prob/(cond_prob.gather(dim=-1, index=x_t[..., None]))


        del cond_prob, x_0, x_t
        torch.cuda.empty_cache()

    """
    Combine all terms
    """
    # [batch, seqlen]
    loss = (weight * (logscore.exp() - ratio * logscore + ratio * (ratio.log() - 1))).sum((-1, -2))
#    loss = (weight * (logscore.exp() - ratio * logscore + ratio * (safe_log(ratio) - 1))).sum((-1, -2))

    # batch mean
    loss = loss.mean()

    return loss


