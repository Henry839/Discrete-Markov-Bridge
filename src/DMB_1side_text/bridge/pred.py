"""Sampling.

Sampling Method:
    1) Euler Method

Batch sampling:
    Start from p_T
    Predicting the p_0 using the score model
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .utils import sample_categorical, safe_log

def reverse_rate(self, 
                 score,
                 x_t, 
                 sigma_t,
                 local_rank=-1):
    """Reverse Rate.
    Calculate the row of the reverse rate, 
        i.e. \bar{Q}_t[x_{t}, x_{t-1}}] = p(x_{t-1})/p(x_t) Q(x_{t-1}, x_t)

    Notice that we suppose the sample time t is the same for all states in x_t, 
    therefore sigma_t is a scalar

    Args:
        score [batch, seqlen, vocab]
        x_t [batch, seqlen]: current state, 
        sigma_t: noise schedule, scalar or [batch, seqlen]
                BE CAREFUL that this is not the integral version

    Returns:
        Q_bar: [batch, seqlen, vocab], reverse rate
    """
    batch = x_t.shape[0]
    seqlen = x_t.shape[1]
    vocab = score.shape[-1]

    # unnormalize reverse rate matrix
    # [batch, seqlen, vocab]
    if local_rank != -1:
        Q_bar_unnorm = score * sigma_t[..., None, None] * self.Q_model.module.get_col(x_t)
    else:
        Q_bar_unnorm = score * sigma_t[..., None, None] * self.Q_model.get_col(x_t)

    # normalize Q_bar
    # set the diagonal to the negative sum of the row
    # [batch * seqlen, vocab]
    Q_bar = torch.scatter_add(Q_bar_unnorm,
                              -1,
                              x_t[..., None],
                              -1.0*Q_bar_unnorm.sum(-1, keepdim=True))

    # [batch, seqlen, vocab]
    Q_bar = Q_bar.view(batch, seqlen, vocab)
    return Q_bar



def euler_pred_step(self, 
                    score,
                    x_t,
                    sigma_t,
                    dt, 
                    local_rank=-1):
    """One Step Euler
    
    Euler method for reverse process
        A single step

    Notice that the sample time t is the same for all states in x_t, 
    therefore sigma_t, h_sigma_t is a scalar

    Args:
        score [`torch.Tensor [batch, seqlen, vocab]`]
        x_t [`torch.Tensor [batch, seqlen]`]: current state 
        sigma_t: noise schedule, scalar
        step_num: number of steps

    Returns:
        x_t_dt: x_{t-dt}, [batch, seqlen]
        p_t_dt: p_{t-dt}(.|x_t), [batch, seqlen, vocab] conditional distribution
    """
    # reverse rate matrix
    # [batch, seqlen, vocab]
    Q_bar = self.reverse_rate(score, x_t, sigma_t, local_rank=local_rank)

    # transition probability 
    # p(x_t-dt|x_t) = kronecker_delta(x_t, x_t-dt) + dt * Q_bar
    # [batch, seqlen, vocab]
    p_t_dt = F.one_hot(x_t.to(torch.int64), 
                       num_classes=self.vocab_size).to(Q_bar) + dt * Q_bar

    # [batch, seqlen]
    x_t_dt = sample_categorical(p_t_dt)

    return p_t_dt, x_t_dt



def euler_pred(self, 
               sample_batch_size,
               prob_T, 
               step_num,
               local_rank=-1):
    """Euler Sampling

    Euler method for reverse process
        Start from p_T

    Args:
        prob_T [`torch.Tensor [seqlen, vocab size]`]: the prior distribution, which is known
        sample_batch_size: number of samples, multi sampling process for a better prediction
        step_num: number of steps

    Returns:
        p_0 [`torch.Tensor [seqlen, vocab size]`]: estimation of data distribution 
        x_0 [`torch.Tensor [batch, seqlen]`]: samples from p_0
    """
    eps = self.eps
    self.Q_model.eval()
    self.score_model.eval()

    with torch.no_grad():

        prob_T = prob_T[None, ...].repeat(sample_batch_size, 1, 1)
        x_t = sample_categorical(prob_T)

        dt = (1 - eps) / step_num

        for step in tqdm(range(step_num, 0, -1)):
            # current time
            # [batch]
            t = torch.tensor(step * dt).to("cuda").view(1).repeat(sample_batch_size)

            # get sigma (noise scheduler)
            # [batch]
            sigma_t = self.scheduler.get_sigma(t).to("cuda").view((sample_batch_size))
            h_sigma_t = self.scheduler.get_integral(t).to("cuda").view((sample_batch_size)) 

            # [batch, seqlen, vocab]
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logscore = self.score_model(data=x_t, h_sigma=h_sigma_t)

            score = logscore.exp()

            # one step sampling
            # [batch, seqlen, vocab], [batch, seqlen]
            p_t, x_t = self.euler_pred_step(score, 
                                            x_t, 
                                            sigma_t, 
                                            dt, 
                                            local_rank=local_rank)

        # batch mean of probability
        # [seqlen, vocab]
        p_0 = p_t.mean(0)

        # normalize
        p_0 = F.relu(p_0)
        p_0 = p_0 / p_0.sum(-1, keepdim=True)

    return p_0, x_t


