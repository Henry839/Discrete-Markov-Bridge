'''
Bridge main process
'''
import torch

class DiscreteBridge():
    """Bridge API.

    bridge process: including the training of Q and Score
    
    A two stage training process:
        1. KL(P_{T|0} || P_T)
        2. s_theta = p(i)/p(j)
    """

    def __init__(self, 
                 Q_model, 
                 Q_optimizer,
                 score_model,
                 score_optimizer,
                 score_lr_scheduler,
                 score_wd_scheduler,
                 Q_lr_scheduler,
                 scaler,
                 scheduler,
                 vocab_size,
                 eps,
                 ) -> None:
        """Initialization.

        Args:
            Q_model
            Q_optimizer
            score_model: for calculating the p(i,t)/p(j,t)
                           p(i,t): probability of staying in i at time t
            score_lr_scheduler: learning rate scheduler (for warmup)
            Q_lr_scheduler: learning rate scheduler (for warmup)
            score_wd_scheduler: weight decay scheduler
            scaler: grad scaler
            scheduler: the noise scheduler
            vocab_size: the size of the finite discrete set
            eps: the epsilon
        """
        self.Q_model = Q_model
        self.Q_optimizer = Q_optimizer 
        self.score_model = score_model
        self.score_optimizer = score_optimizer
        self.score_lr_scheduler = score_lr_scheduler
        self.score_wd_scheduler = score_wd_scheduler
        self.Q_lr_scheduler = Q_lr_scheduler
        self.scaler = scaler
        self.scheduler = scheduler 
        self.vocab_size = vocab_size
        self.eps = eps

    from .Q import train_Q, infer_Q
    from .evaluate import evaluate
    from .score import (add_noise, train_score, score_entropy_loss,)
    from .pred import reverse_rate, euler_pred_step, euler_pred



