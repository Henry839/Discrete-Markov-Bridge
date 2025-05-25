"""Q Learning.

Minimize KL(P_{T|0}||P_T)

"""
import torch
import wandb
import torch.nn.functional as F
from ..utils import safe_log, safe_division

#eps = 1e-20

def train_Q(self,
            prob_0,
            train_epoch, 
            accumulation_step,
            grad_clip,
            train_dataLoader=None,
            local_rank=-1):
    """Q training.

    Minimize KL(P_{T|0}||P_T)

    Args:
        prob_0 [`torch.Tensor [seqlen, vocab]`]: estimation of mu
        train_epoch: number of epochs for training
        accumulation_step: gradient accumulation
        grad_clip: gradient clipping
        train_dataLoader: data from mu when using KL loss

    Returns:
    - pred_prob_T
    """

    eps = self.eps
    self.Q_model.train()
    self.score_model.eval()  # freeze the score model

    optimizer = self.Q_optimizer
    optimizer.zero_grad()


    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if local_rank in [-1, 0]:
        print("=============================")
        print("|          Q model          |")
        print("=============================")

    total_loss = 0
    for epoch in range(train_epoch):
        epoch_loss = 0
        for step, raw_batch in enumerate(train_dataLoader):
            # [batch, seqlen]
            if isinstance(raw_batch, dict):
                batch = raw_batch['input_ids'].to(device)
            else:
                batch = raw_batch.to(device)

            batch_size = batch.shape[0]
            seqlen = batch.shape[1]

            # [seqlen, vocab]
            # int_0^T sigma(t) dt, scalar
            h_sigma_T = self.scheduler.get_integral(torch.tensor([1],device=device))

            if local_rank != -1:
                pred_prob_T = self.Q_model.module.cal_p_T(p_0=prob_0, 
                                                          h_sigma=h_sigma_T)
            else:
                pred_prob_T = self.Q_model.cal_p_T(p_0=prob_0, 
                                                       h_sigma=h_sigma_T)

            # [batch, 1]
            time = torch.ones(
                    (batch_size, 1), device=device
                    )

            # [batch]
            h_sigma_T = self.scheduler.get_integral(time).view((batch_size))

            # p_{T|0}(.|x_0)
            # [batch, seqlen, vocab size]
            # p(y|x_0): the x_0-th row of Q_t
            if local_rank != -1:
                cond_prob = self.Q_model.module.get_exp_row(idx=batch, 
                                                            h_sigma=h_sigma_T)
            else:
                cond_prob = self.Q_model.get_exp_row(idx=batch, 
                                                     h_sigma=h_sigma_T)

            cond_prob = (cond_prob + eps)/(cond_prob + eps).sum(-1, keepdim=True)
            pred_prob_T = (pred_prob_T + eps)/(pred_prob_T + eps).sum(-1, keepdim=True)

            loss = cond_prob*((cond_prob/pred_prob_T).log())
            loss = loss.sum()/batch_size
            assert loss >= 0, f"loss {loss} is smaller than zero"

            loss.backward()

            # gradient accumulation & gradient clip
            if ((step+1) % accumulation_step == 0 
                or (step+1) == len(train_dataLoader)):
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.Q_model.parameters(), grad_clip)
                optimizer.step()
                if self.Q_lr_scheduler is not None:
                    self.Q_lr_scheduler.step()
                optimizer.zero_grad()
 
            total_loss += loss.item()/len(train_dataLoader)
            epoch_loss += loss.item()/len(train_dataLoader)

        if local_rank in [-1, 0]:
            print(f"-- Epoch {epoch}: {epoch_loss},")
            wandb.log({"Q epoch Loss": epoch_loss})

    if local_rank in [-1, 0]:
        print(f"Q model total loss: {total_loss/(train_epoch)}")
        wandb.log({"Q Loss": total_loss/(train_epoch)})

    # double check
    optimizer.zero_grad()
    self.score_optimizer.zero_grad()
    del batch, h_sigma_T, time, cond_prob, loss
    torch.cuda.empty_cache()
