"""Evaluation.
Evaluate the score model: 
    Calculate two things: 
        1) the score entropy loss
        2) the KL divergence term (Lou et al)
"""
import torch
import torch.nn.functional as F
import torch.distributed as dist
from .utils import safe_log, safe_division

def evaluate(self, 
             eval_dataLoader, 
             prob_T, 
             local_rank=-1, 
             mode="valid"):
    """Evaluation

    Evaluation once: Score entropy + KL(P_{T|0} || P_T)

    Args:
        eval_dataLoader
        prob_T [`torch.Tensor [seqlen, vocab size]`]: 
                    the prior distribution, the estimation of phi, 
                    which is the starting point of our diffusion process
        local_rank: -1 for not ddp
        mode: valid, test
    """
    batch_size = eval_dataLoader.batch_size
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    eps = self.eps

    
    self.Q_model.eval()
    self.score_model.eval()

    total_loss = 0
    with torch.no_grad():
        for _, raw_batch in enumerate(eval_dataLoader):
            # [batch, seqlen]
            if isinstance(raw_batch, dict):
                batch = raw_batch['input_ids'].to(device)
            else:
                batch = raw_batch.to(device)

            batch_size = batch.shape[0]

            # randomly sample time 
            # [batch, 1]
            time = (1 - eps) * torch.rand((batch_size, 1), device=device) + eps

            # [batch]
            h_sigma_t = self.scheduler.get_integral(time).view((batch_size))
            sigma_t = self.scheduler.get_sigma(time).view((batch_size))

            # noise added data
            # [batch, seqlen]
            perturbed_batch = self.add_noise(batch, h_sigma_t, local_rank=local_rank)

            
            ########################################################
            # Score Entropy (stems from Andrew et al and Lou at al)#
            ########################################################
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logscore = self.score_model(data=perturbed_batch, 
                                            h_sigma=h_sigma_t)
            loss = self.score_entropy_loss(x_t=perturbed_batch, 
                                           x_0=batch, 
                                           h_sigma_t=h_sigma_t,
                                           sigma_t=sigma_t, 
                                           logscore=logscore,
                                           local_rank=local_rank,)


            #######################################################
            #                    KL Divergence                    #
            #######################################################
            # calculate the Expected KL divergence
            # between cond prob and the starting point
            # of the diffusion process
            # [batch, 1]
            time = torch.ones((batch_size, 1), device=device)

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

#            prob_T = (prob_T + 1e-10)/(prob_T + 1e-10).sum(-1, keepdim=True)
            cond_prob = (cond_prob + eps)/(cond_prob + eps).sum(-1, keepdim=True)
            prob_T = (prob_T + eps)/(prob_T + eps).sum(-1, keepdim=True)
            # KL divergence
            kl_loss = F.kl_div(target=cond_prob.log(), 
                               input=prob_T.log(),
                               reduction='none',
                               log_target=True,)
            kl_loss = kl_loss.sum()/batch_size
            total_loss += loss + kl_loss

        # gather total loss
        world_size = dist.get_world_size()
        gathered_total_loss = [torch.zeros_like(total_loss) for _ in range(world_size)]
        dist.all_gather(gathered_total_loss, total_loss)
        new_total_loss = torch.stack(gathered_total_loss, dim=0)
        new_total_loss = torch.mean(new_total_loss, dim=0)
        total_loss = new_total_loss.item()

        total_loss = total_loss/len(eval_dataLoader)

        if local_rank in [-1, 0]:
            print(f"{mode} loss: {total_loss}")
    return total_loss
