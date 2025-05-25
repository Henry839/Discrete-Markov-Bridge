import torch
def infer_Q(self,
            prob_0,
            local_rank=-1):
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    # inference the estimated p_T
    self.Q_model.eval()
    with torch.no_grad():
        h_sigma_T = self.scheduler.get_integral(torch.tensor([1],device=device))
        if local_rank != -1:
            pred_prob_T = self.Q_model.module.cal_p_T(
                    p_0=prob_0,
                    h_sigma=h_sigma_T,)
        else:    
            pred_prob_T = self.Q_model.cal_p_T(
                    p_0=prob_0,
                    h_sigma=h_sigma_T,)
    return pred_prob_T

