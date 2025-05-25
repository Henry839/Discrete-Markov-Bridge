import torch
class sigma_scheduler():
    def __init__(self,
                 name,
                 sigma_min,
                 sigma_max,
                 local_rank=-1):
        '''
        Args:
        - name: type of scheduler

        - Geometric noise:
            - sigma_min: min sigma, use for geometric noise
            - sigma_max: max sigma, use for geometric noise

        '''
        self.eps = 1e-3
        self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])

        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        # copy from SEDD by Chenlin Meng
        if name == "loglinear":
            self.scheduler =  lambda t: (1 - self.eps) / (1 - (1 - self.eps) * t)
            self.integral = lambda t: -torch.log1p(-(1 - self.eps)*t)
        elif name == "geometric":
            # geometric noise
            self.scheduler = lambda t: self.sigmas[0]**(1 - t) * self.sigmas[1]**t * (self.sigmas[1].log() - self.sigmas[0].log())
            # int_0^t sigma(s) ds = sigma_min^(1-t) * sigma_max^t
            self.integral = lambda t: self.sigmas[0]**(1 - t) * self.sigmas[1]**t
        elif name == "linear":
            self.scheduler = lambda t: self.sigmas[1] * t
            self.integral = lambda t: 0.5 * self.sigmas[1] * t**2
        else:
            raise ValueError("Unknown scheduler")

    from .get import get_sigma, get_integral









