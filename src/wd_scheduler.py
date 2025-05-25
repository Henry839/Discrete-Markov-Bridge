"""
Weight decay scheduler
"""

class WeightDecayScheduler:
    def __init__(self, init_wd, step=0, threshold=8e5, frequency=1e5, decay_method="smooth"):
        self.wd = init_wd
        self.num_step = step
        self.threshold = threshold
        self.frequency = frequency
        self.decay_method = decay_method
        self.ema_loss = None 
        self.sharp_time = 0


    def smooth_step(self, optimizer):
        self.num_step += 1

        if (self.num_step + 1) % self.frequency == 0 and self.num_step < self.threshold:
            self.wd /= 10
            optimizer.param_groups[0]['weight_decay'] = self.wd

        elif self.num_step >= self.threshold and self.wd != 0:
            self.wd = 0
            optimizer.param_groups[0]['weight_decay'] = self.wd
        return 
    

    def sharp_step(self, optimizer, loss):
        """Sharp decay.

        Calculate the ema loss and check the variance of the loss, if the variance is small
        then decay the weight decay sharply. 

        Args:
            optimizer: torch.optim.Optimizer
            loss: float

        """
        if loss is None:
            raise ValueError("Loss is required for sharp decay")

        self.num_step += 1

        if self.ema_loss is None:
            ema_loss = loss
        else:
            ema_loss = 0.99 * self.ema_loss + 0.01 * loss
        if self.ema_loss is not None:
            variance = abs(ema_loss - self.ema_loss)
            if variance < 0.05:
                # culmulate
                self.sharp_time += 1
            else:
                # reset
                self.sharp_time = 0

            if self.sharp_time >= 10:
                self.wd = 0
                optimizer.param_groups[0]['weight_decay'] = self.wd
        self.ema_loss = ema_loss


    def step(self, optimizer, loss=None):
        if self.decay_method == "smooth":
            self.smooth_step(optimizer)
        elif self.decay_method == "sharp":
            self.sharp_step(optimizer, loss)
        else:
            raise ValueError("Invalid decay method")
        return


    def set(self, optimizer):
        optimizer.param_groups[0]['weight_decay'] = self.wd
        return


    def state_dict(self,):
        return {"num_step": self.num_step, "wd": self.wd, 
                "threshold": self.threshold, "frequency": self.frequency, 
                "decay_method": self.decay_method, "ema_loss": self.ema_loss, 
                "sharp_time": self.sharp_time}

    
    def load_state_dict(self, state_dict):
        self.num_step = state_dict["num_step"]
        self.wd = state_dict["wd"]
        self.threshold = state_dict["threshold"]
        self.frequency = state_dict["frequency"]
        if "decay_method" in state_dict:
            self.decay_method = state_dict["decay_method"]
        if "ema_loss" in state_dict:
            self.ema_loss = state_dict["ema_loss"]
        if "sharp_time" in state_dict:
            self.sharp_time = state_dict["sharp_time"]
        return





