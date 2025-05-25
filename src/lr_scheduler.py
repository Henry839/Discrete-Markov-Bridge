from torch.optim.lr_scheduler import LambdaLR
import math
def get_cosine_schedule_with_warmup(optimizer, 
                                    num_warmup_steps, 
                                    num_training_steps, 
                                    num_cycles=0.5):
    """Cosine Learning Rate.

    Args:
        optimizer
        num_warmup_steps
        num_training_steps: total training steps
        num_cycles

    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = (float(current_step - num_warmup_steps) / 
                    float(max(1, num_training_steps - num_warmup_steps)))

        if current_step < num_training_steps:
            return max(2e-5, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
        else:
            return 2e-5

    return LambdaLR(optimizer, 
                    lr_lambda)


def get_fix_schedule_with_warmup(optimizer,
                                 num_warmup_steps,
                                 num_training_steps,):
    """Fix Learning Rate.

    Args:
        optimizer
        num_warmup_steps
        num_training_steps: total training steps
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    return LambdaLR(optimizer, lr_lambda)



