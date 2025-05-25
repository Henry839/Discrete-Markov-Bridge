"""Training.

For Score Model
score matching denoising
"""
import torch
import wandb

def train_score(self, 
                train_dataLoader, 
                accumulation_step,
                ema, 
                train_epoch, 
                grad_clip, 
                local_rank=-1,):
    """Training Score.

    Training score entropy

    Args:
        train_dataLoader: the true data for learning, data are the observations from the target distribution
        accumulation_step: gradient accumulation step
        ema: exponential moving average
        grad_clip: gradient clipping
        train epoch
    """
    batch_size = train_dataLoader.batch_size
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    eps = self.eps
    
    self.Q_model.eval()     # freeze the Q model
    self.Q_model.requires_grad = False
    self.score_model.train()

    optimizer = self.score_optimizer
    optimizer.zero_grad()
    scaler = self.scaler
    self.Q_optimizer.zero_grad()

    if local_rank in [-1, 0]:
        print("=============================")
        print("|      Score model          |")
        print("=============================")

    epoch_total_loss = 0
    for epoch in range(train_epoch):
        total_loss = 0
        for step, raw_batch in enumerate(train_dataLoader):
            with torch.no_grad():
                # [batch, seqlen]
                if isinstance(raw_batch, dict):
                    batch = raw_batch['input_ids'].to(device)
                else:
                    batch = raw_batch.to(device)

                batch_size = batch.shape[0]
                seqlen = batch.shape[1]

                # randomly sample time 
                # [batch, 1]
                time = (1 - eps) * torch.rand((batch_size, 1), device=device) + eps

                # [batch]
                h_sigma_t = self.scheduler.get_integral(time).view((batch_size))
                sigma_t = self.scheduler.get_sigma(time).view((batch_size))

                # noise added data
                # [batch, seqlen]
                perturbed_batch = self.add_noise(batch, h_sigma_t, local_rank=local_rank)

            # Calculate Loss
            # score entropy (stems from Andrew et al and Lou at al)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logscore = self.score_model(data=perturbed_batch, 
                                            h_sigma=h_sigma_t)
            loss = self.score_entropy_loss(x_t=perturbed_batch, 
                                           x_0=batch, 
                                           h_sigma_t=h_sigma_t,
                                           sigma_t=sigma_t, 
                                           logscore=logscore,
                                           local_rank=local_rank,) / accumulation_step
            scaler.scale(loss).backward()
            total_loss += loss.item()
            epoch_total_loss += loss.item()
 
            # grad accumulation & grad clip
            if (step+1) % accumulation_step == 0 or (step+1) == len(train_dataLoader):
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.score_model.parameters(), grad_clip)
                self.score_lr_scheduler.step()
                if self.score_wd_scheduler is not None:
                    self.score_wd_scheduler.step(optimizer, loss.detach())
                scaler.step(optimizer)
                scaler.update()
                ema.update(self.score_model.parameters())
                optimizer.zero_grad()
                self.Q_optimizer.zero_grad()
                if local_rank in [-1, 0]:
                    wandb.log({"Score step Loss": loss.item(), 
                               "LR": optimizer.param_groups[0]['lr'], 
                               "WD": optimizer.param_groups[0]['weight_decay']})
            if (step + 1) % 50 == 0 and local_rank in [-1, 0]:
                print((f"-- Step: {step+1} "
                       f"Loss: {loss.item()} "
                       f"LR: {optimizer.param_groups[0]['lr']}"
                       f"WD: {optimizer.param_groups[0]['weight_decay']}"
                      ))

        if local_rank in [-1, 0]:
            print(f"Epoch: {epoch} "
                  f"Score Loss: {total_loss/len(train_dataLoader)}"
                  f"LR: {optimizer.param_groups[0]['lr']}"
                  f"WD: {optimizer.param_groups[0]['weight_decay']}"
                  )
            wandb.log({"Score Epoch Loss": total_loss/(len(train_dataLoader))})

    # double check
    optimizer.zero_grad()
    self.Q_optimizer.zero_grad()
    self.Q_model.requires_grad = True

    return 
