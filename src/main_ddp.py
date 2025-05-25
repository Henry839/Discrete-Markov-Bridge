import os
import pathlib
import torch
import torch.distributed as dist
import numpy as np
import random
from transformers import GPT2TokenizerFast
from torch.nn.parallel import DistributedDataParallel as DDP
from DMB_1side_text import (DiscreteBridge, Q_model, ScoreModel, 
                       sigma_scheduler, get_dataloaders, 
                       ExponentialMovingAverage)
import wandb
from tqdm import tqdm
from lr_scheduler import (get_cosine_schedule_with_warmup, 
                                get_fix_schedule_with_warmup)
from wd_scheduler import WeightDecayScheduler
from parser import get_parser



def main():
    ######################
    #   ArgumentParser   #
    ######################
    parser = get_parser()
    args = parser.parse_args()
    path = f"{args.path}/{args.run_name}"
    
    # reproducibility
#    torch.manual_seed(args.random_seed)
#    torch.cuda.manual_seed_all(args.random_seed)
#    torch.backends.cudnn.deterministic = True
#    torch.backends.cudnn.benchmark = False
#    np.random.seed(args.random_seed)
#    random.seed(args.random_seed)
#    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    wandb_id = args.run_name

    # DDP initialization
    if args.local_rank != -1:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create log directory
    if args.local_rank in [-1, 0]:
        if not os.path.exists(f"{path}"):
            os.makedirs(path)
            os.makedirs(f"{path}/model")
            os.makedirs(f"{path}/distribution")

    tokenizer = None
    # vocab size
    vocab_size = args.vocab_size
    print(f"Vocab size: {vocab_size}")

    ##############################
    #        mu (reverse)        #
    ##############################
    #    estimate by prob_0      #
    ##############################
    if args.local_rank in [-1, 0]:
        print((f"<Train> {args.mu_train_dataset_name} "
               f"<Eval> {args.mu_eval_dataset_name} "
               f"<Test> {args.mu_test_dataset_name}"))
    # training dataset
    train_loader, tokenized_train_dataset = get_dataloaders(
            name=args.mu_train_dataset_name,
            tokenizer=tokenizer,
            batch_size=args.score_train_batch_size,
            accumulation_step=args.score_accum,
            cache_dir=args.cache_dir,
            seqlen=args.seqlen,
            shuffle=True,
            ngpus=args.ngpus,
            mode="train",
            distributed=(args.local_rank != -1),
            debug=args.debug)

    # eval dataset
    eval_loader, _ = get_dataloaders(
            name=args.mu_eval_dataset_name,
            tokenizer=tokenizer,
            batch_size=args.score_eval_batch_size,
            accumulation_step=None,
            cache_dir=args.cache_dir,
            seqlen=args.seqlen,
            shuffle=False,
            ngpus=args.ngpus,
            mode="validation",
            distributed=(args.local_rank != -1),
            debug=args.debug)

    # test dataset
    test_loader, _ = get_dataloaders(
            name=args.mu_test_dataset_name,
            tokenizer=tokenizer,
            batch_size=args.score_eval_batch_size,
            accumulation_step=None,
            cache_dir=args.cache_dir,
            seqlen=args.seqlen,
            shuffle=False,
            ngpus=args.ngpus,
            mode="test",
            distributed=(args.local_rank != -1),
            debug=args.debug)


    ##############################
    #       phi (forward)        #
    ##############################
    #    estimate by prob_T      #
    ##############################
    # training dataset
    # use uniform
    phi = 1.0 * torch.tensor([1/vocab_size 
                              for _ in range(vocab_size)]).view(1,-1).repeat(args.seqlen,1)
    phi = phi/phi.sum(-1, keepdim=True)
    phi = phi.to(device)


    ##############################
    #      Discrete Bridge       #
    ##############################
    # noise scheduler
    scheduler = sigma_scheduler(name=args.sche_name,
                                sigma_min=args.sigma_min,
                                sigma_max=args.sigma_max,
                                local_rank=args.local_rank,)

    # rand initialization of Q matrix
    Q = Q_model(vocab_size=vocab_size,
                seqlen=args.seqlen,
                mu_data=train_loader.dataset.data.squeeze(1),
                phi=phi,
                h_sigma_T=scheduler.get_integral(
                    torch.tensor([1], device=device)
                    ),
                initialization=args.Q_initialization,
                negative_func=args.Q_negative_func,)

    if args.local_rank == 0:
        print(f"Initialization Method: {args.Q_initialization}")
        print(f"Q_model lambda: {Q.get_lambda()[0]}")
    Q.to(device)

    # score model
    score = ScoreModel(vocab_size=vocab_size,
                       hidden_size=args.hidden_size,
                       time_hidden_size=args.time_hidden_size,
                       n_heads=args.n_heads,
                       dropout=args.dropout,
                       n_blocks=args.n_blocks,
                       scale_by_sigma=args.score_scale_by_sigma,)
    score.to(device)

    # Wrap models with DDP
    if args.local_rank != -1:
        Q = DDP(Q, device_ids=[args.local_rank], output_device=args.local_rank)
        score = DDP(score, device_ids=[args.local_rank], output_device=args.local_rank)

    # Trick
    ema = ExponentialMovingAverage(score.parameters(),
                                   decay=args.ema,)


    # optimizers
    Q_optimizer = torch.optim.AdamW(Q.parameters(),
                                    lr=args.Q_lr,
                                    betas=(0.9, 0.999),
                                    weight_decay=args.Q_weight_decay)

    score_optimizer = torch.optim.AdamW(score.parameters(),
                                        lr=args.score_lr,
                                        betas=(0.9, 0.999),
                                        weight_decay=args.score_weight_decay,
                                        )
    scaler = torch.cuda.amp.GradScaler()

    if args.score_lr_scheduler == "cosine":
        score_lr_scheduler = get_cosine_schedule_with_warmup(
                score_optimizer, 
                args.score_warmup_steps, 
                num_training_steps=args.score_lr_num_train_step)
    elif args.score_lr_scheduler == "fix":
        score_lr_scheduler = get_fix_schedule_with_warmup(
                score_optimizer, 
                args.score_warmup_steps, 
                num_training_steps=args.score_lr_num_train_step)
    else:
        raise NotImplementedError

    if args.score_wd_scheduler:
        score_wd_scheduler = WeightDecayScheduler(
                init_wd=args.score_weight_decay, 
                decay_method=args.score_wd_scheduler_decay_method)
    else:
        score_wd_scheduler = None


    if args.Q_lr_scheduler == "cosine":
        Q_lr_scheduler = get_cosine_schedule_with_warmup(
                Q_optimizer, 
                args.Q_warmup_steps, 
                num_training_steps=args.Q_lr_num_train_step)
    elif args.Q_lr_scheduler == "fix":
        Q_lr_scheduler = get_fix_schedule_with_warmup(
                Q_optimizer, 
                args.Q_warmup_steps, 
                num_training_steps=args.Q_lr_num_train_step)
    else:
        Q_lr_scheduler = None

    # random initialization of p_0
    # estimation of mu
    prob_0 = torch.rand(vocab_size).view(1,-1).repeat(args.seqlen,1)
    prob_0 = prob_0/prob_0.sum(-1, keepdim=True)
    prob_0 = prob_0.to(device)
    prob_0.requires_grad = False
    print("prob_0: ", prob_0)

    # estimation of phi
    prob_T = phi
    prob_T = prob_T.to(device)
    prob_T.requires_grad = False
    print("prob_T: ", prob_T)

    start = 0
    if args.resume:
        checkpoint_list = list(pathlib.Path(f"{path}/model/").glob("checkpoint-*.pt"))
        print(checkpoint_list)
        if len(checkpoint_list) == 1:
            ckpt = torch.load(f"{str(checkpoint_list[0])}", map_location=device)
            Q.load_state_dict(ckpt['Q_model'])
            score.load_state_dict(ckpt['score_model'])
            Q_optimizer.load_state_dict(ckpt['Q_optimizer'])
            score_optimizer.load_state_dict(ckpt['score_optimizer'])
            score_lr_scheduler.load_state_dict(ckpt['score_lr_scheduler'])
            if args.score_wd_scheduler:
                score_wd_scheduler.load_state_dict(ckpt['score_wd_scheduler'])
                score_wd_scheduler.set(score_optimizer)
            if "scaler" in ckpt.keys():
                scaler.load_state_dict(ckpt['scaler'])
            start = ckpt['epoch'] + 1
            prob_0 = ckpt['prob_0']
            prob_T = ckpt['prob_T']
            # if exists
            Q_lr_scheduler.load_state_dict(ckpt['Q_lr_scheduler'])
            wandb_id = ckpt['wandb_id']
            vocab_size = ckpt['vocab_size']
            ema.load_state_dict(ckpt['ema'])
            print(f"resume from {str(checkpoint_list[0])}")
        else:
            raise ValueError

    # wandb
    if args.local_rank in [-1, 0]:  # Only initialize wandb in the main process
        if args.debug:
            wandb.init(project="text", 
                       name=f"{args.run_name}",
                       config=args)
        else:
            wandb.init(project="text", 
                       name=f"{args.run_name}",
                       config=args,
                       resume="allow",
                       id=wandb_id)

    ##############################
    #      bridge process        #
    ##############################
    diffBridge = DiscreteBridge(
            Q_model=Q,
            Q_optimizer=Q_optimizer,
            score_model=score,
            score_optimizer=score_optimizer,
            score_lr_scheduler=score_lr_scheduler,
            score_wd_scheduler=score_wd_scheduler,
            Q_lr_scheduler=Q_lr_scheduler, 
            scaler=scaler,
            scheduler=scheduler,
            vocab_size=vocab_size,
            eps=args.eps)


    if not args.resume:
        # Evaluate in advanced
        ####### Evaluate Score Model #######
        # evaluation, calculate per token loss
        diffBridge.evaluate(
                eval_dataLoader=eval_loader,
                mode="valid",
                prob_T=prob_T,
                local_rank=args.local_rank)

        ####### Test Score Model #######
        # test, calculate per token loss
        diffBridge.evaluate(
                eval_dataLoader=test_loader,
                mode="test",
                prob_T=prob_T,
                local_rank=args.local_rank)

    min_valid_loss = 1e5
    min_valid_epoch = 0
    min_test_loss = 1e5
    min_test_epoch = 0
    for i in tqdm(range(start, args.epoch)):

        if args.reset_optimizer:
            Q_optimizer = torch.optim.AdamW(
                    Q.parameters(),
                    lr=args.Q_lr,
                    betas=(0.9, 0.999),
                    weight_decay=args.Q_weight_decay
                    )
            score_optimizer = torch.optim.AdamW(
                    score.parameters(),
                    lr=args.score_lr,
                    betas=(0.9, 0.999),
                    weight_decay=args.score_weight_decay,
                    )
            if args.score_lr_scheduler == "cosine":
                score_lr_scheduler = get_cosine_schedule_with_warmup(
                        diffBridge.score_optimizer, 
                        0,
                        num_training_steps=args.score_lr_num_train_step)
            elif args.score_lr_scheduler == "fix":
                score_lr_scheduler = get_fix_schedule_with_warmup(
                        diffBridge.score_optimizer, 
                        0,
                        num_training_steps=args.score_lr_num_train_step)
            else:
                raise NotImplementedError

            # reset in diffBridge api
            diffBridge.Q_optimizer = Q_optimizer
            diffBridge.score_optimizer = score_optimizer
            if args.score_wd_scheduler:
                diffBridge.score_wd_scheduler.set(score_optimizer)
            diffBridge.score_lr_scheduler = score_lr_scheduler

        ######## Forward Process ########
        diffBridge.train_Q(
                prob_0=prob_0, 
                train_epoch=args.Q_epochs,
                accumulation_step=args.Q_accum,
                grad_clip=args.Q_grad_clip,
                train_dataLoader=train_loader,
                local_rank=args.local_rank)

        # update prob T
        prob_T = diffBridge.infer_Q(prob_0=prob_0,
                                    local_rank=args.local_rank,)
        world_size = dist.get_world_size()
        gathered_prob_T = [torch.zeros_like(prob_T) for _ in range(world_size)]
        dist.all_gather(gathered_prob_T, prob_T)
        new_prob_T = torch.stack(gathered_prob_T, dim=0)
        new_prob_T = torch.mean(new_prob_T, dim=0)
        prob_T.copy_(new_prob_T)

        if args.local_rank in [-1, 0]:
            print(f"prob_T: {prob_T}")

        ######## Reverse Process ########
        diffBridge.train_score(
                train_dataLoader=train_loader,
                accumulation_step=args.score_accum,
                ema=ema,
                train_epoch=args.score_epoch,
                grad_clip=args.score_grad_clip,
                local_rank=args.local_rank)

        # store training parameter and grab ema parameter for evaluation
        ema.store(score.parameters())
        ema.copy_to(score.parameters())

        ####### Estimate p_0 #######
        # estimation of p_0 (corresponds to mu)
        # [seqlen, vocab], [batch, seqlen]
        prob_0, x_0  = diffBridge.euler_pred(
                sample_batch_size=args.sample_batch_size,
                prob_T=prob_T,
                step_num=args.step_num,
                local_rank=args.local_rank)

        ####### Evaluate Score Model #######
        valid_loss = diffBridge.evaluate(
                eval_dataLoader=eval_loader,
                prob_T=prob_T,
                local_rank=args.local_rank,
                mode="valid")

        tot_test_loss = 0
        for _ in tqdm(range(args.eval_times)):
            test_loss = diffBridge.evaluate(
                    eval_dataLoader=test_loader,
                    prob_T=prob_T,
                    local_rank=args.local_rank,
                    mode="test")
            tot_test_loss += test_loss

        tot_test_loss /= args.eval_times 

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            min_valid_epoch = i
        if tot_test_loss < min_test_loss:
            min_test_loss = tot_test_loss
            min_test_epoch = i

        if args.local_rank in [-1, 0]:
            print(f"Epoch: {i}")
            print((f"   Valid Loss: {valid_loss},"
                   f"   Test loss: {tot_test_loss}"))
            print(f"Min Valid Epoch: {min_valid_epoch}")
            print(f"   Valid Loss: {min_valid_loss}")
            print(f"Min Test Epoch: {min_test_epoch}")
            print(f"   Test Loss: {min_test_loss}")

            wandb.log({"Valid loss": valid_loss, "Test loss": tot_test_loss,})
        # restore ema trick
        ema.restore(score.parameters())
        
        # gather prob_0: seqlen, vocab
        world_size = dist.get_world_size()
        gathered_prob_0 = [torch.zeros_like(prob_0) for _ in range(world_size)]
        dist.all_gather(gathered_prob_0, prob_0)
        new_prob_0 = torch.stack(gathered_prob_0, dim=0)
        new_prob_0 = torch.mean(new_prob_0, dim=0)
        prob_0.copy_(new_prob_0)
        if args.local_rank in [-1, 0]:
            print(f"prob_0: {prob_0}")

       
        # tokenizer decode x_0
        if args.mu_train_dataset_name == "text8":
            example_text = tokenized_train_dataset.tensor2text(x_0.unsqueeze(1))[0]
        else:
            example_text = tokenizer.batch_decode(x_0)
            
        if args.local_rank in [-1, 0]:
            print(f"ids example: {x_0[0]}")
            print(f"Generated data from mu: {example_text}")

        if args.score_wd_scheduler:
            # save model
            torch.save({'epoch': i,
                        'Q_model': Q.state_dict(),
                        'score_model': score.state_dict(),
                        'Q_optimizer': Q_optimizer.state_dict(),
                        'score_optimizer': score_optimizer.state_dict(),
                        'score_lr_scheduler': score_lr_scheduler.state_dict(),
                        'score_wd_scheduler': score_wd_scheduler.state_dict(),
                        'Q_lr_scheduler': Q_lr_scheduler.state_dict() if Q_lr_scheduler is not None else None,
                        'args': args,
                        'vocab_size': vocab_size,
                        'ema': ema.state_dict(),
                        'prob_0': prob_0,
                        'prob_T': prob_T,
                        'wandb_id': wandb_id,
                        'scaler': scaler.state_dict(),
                        },
                    f"{path}/model/checkpoint-{i}.pt")
        else:
            torch.save({'epoch': i,
                        'Q_model': Q.state_dict(),
                        'score_model': score.state_dict(),
                        'Q_optimizer': Q_optimizer.state_dict(),
                        'score_optimizer': score_optimizer.state_dict(),
                        'score_lr_scheduler': score_lr_scheduler.state_dict(),
                        'Q_lr_scheduler': Q_lr_scheduler.state_dict() if Q_lr_scheduler is not None else None,
                        'args': args,
                        'vocab_size': vocab_size,
                        'ema': ema.state_dict(),
                        'prob_0': prob_0,
                        'prob_T': prob_T,
                        'wandb_id': wandb_id,
                        'scaler': scaler.state_dict(),
                        },
                    f"{path}/model/checkpoint-{i}.pt")

        dist.barrier()
    # Clean up the process group
    if args.local_rank != -1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
