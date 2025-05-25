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
from tqdm import tqdm
from score_lr_scheduler import (get_cosine_schedule_with_warmup, 
                                get_fix_schedule_with_warmup)
from parser import get_parser



def main():
    ######################
    #   ArgumentParser   #
    ######################
    parser = get_parser()
    args = parser.parse_args()
    path = f"logs/{args.run_name}"
    
    # reproducibility
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)


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
    if args.mu_train_dataset_name != "text8":
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
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
                                T=args.T,
                                local_rank=args.local_rank,)

    # rand initialization of Q matrix
    Q = Q_model(vocab_size=vocab_size,
                seqlen=args.seqlen,
                mu_data=train_loader.dataset.data.squeeze(1),
                phi=phi,
                h_sigma_T=scheduler.get_integral(torch.tensor([args.T],device=device)),
                initialization=args.Q_initialization)
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
                                   decay=args.ema,
                                   use_num_updates=(not args.ema_not_use_num_update))


    # optimizers
    Q_optimizer = torch.optim.AdamW(Q.parameters(),
                                   lr=args.Q_lr,
                                   weight_decay=args.Q_weight_decay)

    score_optimizer = torch.optim.AdamW(score.parameters(),
                                       lr=args.score_lr,
                                       betas=(0.9, 0.999),)

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

    # random initialization of p_0
    # estimation of mu
    prob_0 = torch.rand(vocab_size).view(1,-1).repeat(args.seqlen,1)
    prob_0 = prob_0/prob_0.sum(-1, keepdim=True)
    prob_0 = prob_0.to(device)
    prob_0.requires_grad = False
    print("prob_0: ", prob_0)

    # random initialization of p_T
    # estimation of phi
    prob_T = phi
    if args.p_T_rand_init:
        prob_T = torch.rand(vocab_size).view(1,-1).repeat(args.seqlen,1)
        prob_T = prob_T/prob_T.sum(-1, keepdim=True)
    prob_T = prob_T.to(device)
    prob_T.requires_grad = False
    print("prob_T: ", prob_T)

    start = 0
    gap = 1e5
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
            start = ckpt['epoch'] + 1
            prob_0 = ckpt['prob_0']
            prob_T = ckpt['prob_T']
            # if exists
            if 'old_prob_T' in ckpt:
                old_prob_T = ckpt['old_prob_T']
                gap = ckpt['gap']
            vocab_size = ckpt['vocab_size']
            ema.load_state_dict(ckpt['ema'])
            print(f"resume from {str(checkpoint_list[0])}")
        else:
            raise ValueError


    ##############################
    #      bridge process        #
    ##############################
    diffBridge = DiscreteBridge(
            Q_model=Q,
            Q_optimizer=Q_optimizer,
            score_model=score,
            score_optimizer=score_optimizer,
            score_lr_scheduler=score_lr_scheduler,
            scheduler=scheduler,
            vocab_size=vocab_size,
            T=args.T,
            Q_add_p0_pT=args.Q_add_p0_pT,)

    # store training parameter and grab ema parameter for evaluation
    ema.store(score.parameters())
    ema.copy_to(score.parameters())

    ####### Evaluate Score Model #######
    tot_test_loss = 0
    for _ in tqdm(range(args.eval_times)):
        test_loss = diffBridge.evaluate(
                eval_dataLoader=test_loader,
                prob_T=prob_T,
                local_rank=args.local_rank,
                mode="test")
        tot_test_loss += test_loss

    tot_test_loss /= args.eval_times 

    if args.local_rank in [-1, 0]:
        print((f"   Test loss: {tot_test_loss}"))

    # restore ema trick
    ema.restore(score.parameters())

    # Clean up the process group
    if args.local_rank != -1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

