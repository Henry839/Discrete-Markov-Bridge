import argparse
import os
import datetime

def get_parser():
    parser = argparse.ArgumentParser(description='Discrete Diffusion Bridge')
    
    # Script-specific arguments
    parser.add_argument("--ngpus", type=int, required=True, help='Number of processes per node')
    
    # Add all the arguments as in your original script
    # scheduler
    parser.add_argument("--sche_name", type=str, default='geometric', help='scheduler type')
    parser.add_argument("--sigma_min", type=float, default=1e-4, help='geometric scheduler')
    parser.add_argument("--sigma_max", type=float, default=20, help='geometric scheduler')
    parser.add_argument("--step_num", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=27)
    parser.add_argument("--mu_train_dataset_name", type=str, default="text8")
    parser.add_argument("--mu_eval_dataset_name", type=str, default="text8")
    parser.add_argument("--mu_test_dataset_name", type=str, default="text8")
    parser.add_argument("--cache_dir", type=str, default="./cache/")
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--sample_batch_size", type=int, default=256)
    parser.add_argument("--eval_times", type=int, default=20)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--run_name", type=str, default=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    parser.add_argument("--path", type=str, default="./logs")
    parser.add_argument("--random_seed", type=int, default=0, help="random seed")
    parser.add_argument("--reset_optimizer", action="store_true")
    parser.add_argument("--eps", type=float, default=1e-10)

    # score
    parser.add_argument("--score_epoch", type=int, default=2)
    parser.add_argument("--score_lr", type=float, default=0.02)
    parser.add_argument("--score_lr_scheduler", type=str, default="fix")
    parser.add_argument("--score_lr_num_train_step", type=int, default=1300001)
    parser.add_argument("--score_warmup_steps", type=int, default=2000)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--time_hidden_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_blocks", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--score_accum", type=int, default=1)
    parser.add_argument("--score_train_batch_size", type=int, default=1024)
    parser.add_argument("--score_eval_batch_size", type=int, default=512)
    parser.add_argument("--score_grad_clip", type=float, default=1.0)
    parser.add_argument("--score_weight_decay", type=float, default=0)
    parser.add_argument("--score_scale_by_sigma", action="store_true")
    parser.add_argument("--score_wd_scheduler", action="store_true")
    parser.add_argument("--score_wd_scheduler_decay_method", type=str, default="smooth")
    parser.add_argument("--ema", type=float, default=0.9999, help="Exponential moving average")

    # Q
    parser.add_argument("--Q_epochs", type=int, default=2)
    parser.add_argument("--Q_train_batch_size", type=int, default=1024)
    parser.add_argument("--Q_grad_clip", type=float, default=1.0)
    parser.add_argument("--Q_initialization", type=str, default="gather")
    parser.add_argument("--Q_lr", type=float, default=0.02)
    parser.add_argument("--Q_lr_scheduler", type=str, default="fix")
    parser.add_argument("--Q_lr_num_train_step", type=int, default=1300001)
    parser.add_argument("--Q_accum", type=int, default=1)
    parser.add_argument("--Q_warmup_steps", type=int, default=0)
    parser.add_argument("--Q_weight_decay", type=float, default=0)
    parser.add_argument("--Q_negative_func", type=str, default="relu")

    # local_rank is provided by torchrun
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)))

    return parser
 
