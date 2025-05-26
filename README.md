<h1 align="left">Discrete Markov Bridge</h1>
<div align="left">
 <a href=""><img src="https://img.shields.io/badge/Arxiv-DMB-b31b1b.svg?logo=arXiv" alt=""></a>
</div>

This is the code for the implementation of the Discrete Markov Bridge. For description and theory, refer to the [paper]() by [Hengli Li](https://github.com/Henry839), [Yuxuan Wang](https://github.com/patrick-tssn), [Song-Chun Zhu](https://zhusongchun.net/), [Ying Nian Wu](http://www.stat.ucla.edu/~ywu/), and [Zilong Zheng](https://zilongzheng.github.io/).

![DMB](./img/CTMB.jpg)

<div align="center">
Figure 1. Overview of the Discrete Markov Bridge framework.
</div>




## Introduction


**Discrete Markov Bridge (DMB)** consists of two component: the *Matrix-learning*
and the *Score-learning*. The *Matrix-learning* process is designed to learn an adaptive transition
rate matrix, which facilitates the estimation of an adapted latent distribution. Concurrently, the
*score-learning* process focuses on estimating the probability ratio necessary for constructing the
inverse transition rate matrix, thereby enabling the reconstruction of the original data distribution.

## Installation
```bash
conda create -n DMB python=3.10
conda activate DMB
pip3 install torch torchvision torchaudio
pip install transformers datasets tqdm accelerate
pip install wandb
```

## Usage

```bash
cd src
sh scripts/example.sh
```
For understanding of the shell scripts, please check [parse-file](./src/parser.py) for description of the args.

The example shell scripts
```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun --nproc_per_node=8 --master_port=29501 main_ddp.py \
    --ngpus 8 \
    --sche_name "loglinear" \ # diffusion scheduler
    --sigma_min 1e-4 \
    --sigma_max 20 \
    --Q_lr 1e-3 \ # learning rate for matrix learning
    --Q_weight_decay 1e-2 \
    --Q_initialization 'gather' \
    --Q_epochs 15 \
    --vocab_size 27 \
    --score_epoch 10 \
    --hidden_size 768 \ # transformer args
    --time_hidden_size 128 \ # transformer args
    --dropout 0.1 \ # transformer args
    --n_blocks 12 \ # transformer args
    --n_heads 12 \ # transformer args
    --score_lr 3e-4 \ # score learning rate
    --score_warmup_steps 2500 \
    --mu_train_dataset_name 'text8' \ # training
    --mu_eval_dataset_name 'text8' \ # eval
    --mu_test_dataset_name 'text8' \ # test
    --cache_dir './cache/' \
    --seqlen 256 \
    --score_accum 1 \
    --score_train_batch_size 512 \
    --score_eval_batch_size 512 \
    --score_grad_clip 1.0 \
    --sample_batch_size 512 \
    --Q_accum 1 \
    --Q_train_batch_size 512 \
    --eval_times 1000 \
    --ema 0.9999 \
    --epoch 1000 \
    --run_name 'text8' \
    --random_seed 42 \

```

## Citation
```bibtex
TBA
```

## Contact
* If you have any questions, please send me an email at: lihengli@stu.pku.edu.cn

