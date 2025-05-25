<h2 align="center">Discrete Markov Bridge</h2>
<div align="center">
 <a href=""><img src="https://img.shields.io/badge/Arxiv-DMB-b31b1b.svg?logo=arXiv" alt=""></a>
</div>


## Introduction

# ![DMB](./img/CTMB.jpg)
*This is the code for implementation of Discrete Markov Bridge, for description and theory, refer to the paper.*

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
