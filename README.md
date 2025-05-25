<h3 align="center">Discrete Markov Bridge</h3>

## Introduction

# ![DMB](./img/CTMB.jpg)
*This is the code for implementation of the Discrete Markov Bridge, for description and theory, refer to the paper.*

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
