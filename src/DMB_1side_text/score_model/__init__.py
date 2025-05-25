'''
Score Estimator

Estimate p_j(t)

where p_j(t) refers to the probability of the j-th type at time t

'''
import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers import AutoConfig
import torch.nn.functional as F
import torch.nn as nn

from .EmbeddingLayer import EmbeddingLayer
from .DDiTBlock import DDiTBlock
from .DDitFinalLayer import DDitFinalLayer
from .TimestepEmbedder import TimestepEmbedder
from . import rotary




class ScoreModel(torch.nn.Module):
    def __init__(self, 
                 vocab_size,
                 hidden_size=768,
                 time_hidden_size=128,
                 n_heads=12,
                 dropout=0.1,
                 n_blocks=12,
                 scale_by_sigma=False,):
        '''
        Calculate p_j(t)
            Notice that the output is a row vector of size n 
               [p_1(t), p_2(t), ..., p_n(t)]

        Model Inputs:
        - perturbed data (row number), time, h_sigma: [batch, 1 + 1 + 1]

        Model Outputs:
        - row number-th type log score: [batch, n]


        Args:
        - vocab_size: size of the finite state set
        - hidden_size: for vocab embed
        - time_hidden_size: for time embed
        - n_heads: number of multi-head attention
        - dropout: dropout ratio
        - n_blocks: number of DiT blocks

        '''
        super(ScoreModel, self).__init__()

        self.vocab_size = vocab_size

        self.vocab_embed = EmbeddingLayer(hidden_size, vocab_size)
        self.sigma_map = TimestepEmbedder(time_hidden_size)
        self.rotary_emb = rotary.Rotary(hidden_size // n_heads)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DDiTBlock(hidden_size, 
                      n_heads, 
                      time_hidden_size, 
                      dropout=dropout) 
            for _ in range(n_blocks)
        ])

        self.output_layer = DDitFinalLayer(hidden_size, vocab_size, time_hidden_size)
        self.scale_by_sigma = scale_by_sigma

    from .forward import forward

from .ema import ExponentialMovingAverage

