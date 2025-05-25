import torch
import torch.nn as nn
import math
class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        """
        Mode arg: 
        0 -> use a learned layer, 
        1 -> use eigenvectors, 
        2-> add in eigenvectors, 
        3 -> use pretrained embedding matrix

        Args:
        - dim: embedding dim
        - vocab_dim: vocab size
        """
        super().__init__()
        
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))

        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))
    from .forward import forward


