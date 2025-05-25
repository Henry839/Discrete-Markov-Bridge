import torch
import torch.nn.functional as F
def forward(self, x):
    with torch.cuda.amp.autocast(enabled=False):
        x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None,None,:]

