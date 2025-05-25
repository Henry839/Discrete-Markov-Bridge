import torch
import torch.nn.functional as F
import numpy as np
eps = 1e-15
def forward(self, 
            h_sigma,
            data=None,):
    '''
    output log score

    Args:
    - h_sigma: int_0^T sigma(s)ds, [batch, 1]
    - data: perturbed data (row number), [batch, length]
    '''
    x = self.vocab_embed(data)
    c = F.silu(self.sigma_map(h_sigma))

    rotary_cos_sin = self.rotary_emb(x)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
        x = self.output_layer(x, c)

    if self.scale_by_sigma:
        esigm1_log = torch.where(h_sigma < 0.5, torch.expm1(h_sigma), h_sigma.exp() - 1).log().to(x.dtype)[:, None, None]
        x = x - esigm1_log - np.log(x.shape[-1] - 1)# this will be approximately averaged at 0


    # p_i/p_i
    # use log score, set 0 here, after exp turns to 1
    x = torch.scatter(x, -1, data[..., None], torch.zeros_like(x[..., :1]))
    return x

