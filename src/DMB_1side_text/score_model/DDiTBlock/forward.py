import torch

from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func

from .. import rotary
from ..fused_add_dropout_scale import (
    bias_dropout_add_scale_fused_train, 
    bias_dropout_add_scale_fused_inference, 
    get_bias_dropout_add_scale, 
    modulate_fused,
)
def forward(self, x, rotary_cos_sin, c, seqlens=None):
    batch_size, seq_len = x.shape[0], x.shape[1]

    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

    # attention operation
    x_skip = x
    x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
    # dtype0 = x.dtype

    qkv = self.attn_qkv(x)
    qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
    with torch.cuda.amp.autocast(enabled=False):
        cos, sin = rotary_cos_sin
        qkv = rotary.apply_rotary_pos_emb(
            qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
        )
    qkv = rearrange(qkv, 'b s ... -> (b s) ...')
    if seqlens is None:
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * seq_len, step=seq_len,
            dtype=torch.int32, device=qkv.device
        )
    else:
        cu_seqlens = seqlens.cumsum(-1)
    x = flash_attn_varlen_qkvpacked_func(
        qkv, cu_seqlens, seq_len, 0., causal=False)
    
    x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)

    x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)

    # mlp operation
    x = bias_dropout_scale_fn(self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), None, gate_mlp, x, self.dropout)
    return x


