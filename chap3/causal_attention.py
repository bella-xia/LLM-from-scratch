import torch.nn as nn
import torch
from self_attention import SelfAttention

class CausalAttention(SelfAttention):
    def __init__(self, d_in, d_out, context_len, dropout, 
                 qkv_bias = False):
        super().__init__(d_in, d_out, qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_len, context_len),
                       diagonal=1)
        )
    
    def forward(self, x):
        _, num_tokens, _ = x.shape
        
        queries = self.w_q(x)
        keys = self.w_k(x)
        values = self.w_v(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf
        )
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec