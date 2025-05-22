import torch.nn as nn
import torch

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out,
                 qkv_bias=False):
        super().__init__()
        self.d_out = d_out

        self.w_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_v = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        keys = self.w_k(x)
        queries = self.w_q(x)
        values = self.w_v(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec