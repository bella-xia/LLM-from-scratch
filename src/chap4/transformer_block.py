import torch
import torch.nn as nn
from src.chap3.multihead_attention import MultiHeadAttention
from src.chap4.feed_forward import FeedForward
from src.chap4.layer_norm import LayerNorm

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, x):
        return x

class TransformerBlock(DummyTransformerBlock):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.att = MultiHeadAttention(
            d_in = cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_len=cfg['context_len'],
            num_heads=cfg['n_heads'],
            dropout = cfg['drop_rate'],
            qkv_bias=cfg['qkv_bias']
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_resid = nn.Dropout(cfg['drop_rate'])
    
    def forward(self, x):
        # multihead attention
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_resid(x)
        x = x + shortcut

        # feed forward
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut
        return x