import torch
import torch.nn as nn


class DummyLayerNorm(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class LayerNorm(DummyLayerNorm):
    def __init__(self, emb_dim: int)->None:
        super().__init__(emb_dim)
        self.eps : float = 1e-5
        self.scale: torch.Tensor = nn.Parameter(torch.ones(emb_dim))
        self.shift : torch.Tensor = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean: torch.Tensor = x.mean(dim=-1, keepdim=True)
        var: torch.Tensor = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x: torch.Tensor = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift