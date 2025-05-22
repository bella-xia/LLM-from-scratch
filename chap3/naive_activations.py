import torch 

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)