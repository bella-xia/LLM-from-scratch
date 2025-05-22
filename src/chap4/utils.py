import torch
import torch.nn as nn

def print_gradients(model : nn.Module, x):
    output = model(x)
    target = torch.tensor([[0.]])

    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)

    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradiante mean of {param.grad.abs().mean()}")
        

def naive_generate_text(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx