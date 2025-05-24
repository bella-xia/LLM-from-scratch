import torch
import torch.nn.functional as F

def calc_loss_batch(input_batch : torch.Tensor, target_batch : torch.Tensor, model : torch.nn.Module, device : torch.device) -> torch.Tensor:
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )
    return loss

def calc_loss_loader(dataloader : torch.utils.data.DataLoader, 
                     model : torch.nn.Module, 
                     device : torch.device, num_batches : int | None =None):
    total_loss = 0.
    if not num_batches:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))
    
    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
            continue

        break
    
    return total_loss / num_batches