import torch, tiktoken
from src.chap5.loss_fn import calc_loss_batch, calc_loss_loader
from src.chap5.utils import text_to_token_ids, token_ids_to_text
from src.chap4.utils import naive_generate_text

def train_model_simple(model : torch.nn.Module,
                       train_dataloader : torch.utils.data.DataLoader,
                       val_dataloader : torch.utils.data.DataLoader,
                       num_epochs: int, device : torch.device,
                       optimizer : torch.optim.Optimizer,
                       eval_freq : int, eval_iter: int,
                       start_context : str):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_dataloader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_dataloader, val_dataloader,
                    device
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch+1} (step {global_step:06d}):")
                print(f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        print("greedy generation:")
        generate_and_print_sample(
            model, train_dataloader.dataset.tokenizer,
            device, start_context, is_naive=True
        )
        print("temperature & top-k generation:")
        generate_and_print_sample(
            model, train_dataloader.dataset.tokenizer,
            device, start_context, is_naive=False
        )
    
    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model : torch.nn.Module, 
                   train_dataloader : torch.utils.data.DataLoader,
                   val_dataloader : torch.utils.data.DataLoader,
                   device : torch.device):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_dataloader, model, device)
        val_loss = calc_loss_loader(val_dataloader, model, device)
    
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model : torch.nn.Module,
                              tokenizer : tiktoken.Encoding,
                              device : torch.device,
                              start_context : str,
                              is_naive = True):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        if is_naive:
            token_ids = naive_generate_text(
                model, encoded, max_new_tokens=50,
                context_size=context_size
            )
        else:
            token_ids = generate(
                model, encoded, max_new_tokens=50,
                context_size=context_size,
                temperature=1.4, top_k=25
            )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))
    model.train()

def generate(model : torch.nn.Module, idx : torch.Tensor,
             max_new_tokens : int, context_size : int,
             temperature : float, top_k : int | None = None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        # top k sampling
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        # temperature
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
        