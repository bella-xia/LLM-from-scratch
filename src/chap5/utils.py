import tiktoken, torch
import matplotlib.pyplot as plt

def text_to_token_ids(text : str, tokenizer : tiktoken.Encoding) -> torch.Tensor:
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids : torch.Tensor, tokenizer: tiktoken.Encoding) -> str:
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

def print_sampled_tokens(probs,
                         num_samples=1000):
    torch.manual_seed(123)
    sample = [torch.multinomial(probs, num_samples=1).item() 
              for _ in range(num_samples)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} X {token_ids_to_text([i])[0]}")

def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)