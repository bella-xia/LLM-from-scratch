import argparse, tiktoken, torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src.chap4.gpt_model import GPTModel
from src.chap4.layer_norm import LayerNorm
from src.chap4.activations import naive_GELU
from src.chap4.feed_forward import FeedForward, ExampleDeepFeedForward
from src.chap4.transformer_block import TransformerBlock
from src.chap4.config import GPT_CONFIG_124M
from src.chap4.utils import print_gradients, naive_generate_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=int, default=0)
    args = parser.parse_args()

    # ----- 4.1 LLM architecture ----- #
    if args.mode == 1:
        print("# ----- 4.1 LLM architecture ----- #")
        tokenizer = tiktoken.get_encoding("gpt2")
        batch = []
        txt1 = "Every effort moves you"
        txt2 = "Every day holds a"

        batch.append(torch.tensor(tokenizer.encode(txt1)))
        batch.append(torch.tensor(tokenizer.encode(txt2)))
        batch = torch.stack(batch, dim=0)
        print("tokenized batch input:\n", batch)

        torch.manual_seed(123)
        model = GPTModel(GPT_CONFIG_124M, is_dummy=True)
        logits = model(batch)
        print("output shape:\n", logits.shape)
        print("output:\n", logits)
        print("")

    # ----- 4.2 Layer Normalization ----- #
    if args.mode == 2:
        print("# ----- 4.2 Layer Normalization ----- #")
        torch.manual_seed(123)
        batch_example = torch.randn(2, 5)
        layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
        out = layer(batch_example)
        print("output of batch from activation layer:\n", out)
        
        mean = out.mean(dim=-1, keepdim=True)
        var = out.var(dim=-1, keepdim=True)
        print("output mean:\n", mean)
        print("output variance:\n", var)

        # normalization
        out_norm = (out - mean) / torch.sqrt(var)
        mean = out_norm.mean(dim=-1, keepdim=True)
        var = out_norm.var(dim=-1, keepdim=True)
        print("normalized layer output:\n", out_norm)
        print("normalized output mean:\n", mean)
        print("normalized output variance:\n", var)

        # implemented layer norm module
        ln : nn.Module = LayerNorm(5)
        out_ln: torch.Tensor = ln(batch_example)
        mean = out_ln.mean(dim=-1, keepdim=True)
        var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
        print("normalized mean via layer norm module:\n", mean)
        print("normalzied variance via layer norm module:\n", var)

    # ----- 4.3 GeLU activation function ----- #
    if args.mode == 3:
        print("# ----- 4.3 GeLU activation function ----- #")
        gelu, relu = naive_GELU(), nn.ReLU()
        x = torch.linspace(-3, 3, 100)
        y_gelu, y_relu = gelu(x), relu(x)
        plt.figure(figsize=(8, 3))
        for i, (y, lab) in enumerate(zip([y_gelu, y_relu], ['GeLU', 'ReLU'])):
            plt.subplot(1, 2, i+1)
            plt.plot(x, y)
            plt.title(f"{lab} activation function")
            plt.xlabel("x")
            plt.ylabel(f"{lab}(x)")
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()

        ffn = FeedForward(GPT_CONFIG_124M)
        x = torch.rand(2, 3, GPT_CONFIG_124M['emb_dim'])
        out = ffn(x)
        print(out.shape)

    # ----- 4.4 skip connections ----- #
    if args.mode == 4:
        print("# ----- 4.4 skip connections ----- #")
        layer_sizes = [3, 3, 3, 3, 3, 1]
        sample_input = torch.tensor([[1., 0., -1.]])
        torch.manual_seed(123)
        model_wo_shortcut = ExampleDeepFeedForward(layer_sizes,
                                                   use_shortcut=False)
        print("symptom of vanishing gradient in deep models without skip connection:")
        print_gradients(model_wo_shortcut, sample_input)

        torch.manual_seed(123)
        model_w_shortcut = ExampleDeepFeedForward(layer_sizes,
                                                  use_shortcut=True)
        print("deep models with skip connection:")
        print_gradients(model_w_shortcut, sample_input)

    # ----- 4.5 connect attention and linear in transformer block ----- #
    if args.mode == 5:
        print("# ----- 4.5 connect attention and linear in transformer block ----- #")
        torch.manual_seed(123)
        x = torch.rand(2, 4, GPT_CONFIG_124M['emb_dim'])
        block = TransformerBlock(GPT_CONFIG_124M)
        output = block(x)

        print("input shape:\n", x.shape)
        print("output shape:\n", output.shape)

    # ----- 4.6 compile the GPT model ----- #
    if args.mode == 6:
        print("# ----- 4.6 compile the GPT model ----- #")
        tokenizer = tiktoken.get_encoding("gpt2")
        batch = []
        txt1 = "Every effort moves you"
        txt2 = "Every day holds a"

        batch.append(torch.tensor(tokenizer.encode(txt1)))
        batch.append(torch.tensor(tokenizer.encode(txt2)))
        batch = torch.stack(batch, dim=0)

        torch.manual_seed(123)
        model = GPTModel(GPT_CONFIG_124M)

        out = model(batch)
        print("input batch:\n", batch)
        print("output shape:\n", out.shape)
        print("output representation:\n", out)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"total number of parameters in the model: {total_params:,}")
        print("token embedding layer shape:", model.tok_emb.weight.shape)
        print("output layer shape:", model.out_head.weight.shape)

        total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
        print(f"number of parameters considering weight tying: {total_params_gpt2:,}")

        total_size_bytes = total_params * 4
        total_size_mb = total_size_bytes / (1024 * 1024)
        print(f"total size of of model: {total_size_mb:.2f} MB")
    
    # ----- 4.7 generating text ----- #
    if args.mode == 7:
        print("# ----- 4.7 generating text ----- #")
        tokenizer = tiktoken.get_encoding("gpt2")
        starting_context = "Hello, I am"
        encoded = tokenizer.encode(starting_context)
        print("encoded starter context:", encoded)
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        print("encoded tensor shape:", encoded_tensor.shape)

        torch.manual_seed(123)
        model = GPTModel(GPT_CONFIG_124M)
        model.eval()
        out = naive_generate_text(
            model=model,
            idx=encoded_tensor,
            max_new_tokens=15,
            context_size=GPT_CONFIG_124M['context_len']
        )
        print("outputted tensor:", out)
        print("outputted tensor shape:", out.shape)

        decoded_text = tokenizer.decode(out.squeeze(0).tolist())
        print(f"decoded text generation: '{decoded_text}'")
