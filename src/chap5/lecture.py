import argparse, torch, tiktoken, os, math
import matplotlib.pyplot as plt
from src.chap2.simple_dataloader import create_dataloader
from src.chap4.gpt_model import GPTModel
from src.chap4.config import GPT_CONFIG_124M
from src.chap4.utils import naive_generate_text
from src.chap5.utils import text_to_token_ids, token_ids_to_text, plot_losses, find_highest_gradient
from src.chap5.loss_fn import calc_loss_loader, calc_loss_batch
from src.chap5.training import train_model_simple, generate_and_print_sample
# from src.chap5.gpt_download import download_and_load_gpt2

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="the-verdict.txt")
    parser.add_argument("-o", "--output", type=str, default="model.pth")
    parser.add_argument("-m", "--mode", type=int, default=0)
    parser.add_argument("-s", "--section", type=int, default=0)
    args = parser.parse_args()


    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    tokenizer = tiktoken.get_encoding("gpt2")

    if args.mode == 1:
        model.eval()

    # ----- 5.1.1 using gpt to generate text ----- #
    if args.mode == 1 and args.section == 1:
        print("# ----- 5.1.1 using gpt to generate text ----- #")
        start_context = "Every effort moves you"

        token_ids = naive_generate_text(
            model=model,
            idx=text_to_token_ids(start_context, tokenizer),
            max_new_tokens=10,
            context_size=GPT_CONFIG_124M['context_len']
        )
        print("output text:\n", token_ids_to_text(token_ids, tokenizer))

    # ----- 5.1.2 calculating text generation loss ----- #
    if args.mode == 1 and args.section == 2:
        print("# ----- 5.1.2 calculating text generation loss ----- #")

        inputs = torch.tensor([[16833, 3626, 6100], # every effort moves
                  [40, 1107, 588]])     # I really like
        targets = torch.tensor([[3626, 6100, 345], # effort moves you
                   [1107, 588, 428]]) # really like ch
        with torch.no_grad():
            logits = model(inputs)
        probs = torch.softmax(logits, dim=-1)
        token_ids = torch.argmax(probs, dim=-1)
        print(f"targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
        print(f"outputs batch 1: {token_ids_to_text(token_ids[0], tokenizer)}")
        print(f"targets batch 2: {token_ids_to_text(targets[1], tokenizer)}")
        print(f"outputs batch 2: {token_ids_to_text(token_ids[1], tokenizer)}")

        text_idx = 0
        target_prob_1 = probs[text_idx, [0, 1, 2], targets[text_idx]]
        print("text 1:", target_prob_1)

        text_idx = 1
        target_prob_2 = probs[text_idx, [0, 1, 2], targets[text_idx]]
        print("text 2:", target_prob_2)

        log_probs = torch.log(torch.cat((target_prob_1, target_prob_2)))
        print("concatenated log probabilities of the target text:", log_probs)

        avg_log_probs = torch.mean(log_probs)
        print(f"current log prob mean: {avg_log_probs}; target log prob mean: 0")

        # using cross entropy loss
        logits_flat = logits.flatten(0, 1)
        targets_flat = targets.flatten()
        print("flattened logits shape:", logits_flat.shape)
        print("flattened targets shape:", targets_flat.shape)

        loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
        print("cross entropy loss:", loss)
    
    if args.section >= 3 or args.mode >= 2:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
        data_dir = os.path.join(project_root, 'data')

        with open(os.path.join(data_dir, args.input), 'r', encoding='utf-8') as f:
            text_data : str = f.read()

        train_ratio = 0.9
        split_idx = int(train_ratio * len(text_data))
        train_data = text_data[:split_idx]
        val_data = text_data[split_idx:]

        train_dataloader = create_dataloader(
            train_data,
            batch_size=2,
            max_len=GPT_CONFIG_124M['context_len'],
            stride=GPT_CONFIG_124M['context_len'],
            tokenizer=tokenizer,
            drop_last=True,
            shuffle=True
        )
        val_dataloader = create_dataloader(
            val_data,
            batch_size=2,
            max_len=GPT_CONFIG_124M['context_len'],
            stride=GPT_CONFIG_124M['context_len'],
            tokenizer=tokenizer,
            drop_last=False,
            shuffle=False
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    # ----- 5.1.3 calculating the training and validation set losses ----- #
    if args.mode == 1 and args.section == 3:
        print("# ----- 5.1.3 calculating the training and validation set losses ----- #")

        print("train loader:")
        for x, y in train_dataloader:
            print(x.shape, y.shape)
        
        print("validation loader:")
        for x, y in val_dataloader:
            print(x.shape, y.shape)

        train_loss = calc_loss_loader(train_dataloader, model, device)
        val_loss = calc_loss_loader(val_dataloader, model, device)
        print("training loss:", train_loss)
        print("validation loss:", val_loss)
    
    # ----- 5.2 training an LLM ----- #
    # ----- 5.3 use topk and temperature for text generation ----- #
    if args.mode >= 2 and args.mode <= 4:
        print("# ----- 5.2 training an LLM ----- #")
        print("# ----- 5.3 use topk and temperature for text generation ----- #")
        optimizer : torch.optim.Optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=4e-4,
            weight_decay=0.1
        )
        num_epochs = 10
        train_losses, val_losses, token_seen = train_model_simple(
            model, train_dataloader, val_dataloader, num_epochs,
            device, optimizer, eval_freq=5, eval_iter=1,
            start_context="Every effort moves you"
        )

        epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
        plot_losses(epochs_tensor, token_seen,
                    train_losses, val_losses)
    
    # ----- 5.4 saving and loading LLM weights ----- #
    if args.mode == 4:
        print("# ----- 5.4 loading LLM weights ----- #")
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
        data_dir = os.path.join(project_root, 'data')

        # first save model 
        torch.save(model.state_dict(), os.path.join(data_dir, args.output))

        # then load model
        model.load_state_dict(torch.load(os.path.join(data_dir, args.output)))
        model.eval()
        start_context="Every effort moves you"
        print("gready generation:")
        generate_and_print_sample(
            model, tokenizer, device,
            start_context, is_naive = True
        )        
        print("temperature & topk generation:")
        generate_and_print_sample(
            model, tokenizer, device,
            start_context, is_naive = False
        )

    # ----- 5.5 loading pretrained weights from OpenAI ----- #
    # currently failed due to problems with tensorflow
    if args.mode == 5:
        print("# ----- 5.5 loading pretrained weights from OpenAI ----- #")
        settings, params = download_and_load_gpt2(
            model_size="124M",
            models_dir="gpt2"
        )
        print("settings:", settings)
        print("parameter dictionary keys:", params.keys())

    # ----- 5.6 adding bells and whistles to training ----- #
    if args.mode == 6:
        print("# ----- 5.6 adding bells and whistles to training ----- #")
    
    if args.mode == 6 and args.section <= 2:
        n_epochs = 15
        initial_lr = 0.0001
        peak_lr = 0.01
        warmup_steps = 30

        optimizer = torch.optim.AdamW(
            model.parameters(),
            weight_decay=0.1
        )
        lr_increment = (peak_lr - initial_lr) / warmup_steps
        global_step = -1
        total_train_steps = len(train_dataloader) * n_epochs
        track_lrs = []

    # ----- 5.6.1 rate warmup ----- #
    if args.mode == 6 and args.section == 1:
        print("# ----- 5.6.1 rate warmup ----- #")

        for epoch in range(n_epochs):
            for input_batch, target_batch in train_dataloader:
                optimizer.zero_grad()
                global_step += 1

                if global_step < warmup_steps:
                    lr = initial_lr + global_step * lr_increment
                else:
                    lr = peak_lr
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                track_lrs.append(optimizer.param_groups[0]['lr'])
            
        plt.ylabel('learning rate')
        plt.xlabel('step')
        plt.plot(range(total_train_steps), track_lrs)
        plt.show()

    # ----- 5.6.2 cosine decay ----- #
    if args.mode == 6 and args.section == 2:
        print("# ----- 5.6.2 cosine decay ----- #")
        min_lr = 0.1 * initial_lr
        for epoch in range(n_epochs):
            for input_batch, target_batch in train_dataloader:
                optimizer.zero_grad()
                global_step += 1

                if global_step < warmup_steps:
                    lr = initial_lr + global_step * lr_increment
                else:
                    progress = ((global_step - warmup_steps) / (total_train_steps - warmup_steps))
                    lr = min_lr + (peak_lr - min_lr) * 0.5 * (
                        1 + math.cos(math.pi * progress))

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                track_lrs.append(optimizer.param_groups[0]['lr'])
            
        plt.ylabel('learning rate')
        plt.xlabel('step')
        plt.plot(range(total_train_steps), track_lrs)
        plt.show()                  

    # ----- 5.6.3 gradient clipping ----- #
    if args.mode == 6 and args.section == 3:

        for input_batch, target_batch in train_dataloader:
            break
        
        loss = calc_loss_batch(
            input_batch, target_batch, model, device
        )
        loss.backward()

        print("hihgest gradient before clipping:", find_highest_gradient(model))

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )
        print("hihgest gradient after clipping:", find_highest_gradient(model))


