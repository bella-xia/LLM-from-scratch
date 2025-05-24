import re, tiktoken, argparse, torch, os
from src.chap2.simple_tokenizer import SimpleTokenizer
from src.chap2.simple_dataloader import create_dataloader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="the-verdict.txt")
    parser.add_argument("-m", "--mode", type=int, default=1)

    args = parser.parse_args()

    # ----- 2.1 initializaing ----- #
    print("# ----- 2.1 initializing ----- #")
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
    data_dir = os.path.join(project_root, 'data')

    with open(os.path.join(data_dir, args.input), 'r', encoding='utf-8') as f:
        raw_text : str = f.read()
    
    print(f"total number of characters: {len(raw_text)}")
    print("fist 50 characters: ", raw_text[:50])
    print("")
        
    # ----- 2.2 tokenization ----- #
    """
    Some re regular expressions
    re.split(r'(\s)', text) --> split all text by 
    occurences of space (space will also be separately)
    split as a token

    re.split(r'([,.]|\s)', text) --> split by either comma,
    period, or space

    text.strip() --> can be used to closely monitor where
    a string is fully space (in which case this is equivalent)
    of a false
    """

    if args.mode >= 2 and args.mode < 5:
        print("# ----- 2.2 tokenization ----- #")
        split_pattern : re.Pattern = re.compile(r'([,.?_!"()]\']|--|\s)')
        preprocessed : list[str] = re.split(split_pattern, raw_text)
        preprocessed : list[str] = [item.strip() for item in preprocessed
                                    if item.strip()]
        print("first 30 tokens: ", preprocessed[:30])
        print("")

    # ----- 2.3 token2id ----- #
    if args.mode >= 3 and args.mode < 5:
        print("# ----- 2.3 token2id ----- #")
        all_words : list[str] = sorted(list(set(preprocessed)))
        all_words.extend(['<|endoftext|>', '<|unk|>'])
        vocab_size : int = len(all_words)
        print(f"the total vocabulary size in the current text is {vocab_size}")

        vocab = {token:idx for idx, token in enumerate(all_words)}
        print("first 5 vocabs:")
        for (i, item) in enumerate(vocab.items()):
            tok, tokid = item

            print(f"{tokid} --> {tok}")
            if i > 5:
                break
        
        # experimenting with tokenizer
        print("--> experimenting with tokenizer:")
        tokenizer = SimpleTokenizer(vocab)
        text = "with that face watching me I couldn't do another stroke. The plain truth was, I didn't know where to put it--_I had never known_. Only, with my sitters and my public,"
        ids = tokenizer.encode(text)
        print(ids)
        print(tokenizer.decode(ids))
        print("")

    # ----- 2.4 context token ----- #
    if args.mode == 4:
        print("# ----- 2.4 context token ----- #")
        # adding unknown and end of text
        print("--> adding unknown and end of text:")
        text1 = "Hello, do you like tea?"
        text2 = "In the sunlit terraces of the palace."
        text = " <|endoftext|> ".join((text1, text2))
        ids = tokenizer.encode(text)
        print(ids)
        print(tokenizer.decode(ids))
        print("")

    # ----- 2.5 byte pair encoding ----- #
    if args.mode == 5:
        print("# ----- 2.5 byte pair encoding ----- #")
        text = "with that face watching me I couldn't do another stroke.<|endoftext|> The plain truth was, I didn't know where to put it--_I had never known_.<|endoftext|> Only, with my sitters and my public,"
        tokenizer = tiktoken.get_encoding("gpt2")
        ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        print(ids)
        words = tokenizer.decode(ids)
        print(words)
        print("")
    
    # ----- 2.6 data sampling with a sliding window ----- #
    if args.mode == 6:
        print("# ----- 2.6 data sampling with a sliding window ----- #")
        tokenizer = tiktoken.get_encoding("gpt2")
        enc_text = tokenizer.encode(raw_text)
        print(f"the encoded token length is {len(enc_text)}")

        enc_sample = enc_text[50:]
        context_size = 4
        x = enc_sample[:context_size]
        y = enc_sample[1:context_size]
        print(f"x: {x}")
        print(f"y:      {y}")

        for i in range(1, context_size + 1):
            context = enc_sample[:i]
            desired = enc_sample[i]
            print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
        
        dataloader = create_dataloader(
            raw_text, batch_size=1, max_len=4, stride=1, shuffle=False,
            tokenizer=tokenizer
        )
        data_iter = iter(dataloader)
        print("first two batches from the dataloader:")
        print(next(data_iter))
        print(next(data_iter))
        print("")
    
    # ----- 2.7 token embeddings ----- #
    if args.mode == 7:
        print("# ----- 2.7 token embeddings ----- #")
        vocab_size = 6
        output_dim = 3

        torch.manual_seed(123)
        embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
        print("embedding weights: ")
        print(embedding_layer.weight)
        print("embedding vector for vocab #4: ")
        print(embedding_layer(torch.tensor([3])))
        print("embedding vector for an array of vocab index: [2, 3, 5, 1]: ")
        input_ids = torch.tensor([2, 3, 5, 1])
        print(embedding_layer(input_ids))
        print("")

    # ----- 2.8 encoding word positions ----- #
    if args.mode == 8:
        print("# ----- 2.8 encoding word positions ----- #")
        max_len = 4
        tokenizer, dataloader = create_dataloader(raw_text, batch_size=8,
                                       max_len=max_len, stride=max_len)
        output_dim = 256
        vocab_size = tokenizer.n_vocab
        token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
        print(f"the loaded tokenizer has {vocab_size} vocabs")

        data_iter = iter(dataloader)
        inputs, targets = next(data_iter)
        print(f"token ids:\n{inputs}")
        print(f"input shape:\n{inputs.shape}")

        token_embeddings = token_embedding_layer(inputs)
        print(f"embedded input vector shape:\n{token_embeddings.shape}")

        context_len = max_len
        pos_embedding_layer = torch.nn.Embedding(context_len, output_dim)
        pos_embeddings = pos_embedding_layer(torch.arange(context_len))
        print(f"position embedding vector shape:\n{pos_embeddings.shape}")

        input_embeddings = token_embeddings + pos_embeddings
        print(f"resulting input embeddings vector size:\n{input_embeddings.shape}")