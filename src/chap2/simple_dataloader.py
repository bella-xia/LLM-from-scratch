import torch, tiktoken
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self, input, tokenizer, max_len, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(input)

        for i in range(0, len(token_ids) - max_len, stride):
            input_chunk = token_ids[i:i+max_len]
            target_chunk = token_ids[i+1:i+max_len+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(input, batch_size=4, 
                      max_len=256, stride=128, 
                      tokenizer=None,
                      shuffle=True, drop_last=True):
        if not tokenizer:
            tokenizer = tiktoken.get_encoding("gpt2")
        dataset = GPTDataset(input=input, tokenizer=tokenizer, 
                             max_len=max_len, stride=stride)
        dataloader = DataLoader(
             dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
        )
        return dataloader