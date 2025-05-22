import re

class SimpleTokenizer:
    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab2id = vocab
        self.id2vocab = {i:v for v, i in vocab.items()}
    
    def encode(self, text: str) -> list[int]:
        preprocessed = re.split(r'([,.?_!"()]\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.vocab2id else '<|unk|>'for item in preprocessed]
        idxs = [self.vocab2id[vocab] for vocab in preprocessed]
        return idxs

    def decode(self, idxs : list[int]) -> str:
        text = " ".join([self.id2vocab[idx] for idx in idxs])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text