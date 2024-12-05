import torch

class Data:
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}  # char to index
        self.itos = {i: ch for i, ch in enumerate(self.chars)}  # index to char

    def encode(self, s):
        """Converts a string to a list of integers."""
        return [self.stoi[c] for c in s]

    def decode(self, l):
        """Converts a list of integers back to a string."""
        return ''.join([self.itos[i] for i in l])

    def get_data(self):
        """Returns the encoded data."""
        return torch.tensor(self.encode(self.text), dtype=torch.long)
