import torch 
#Processes text into tensor with encoding.
class Data:
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        each_char = sorted(list(set(self.text)))
        self.vocab_size = len(each_char)
        self.stoi = {ch: i for i, ch in enumerate(each_char)} 
        self.itos = {i: ch for i, ch in enumerate(each_char)}  
    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

    def get_data(self):
        return torch.tensor(self.encode(self.text), dtype=torch.long)
    
    

