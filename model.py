import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn = nn.MultiheadAttention(embed_dim=n_embd, num_heads=n_head)

    def forward(self, x):
        return self.attn(x, x, x)[0]

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.attn = Head(n_embd, n_head)
        self.ff = FeedForward(n_embd)

    def forward(self, x):
        x = self.attn(x) + x
        return self.ff(x) + x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd=32, n_head=4, n_layer=3):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        x = self.token_embeddings(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def generate(self, context, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token), dim=1)
        return context
