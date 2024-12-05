import torch
import torch.nn.functional as F

class Trainer:
    def __init__(self, data, model, batch_size=16, block_size=8, learning_rate=0.001):
        self.data = data
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.block_size = block_size

    def get_batch(self, split):
        data = self.data.get_data()
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x, y

    def train(self, iterations):
        for i in range(iterations):
            self.model.train()
            xb, yb = self.get_batch('train')
            logits = self.model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss.item()}")
