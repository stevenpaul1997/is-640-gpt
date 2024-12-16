#import the torch function
import torch
import torch.nn.functional as F
import torch.nn as nn



# class BigramLanguageModel(nn.Module):
#     def __init__(self, data,vocab_size, const_var):
#         super().__init__()
#         # each token directly reads off the logits for the next token from a lookup table
#         self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

#     def forward(self, idx, targets=None):

#         # idx and targets are both (B,T) tensor of integers
#         logits = self.token_embedding_table(idx) # (B,T,C)

#         if targets is None:
#             loss = None
#         else:
#             B, T, C = logits.shape
#             logits = logits.view(B*T, C)
#             targets = targets.view(B*T)
#             loss = F.cross_entropy(logits, targets)

#         return logits, loss

#     def generate(self, idx, max_new_tokens):
#         # idx is (B, T) array of indices in the current context
#         for _ in range(max_new_tokens):
#             # get the predictions
#             logits, loss = self(idx)
#             # focus only on the last time step
#             logits = logits[:, -1, :] # becomes (B, C)
#             # apply softmax to get probabilities
#             probs = F.softmax(logits, dim=-1) # (B, C)
#             # sample from the distribution
#             idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
#             # append sampled index to the running sequence
#             idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
#         return idx



class Trainer() :
    def __init__( self,data,gpt_model,const_var) :
        self.data = data
        self.model = gpt_model
        self.m =  self.model.to(const_var.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=const_var.LEARNING_RATE)
        self.const_var  = const_var

    
    def __init_train_val_data(self) :
            n = int(0.9*len(self.data)) # first 90% will be train, rest val
            self.train_data = self.data[:n]
            self. val_data = self.data[n:]

    # data loading
    def __get_batch(self,split):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.const_var.BLOCK_SIZE, (self.const_var.BATCH_SIZE,))
        x = torch.stack([data[i:i+self.const_var.BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i+1:i+self.const_var.BLOCK_SIZE+1] for i in ix])
        x, y = x.to(self.const_var.device), y.to(self.const_var.device)
        return x, y

    #This is a PyTorch decorator that disables gradient tracking during the execution of the method it decorates.
    @torch.no_grad()
    def __estimate_loss(self,):
            out = {}
            self.model.eval()
            for split in ['train', 'val']:
                losses = torch.zeros(self.const_var.EVAL_ITERS)
                for k in range(self.const_var.EVAL_ITERS):
                    X, Y = self.__get_batch(split)
                    logits, loss = self.model(X, Y)
                    losses[k] = loss.item()
                out[split] = losses.mean()
            self.model.train()
            return out

    def train(self,train_iterations) :
 
        self.__init_train_val_data()
        for iter in range(train_iterations):

            # every once in a while evaluate the loss on train and val sets
            if iter % self.const_var.EVAL_INTERVAL == 0:
                losses = self.__estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # sample a batch of data
            xb, yb = self.__get_batch('train')

            # evaluate the loss
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        # generate from the model
        return  torch.zeros((1, 1), dtype=torch.long, device=self.const_var.device)

