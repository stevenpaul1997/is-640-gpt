import torch
from trainer import Trainer
from data import Data
from model import GPTLanguageModel
 


MAX_ITERS = 1000
RANDOM_SEED = 1337
WORD_COUNT = 100
DATA_FILE = "input.txt"

def main():
    torch.manual_seed(RANDOM_SEED)
    data = Data(DATA_FILE)
    model = GPTLanguageModel(data.vocab_size)
    trainer = Trainer(data.get_data(), model)
    context =trainer.train(MAX_ITERS)
    generated = model.generate(context, max_new_tokens=WORD_COUNT)
    print(data.decode(generated[0].tolist()))

if __name__ == "__main__":
    main()
