import torch
from trainer import Trainer
from data import Data
from model import GPTLanguageModel

RANDOM_SEED = 1337
TRAIN_ITERATIONS = 10000
WORD_COUNT = 100
DATA_FILE = "input.txt"

def main():
    torch.manual_seed(RANDOM_SEED)

    data = Data(DATA_FILE)
    model = GPTLanguageModel(data.vocab_size)

    trainer = Trainer(data, model)
    trainer.train(TRAIN_ITERATIONS)

    context = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(context, WORD_COUNT)[0].tolist()

    print(data.decode(generated))


if __name__ == "__main__":
    main()
