import torch
from trainer import Trainer
from data import Data
from model import GPTLanguageModel
 

class static_variables() :


    BATCH_SIZE = 16 
    BLOCK_SIZE = 32 
    MAX_ITERS = 10000
    EVAL_INTERVAL = 1000
    LEARNING_RATE = 1E-3
    EVAL_ITERS = 200
    RANDOM_SEED = 1337
    WORD_COUNT = 150
    n_embd =  64
    n_head =  4
    n_layer =  4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dropout = 0.2



def main():

    DATA_FILE = "input.txt"
    torch.manual_seed(static_variables.RANDOM_SEED)
    data = Data(DATA_FILE)
    model = GPTLanguageModel(data.vocab_size,static_variables)
    trainer = Trainer(data.get_data(), model,static_variables)
    context =trainer.train(static_variables.MAX_ITERS)
    
    generated = model.generate(context, static_variables.WORD_COUNT)[0].tolist()
    print(data.decode(generated))

if __name__ == "__main__":
    main()
