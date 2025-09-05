from datasets import load_dataset
from utils import * 
from tokenizer import Tokenizer 



def main():
    iterator = DatasetIterator()
    tokenizer = Tokenizer(vocab_size=300)
    tokenizer.train_from_iterator(iterator)
    tokenizer.save(merges_path='data/merges.json',vocab_path='data/vocab.json')
    print(tokenizer.merges)

if __name__ == "__main__":
    main()
