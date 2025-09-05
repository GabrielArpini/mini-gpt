from datasets import load_dataset


class DatasetIterator:
    def __init__(self):
        self.data = load_dataset("AIGym/tokenizer-training-v2", split='train')

    def __iter__(self):
        for item in self.data:
            yield item 
