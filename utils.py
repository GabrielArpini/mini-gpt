from datasets import load_dataset
import os

class DatasetIterator:
    def __init__(self):
        os.makedirs('data', exist_ok=True)
        print("Loading C4 for tokenizer training...")
        try:
            self.dataset = load_dataset(
                "allenai/c4",
                "en",
                split='train',
                streaming=True,
                trust_remote_code=False
            ).take(100_000)
            print("Dataset loaded for tokenizer training")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.dataset = []

    def __iter__(self):
        for item in self.dataset:
            text = item.get('text', '')
            if text and isinstance(text, str):
                yield text 
