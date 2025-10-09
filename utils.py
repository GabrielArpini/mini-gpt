from datasets import load_dataset
import unicodedata

class DatasetIterator:
    def __init__(self):
        self.data = load_dataset("iara-project/raw_dataset_with_embeddings_bert-base-portuguese-cased-nli-assin-2")["train"]

    def __iter__(self):
        for item in self.data:
            # Yield a dictionary with "text" key, using the "sentence" field
            yield {"text": unicodedata.normalize("NFC", item["sentence"]).lower()}
