from datasets import load_dataset
import kagglehub
import unicodedata
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')
nltk.download('machado', quiet=True)
from nltk.corpus import machado

class DatasetIterator:
    def __init__(self):
        # Load Machado corpus and split into sentences
        raw_text = machado.raw()
        # Normalize text (similar to your old Tokenizer._normalize)
        normalized_text = unicodedata.normalize('NFC', raw_text).lower()
        # Split into sentences
        self.data = sent_tokenize(normalized_text, language='portuguese')

    def __iter__(self):
        for item in self.data:
            yield {'sentence': item}
