import unicodedata
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import machado
import os
nltk.download('punkt', quiet=True)
nltk.download('machado', quiet=True)

class DatasetIterator:
    def __init__(self):
        os.makedirs('data', exist_ok=True)
        data_file = "data/machado_texts.txt"
        try:
            
            raw_text = machado.raw()
            if not raw_text:
                raise ValueError("machado.raw() returned empty data. Ensure NLTK 'machado' corpus is downloaded.")
            print(f"Loaded machado.raw(): {len(raw_text)} characters")
            # Normalize text
            normalized_text = unicodedata.normalize('NFC', raw_text).lower()
            
            self.data = sent_tokenize(normalized_text, language='portuguese')
            print(f"Split into {len(self.data)} sentences")
    
            print(f"Saving dataset to {data_file}")
            with open(data_file, "w", encoding="utf-8") as f:
                for sentence in self.data:
                    f.write(sentence + "\n")
            print(f"Saved {os.path.getsize(data_file)} bytes to {data_file}")
        except Exception as e:
            print(f"Error in DatasetIterator: {e}")
            self.data = []

    def __iter__(self):
        for item in self.data:
            yield {'sentence': item} 
