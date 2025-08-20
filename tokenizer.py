from __future__ import annotations


import re
import unicodedata
from collections import Counter,deque

class Tokenizer:
    def __init__(self, corpus, vocab_size):
        #Normalize corpus when instantiating class.
        self.corpus = self._normalize(corpus)
        # Create empty vocabulary.
        self.vocab = {}
        # Set vocabulary vocabulary size.
        self.vocab_size = vocab_size
        self.merges = {}
    def _normalize(self,corpus):
        unicode_normalized = unicodedata.normalize("NFC",corpus)
        return unicode_normalized.lower()
    
    def train(self):
        # Create initial state with all bytes from ascii table(256 total).
        initial_state = [chr(i) for i in range(256)]
        # Extend initial stat with new ids if it is not in initial_state.
        initial_state.extend(c for c in set(self.corpus) if c not in initial_state)
        # Fills vocabulary with id to char from initial state
        self.vocab ={i: char for i,char in enumerate(initial_state)}
        # Fills an inverse vocab which is usefull to tokenize corpus into ids as bellow
        self.inverse_vocab = {char: i for i,char in enumerate(initial_state)}
        # Tokenizes the corpus into the correspondent ids.
        tokenized_ids = [self.inverse_vocab[char] for char in self.corpus]

        # Now training process can start by iterating to find frequent pairs.
        for idx in range(len(self.vocab), self.vocab_size):
            pair = self.find_pairs(tokenized_ids)
            if pair is None:
                break
            tokenized_ids = self.replace_pair(pair,tokenized_ids,idx)
            self.merges[pair] = idx


        # Now we can fill the vocabulary with the tokens.
        for (p0,p1), new_id in self.merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id
        return self.vocab
        
    # Helper function to find pairs given an tokenized list.
    def find_pairs(self, tokenized_list):
        id_frequency = Counter(zip(tokenized_list,tokenized_list[1:]))
        
        if not id_frequency:
            return None 
        return max(id_frequency.items(), key=lambda x: x[1])[0]
    
    def replace_pair(self,pair,tokenized_list,new_id):
        """
        The idea here is to replace the portions with the pair
        of ids with a new id which represents the combination of both.
        """

        dq = deque(tokenized_list)
        replaced = []
        while dq:
            current = dq.popleft()
            if dq and (current,dq[0]) == pair:
                replaced.append(new_id)
                # Because we tested the next one, so it should be popped.
                dq.popleft()
            else:
                replaced.append(current)

        return replaced


if __name__ == "__main__":
    corpus1 = "É Ó ś Sketch Engine is the ultimate tool to explore how language works. Its algorithms analyze authentic texts of billions of words (text corpora) to identify instantly what is typical in language and what is rare, unusual or emerging usage. It is also designed for text analysis or text mining applications."
    corpus = "aaabbb bbaaa basaba babaaba babba"
    print("Original:")
    print(corpus)
    print("\n")
    print("Initial State:")
    tokenizer = Tokenizer(corpus,1024)

    print(tokenizer.train())
