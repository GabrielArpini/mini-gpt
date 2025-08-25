from __future__ import annotations


import re
import unicodedata
from collections import Counter,deque
import json
import os 


class Tokenizer:
    def __init__(self, corpus=None, vocab_size=None):
        # TODO: CHECK IF VOCAB JSON AND MERGES EXISTS
        #Normalize corpus when instantiating class.
        if corpus is not None:
            self.corpus = self._normalize(corpus)
        # Create empty vocabulary.
        self.vocab = {}
        # Set vocabulary vocabulary size.
        if vocab_size is not None:
            self.vocab_size = vocab_size
        self.merges = {}
    def _normalize(self,corpus):
        # Normalize the bytes of the corpus by similarity.
        # For example:
        # e = é
        unicode_normalized = unicodedata.normalize("NFC",corpus)
        # Returns the lowered version of the corpus
        # and encodes it into bytes
        # order of operations matters here 
        return unicode_normalized.lower().encode(encoding="utf-8")  
    
    def train(self):
        print("Starting tokenizer training...")
        # Create initial state with all bytes from ascii table(256 total).
        initial_state = [bytes([i]) for i in range(256)]
        # Fills vocabulary with id to char from initial state
        self.vocab ={i: char for i,char in enumerate(initial_state)}
        # Fills an inverse vocab which is usefull to tokenize corpus into ids as bellow
        self.inverse_vocab = {char: i for i,char in enumerate(initial_state)}
        # Tokenizes the corpus into the correspondent ids.
        tokenized_ids = [i for i in self.corpus]

        # Now training process can start by iterating to find frequent pairs.
        # The idx(id to be assigned) starts from the end of the initial_state size(256)
        # towards the defined vocab_size, so essentially it learns len(vocab_size) - len(initial_state) combinations.
        for idx in range(len(self.vocab), self.vocab_size):
            # I think the functions and operations bellow explains themselves.
            pair = self.find_pairs(tokenized_ids)
            if pair is None:
                break
            tokenized_ids = self.replace_pair(pair,tokenized_ids,idx)
            self.merges[pair] = idx


        # Now we can fill the vocabulary with the tokens.
        for (p0,p1), new_id in self.merges.items():
            # Concat the bytes 
            merged_token = self.vocab[p0] + self.vocab[p1]
            # Adds the merged token and respectively id into the vocabulary
            self.vocab[new_id] = merged_token
            # Adds the reverse of the above as well. 
            self.inverse_vocab[merged_token] = new_id
        
        print("Tokenizer successfully trained!")
        
    # Helper function to find pairs given an tokenized list.
    def find_pairs(self, tokenized_list):
        # Uses Counter to get frequency of ids by comparing pairs.
        id_frequency = Counter(zip(tokenized_list,tokenized_list[1:]))

        if not id_frequency:
            return None
        # Returns the pair with most frequency. 
        return max(id_frequency.items(), key=lambda x: x[1])[0]
    
    def replace_pair(self,pair,tokenized_list,new_id):
        """
        The idea here is to replace the portions with the pair
        of ids with a new id which represents the combination of both.
        """
        # uses deque because of popleft function 
        dq = deque(tokenized_list)
        # Initial list of tokenized ids.
        replaced = []
        while dq:
            # Removes first id 
            current = dq.popleft()
            # Checks if dq still exists(otherwise it will raise an error when indexing 0)
            # Then checks if its equal to the inputed pair.
            if dq and (current,dq[0]) == pair:
                # If true it will append the new id into the replaced list.
                replaced.append(new_id)
                # Because we tested the next one, so it should be popped.
                dq.popleft()
            else:
                # Otherwise it will just append current value
                replaced.append(current)

        return replaced
    def encode(self,text):
        # TODO: APPLY TESTS 
        # First it needs to normalize text 
        normalized_text = self._normalize(text)

        # Create list of bytes 
        list_bytes = [i for i in normalized_text]
        # Iterate the merges cheatsheet
        for pair, id in self.merges.items():
            # Use replce_pair to replace the pairs from pairs in merges dict 
            replaced_pairs = self.replace_pair(pair,list_bytes,id)
            # Updates list_bytes so next iteration uses an updated version.
            list_bytes = replaced_pairs
        return list_bytes
    
    def decode(self,tokenized_ids):
        decoded_bytes = [self.vocab[item] for item in tokenized_ids]
        decoded_str = "".join([i.decode("UTF-8") for i in decoded_bytes])
        return decoded_str

    def save(self, merges_path, vocab_path):
        # Json doesnt save bytes or sets, so we need to address that first
        if not os.path.exists(merges_path):
            with open(merges_path, "w", encoding="utf-8") as f:
                merges_list = [{"pair": list(pair), "id": id} for pair,id in self.merges.items()]
                json.dump(merges_list, f)
        else:
            print("Specified merge path already exists, skipping...")
        if not os.path.exits(vocab_path):
            with open(vocab_path, "w", encoding="utf-8") as f:
                vocab_list = [{"id": id, "merged_token": [token for token in tokens]} for id, tokens in self.vocab.items()]
                json.dump(vocab_list, f)
        else:
            print("Specified vocab path already exists, skipping...")

        
    def load(self, merges_path, vocab_path):
        # Loads the json from specified path.
        
        if os.path.exists(merges_path) and os.path.exists(vocab_path):
            with open(merges_path, "r", encoding="utf-8") as f:
                loaded_merges = json.load(f)
                for item in loaded_merges:
                    pair_list = item["pair"]
                    item_id = item["id"]
                    self.merges[tuple(pair_list)] = int(item_id)
            
        
            with open(vocab_path, "r", encoding="utf-8") as f:
                loaded_merges = json.load(f)
                for item in loaded_merges:
                    item_id = item["id"]
                    merged_t_id = item["merged_token"]
                    self.vocab[item_id] = bytearray(merged_t_token)
                    self.inverse_vocab[bytearray(merged_t_token)] = item_id
        else:
            print("Provided path(s) doesnt exists.")
    
            

if __name__ == "__main__":
    corpus1 = "É Ó ś Sketch Engine is the ultimate tool to explore how language works. Its algorithms analyze authentic texts of billions of words (text corpora) to identify instantly what is typical in language and what is rare, unusual or emerging usage. It is also designed for text analysis or text mining applications."
    corpus = "aaabbb bbaaa basaba babaaba babba"
    print("Original:")
    #print(corpus1)
    #print("\n")
    #print("Initial State:")
    tokenizer = Tokenizer(corpus1,350)
    tokenized_example = tokenizer.encode("Alahu akbarrrrrrr")
    print("tokenized_example")
    print(tokenized_example)
    tokenizer.train()
    #print(tokenizer.train())
    decoded_result = tokenizer.decode(tokenized_example)
    print("decoded result:")
    print(decoded_result)
    tokenizer.save("data/merges.json", "data/vocabulary.json")
    
