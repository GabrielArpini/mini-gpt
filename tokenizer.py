from __future__ import annotations


import re
import unicodedata
from collections import Counter,deque
import json
import os 
from typing import List, Tuple, Optional
import regex as re
import time 


class Tokenizer:
    def __init__(self, corpus=None, vocab_size=None) -> None:
        #Normalize corpus when instantiating class.
        if corpus is not None:
            self.corpus = self._normalize(corpus)
        # Create empty vocabulary.
        self.vocab = {}
        self.inverse_vocab = {}
        # Set vocabulary vocabulary size.
        if vocab_size is not None:
            self.vocab_size = vocab_size
        self.merges = {}

        # Special tokens 
        self.special_tokens = ['[PAD]','[BOS]','[EOS]']
        self.encoded_specials = [i.encode('utf-8') for i in self.special_tokens]
        self.pad = self.encoded_specials[0]
        self.bos = self.encoded_specials[1]
        self.eos = self.encoded_specials[2]

    def _normalize(self,corpus: str) -> List[int]:   
        """
        Normalizes the input corpus and returns a list of byte integers. 
        """
        # Normalize the bytes of the corpus by similarity.
        # For example:
        # e = é
        unicode_normalized = unicodedata.normalize("NFC",corpus).lower()

        #regex pattern from GPT-2
        pattern = re.compile(
                r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
        )
        # Applies the pattern into the corpus
        # returns: ["I","love","cats"]
        token_list = re.findall(pattern,unicode_normalized)
        # Joins the tokens into a continuous byte object 
        joined_bytes = b''.join(t.encode(encoding="utf-8",errors='replace') for t in token_list)

        # Unpack the byte object into an list of integers
        return list(joined_bytes)
    
    def train(self) -> None:
        print("Starting tokenizer training...")
        # Create initial state with all bytes from ascii table(256 total).
        initial_state = [bytes([i]) for i in range(256)]

         

        # Fills vocabulary with id to char from initial state
        self.vocab ={i: char for i,char in enumerate(initial_state)}
        # Add specials 
        special_tokens_vocab = {i+256: special for i, special in enumerate(self.encoded_specials)}
        special_tokens_vocab_inverse = {special: i+256 for i, special in enumerate(self.encoded_specials)}

        self.vocab.update(special_tokens_vocab)


        # Fills an inverse vocab which is useful to tokenize corpus into ids as bellow
        self.inverse_vocab = {char: i for i,char in enumerate(initial_state)}
        self.inverse_vocab.update(special_tokens_vocab_inverse)

        self.pad_id = self.inverse_vocab[self.pad]
        self.eos_id = self.inverse_vocab[self.eos]
        self.bos_id = self.inverse_vocab[self.bos]
        # Tokenizes the corpus into the correspondent ids.
        tokenized_ids = [i for i in self.corpus]

        # Now training process can start by iterating to find frequent pairs.
        # The idx(id to be assigned) starts from the end of the initial_state size(256)
        # towards the defined vocab_size, so essentially it learns len(vocab_size) - len(initial_state) combinations.
        for idx in range(len(initial_state) + len(self.encoded_specials), self.vocab_size):
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
    

    def train_from_iterator(self, iterator, verbose=False):
        #TODO: implement verbose prints and fix time error. 
        print("Starting training...")
        start = time.time()
        initial_state = [bytes([i]) for i in range(256)]

        # Fills vocabulary with id to char from initial state
        self.vocab ={i: char for i,char in enumerate(initial_state)}
        # Add specials 
        special_tokens_vocab = {i+256: special for i, special in enumerate(self.encoded_specials)}
        special_tokens_vocab_inverse = {special: i+256 for i, special in enumerate(self.encoded_specials)}

        self.vocab.update(special_tokens_vocab)


        # Fills an inverse vocab which is useful to tokenize corpus into ids as bellow
        self.inverse_vocab = {char: i for i,char in enumerate(initial_state)}
        self.inverse_vocab.update(special_tokens_vocab_inverse)
        
        self.pad_id = self.inverse_vocab[self.pad]
        self.eos_id = self.inverse_vocab[self.eos]
        self.bos_id = self.inverse_vocab[self.bos]
        
        for i in range(len(initial_state) + len(self.encoded_specials), self.vocab_size):
            _count = Counter()
            for chunk in iterator:
                if not isinstance(chunk, dict) or 'text' not in chunk:
                    raise ValueError("Iterator must yield dictionaries with a 'text' key")
                chunk = chunk['text']
                chunk_normalized = self._normalize(chunk)
                # Updates list of bytes with merges 
                for pair, id in self.merges.items():
                    if pair and id:
                        # Use replce_pair to replace the pairs from pairs in merges dict 
                        replaced_pairs = self.replace_pair(pair,chunk_normalized,id)
                        # Updates list_bytes so next iteration uses an updated version.
                        chunk_normalized = replaced_pairs
                
                # Counts the frequency of each pair 
                _count.update(zip(chunk_normalized,chunk_normalized[1:]))
                
            most_frequent_pair = _count.most_common()[0][0]
            self.merges[most_frequent_pair] = i 
            item_1 = self.vocab[most_frequent_pair[0]]
            item_2 = self.vocab[most_frequent_pair[1]]
        
            merged_token = item_1 + item_2

            self.vocab[i] = merged_token
            self.inverse_vocab[merged_token] = i 

            total_merges = self.vocab_size - 256
            merges_done = i - 256 + 1 
            elapsed_time = time.time()-start
            avg_time_per_merge = elapsed_time / merges_done 
            print(f"{merges_done} merges done in {elapsed_time:.2f}s. Remaining: {total_merges - merges_done}. Time prediction: {avg_time_per_merge*(total_merges-merges_done)}")
            
        end = time.time()
        print(f"Training completed in {end-start} seconds.")

            

                

                
                
                



    # Helper function to find pairs given an tokenized list.
    def find_pairs(self, tokenized_list: List) -> Optional[Tuple[int, int]]:
        # Uses Counter to get frequency of ids by comparing pairs.
        id_frequency = Counter(zip(tokenized_list,tokenized_list[1:]))

        if not id_frequency:
            return None
        # Returns the pair with most frequency. 
        return max(id_frequency.items(), key=lambda x: x[1])[0]
    
    def replace_pair(self, pair: tuple, tokenized_list: List, new_id: int) -> List:
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
    def encode(self,text: str) -> List[int]:

        # First it needs to normalize text 
        normalized_text = self._normalize(text)

        # Create list of bytes 
        list_bytes = normalized_text 
        # Iterate the merges cheatsheet
        for pair, id in self.merges.items():
            # Use replce_pair to replace the pairs from pairs in merges dict 
            replaced_pairs = self.replace_pair(pair,list_bytes,id)
            # Updates list_bytes so next iteration uses an updated version.
            list_bytes = replaced_pairs

        # Add EOS and BOS 
        out_list = [self.bos_id]
        out_list.extend(list_bytes)
        out_list.append(self.eos_id)
        
        return out_list 
    
    def decode(self,tokenized_ids: List) -> str:
        special_tokens = [self.eos_id, self.bos_id, self.pad_id]
        decoded_bytes = [self.vocab[item] for item in tokenized_ids if item not in special_tokens]
        decoded_str = "".join([i.decode("UTF-8",errors='replace') for i in decoded_bytes])
        


        return decoded_str

    def save(self, merges_path: str, vocab_path: str) -> None:
        # Json doesnt save bytes or sets, so we need to address that first
        if not os.path.exists(merges_path):
            with open(merges_path, "w", encoding="utf-8") as f:
                print("Saving merges state...")
                merges_list = [{"pair": list(pair), "id": id} for pair,id in self.merges.items()]
                json.dump(merges_list, f)
        else:
            print("Specified merge path already exists, skipping...")
        if not os.path.exists(vocab_path):
            with open(vocab_path, "w", encoding="utf-8") as f:
                print("Saving vocabulary state...")
                vocab_list = [{"id": id, "merged_token": [token for token in tokens]} for id, tokens in self.vocab.items()]
                json.dump(vocab_list, f)
        else:
            print("Specified vocab path already exists, skipping...")
        
        
    def load(self, merges_path:str, vocab_path: str) -> None:
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
                    self.vocab[item_id] = bytes(merged_t_id)
                    self.inverse_vocab[bytes(merged_t_id)] = item_id
                    
                self.pad_id = self.inverse_vocab[self.pad]
                self.eos_id = self.inverse_vocab[self.eos]
                self.bos_id = self.inverse_vocab[self.bos]
        else:
            print("Provided path(s) doesnt exists.")
    
            

if __name__ == "__main__":
    # Corpus to train the tokenizer
    training_corpus = "Quantum decoherence explains why a system interacting with an environment transitions from being a pure state, exhibiting superpositions, to a mixed state, an incoherent combination of classical alternatives.[14] This transition is fundamentally reversible, as the combined state of system and environment is still pure, but for all practical purposes irreversible in the same sense as in the second law of thermodynamics: the environment is a very large and complex quantum system, and it is not feasible to reverse their interaction. Decoherence is thus very important for explaining the classical limit of quantum mechanics, but cannot explain wave function collapse, as all classical alternatives are still present in the mixed state, and wave function collapse selects only one of them."
    
    # Text we want to encode and then decode
    text_to_process = "Von Neumann's mathematical analysis of the structure of self-replication preceded the discovery of the structure of DNA."
    
    tokenizer = Tokenizer(corpus=training_corpus, vocab_size=350)
    tokenizer.train()
    print("-" * 50)
    
    print(f"Encoding the text: '{text_to_process}'")
    tokenized_ids = tokenizer.encode(text_to_process)
    print("Encoded Token IDs:")
    print(tokenized_ids)
    print(f"(Length: {len(tokenized_ids)} tokens)")
    print("-" * 50)

    print("Decoding the token IDs to text...")
    decoded_text = tokenizer.decode(tokenized_ids)
    print("Decoded Result:")
    print(decoded_text)
    print("-" * 50)

   
    tokenizer.save("data/merges.json", "data/vocabulary.json")
    print("\n" + "-" * 50)



