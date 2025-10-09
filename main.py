from __future__ import annotations 

from datasets import load_dataset
from utils import * 
#from tokenizer import Tokenizer 
import tiktoken
import os 
from pos_encoding import RoPE
from MultiHeadAttention import MultiHeadAttention
from transformer import Transformer, TransformerBlock
import kagglehub
from torch.utils.data import DataLoader
import torch 
import torch.nn as nn


import torch 
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_data():
    root_path = os.getcwd()
    os.makedirs("data", exist_ok=True)
    path_to_save = f'{root_path}/data/brazilian_lit' 
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    
    print("Downloading dataset...")
    dataset_path = kagglehub.dataset_download("rtatman/brazilian-portuguese-literature-corpus")
    print("Success!")

    return dataset_path 


    
def train(
    enc, # encoder from tiktoken 
    mha_params: dict,
    vocab_size: int,
    tokenizer: Tokenizer = None,
    N: int = 8,
    batch_size: int = 8,
    max_seq_len: int = 400,
    test_size: int = 0.1,
    epochs: int = 10,
    lr=1e-4
):

    # This function is inside train, because it needs the arguments
    # if it is outside i will need to use fn_kwargs inside map function
    # which gets messy easily.
    
    def collate_fn(batch):
        # batch is list of dicts, each with 'sentence'
        sentences = [item['sentence'] for item in batch]
        
        # Tokenize all sentences in batch
        all_input_ids = []
        all_labels = []
        pad_token_id = enc.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]  # Get pad token ID
        for text in sentences:
            tokens = enc.encode(text, allowed_special={'<|endoftext|>'})
            #tokens = tokenizer.encode(text)  # Full sentence tokens (includes BOS/EOS)
            if not tokens:
                print(f"Warning: Empty token list for sentence {i}: {text}")
                tokens = [pad_token_id] * max_seq_len

            # Truncate to max_seq_len (keep BOS, drop tail if needed)
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]
            # Pad to max_seq_len
            tokens += [pad_token_id] * (max_seq_len - len(tokens))
            input_ids = tokens 
            labels = tokens[1:] + [pad_token_id]
            
            all_input_ids.append(input_ids)
            all_labels.append(labels)
        
        # Stack into tensors: (batch_size, max_seq_len)
        input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long).to(device)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long).to(device)
        
        return {"input_ids": input_ids_tensor, "labels": labels_tensor}
    



    if not isinstance(mha_params, dict):
        raise TypeError(f"Expected mha_params to be a dict, got {type(mha_params)}: {mha_params}") 
    # Get dataset 
    #dataset_path = download_data() + '/Brazilian_Portugese_Corpus'
    #print(dataset_path)
    #dataset = load_dataset("text", data_files={"train": dataset_path + "/*.txt"}, encoding="iso-8859-1")
    dataset = load_dataset("iara-project/raw_dataset_with_embeddings_bert-base-portuguese-cased-nli-assin-2")


    dataset_splits = dataset['train'].train_test_split(test_size=test_size)
    train_dataset = dataset_splits['train']
    test_dataset = dataset_splits['test']


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Setup training instances.
    model = Transformer(vocab_size=vocab_size,mha_params=mha_params,N=N,block_dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Start training loop 
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            target_labels = batch['labels']
            y_pred = model(input_ids) 
            loss = criterion(y_pred.view(-1, y_pred.size(-1)), target_labels.view(-1))
            loss.backward()
            # Maybe grad clip later 
            
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch: {epoch+1}, AVG loss: {avg_loss:.4f}")
    return model






def main():
    merges_path = 'data/merges.json'
    vocab_path = 'data/vocab.json'
    merges_path = 'data/merges.json'
    os.makedirs('data', exist_ok=True)


    #tokenizer = Tokenizer(vocab_size=1000)
    #if not os.path.exists(merges_path) and not os.path.exists(vocab_path):
        #iterator = DatasetIterator()
        #tokenizer.train_from_iterator(iterator)
        #tokenizer.save(merges_path,vocab_path)
    
    #if not tokenizer.merges or not tokenizer.vocab:
        #tokenizer.load(merges_path, vocab_path)

 
    enc = tiktoken.get_encoding("gpt2")  # ~50,257 tokens
    vocab_size = enc.n_vocab  # Use tiktoken's vocab size

    # HYPERPARAMETERS

    batch_size = 8
    d_model = 128
    max_seq_len = 400
    num_heads = 8
    dropout = 0.1
    N = 8

    #vocab_size = tokenizer.vocab_size

    
    mha_params = {
        'd_model':d_model, 
        'h': num_heads, 
        'max_seq_len': max_seq_len, 
        'dropout': dropout         
    }
    print("\n starting model training")
    
    base_model_path = "models/base_model.pt"
    os.makedirs("models", exist_ok=True)
    model = Transformer(vocab_size=vocab_size, mha_params=mha_params, N=N, block_dropout=0.2).to(device)
    if not os.path.exists(base_model_path):
        model = train(enc,
            mha_params = mha_params,
            vocab_size = vocab_size,
            batch_size = 8,
            max_seq_len = 400,
            test_size = 0.1,
            epochs = 10,
            lr=1e-4
        )
        torch.save(model.state_dict(), base_model_path)
    else:
        state_dict = torch.load(base_model_path, weights_only=True)
        model.load_state_dict(state_dict)
   
    print("Testing model...")
    
    prompt = "Brasil"
    input_ids = tokenizer.encode(prompt)
    # Pad to max_seq_len
    input_ids = input_ids + [tokenizer.pad_id] * (max_seq_len - len(input_ids))
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)  # Shape: [1, 400]

    generated_tokens = input_ids[0].tolist()  # Start with prompt tokens
    max_new_tokens = 200

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)[:, -1, :]  # Logits for last token: [1, vocab_size]
            logits[:, tokenizer.pad_id] = float('-inf')
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1).item()  # Get highest-probability token

            generated_tokens.append(next_token)
            # Update input_ids with new token
            input_ids = torch.tensor(generated_tokens[-max_seq_len:], dtype=torch.long).unsqueeze(0).to(device)
    
    generated_text = tokenizer.decode(generated_tokens)
    print(f"Generated text: {generated_text}")

   

if __name__ == "__main__":
    main()
