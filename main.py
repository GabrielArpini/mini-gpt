from __future__ import annotations 

from datasets import load_dataset
from utils import * 
import tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
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
import torch.nn.functional as F 
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




    
def train(
    #enc, # encoder from tiktoken 
    mha_params: dict,
    vocab_size: int,
    tokenizer: tokenizers.Tokenizer = None,
    N: int = 8,
    batch_size: int = 8,
    max_seq_len: int = 400,
    test_size: int = 0.1,
    epochs: int = 50,
    lr=1e-4,
    checkpoint=False 
):

    # This function is inside train, because it needs the arguments
    # if it is outside i will need to use fn_kwargs inside map function
    # which gets messy easily.
    
    def collate_fn(batch):
        sentences = [item['text'] for item in batch if item['text'].strip()]
        
        # Tokenize all sentences in batch
        all_input_ids = []
        all_labels = []
        #pad_token_id = enc.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]  # Get pad token ID tiktoken 
        
        for i,text in enumerate(sentences):
            #tokens = enc.encode(text, allowed_special={'<|endoftext|>'}) #tiktoken 
            tokens = tokenizer.encode(text).ids  # Full sentence tokens (includes BOS/EOS)
            if not tokens:
                print(f"Warning: Empty token list for sentence {i}: {text}")
                tokens = [tokenizer.token_to_id("[PAD]")] * max_seq_len
                #tokens = [pad_token_id] * max_seq_len # tiktoken 

            # Truncate to max_seq_len (keep BOS, drop tail if needed)
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]
            # Pad to max_seq_len
            tokens += [tokenizer.token_to_id("[PAD]")] * (max_seq_len - len(tokens))

            #tokens += [pad_token_id] * (max_seq_len - len(tokens)) #tiktoken 

            input_ids = tokens 
            #labels = tokens[1:] + [pad_token_id]
            labels = tokens[1:] + [tokenizer.token_to_id("[PAD]")]
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

    #dataset = load_dataset("iara-project/raw_dataset_with_embeddings_bert-base-portuguese-cased-nli-assin-2")
    try:
        dataset = load_dataset("text",data_files="data/machado_texts.txt", encoding="utf-8")
    except Exception as e:
        print(f"Error while loading dataset: {e}")
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
        if checkpoint:
            torch.save(model.state_dict(), f"models/checkpoints/epoch_{epoch}.pt")
    return model



def temperature_sampling(logits, temperature):
    #P(token) = (e^{logits/T}) / sum{e^{logits/T}}
    scaled_logits = logits / temperature
    return F.softmax(scaled_logits,dim=-1)

def main():
    merges_path = 'data/merges.json'
    vocab_path = 'data/vocab.json'
    os.makedirs('data', exist_ok=True)

    # Initialize Hugging Face tokenizers BPE
    

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
    if not os.path.exists(merges_path) or not os.path.exists(vocab_path):
        iterator = DatasetIterator()
        print("Starting tokenizer training...")
        trainer = BpeTrainer(vocab_size=20000, special_tokens=special_tokens)
        tokenizer.train_from_iterator(
            (item['sentence'] if isinstance(item, dict) and 'sentence' in item else item for item in iterator),
            trainer=trainer
        )
        tokenizer.save(vocab_path)
        print("Tokenizer successfully trained!")
    
    if not tokenizer.get_vocab():
        tokenizer = Tokenizer.from_file(vocab_path)

    vocab_size = tokenizer.get_vocab_size()
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
    
    base_model_path = "models/base_model.pt"
    os.makedirs("models", exist_ok=True)
    model = Transformer(vocab_size=vocab_size, mha_params=mha_params, N=N, block_dropout=0.2).to(device)
    if not os.path.exists(base_model_path):
        print("\n starting model training")

        model = train(
            mha_params = mha_params,
            vocab_size = vocab_size,
            tokenizer = tokenizer,
            batch_size = 8,
            max_seq_len = 400,
            test_size = 0.1,
            epochs = 25,
            lr=1e-4,
            checkpoint=True
        )
        torch.save(model.state_dict(), base_model_path)
    else:
        if torch.cuda.is_available():
            state_dict = torch.load(base_model_path, weights_only=True)
        else:
            state_dict = torch.load(base_model_path,weights_only=True,map_location=torch.device('cpu'))

        model.load_state_dict(state_dict)
   
    print("Testing model...")
    
    prompt = "Brasil"
    #input_ids = enc.encode(prompt)
    input_ids = tokenizer.encode(prompt).ids

    #pad_id = enc.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]  # Get pad token ID

    # Pad to max_seq_len
    #input_ids = input_ids + [pad_id] * (max_seq_len - len(input_ids))
    input_ids = input_ids + [tokenizer.token_to_id("[PAD]")] * (max_seq_len - len(input_ids))
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)  # Shape: [1, 400]

    generated_tokens = input_ids[0].tolist()  # Start with prompt tokens
    max_new_tokens = 200

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)[:, -1, :]  # Logits for last token: [1, vocab_size]
            #logits[:, pad_id] = float('-inf')
            logits[:,tokenizer.token_to_id("[PAD]")] = float('-inf')

            probs = temperature_sampling(logits, temperature=0.7)
            next_token = torch.multinomial(probs,num_samples=1) # Random sampling.
            next_token = next_token.item()

            generated_tokens.append(next_token)
            # Update input_ids with new token
            input_ids = torch.tensor(generated_tokens[-max_seq_len:], dtype=torch.long).unsqueeze(0).to(device)
    
    #generated_text = enc.decode(generated_tokens)
    generated_text = tokenizer.decode(generated_tokens)
    print(f"Generated text: {generated_text}")

   

if __name__ == "__main__":
    main()
