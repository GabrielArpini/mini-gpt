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

from dataclasses import dataclass

from datetime import datetime

import torch
import torch.nn.functional as F 
from torch.cuda.amp import autocast, GradScaler

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass 
class HyperParameters:
    N: int = 8
    batch_size: int = 8
    max_seq_len: int = 512
    test_size: float = 0.1 
    epochs: int = 20 
    lr: float = 1e-4 
    checkpoint: bool = True
    d_model: int = 128
    num_heads: int = 8
    dropout: float = 0.1 
    vocab_size: int = 20_000
    max_new_tokens: int = 300 
    temperature: float = 0.7 # 0 <= x <= 1 

def split_into_chunks(tokens, chunk_size=512, overlap=50):
    chunks = []
    current = 0 
    step = chunk_size - overlap
    while current < len(tokens):
        chunk = tokens[current: current + chunk_size]
        chunks.append(chunk)
        current += step 
    return chunks 
    
def train(
    #enc, # encoder from tiktoken 
    mha_params: dict,
    vocab_size: int,
    tokenizer: tokenizers.Tokenizer = None,
    N: int = 8,
    batch_size: int = 8,
    max_seq_len: int = 7000,
    test_size = 0.1,
    epochs: int = 20,
    lr=1e-4,
    checkpoint=False 
):

    # This function is inside train, because it needs the arguments
    # if it is outside i will need to use fn_kwargs inside map function
    # which gets messy easily.
    


    def collate_fn(batch):
        bos_id = tokenizer.token_to_id("[BOS]")
        eos_id = tokenizer.token_to_id("[EOS]")
        pad_id = tokenizer.token_to_id("[PAD]")
        all_input_ids = []
        all_labels = []

        for item in batch:
            text = item['text']
            tokens = tokenizer.encode(text).ids
            # - 2 from eos and bos, that should not be included.
            chunks = split_into_chunks(tokens, chunk_size=512 - 2 , overlap=50)
            for chunk in chunks:
                chunk_specials = [bos_id] + chunk + [eos_id] 
                if len(chunk_specials) < max_seq_len:
                    pad_quantity = max_seq_len - len(chunk_specials)
                    chunk_specials += [pad_id] * pad_quantity

                input_ids = chunk_specials[:-1]
                labels = chunk_specials[1:]
            
                all_input_ids.append(input_ids)
                all_labels.append(labels)
        input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long).to(device)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long).to(device)

        return {
            "input_ids": input_ids_tensor,
            "labels": labels_tensor
        }
                    




    if not isinstance(mha_params, dict):
        raise TypeError(f"Expected mha_params to be a dict, got {type(mha_params)}: {mha_params}") 
    # Get dataset 
    #dataset_path = download_data() + '/Brazilian_Portugese_Corpus'
    #print(dataset_path)

    #dataset = load_dataset("iara-project/raw_dataset_with_embeddings_bert-base-portuguese-cased-nli-assin-2")
    #try:
    #   dataset = load_dataset("text",data_files="data/machado_texts.txt", encoding="utf-8")
    #except Exception as e:
    #   print(f"Error while loading dataset: {e}")
    try:
        train_dataset, test_dataset = load_dataset("dominguesm/wikipedia-ptbr-20230601", split=['train[:60%]', 'test[:60%]']) 
        #print(f"Loaded Wikipedia dataset: {len(dataset)} examples")
    except Exception as e:
        print(f"Error while loading dataset: {e}")
        return None


    #dataset_splits = dataset['train'].train_test_split(test_size=test_size)
    #train_dataset = dataset['train']
    #test_dataset = dataset['test']


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Setup training instances.
    model = Transformer(vocab_size=vocab_size,mha_params=mha_params,N=N,block_dropout=0.2).to(device)
    print("Compiling model...")
    model = torch.compile(model, mode='default')
    print("Finished.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    pad_id = tokenizer.token_to_id("[PAD]")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id) # Ignores pad_id when calculatin loss. 
    os.makedirs('models/checkpoints',exist_ok=True)
    # Start training loop 
    # Save now before training so every checkpoint can have the same exact time 
    now = datetime.now()
    
    # Optimization with mixed precision 
    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            # For mixed precision 
            with autocast():

                input_ids = batch['input_ids'].to(device)
                target_labels = batch['labels']
                y_pred = model(input_ids) 
                loss = criterion(y_pred.view(-1, y_pred.size(-1)), target_labels.view(-1))
            
            # For mixed precision 
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        if checkpoint:
            torch.save(model.state_dict(), f"models/checkpoints/{now.year}_{now.month}_{now.day}_epoch_{epoch}.pt")
        
        # Evaluation of epoch 
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                target_labels = batch['labels'].to(device)
                y_pred = model(input_ids)
                loss = criterion(y_pred.view(-1, y_pred.size(-1)), target_labels.view(-1))
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(test_loader)
        perplexity = torch.exp(torch.tensor(avh_val_loss)).item()
        print(f"Epoch: {epoch+1}, AVG loss: {avg_loss:.4f} AVG Val loss: {avg_val_loss} Perplexity: {perplexity}")
    return model



def temperature_sampling(logits, temperature):
    #P(token) = (e^{logits/T}) / sum{e^{logits/T}}
    scaled_logits = logits / temperature
    return F.softmax(scaled_logits,dim=-1)

def main():

    # HYPERPARAMETERS
    hp = HyperParameters()
    batch_size = hp.batch_size
    d_model = hp.batch_size 
    max_seq_len = hp.max_seq_len 
    num_heads = hp.num_heads
    dropout = hp.dropout 
    N = hp.N 
    lr = hp.lr 
    test_size = hp.test_size 
    epochs = hp.epochs 
    checkpoint = hp.checkpoint 
    vocab_size = hp.vocab_size
    max_new_tokens = hp.max_new_tokens 
    temperature = hp.temperature 


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
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
        tokenizer.train_from_iterator(
            (item['sentence'] if isinstance(item, dict) and 'sentence' in item else item for item in iterator),
            trainer=trainer
        )
        tokenizer.save(vocab_path)
        print("Tokenizer successfully trained!")
    
    if not tokenizer.get_vocab():
        tokenizer = Tokenizer.from_file(vocab_path)

    vocab_size = tokenizer.get_vocab_size()
    
    


    
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
            batch_size = batch_size,
            max_seq_len = max_seq_len,
            test_size = test_size,
            epochs = epochs,
            lr=lr,
            checkpoint=checkpoint
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


    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)[:, -1, :]  # Logits for last token: [1, vocab_size]
            #logits[:, pad_id] = float('-inf')
            logits[:,tokenizer.token_to_id("[PAD]")] = float('-inf')

            probs = temperature_sampling(logits, temperature=temperature)
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
