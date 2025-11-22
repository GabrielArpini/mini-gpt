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
from torch.amp import autocast, GradScaler
import glob
import re

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

@dataclass
class HyperParameters:
    N: int = 8
    batch_size: int = 32  # Increased from 6 for more stable gradients
    max_seq_len: int = 512
    test_size: float = 0.1
    epochs: int = 30
    lr: float = 3e-4  # Increased from 1e-4 (standard for AdamW with transformers)
    checkpoint: bool = True
    d_model: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    vocab_size: int = 20_000
    max_new_tokens: int = 300
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95 

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
    mha_params: dict,
    vocab_size: int,
    tokenizer: tokenizers.Tokenizer = None,
    N: int = 8,
    batch_size: int = 8,
    max_seq_len: int = 1024,
    test_size = 0.1,
    epochs: int = 20,
    lr=1e-4,
    checkpoint=False,
    start_epoch: int = 0,
    initial_model = None
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
        max_chunks = batch_size  # Limit total chunks to batch_size to control memory

        for item in batch:
            if len(all_input_ids) >= max_chunks:
                break

            try:
                text = item['text']
                if not isinstance(text, str):
                    continue
                tokens = tokenizer.encode(text).ids
                # - 2 from eos and bos, that should not be included.
                chunks = split_into_chunks(tokens, chunk_size=512 - 2 , overlap=50)

                # Only take first chunk from each text to avoid memory explosion
                for chunk in chunks[:1]:
                    chunk_specials = [bos_id] + chunk + [eos_id]
                    if len(chunk_specials) < max_seq_len:
                        pad_quantity = max_seq_len - len(chunk_specials)
                        chunk_specials += [pad_id] * pad_quantity

                    input_ids = chunk_specials[:-1]
                    labels = chunk_specials[1:]

                    all_input_ids.append(input_ids)
                    all_labels.append(labels)
            except (UnicodeDecodeError, UnicodeError, Exception):
                # Skip corrupted examples
                continue

        if len(all_input_ids) == 0:
            # Return dummy batch if all examples were corrupted
            dummy = [pad_id] * max_seq_len
            all_input_ids = [dummy]
            all_labels = [dummy]

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
        print("Loading C4 dataset with streaming...")
        num_train = 5_000_000
        num_test = 100_000

        def is_valid_example(example):
            try:
                text = example.get('text', '')
                if not isinstance(text, str) or len(text) == 0:
                    return False
                text.encode('utf-8', errors='strict')
                return True
            except (UnicodeDecodeError, UnicodeError, Exception):
                return False

        train_dataset = load_dataset(
            "allenai/c4",
            "en",
            split='train',
            streaming=True,
            trust_remote_code=False
        ).filter(is_valid_example).shuffle(seed=42, buffer_size=10000).take(num_train)

        test_dataset = load_dataset(
            "allenai/c4",
            "en",
            split='validation',
            streaming=True,
            trust_remote_code=False
        ).filter(is_valid_example).take(num_test)

        print(f"Loaded streaming corpus: train={num_train:,}, test={num_test:,}")

    except Exception as e:
        print(f"Error: {e}")
        return None

    #dataset_splits = dataset['train'].train_test_split(test_size=test_size)
    #train_dataset = dataset['train']
    #test_dataset = dataset['test']


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    if initial_model is not None:
        model = initial_model
    else:
        model = Transformer(vocab_size=vocab_size,mha_params=mha_params,N=N,block_dropout=0.2).to(device)
        print("Compiling model...")
        model = torch.compile(model, mode='default')
        print("Finished.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    pad_id = tokenizer.token_to_id("[PAD]")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    os.makedirs('models/checkpoints',exist_ok=True)
    now = datetime.now()

    scaler = GradScaler('cuda')

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()

            with autocast('cuda'):
                input_ids = batch['input_ids'].to(device)
                target_labels = batch['labels']
                y_pred = model(input_ids)
                loss = criterion(y_pred.view(-1, y_pred.size(-1)), target_labels.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        if checkpoint:
            torch.save(model.state_dict(), f"models/checkpoints/{now.year}_{now.month}_{now.day}_epoch_{epoch}.pt")
        
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                target_labels = batch['labels'].to(device)
                y_pred = model(input_ids)
                loss = criterion(y_pred.view(-1, y_pred.size(-1)), target_labels.view(-1))
                total_val_loss += loss.item()
                num_val_batches += 1
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
        print(f"Epoch: {epoch+1}, AVG loss: {avg_loss:.4f} AVG Val loss: {avg_val_loss} Perplexity: {perplexity}")
    return model



def temperature_sampling(logits, temperature):
    scaled_logits = logits / temperature
    return F.softmax(scaled_logits, dim=-1)

def top_k_filtering(logits, top_k):
    if top_k > 0:
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_value = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_value, torch.tensor(float('-inf')).to(logits.device), logits)
    return logits

def top_p_filtering(logits, top_p):
    if top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    return logits

def main():

    # HYPERPARAMETERS
    hp = HyperParameters()
    batch_size = hp.batch_size
    d_model = hp.d_model 
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
    top_k = hp.top_k
    top_p = hp.top_p


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
        tokenizer.train_from_iterator(iterator, trainer=trainer)
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
    os.makedirs("models/checkpoints", exist_ok=True)

    start_epoch = 0
    model = Transformer(vocab_size=vocab_size, mha_params=mha_params, N=N, block_dropout=0.2).to(device)

    if not os.path.exists(base_model_path):
        checkpoint_files = glob.glob("models/checkpoints/*_epoch_*.pt")
        if checkpoint_files:
            epoch_numbers = []
            for f in checkpoint_files:
                match = re.search(r'epoch_(\d+)\.pt$', f)
                if match:
                    epoch_numbers.append((int(match.group(1)), f))

            if epoch_numbers:
                latest_epoch, latest_checkpoint = max(epoch_numbers, key=lambda x: x[0])
                print(f"Resuming from checkpoint: {latest_checkpoint} (epoch {latest_epoch})")

                if torch.cuda.is_available():
                    state_dict = torch.load(latest_checkpoint, weights_only=True)
                else:
                    state_dict = torch.load(latest_checkpoint, weights_only=True, map_location=torch.device('cpu'))

                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                start_epoch = latest_epoch + 1

        if start_epoch < epochs:
            print(f"\nStarting training from epoch {start_epoch}")
            torch.cuda.empty_cache()
            print("Compiling model...")
            model = torch.compile(model, mode='default')
            print("Finished.")

            model = train(
                mha_params=mha_params,
                vocab_size=vocab_size,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                test_size=test_size,
                epochs=epochs,
                lr=lr,
                checkpoint=checkpoint,
                start_epoch=start_epoch,
                initial_model=model
            )
            if model is None:
                print("Training failed. Model could not be created. Exiting.")
                return
            torch.save(model.state_dict(), base_model_path)
        else:
            print("Training already completed")
    else:
        if torch.cuda.is_available():
            state_dict = torch.load(base_model_path, weights_only=True)
        else:
            state_dict = torch.load(base_model_path, weights_only=True, map_location=torch.device('cpu'))
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
   
    print("Testing model...")
    
    prompt = "The future of artificial intelligence"
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
            logits = model(input_ids)[:, -1, :]
            logits[:, tokenizer.token_to_id("[PAD]")] = float('-inf')

            logits = top_k_filtering(logits, top_k)
            logits = top_p_filtering(logits, top_p)
            probs = temperature_sampling(logits, temperature=temperature)

            next_token = torch.multinomial(probs, num_samples=1)
            next_token = next_token.item()

            generated_tokens.append(next_token)
            input_ids = torch.tensor(generated_tokens[-max_seq_len:], dtype=torch.long).unsqueeze(0).to(device)
    
    #generated_text = enc.decode(generated_tokens)
    generated_text = tokenizer.decode(generated_tokens)
    print(f"Generated text: {generated_text}")

   

if __name__ == "__main__":
    main()
