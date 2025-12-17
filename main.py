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

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from time import time


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizers parallelism to avoid fork issues

@dataclass
class HyperParameters:
    N: int = 8
    batch_size: int = 8
    accumulation_steps: int = 4
    max_seq_len: int = 512
    test_size: float = 0.1
    epochs: int = 10  
    lr: float = 6e-4  # Slightly increased from 3e-4, scheduler will handle warmup
    checkpoint: bool = False
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
    epochs: int = 10,
    lr=1e-4,
    checkpoint=False,
    start_epoch: int = 0,
    initial_model = None,
    accumulation_steps: int = 1
):

    # Simplified collate function for pre-tokenized data
    def collate_fn(batch):
        pad_id = tokenizer.token_to_id("[PAD]")
        all_input_ids = []
        all_labels = []

        for item in batch:
            input_ids = item['input_ids']
            labels = item['labels']

            # Pad to max_seq_len if needed
            if len(input_ids) < max_seq_len:
                pad_quantity = max_seq_len - len(input_ids)
                input_ids = input_ids + [pad_id] * pad_quantity
                labels = labels + [pad_id] * pad_quantity

            all_input_ids.append(input_ids[:max_seq_len])
            all_labels.append(labels[:max_seq_len])

        # Create tensors on CPU (worker processes can't use CUDA)
        # Training loop will move to GPU
        input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long)

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
    

    # Load preprocessed dataset from disk
    try:
        from datasets import load_from_disk

        preprocessed_dir = "data/preprocessed"
        train_path = f"{preprocessed_dir}/train"
        test_path = f"{preprocessed_dir}/test"

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"ERROR: Preprocessed dataset not found at {preprocessed_dir}")
            print("Please run 'python preprocess_dataset.py' first to create the preprocessed dataset.")
            return None

        print("Loading preprocessed C4 dataset from disk...")
        train_dataset = load_from_disk(train_path)
        test_dataset = load_from_disk(test_path)

        print(f"Loaded preprocessed dataset:")
        print(f"  Train chunks: {len(train_dataset):,}")
        print(f"  Test chunks: {len(test_dataset):,}")

    except Exception as e:
        print(f"Error loading preprocessed dataset: {e}")
        return None

    #dataset_splits = dataset['train'].train_test_split(test_size=test_size)
    #train_dataset = dataset['train']
    #test_dataset = dataset['test']


    # Shuffle train data for better training, no shuffle for test
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    if initial_model is not None:
        model = initial_model
    else:
        model = Transformer(vocab_size=vocab_size,mha_params=mha_params,N=N,block_dropout=0.2).to(device)
        print("Compiling model...")
        model = torch.compile(model, mode='reduce-overhead')
        print("Finished.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    pad_id = tokenizer.token_to_id("[PAD]")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # Add learning rate scheduler with warmup
    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    os.makedirs('models/checkpoints',exist_ok=True)
    now = datetime.now()

    scaler = GradScaler('cuda')
    print("First run will take a bit longer...")

    # Start timing total training
    training_start_time = time()

    for epoch in range(start_epoch, epochs):
        # Start timing this epoch
        epoch_start_time = time()

        model.train()
        total_loss = 0
        num_batches = 0
        for i, batch in enumerate(train_loader):
            with autocast('cuda'):
                input_ids = batch['input_ids'].to(device)
                target_labels = batch['labels'].to(device)
                y_pred = model(input_ids)
                loss = criterion(y_pred.view(-1, y_pred.size(-1)), target_labels.view(-1))
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                # Unscale gradients for clipping
                scaler.unscale_(optimizer)

                # Calculate gradient norm before clipping
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Log gradient norm
                global_step = epoch * len(train_loader) + i
                writer.add_scalar("Gradients/norm", total_norm, global_step)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # Update learning rate
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            num_batches += 1

            # Log batch-level loss every 100 batches
            if i % 100 == 0:
                global_step = epoch * len(train_loader) + i
                writer.add_scalar("Loss/batch", loss.item() * accumulation_steps, global_step)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        writer.add_scalar("AvgLoss/train", avg_loss, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
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
        writer.add_scalar("AvgValLoss/val", avg_val_loss, epoch)
        perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
        writer.add_scalar("Perplexity", perplexity, epoch)

        # Calculate and log epoch time
        epoch_time_minutes = (time() - epoch_start_time) / 60
        writer.add_scalar("Time/epoch_minutes", epoch_time_minutes, epoch)

        print(f"Epoch: {epoch+1}, AVG loss: {avg_loss:.4f} AVG Val loss: {avg_val_loss} Perplexity: {perplexity} Time: {epoch_time_minutes:.2f}m")

    # Calculate and log total training time
    total_training_minutes = (time() - training_start_time) / 60
    writer.add_scalar("Time/total_training_minutes", total_training_minutes, epochs - 1)

    print(f"\nTotal training time: {total_training_minutes:.2f} minutes ({total_training_minutes/60:.2f} hours)")

    writer.close()
    writer.flush()
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

    hp = HyperParameters()
    batch_size = hp.batch_size
    accumulation_steps = hp.accumulation_steps
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
            print("Compiling model (first epoch will be slower)...")
            model = torch.compile(model, mode='max-autotune')
            print("Compilation finished.")

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
                initial_model=model,
                accumulation_steps=accumulation_steps
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
    
    prompt = "The future of artificial intelligence is "
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
