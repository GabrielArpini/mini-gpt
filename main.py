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
from torch.profiler import profile, ProfilerActivity, record_function

import glob
import re

import wandb

from time import time
from contextlib import nullcontext

from muon import Muon # Karpathy's nanochat muon implementation. 

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizers parallelism to avoid fork issues

@dataclass
class HyperParameters:
    N: int = 8
    batch_size: int = 10
    accumulation_steps: int = 4
    max_seq_len: int = 512
    test_size: float = 0.1
    epochs: int = 3  # Reduced from 20 to prevent overfitting
    lr: float = 6e-4  # Slightly increased from 3e-4, scheduler will handle warmup
    checkpoint: bool = True
    d_model: int = 512
    num_heads: int = 8
    dropout: float = 0.3  # Increased to prevent degenerate repetition patterns
    vocab_size: int = 20_000
    max_new_tokens: int = 300
    temperature: float = 1.5
    top_k: int = 50
    top_p: float = 0.9
    penalty_factor: float = 1.6
    window_size: int = 30
    prompt: str = "Aqui no Brasil, as pessoas costumam "
    enable_profiler: bool = True  # Profiling can slow down training significantly

def split_into_chunks(tokens, chunk_size=512, overlap=50):
    chunks = []
    current = 0
    step = chunk_size - overlap
    while current < len(tokens):
        chunk = tokens[current: current + chunk_size]
        chunks.append(chunk)
        current += step
    return chunks


def split_params_for_optimizer(model):
    """
    Split model parameters into two groups:
    - muon_params: 2D+ parameters from hidden layers (TransformerBlocks)
    - adamw_params: Input embedding, output layer, and all 1D parameters (LayerNorms, biases)

    According to Muon paper, Muon should only be used for 2D+ weight matrices,
    not for embeddings, final layers, or LayerNorms.
    """
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Embedding layer -> AdamW
        if 'embedding' in name:
            adamw_params.append(param)
        # Final output layer -> AdamW
        elif 'final_layer' in name:
            adamw_params.append(param)
        # LayerNorm parameters (1D) -> AdamW
        elif 'norm' in name:
            adamw_params.append(param)
        # Biases (1D) -> AdamW
        elif 'bias' in name:
            adamw_params.append(param)
        # 2D+ parameters in TransformerBlocks -> Muon
        elif param.ndim >= 2:
            muon_params.append(param)
        # Any remaining 1D parameters -> AdamW
        else:
            adamw_params.append(param)

    return muon_params, adamw_params 
  





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
    checkpoint=True,
    start_epoch: int = 0,
    initial_model = None,
    accumulation_steps: int = 1,
    optimizer = None,
    scheduler = None,
    enable_profiler: bool = False
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

        print("Loading preprocessed dataset from disk...")
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
        model = Transformer(vocab_size=vocab_size,mha_params=mha_params,N=N,block_dropout=0.3).to(device)

    # Create optimizer and scheduler only if not provided
    if optimizer is None:
        # Split parameters for Muon and AdamW
        muon_params, adamw_params = split_params_for_optimizer(model)

        print(f"Optimizer split: {len(muon_params)} params for Muon, {len(adamw_params)} params for AdamW")

        # Create both optimizers
        # Muon for 2D+ weight matrices in transformer blocks
        muon_optimizer = Muon(muon_params, lr=0.01, momentum=0.95)

        # AdamW for embeddings, final layer, and 1D parameters
        adamw_optimizer = torch.optim.AdamW(adamw_params, lr=lr, weight_decay=0.1)

        # Store both in a dict for easier handling
        optimizer = {'muon': muon_optimizer, 'adamw': adamw_optimizer,
                     'muon_params': muon_params, 'adamw_params': adamw_params}

    pad_id = tokenizer.token_to_id("[PAD]")
    # Add label_smoothing to prevent overconfident predictions that lead to repetition
    # 0.1 means: true label gets 90% probability mass, other tokens share 10%
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)

    # Add learning rate scheduler with warmup (only if not provided)
    # Scheduler only for AdamW (Muon has fixed LR)
    if scheduler is None:
        num_training_steps = epochs * len(train_loader)
        num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

        # Scheduler only applies to AdamW optimizer
        adamw_opt = optimizer['adamw'] if isinstance(optimizer, dict) else optimizer
        scheduler = torch.optim.lr_scheduler.LambdaLR(adamw_opt, lr_lambda)
    os.makedirs('models/checkpoints',exist_ok=True)
    now = datetime.now()

    # Initialize wandb
    wandb.init(
        project="mini-gpt",
        config={
            "N": N,
            "batch_size": batch_size,
            "accumulation_steps": accumulation_steps,
            "max_seq_len": max_seq_len,
            "epochs": epochs,
            "lr": lr,
            "vocab_size": vocab_size,
            "d_model": mha_params.get('d_model'),
            "num_heads": mha_params.get('h'),
            "dropout": mha_params.get('dropout'),
        }
    )

    scaler = GradScaler('cuda')
    print("First run will take a bit longer...")

    # Start timing total training
    training_start_time = time()

    # Setup profiler for training (profile first epoch only) - optional
    prof = None
    if enable_profiler:
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=lambda p: p.export_chrome_trace("training_trace.json")
        )

    # Helper function to conditionally use record_function or nullcontext
    def profiler_context(name):
        return record_function(name) if enable_profiler else nullcontext()

    for epoch in range(start_epoch, epochs):
        # Start timing this epoch
        epoch_start_time = time()

        model.train()
        total_loss = 0
        num_batches = 0

        # Start profiling on first epoch (if enabled)
        if enable_profiler and epoch == start_epoch:
            prof.start()

        for i, batch in enumerate(train_loader):
            with profiler_context("## Data Loading"):
                input_ids = batch['input_ids'].to(device)
                target_labels = batch['labels'].to(device)

            with profiler_context("## Forward Pass"):
                with autocast('cuda'):
                    y_pred = model(input_ids)
                    loss = criterion(y_pred.view(-1, y_pred.size(-1)), target_labels.view(-1))
                    loss = loss / accumulation_steps

            with profiler_context("## Backward Pass"):
                scaler.scale(loss).backward()

            # Stop profiling after 15 batches (if enabled)
            if enable_profiler and epoch == start_epoch and i == 15:
                prof.stop()
                print("Training profiling complete. Saved to 'training_trace.json'")

            if (i + 1) % accumulation_steps == 0:
                with profiler_context("## Optimizer Step"):
                    # Handle both optimizers
                    if isinstance(optimizer, dict):
                        # Unscale gradients for clipping (both optimizers)
                        scaler.unscale_(optimizer['muon'])
                        scaler.unscale_(optimizer['adamw'])

                        # Use pre-computed parameter lists (no need to rebuild every step!)
                        muon_params_list = optimizer['muon_params']
                        adamw_params_list = optimizer['adamw_params']

                        # Calculate gradient norms efficiently (single GPU operation per optimizer)
                        # Muon: compute norm without clipping (orthogonalization handles normalization)
                        muon_norm = torch.nn.utils.clip_grad_norm_(muon_params_list, float('inf'))

                        # AdamW: compute norm AND clip to max_norm=1.0 in one call
                        adamw_norm = torch.nn.utils.clip_grad_norm_(adamw_params_list, max_norm=1.0)

                        # Log gradient norms separately
                        global_step = epoch * len(train_loader) + i
                        wandb.log({
                            "gradients/muon_norm": muon_norm.item(),
                            "gradients/adamw_norm": adamw_norm.item(),
                            "global_step": global_step
                        })

                        # Step both optimizers
                        scaler.step(optimizer['muon'])
                        scaler.step(optimizer['adamw'])
                        scaler.update()
                        scheduler.step()  # Update learning rate for AdamW only

                        # Zero gradients for both
                        optimizer['muon'].zero_grad()
                        optimizer['adamw'].zero_grad()
                    else:
                        # Fallback for old single optimizer
                        scaler.unscale_(optimizer)
                        total_norm = 0
                        for p in model.parameters():
                            if p.grad is not None:
                                total_norm += p.grad.data.norm(2).item() ** 2
                        total_norm = total_norm ** 0.5
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        global_step = epoch * len(train_loader) + i
                        wandb.log({"gradients/norm": total_norm, "global_step": global_step})
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            num_batches += 1

            # Log batch-level loss every 100 batches
            if i % 100 == 0:
                global_step = epoch * len(train_loader) + i
                wandb.log({"loss/batch": loss.item() * accumulation_steps, "global_step": global_step})
        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        # Log learning rates for both optimizers
        if isinstance(optimizer, dict):
            wandb.log({
                "loss/train_avg": avg_loss,
                "lr/adamw": optimizer['adamw'].param_groups[0]['lr'],
                "lr/muon": optimizer['muon'].param_groups[0]['lr'],
                "epoch": epoch
            })
        else:
            wandb.log({
                "loss/train_avg": avg_loss,
                "lr": optimizer.param_groups[0]['lr'],
                "epoch": epoch
            })
        if checkpoint:
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch
            }

            # Save optimizer state (handle both single and dual optimizer setup)
            if isinstance(optimizer, dict):
                checkpoint_data['muon_optimizer_state_dict'] = optimizer['muon'].state_dict()
                checkpoint_data['adamw_optimizer_state_dict'] = optimizer['adamw'].state_dict()
            else:
                checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()

            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()

            torch.save(checkpoint_data, f"models/checkpoints/{now.year}_{now.month}_{now.day}_epoch_{epoch}.pt")
        
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

        # Calculate and log epoch time
        epoch_time_minutes = (time() - epoch_start_time) / 60

        wandb.log({
            "loss/val_avg": avg_val_loss,
            "perplexity": perplexity,
            "time/epoch_minutes": epoch_time_minutes,
            "epoch": epoch
        })

        print(f"Epoch: {epoch+1}, AVG loss: {avg_loss:.4f} AVG Val loss: {avg_val_loss} Perplexity: {perplexity} Time: {epoch_time_minutes:.2f}m")

    # Calculate and log total training time
    total_training_minutes = (time() - training_start_time) / 60
    wandb.log({
        "time/total_training_minutes": total_training_minutes,
        "time/total_training_hours": total_training_minutes / 60
    })

    print(f"\nTotal training time: {total_training_minutes:.2f} minutes ({total_training_minutes/60:.2f} hours)")

    wandb.finish()
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

def apply_ngram_penalty(logits, generated_tokens, window_size=20, penalty_factor=1.2):
    """
    Apply n-gram penalty: P'(x_i) = P(x_i) / penalty_factor if x_i in last w tokens

    Args:
        logits: Model logits before softmax
        generated_tokens: List of previously generated token IDs
        window_size: Look back w tokens
        penalty_factor: Penalty strength (> 1.0 reduces probability of repeated tokens)
    """
    if len(generated_tokens) == 0 or penalty_factor == 1.0:
        return logits

    # Get tokens in the window
    window = generated_tokens[-window_size:] if len(generated_tokens) > window_size else generated_tokens

    # Apply penalty by subtracting log(penalty_factor) from logits
    # This is equivalent to dividing probabilities by penalty_factor
    penalty = torch.log(torch.tensor(penalty_factor, device=logits.device))

    for token_id in set(window):
        logits[0, token_id] = logits[0, token_id] - penalty

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
    penalty_factor = hp.penalty_factor
    window_size = hp.window_size
    prompt = hp.prompt
    enable_profiler = hp.enable_profiler

    vocab_path = 'data/vocab.json'
    os.makedirs('data', exist_ok=True)

    # Initialize Hugging Face tokenizers BPE


    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
    if not os.path.exists(vocab_path):
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

    # Load dataset info early (needed for scheduler calculation when resuming)
    from datasets import load_from_disk
    preprocessed_dir = "data/preprocessed"
    train_path = f"{preprocessed_dir}/train"
    if os.path.exists(train_path):
        train_dataset_info = load_from_disk(train_path)
        train_dataset_size = len(train_dataset_info)
    else:
        train_dataset_size = None  # Will use default if no dataset found

    start_epoch = 0
    model = Transformer(vocab_size=vocab_size, mha_params=mha_params, N=N, block_dropout=0.3).to(device)

    # Initialize optimizer and scheduler (will be restored from checkpoint if resuming)
    resumed_optimizer = None
    resumed_scheduler = None

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
                    checkpoint_data = torch.load(latest_checkpoint, weights_only=False)
                else:
                    checkpoint_data = torch.load(latest_checkpoint, weights_only=False, map_location=torch.device('cpu'))

                # Handle both old (state_dict only) and new (dict with optimizer/scheduler) checkpoint formats
                if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
                    # New checkpoint format
                    state_dict = checkpoint_data['model_state_dict']
                    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                    model.load_state_dict(state_dict)

                    # Create optimizers and scheduler first, then load their state
                    # Check if checkpoint has dual optimizer setup (Muon + AdamW)
                    if 'muon_optimizer_state_dict' in checkpoint_data and 'adamw_optimizer_state_dict' in checkpoint_data:
                        # New dual optimizer setup
                        muon_params, adamw_params = split_params_for_optimizer(model)
                        muon_opt = Muon(muon_params, lr=0.01, momentum=0.95)
                        adamw_opt = torch.optim.AdamW(adamw_params, lr=lr, weight_decay=0.1)
                        resumed_optimizer = {'muon': muon_opt, 'adamw': adamw_opt,
                                           'muon_params': muon_params, 'adamw_params': adamw_params}

                        # Load optimizer states
                        muon_opt.load_state_dict(checkpoint_data['muon_optimizer_state_dict'])
                        adamw_opt.load_state_dict(checkpoint_data['adamw_optimizer_state_dict'])
                        print(f"Restored Muon and AdamW optimizers (AdamW LR: {adamw_opt.param_groups[0]['lr']:.2e})")
                    else:
                        # Old single optimizer setup
                        resumed_optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
                        if 'optimizer_state_dict' in checkpoint_data:
                            resumed_optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                            print(f"Restored single AdamW optimizer (LR: {resumed_optimizer.param_groups[0]['lr']:.2e})")

                    # Create scheduler with same config (need dataset size for this)
                    if train_dataset_size is not None:
                        num_training_steps = epochs * (train_dataset_size // (batch_size * accumulation_steps))
                        num_warmup_steps = int(0.1 * num_training_steps)

                        def lr_lambda(current_step):
                            if current_step < num_warmup_steps:
                                return float(current_step) / float(max(1, num_warmup_steps))
                            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

                        # Scheduler applies to AdamW optimizer
                        adamw_for_scheduler = resumed_optimizer['adamw'] if isinstance(resumed_optimizer, dict) else resumed_optimizer
                        resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(adamw_for_scheduler, lr_lambda)

                        # Restore scheduler state
                        if 'scheduler_state_dict' in checkpoint_data:
                            resumed_scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                            print(f"Restored scheduler state")
                    else:
                        print("Warning: Could not calculate scheduler parameters without dataset. Optimizer and scheduler will restart.")
                else:
                    # Old checkpoint format (just state_dict)
                    state_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint_data.items()}
                    model.load_state_dict(state_dict)
                    print("Warning: Old checkpoint format detected. Optimizer and scheduler will restart.")

                start_epoch = latest_epoch + 1

        if start_epoch < epochs:
            print(f"\nStarting training from epoch {start_epoch}")
            torch.cuda.empty_cache()
            print("Compiling model (first epoch will be slower)...")
            model = torch.compile(model, options={'triton.cudagraphs': False}) # Cant add mode, otherwise it will raise RuntimeError.
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
                accumulation_steps=accumulation_steps,
                optimizer=resumed_optimizer,
                scheduler=resumed_scheduler,
                enable_profiler=enable_profiler
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

    input_ids = tokenizer.encode(prompt).ids

    # Don't pad during generation - only use actual tokens
    generated_tokens = input_ids.copy()  # Start with prompt tokens

    eos_token_id = tokenizer.token_to_id("[EOS]")

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Only pass the actual tokens (no padding)
            input_tensor = torch.tensor([generated_tokens[-max_seq_len:]], dtype=torch.long).to(device)

            # Get logits at the last position (which is now the last actual token)
            logits = model(input_tensor)[:, -1, :]

            # Mask out PAD token from being generated
            logits[:, tokenizer.token_to_id("[PAD]")] = float('-inf')

            # Apply n-gram penalty to reduce repetition
            logits = apply_ngram_penalty(logits, generated_tokens, window_size=window_size, penalty_factor=penalty_factor)

            logits = top_k_filtering(logits, top_k)
            logits = top_p_filtering(logits, top_p)
            probs = temperature_sampling(logits, temperature=temperature)

            next_token = torch.multinomial(probs, num_samples=1)
            next_token = next_token.item()

            generated_tokens.append(next_token)

            # Stop if EOS token is generated
            if next_token == eos_token_id:
                break
    
    #generated_text = enc.decode(generated_tokens)
    generated_text = tokenizer.decode(generated_tokens).replace(" ,", ",").replace(" .", ".")
    print(f"Generated text: {generated_text}")

   

if __name__ == "__main__":
    main()
