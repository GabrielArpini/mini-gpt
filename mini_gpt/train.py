from __future__ import annotations

import os
from datetime import datetime
from time import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import tokenizers
from torch.amp import autocast, GradScaler
from torch.profiler import profile, ProfilerActivity, record_function
from torch.utils.data import DataLoader

import wandb

from mini_gpt.model import Transformer
from mini_gpt.optim import Muon

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    # Load preprocessed dataset from disk
    try:
        from datasets import load_from_disk

        preprocessed_dir = "data/preprocessed"
        train_path = f"{preprocessed_dir}/train"
        test_path = f"{preprocessed_dir}/test"

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"ERROR: Preprocessed dataset not found at {preprocessed_dir}")
            print("Please run 'python -m mini_gpt.data.preprocess' first to create the preprocessed dataset.")
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
