from __future__ import annotations

import os
import glob
import re

import torch
import tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from mini_gpt.config import HyperParameters
from mini_gpt.model import Transformer
from mini_gpt.data import DatasetIterator
from mini_gpt.optim import Muon
from mini_gpt.train import train, split_params_for_optimizer
from mini_gpt.generate import (
    temperature_sampling,
    top_k_filtering,
    top_p_filtering,
    apply_ngram_penalty,
)

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizers parallelism to avoid fork issues


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

    generated_text = tokenizer.decode(generated_tokens).replace(" ,", ",").replace(" .", ".")
    print(f"Generated text: {generated_text}")



if __name__ == "__main__":
    main()
