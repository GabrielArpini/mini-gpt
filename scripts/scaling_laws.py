
"""
Scaling Laws Experiment for mini-gpt.

Trains multiple model scales on FineWeb-Edu to empirically derive
scaling law curves (loss vs compute, loss vs params, loss vs tokens).
Each model trains on ~20x its param count in tokens (Chinchilla Hoffmann et al.).
Results saved to scaling_results.json for plotting.

Usage:
    python scripts/scaling_laws.py                     # all scales
    python scripts/scaling_laws.py --scales tiny small # specific scales
    python scripts/scaling_laws.py --resume            # skip already completed
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from time import time

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset

import wandb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mini_gpt.model import Transformer
from mini_gpt.optim import Muon
from mini_gpt.train import split_params_for_optimizer
from mini_gpt.data.fineweb import FineWebStreamer


# Data class for scaling configs 
@dataclass
class ScaleConfig:
    name: str
    d_model: int
    n_layers: int
    num_heads: int
    max_seq_len: int
    batch_size: int
    accumulation_steps: int
    tokens_to_train: int  # Chinchilla optimal: ~20x params
    lr: float
    dropout: float = 0.1


# Each scale trains on ~20 tokens per parameter (Chinchilla ratio).
# Smaller models tolerate higher LR, larger ones need lower.
# Batch size shrinks for larger models to fit in VRAM,
# accumulation steps grow to compensate.
SCALES = {
    "tiny": ScaleConfig(
        name="tiny",
        d_model=256, n_layers=4, num_heads=4,
        max_seq_len=512, batch_size=32, accumulation_steps=2,
        tokens_to_train=200_000_000,   # ~10M params -> 200M tokens
        lr=1e-3,
    ),
    "small": ScaleConfig(
        name="small",
        d_model=384, n_layers=6, num_heads=6,
        max_seq_len=512, batch_size=24, accumulation_steps=2,
        tokens_to_train=500_000_000,   # ~25M params -> 500M tokens
        lr=8e-4,
    ),
    "base": ScaleConfig(
        name="base",
        d_model=512, n_layers=8, num_heads=8,
        max_seq_len=512, batch_size=16, accumulation_steps=4,
        tokens_to_train=840_000_000,   # ~42M params -> 840M tokens
        lr=6e-4,
    ),
    "medium": ScaleConfig(
        name="medium",
        d_model=768, n_layers=12, num_heads=12,
        max_seq_len=512, batch_size=8, accumulation_steps=8,
        tokens_to_train=2_200_000_000,  # ~110M params -> 2.2B tokens
        lr=3e-4,
    ),
    "large": ScaleConfig(
        name="large",
        d_model=1024, n_layers=16, num_heads=16,
        max_seq_len=512, batch_size=4, accumulation_steps=16,
        tokens_to_train=5_000_000_000,  # ~250M params -> 5B tokens
        lr=2e-4,
    ),
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_PATH = "scaling_results.json"


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops_per_token(n_params: int) -> int:
    # ~6N FLOPs per token: 2N forward + 4N backward
    return 6 * n_params


def load_results() -> list[dict]:
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []


def save_results(results: list[dict]):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def get_or_train_tokenizer(vocab_size: int = 20_000) -> Tokenizer:
    """Load existing vocab or train a new one from FineWeb-Edu sample."""
    # Separate vocab from the main one
    vocab_path = "data/fineweb_vocab.json"
    os.makedirs("data", exist_ok=True)

    if os.path.exists(vocab_path):
        print(f"Loading tokenizer from {vocab_path}")
        return Tokenizer.from_file(vocab_path)

    # Train on first 100k docs from FineWeb-Edu
    print("Training tokenizer on FineWeb-Edu sample...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

    def text_iterator():
        for i, example in enumerate(ds):
            if i >= 100_000:
                break
            yield example["text"]

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    tokenizer.save(vocab_path)
    print(f"Tokenizer saved to {vocab_path}")
    return tokenizer


def train_scale(cfg: ScaleConfig, tokenizer: Tokenizer) -> dict:
    """Train one model scale, return metrics for the scaling law curve."""

    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("[PAD]")

    print(f"\n{'='*60}")
    print(f"  SCALE: {cfg.name}")
    print(f"  d_model={cfg.d_model}, layers={cfg.n_layers}, heads={cfg.num_heads}")
    print(f"  Target tokens: {cfg.tokens_to_train:,}")
    print(f"{'='*60}\n")

    # Build model with same architecture as main.py
    mha_params = {
        "d_model": cfg.d_model,
        "h": cfg.num_heads,
        "max_seq_len": cfg.max_seq_len,
        "dropout": cfg.dropout,
    }
    model = Transformer(
        vocab_size=vocab_size,
        mha_params=mha_params,
        N=cfg.n_layers,
        block_dropout=cfg.dropout,
    ).to(device)

    n_params = count_parameters(model)
    flops_per_token = estimate_flops_per_token(n_params)
    total_flops = flops_per_token * cfg.tokens_to_train

    print(f"Parameters: {n_params:,}")
    print(f"Est. total FLOPs: {total_flops:.2e}")

    torch.cuda.empty_cache()
    model = torch.compile(model, options={"triton.cudagraphs": False})

    # Muon + AdamW split from main model 
    muon_params, adamw_params = split_params_for_optimizer(model)
    muon_optimizer = Muon(muon_params, lr=0.01, momentum=0.95)
    adamw_optimizer = torch.optim.AdamW(adamw_params, lr=cfg.lr, weight_decay=0.1)

    # Cosine decay with linear warmup 
    tokens_per_step = cfg.batch_size * cfg.accumulation_steps * cfg.max_seq_len
    total_steps = cfg.tokens_to_train // tokens_per_step
    warmup_steps = int(0.1 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(adamw_optimizer, lr_lambda)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)
    scaler = GradScaler("cuda")

    # Stream data from HuggingFace, no disk storage needed
    streamer = FineWebStreamer(tokenizer, max_seq_len=cfg.max_seq_len)
    # num_workers=0 because IterableDataset with workers duplicates data (no sharding)
    loader = DataLoader(streamer, batch_size=cfg.batch_size, num_workers=0, pin_memory=True)

    # Fall back to offline mode if wandb is not logged in
    wandb.init(
        project="mini-gpt-scaling",
        name=f"scale_{cfg.name}",
        mode=os.environ.get("WANDB_MODE", "online"),
        config={
            "scale": cfg.name,
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "num_heads": cfg.num_heads,
            "n_params": n_params,
            "tokens_target": cfg.tokens_to_train,
            "total_flops": total_flops,
            "batch_size": cfg.batch_size,
            "accumulation_steps": cfg.accumulation_steps,
            "lr": cfg.lr,
        },
    )

    # Training loop. Stops after token budget is reached, not after N epochs.
    model.train()
    tokens_seen = 0
    step = 0
    running_loss = 0.0
    loss_samples = 0
    losses_at_checkpoints = []
    start_time = time()

    print(f"Total steps: {total_steps:,} | Tokens per step: {tokens_per_step:,}")
    print(f"Training started at {datetime.now().strftime('%H:%M:%S')}\n")

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with autocast("cuda"):
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = loss / cfg.accumulation_steps

        scaler.scale(loss).backward()
        running_loss += loss.item() * cfg.accumulation_steps
        loss_samples += 1
        tokens_seen += input_ids.numel()

        if (step + 1) % cfg.accumulation_steps == 0:
            scaler.unscale_(muon_optimizer)
            scaler.unscale_(adamw_optimizer)
            torch.nn.utils.clip_grad_norm_(adamw_params, max_norm=1.0)

            scaler.step(muon_optimizer)
            scaler.step(adamw_optimizer)
            scaler.update()
            scheduler.step()

            muon_optimizer.zero_grad()
            adamw_optimizer.zero_grad()

        step += 1

        # Log every 500 steps
        if step % 500 == 0:
            avg_loss = running_loss / loss_samples
            elapsed = time() - start_time
            tokens_per_sec = tokens_seen / elapsed
            eta_hours = (cfg.tokens_to_train - tokens_seen) / tokens_per_sec / 3600

            wandb.log({
                "loss": avg_loss,
                "tokens_seen": tokens_seen,
                "flops": tokens_seen * flops_per_token,
                "lr/adamw": adamw_optimizer.param_groups[0]["lr"],
                "throughput/tokens_per_sec": tokens_per_sec,
                "step": step,
            })

            print(
                f"  step {step:>7,} | loss {avg_loss:.4f} | "
                f"tokens {tokens_seen:>13,}/{cfg.tokens_to_train:,} "
                f"({100*tokens_seen/cfg.tokens_to_train:.1f}%) | "
                f"{tokens_per_sec:,.0f} tok/s | ETA {eta_hours:.1f}h"
            )

            losses_at_checkpoints.append({
                "step": step,
                "loss": avg_loss,
                "tokens": tokens_seen,
                "flops": tokens_seen * flops_per_token,
            })
            running_loss = 0.0
            loss_samples = 0

        if tokens_seen >= cfg.tokens_to_train:
            break

    elapsed = time() - start_time
    final_loss = losses_at_checkpoints[-1]["loss"] if losses_at_checkpoints else float("nan")

    result = {
        "name": cfg.name,
        "n_params": n_params,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "num_heads": cfg.num_heads,
        "tokens_trained": tokens_seen,
        "total_flops": tokens_seen * flops_per_token,
        "final_loss": final_loss,
        "training_hours": elapsed / 3600,
        "tokens_per_sec": tokens_seen / elapsed,
        "losses": losses_at_checkpoints,
        "timestamp": datetime.now().isoformat(),
    }

    wandb.log({"final_loss": final_loss, "total_flops": result["total_flops"]})
    wandb.finish()

    # Save checkpoint per scale
    os.makedirs("models/scaling", exist_ok=True)
    torch.save(model.state_dict(), f"models/scaling/{cfg.name}.pt")

    print(f"\n  DONE: {cfg.name} | loss={final_loss:.4f} | {elapsed/3600:.2f}h | {tokens_seen:,} tokens\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scales", nargs="*", default=None,
        help="Which scales to run (default: all). Options: tiny, small, base, medium, large",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip scales that already have results in scaling_results.json",
    )
    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(42)

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    tokenizer = get_or_train_tokenizer()

    scale_names = args.scales if args.scales else list(SCALES.keys())
    existing_results = load_results()
    completed_names = {r["name"] for r in existing_results}

    if args.resume:
        scale_names = [s for s in scale_names if s not in completed_names]
        print(f"Resuming. Already completed: {completed_names}")

    print(f"Scales to run: {scale_names}\n")

    results = existing_results.copy()

    for name in scale_names:
        if name not in SCALES:
            print(f"Unknown scale: {name}, skipping")
            continue

        cfg = SCALES[name]
        result = train_scale(cfg, tokenizer)
        results.append(result)
        # Save after each scale so we dont lose progress if instance dies
        save_results(results)
        print(f"Results saved to {RESULTS_PATH}")

    # Summary table 
    print("\n" + "=" * 60)
    print("  SCALING LAWS SUMMARY")
    print("=" * 60)
    print(f"{'Scale':<10} {'Params':>12} {'Tokens':>14} {'FLOPs':>12} {'Loss':>8} {'Time':>8}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['name']:<10} {r['n_params']:>12,} {r['tokens_trained']:>14,} "
            f"{r['total_flops']:>12.2e} {r['final_loss']:>8.4f} {r['training_hours']:>7.2f}h"
        )


if __name__ == "__main__":
    main()
