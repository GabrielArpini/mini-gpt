# mini-gpt

A GPT-style language model built from scratch in PyTorch.

## Architecture

Decoder-only transformer with the following components:

- **Rotary Position Embeddings (RoPE)** for relative position encoding
- **Multi-Query Attention (MQA)** for efficient inference (single K/V head, multiple Q heads)
- **SwiGLU** feed-forward network
- **RMSNorm** instead of LayerNorm
- **Muon optimizer** (momentum orthogonalized by Newton-Schulz) for 2D+ weight matrices, paired with AdamW for embeddings and normalization layers
- Mixed-precision training with gradient accumulation
- Linear warmup + cosine decay learning rate schedule

## Project Structure

```
mini-gpt/
├── mini_gpt/                    # Package
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── transformer.py       # Transformer, TransformerBlock, SwiGLU
│   │   ├── attention.py         # MultiHeadAttention (MQA)
│   │   ├── rope.py              # Rotary Position Embeddings
│   │   └── rmsnorm.py           # RMSNorm
│   ├── optim/
│   │   ├── __init__.py
│   │   └── muon.py              # Muon optimizer
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # DatasetIterator for tokenizer training
│   │   └── preprocess.py        # Dataset preprocessing pipeline
│   ├── tokenizer.py             # Custom BPE tokenizer (reference implementation)
│   ├── config.py                # HyperParameters dataclass
│   ├── generate.py              # Sampling: top-k, top-p, temperature, n-gram penalty
│   └── train.py                 # Training loop with wandb logging
├── scripts/
│   └── inspect_params.py        # Debug utility: inspect model parameter split
├── main.py                      # Entry point
├── pyproject.toml
└── .gitignore
```

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

### 1. Preprocess the dataset

Downloads and tokenizes FineWeb-Edu into chunked training/test splits:

```bash
uv run python -m mini_gpt.data.preprocess
```

### 2. Train

Trains the model with checkpoint saving and wandb logging:

```bash
uv run python main.py
```

Training automatically resumes from the latest checkpoint if one exists.

### 3. Generate

After training completes (or a `models/base_model.pt` exists), `main.py` generates text from the prompt configured in `mini_gpt/config.py`.

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | 8 | Number of transformer blocks |
| `d_model` | 512 | Embedding dimension |
| `num_heads` | 8 | Number of query heads (MQA) |
| `max_seq_len` | 512 | Maximum sequence length |
| `batch_size` | 10 | Batch size |
| `accumulation_steps` | 4 | Gradient accumulation steps |
| `lr` | 6e-4 | AdamW learning rate |
| `epochs` | 3 | Training epochs |
| `dropout` | 0.3 | Dropout rate |
| `vocab_size` | 20,000 | BPE vocabulary size |

See `mini_gpt/config.py` for the full list including generation parameters (temperature, top-k, top-p, n-gram penalty).
