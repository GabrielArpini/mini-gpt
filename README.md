# mini-gpt

A decoder-only transformer built from scratch in PyTorch — no shortcuts, no wrappers. Every component implemented and understood from first principles.

---

## Architecture

```
Input Tokens
     │
     ▼
┌─────────────┐
│  Embedding  │
└─────────────┘
     │
     ▼
┌─────────────────────────────────┐
│         Transformer Block × N   │
│                                 │
│  ┌──────────────────────────┐   │
│  │  RMSNorm                 │   │
│  │  Multi-Query Attention   │   │
│  │  + RoPE + KV Cache       │   │
│  └──────────────────────────┘   │
│             +  (residual)       │
│  ┌──────────────────────────┐   │
│  │  RMSNorm                 │   │
│  │  SwiGLU FFN              │   │
│  └──────────────────────────┘   │
│             +  (residual)       │
└─────────────────────────────────┘
     │
     ▼
┌─────────────┐
│  RMSNorm    │
│  Linear     │
└─────────────┘
     │
     ▼
  Logits
```

**Key design choices:**

| Component | Choice | Why |
|-----------|--------|-----|
| Positional encoding | RoPE | Relative positions, extrapolates to longer sequences |
| Attention | Multi-Query (MQA) | Single K/V head — memory efficient, fast inference |
| Normalization | RMSNorm | Simpler than LayerNorm, no mean centering overhead |
| Activation | SwiGLU | Gated activation, better gradient flow than ReLU/GELU |
| Optimizer | Muon + AdamW | Muon for 2D+ weights, AdamW for embeddings and norms |
| Inference | KV Cache | Prefill + decode phases — no recomputation of past tokens |

---

## Project Structure

```
mini-gpt/
├── mini_gpt/
│   ├── config.py                # HyperParameters and SamplingParameters
│   ├── tokenizer.py             # Custom BPE tokenizer (reference implementation)
│   ├── train.py                 # Training loop: mixed precision, grad accumulation, WandB
│   ├── model/
│   │   ├── transformer.py       # Transformer, TransformerBlock, SwiGLU
│   │   ├── attention.py         # MultiHeadAttention (MQA) with KV cache
│   │   ├── rope.py              # Rotary Position Embeddings
│   │   └── rmsnorm.py           # RMSNorm
│   ├── data/
│   │   ├── dataset.py           # Streaming dataset iterator
│   │   └── preprocess.py        # Tokenization and chunking pipeline
│   ├── optim/
│   │   └── muon.py              # Muon optimizer
│   └── inference/
│       ├── engine.py            # Generation engine (prefill + decode with KV cache)
│       └── sampling.py          # top-k, top-p, temperature, n-gram penalty
├── scripts/
│   └── inspect_params.py        # Inspect Muon vs AdamW parameter split
└── main.py                      # Entry point
```

---

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

---

## Usage

**1. Preprocess** — downloads FineWeb-Edu and tokenizes into chunked splits:
```bash
uv run python -m mini_gpt.data.preprocess
```

**2. Train** — resumes automatically from the latest checkpoint if one exists:
```bash
uv run python main.py
```

**3. Generate** — runs after training or with an existing `models/base_model.pt`.

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | 8 | Transformer blocks |
| `d_model` | 512 | Embedding dimension |
| `num_heads` | 8 | Query heads (MQA: 1 K/V head) |
| `max_seq_len` | 512 | Maximum sequence length |
| `batch_size` | 10 | Batch size |
| `accumulation_steps` | 4 | Gradient accumulation steps |
| `lr` | 6e-4 | Learning rate |
| `epochs` | 3 | Training epochs |
| `dropout` | 0.3 | Dropout rate |
| `vocab_size` | 20,000 | BPE vocabulary size |

Generation parameters (temperature, top-k, top-p, n-gram penalty) are in `SamplingParameters` inside `mini_gpt/config.py`.
