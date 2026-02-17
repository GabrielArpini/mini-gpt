from dataclasses import dataclass


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
