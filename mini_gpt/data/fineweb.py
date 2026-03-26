from __future__ import annotations

import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from tokenizers import Tokenizer


class FineWebStreamer(IterableDataset):
    """
    Streams FineWeb-Edu and tokenizes on the fly.
    Avoids downloading the full dataset to disk.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        max_seq_len: int = 512,
        split: str = "train",
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split
        self.seed = seed
        self.bos_id = tokenizer.token_to_id("[BOS]")
        self.eos_id = tokenizer.token_to_id("[EOS]")

    def __iter__(self):
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split=self.split,
            streaming=True,
        )
        ds = ds.shuffle(seed=self.seed, buffer_size=10_000)

        # Accumulate tokens from multiple documents into a buffer.
        # When buffer is full, slice off a training example.
        # This packs short docs together so we waste no padding.
        buffer = []
        for example in ds:
            text = example["text"]
            token_ids = self.tokenizer.encode(text).ids
            # Wrap each document with BOS/EOS so the model learns boundaries
            buffer.append(self.bos_id)
            buffer.extend(token_ids)
            buffer.append(self.eos_id)

            # Yield chunks as they fill up
            while len(buffer) >= self.max_seq_len + 1:
                chunk = buffer[: self.max_seq_len + 1]
                buffer = buffer[self.max_seq_len:]  # overlap by 1 for next token prediction
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}
