"""
Pre-tokenize and save FineWeb-Edu dataset locally for faster training.
This script will download, tokenize, chunk, and save the dataset to disk.
"""
from __future__ import annotations

from datasets import load_dataset, Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
from tqdm import tqdm
import torch

def split_into_chunks(tokens, chunk_size=512, overlap=50):
    """Split tokens into overlapping chunks."""
    chunks = []
    current = 0
    step = chunk_size - overlap
    while current < len(tokens):
        chunk = tokens[current: current + chunk_size]
        if len(chunk) > 0:  # Only add non-empty chunks
            chunks.append(chunk)
        current += step
    return chunks


def preprocess_dataset(
    num_train: int = 100_000,
    num_test: int = 10_000,
    vocab_size: int = 20_000,
    chunk_size: int = 512,
    overlap: int = 50,
    output_dir: str = "data/preprocessed"
):
    """
    Download, tokenize, and save FineWeb-Edu dataset.

    Args:
        num_train: Number of training examples to process
        num_test: Number of test examples to process
        vocab_size: Tokenizer vocabulary size
        chunk_size: Size of text chunks (including BOS/EOS)
        overlap: Overlap between consecutive chunks
        output_dir: Directory to save preprocessed data
    """

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Load or train tokenizer
    vocab_path = 'data/vocab.json'
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]

    if not os.path.exists(vocab_path):
        print("Training tokenizer...")
        from utils import DatasetIterator
        iterator = DatasetIterator()
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
        tokenizer.train_from_iterator(iterator, trainer=trainer)
        tokenizer.save(vocab_path)
        print("Tokenizer trained and saved!")
    else:
        print("Loading existing tokenizer...")
        tokenizer = Tokenizer.from_file(vocab_path)

    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")

    print(f"\nPreprocessing FineWeb-Edu dataset...")
    print(f"Train examples: {num_train:,}")
    print(f"Test examples: {num_test:,}")
    print(f"Chunk size: {chunk_size} (including BOS/EOS)")
    print(f"Overlap: {overlap}\n")

    def is_valid_example(example):
        """Filter out invalid examples."""
        try:
            text = example.get('text', '')
            if not isinstance(text, str) or len(text) == 0:
                return False
            text.encode('utf-8', errors='strict')
            return True
        except (UnicodeDecodeError, UnicodeError, Exception):
            return False

    def process_split(split_name, num_examples, num_train_to_skip=0):
        """Process a dataset split and return tokenized chunks."""
        print(f"\nProcessing {split_name} split...")

        # Load streaming dataset (fineweb-edu only has 'train' split)
        dataset = load_dataset(
            "karpathy/fineweb-edu-100b-shuffle",
            split='train',
            streaming=True,
            trust_remote_code=False
        )

        # Skip training examples for validation split (non-overlapping)
        if num_train_to_skip > 0:
            dataset = dataset.skip(num_train_to_skip)

        dataset = dataset.take(num_examples)

        all_input_ids = []
        all_labels = []

        for example in tqdm(dataset, total=num_examples, desc=f"Tokenizing {split_name}"):
            try:
                text = example['text']
                if not isinstance(text, str):
                    continue

                # Tokenize
                tokens = tokenizer.encode(text).ids

                # Split into chunks (leaving room for BOS/EOS)
                chunks = split_into_chunks(tokens, chunk_size=chunk_size - 2, overlap=overlap)

                for chunk in chunks:
                    # Add BOS and EOS
                    chunk_with_specials = [bos_id] + chunk + [eos_id]

                    # Create input_ids and labels (shifted by 1)
                    input_ids = chunk_with_specials[:-1]
                    labels = chunk_with_specials[1:]

                    all_input_ids.append(input_ids)
                    all_labels.append(labels)

            except (UnicodeDecodeError, UnicodeError, Exception) as e:
                continue

        print(f"{split_name} split: {len(all_input_ids):,} chunks created from {num_examples:,} examples")

        # Create HuggingFace Dataset
        return Dataset.from_dict({
            'input_ids': all_input_ids,
            'labels': all_labels
        })

    # Process train and test splits (non-overlapping from same dataset)
    train_dataset = process_split('train', num_train, num_train_to_skip=0)
    test_dataset = process_split('validation', num_test, num_train_to_skip=num_train)

    # Save to disk
    train_path = os.path.join(output_dir, 'train')
    test_path = os.path.join(output_dir, 'test')

    print("\nSaving datasets to disk...")
    train_dataset.save_to_disk(train_path)
    test_dataset.save_to_disk(test_path)

    print(f"\nPreprocessing complete!")
    print(f"Train dataset: {train_path}")
    print(f"Test dataset: {test_path}")
    print(f"\nDataset stats:")
    print(f"  Train chunks: {len(train_dataset):,}")
    print(f"  Test chunks: {len(test_dataset):,}")
    print(f"  Sequence length: {chunk_size}")


if __name__ == "__main__":
    # Adjust these parameters as needed
    preprocess_dataset(
        num_train=100_000,     # Increased from 20k to 100k (5x more data)
        num_test=10_000,       # 10% of train
        vocab_size=20_000,
        chunk_size=512,
        overlap=50
    )
