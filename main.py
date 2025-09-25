from __future__ import annotations 

from datasets import load_dataset
from utils import * 
from tokenizer import Tokenizer 
import os 
from pos_encoding import RoPE
from MultiHeadAttention import MultiHeadAttention
from transformer import Transformer, TransformerBlock
import kagglehub
from torch.utils.data import DataLoader
import torch 
import torch.nn as nn


import torch 
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_data():
    root_path = os.getcwd()
    path_to_save = f'{root_path}/data/brazilian_lit' 
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    
    print("Downloading dataset...")
    dataset_path = kagglehub.dataset_download("rtatman/brazilian-portuguese-literature-corpus")
    print("Success!")

    return dataset_path 


    
def train(
    mha_params: dict,
    vocab_size: int,
    tokenizer: Tokenizer = None,
    N: int = 8,
    batch_size: int = 8,
    max_seq_len: int = 400,
    test_size: int = 0.1,
    epochs: int = 10,
    lr=1e-4
):

    # This function is inside train, because it needs the arguments
    # if it is outside i will need to use fn_kwargs inside map function
    # which gets messy easily.
    
    def collate_fn(batch):
        # batch is list of dicts, each with 'sentence'
        sentences = [item['sentence'] for item in batch]
        
        # Tokenize all sentences in batch
        all_input_ids = []
        all_labels = []
        for text in sentences:
            tokens = tokenizer.encode(text)  # Full sentence tokens (includes BOS/EOS)
            if not tokens:
                print(f"Warning: Empty token list for sentence {i}: {text}")
                tokens = [tokenizer.pad_id] * max_seq_len

            # Truncate to max_seq_len (keep BOS, drop tail if needed)
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]
            # Pad to max_seq_len
            tokens += [tokenizer.pad_id] * (max_seq_len - len(tokens))
            
            input_ids = tokens
            labels = tokens[1:] + [tokenizer.pad_id]  # Shift for causal LM
            
            all_input_ids.append(input_ids)
            all_labels.append(labels)
        
        # Stack into tensors: (batch_size, max_seq_len)
        input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long).to(device)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long).to(device)
        
        return {"input_ids": input_ids_tensor, "labels": labels_tensor}
    



    if not isinstance(mha_params, dict):
        raise TypeError(f"Expected mha_params to be a dict, got {type(mha_params)}: {mha_params}") 
    # Get dataset 
    #dataset_path = download_data() + '/Brazilian_Portugese_Corpus'
    #print(dataset_path)
    #dataset = load_dataset("text", data_files={"train": dataset_path + "/*.txt"}, encoding="iso-8859-1")
    dataset = load_dataset("iara-project/raw_dataset_with_embeddings_bert-base-portuguese-cased-nli-assin-2")

    print(dataset['train'][:5])  # Print first 5 processed examples

    print(len(dataset)) 

    dataset_splits = dataset['train'].train_test_split(test_size=test_size)
    train_dataset = dataset_splits['train']
    test_dataset = dataset_splits['test']
    print(len(train_dataset))
    print(len(test_dataset))
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Setup training instances.
    model = Transformer(vocab_size=vocab_size,mha_params=mha_params,N=N,block_dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Start training loop 
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            target_labels = batch['labels']
            y_pred = model(input_ids) 
            loss = criterion(y_pred.view(-1, y_pred.size(-1)), target_labels.view(-1))
            loss.backward()
            # Maybe grad clip later 
            
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch: {epoch+1}, AVG loss: {avg_loss:.4f}")






def main():
    merges_path = 'data/merges.json'
    vocab_path = 'data/vocab.json'
    merges_path = 'data/merges.json'

    tokenizer = Tokenizer(vocab_size=260)
    if not os.path.exists(merges_path) and not os.path.exists(vocab_path):
        iterator = DatasetIterator()
        tokenizer.train_from_iterator(iterator)
        tokenizer.save(merges_path,vocab_path)
    
    if not tokenizer.merges or not tokenizer.vocab:
        tokenizer.load(merges_path, vocab_path)

    print(len(tokenizer.vocab))
    print("Original text:")
    random_text = "In Brazil, Feynman was impressed with samba music, and learned to play the frigideira,[123] a metal percussion instrument based on a frying pan.[124] He was an enthusiastic amateur player of bongo and conga drums and often played them in the pit orchestra in musicals.[125][126] He spent time in Rio with his friend Bohm, but Bohm could not convince Feynman to investigate Bohm's ideas on physics."
    print(random_text)
    encoded_text = tokenizer.encode(random_text)
    


    print("Encoded text")
    print(encoded_text)

    print("Decoded text")
    print(tokenizer.decode(encoded_text))

    

    # Test positional posisitional encoding 
    # Need shape (batch, seq_len, d_model)
    print(len(encoded_text))

    # HYPERPARAMETERS
    seq_len = len(encoded_text)
    batch_size = 8
    d_model = 128
    max_seq_len = 400
    num_heads = 8
    dropout = 0.1
    N = 8
    # Create an embedding layer to convert token IDs to embeddings
    vocab_size = tokenizer.vocab_size
    embedding = torch.nn.Embedding(vocab_size, d_model).to(device)
    
    # Convert encoded text to tensor and repeat for batch_size
    input_ids = torch.tensor(encoded_text, dtype=torch.long).unsqueeze(0).to(device)  # Shape: (1, seq_len)
    input_ids = input_ids.repeat(batch_size, 1)  # Shape: (batch_size, seq_len)
    input_tensor = embedding(input_ids)  # Shape: (batch_size, seq_len, d_model)

    # Instantiate the RoPE module
    rope = RoPE(d_model=d_model, max_seq_len=max_seq_len).to(device)

    # Apply positional encoding
    output_tensor = rope(input_tensor)

    # Validate RoPE output shape
    print(f"\nRoPE Input shape: {input_tensor.shape}")
    print(f"RoPE Output shape: {output_tensor.shape}")
    assert output_tensor.shape == input_tensor.shape, "RoPE output shape should match input shape"
    
    # Visualize RoPE angles
    cos_angles = rope.cos_angles[:5, :].numpy()
    sin_angles = rope.sin_angles[:5, :].numpy()
    
    # Uncomment to see plot
    #print("\nVisualizing RoPE angles for first 5 positions:")
    #rope.plot(cos_angles, sin_angles)  # Plot sine and cosine angles

    # Test MultiHeadAttention
    print("\nTesting MultiHeadAttention:")
    mha = MultiHeadAttention(d_model=d_model, h=num_heads, dropout=dropout, max_seq_len=max_seq_len).to(device)

    input_tensor = input_tensor.to(device)
    
    # Forward pass through MultiHeadAttention
    attention_output = mha(input_tensor)
    # Validate output shape
    print(f"MHA Input shape: {input_tensor.shape}")
    print(f"MHA masked Output shape: {attention_output.shape}")
    assert attention_output.shape == input_tensor.shape, "MHA output shape should match input shape"
    
    # Print a sample of the output for inspection
    print(f"Sample of MHA output (first batch, first 5 tokens, first 5 dimensions):\n{attention_output[0, :5, :5]}")


    mha_params = {
        'd_model':d_model, 
        'h': num_heads, 
        'max_seq_len': max_seq_len, 
        'dropout': dropout         
    }

    
    

    print("\n starting training")
    mha_params = {
        'd_model':d_model, 
        'h': num_heads, 
        'max_seq_len': max_seq_len, 
        'dropout': dropout         
    }
    train(tokenizer = tokenizer,
        mha_params = mha_params,
        vocab_size = vocab_size,
        batch_size = 8,
        max_seq_len = 400,
        test_size = 0.1,
        epochs = 10,
        lr=1e-4
    )


if __name__ == "__main__":
    main()
