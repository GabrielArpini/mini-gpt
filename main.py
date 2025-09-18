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

def process_example(examples):
        all_input_ids = []
        all_labels = []
        for text in examples["text"]:
            tokens = tokenizer.encode(text)
            if len(tokens) < 1: 
                print(f"Warning: Empty tokens for text: {text[:50]}...")
                continue
            # Context window.
            for i in range(0, len(tokens) - max_seq_len + 1, max_seq_len // 2):
                chunk = tokens[i:i + max_seq_len]
                if len(chunk) < max_seq_len:
                    chunk += [tokenizer.pad_token_id] * (max_seq_len - len(chunk))
                all_input_ids.append(chunk)
                all_labels.append(chunk[1:] + [tokenizer.pad_token_id])
        return {"input_ids": all_input_ids, "labels": all_labels}
    
def train(
    tokenizer: Tokenizer = None,
    batch_size: int = 8,
    max_seq_len: int = 400,
    test_size: int = 0.1,
    epochs: int = 10,
    lr=1e-4
):
    # This function is inside train, because it needs the arguments
    # if it is outside i will need to use fn_kwargs inside map function
    # which gets messy easily. 
    def process_example(examples):
        all_input_ids = []
        all_labels = []
        for text in examples["text"]:
            if not text.strip():
                continue
            tokens = tokenizer.encode(text)
            # Context window.
            for i in range(0, len(tokens) - max_seq_len + 1, max_seq_len // 2):
                chunk = tokens[i:i + max_seq_len]
                if len(chunk) < max_seq_len:
                    chunk += [tokenizer.pad_token_id] * (max_seq_len - len(chunk))
                all_input_ids.append(chunk)
                all_labels.append(chunk[1:] + [tokenizer.pad_token_id])
        return {"input_ids": all_input_ids, "labels": all_labels}


    # Get dataset 
    dataset_path = download_data() + '/Brazilian_Portugese_Corpus'
    print(dataset_path)
    dataset = load_dataset("text", data_files={"train": dataset_path + "/*.txt"}, encoding="iso-8859-1")
    dataset = dataset.map(process_example, batched=True, remove_columns=["text"])
    

    dataset_splits = dataset['train'].train_test_split(test_size=test_size)
    train_dataset = dataset_splits['train']
    test_dataset = dataset_splits['test']
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup training instances.
    model = Transformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Start training loop 
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            target_labels = batch['labels'][1:]
            y_pred = model(input_ids)
            logits = y_pred.logits 
            loss = criterion(logits.view(-1, logits.size(-1)), target_labels.view(-1))
            loss.backward()
            # Maybe grad clip later 
            
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch: {epoch+1}, AVG loss: {avg_loss:.4f}")






def main():
    merges_path = 'data/merges.json'
    vocab_path = 'data/vocab.json'
    tokenizer = Tokenizer(vocab_size=300)
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
    transformer = Transformer(N,vocab_size, mha_params,block_dropout=0.1)
    output_logits = transformer(input_tensor).to(device)
    expected_shape = (batch_size, input_tensor.size(1), vocab_size)
    assert output_logits.shape == expected_shape, f"Expected shape {expected_shape}, got {output_logits.shape}"
    
    probs_result = torch.nn.functional.softmax(output_logits,dim=-1)
    print(probs_result)
    
    

    print("\n starting training")

    train(tokenizer=tokenizer,
        batch_size = 8,
        max_seq_len = 400,
        test_size = 0.1,
        epochs = 10,
        lr=1e-4
    )


if __name__ == "__main__":
    main()
