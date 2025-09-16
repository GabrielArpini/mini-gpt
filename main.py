from datasets import load_dataset
from utils import * 
from tokenizer import Tokenizer 
import os 
from pos_encoding import RoPE
from MultiHeadAttention import MultiHeadAttention
from transformer import Transformer, TransformerBlock


import torch 
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    

if __name__ == "__main__":
    main()
