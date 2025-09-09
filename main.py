from datasets import load_dataset
from utils import * 
from tokenizer import Tokenizer 
import os 
from pos_encoding import RoPE 
import torch 
torch.manual_seed(42)

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
    
    input_tensor = torch.randn(batch_size,seq_len,d_model)

    # Instantiate the RoPE module
    rope = RoPE(d_model=d_model, max_seq_len=max_seq_len)

    # Apply positional encoding
    output_tensor = rope(input_tensor)

    # Validate output shape
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    assert output_tensor.shape == input_tensor.shape, "Output shape should match input shape"
    cos_angles = rope.cos_angles[:5, :].numpy()
    sin_angles = rope.sin_angles[:5, :].numpy()
    # Visualize sine and cosine angles for the first few positions
    print("\nVisualizing angles for first 5 positions:")
    rope.plot(cos_angles,sin_angles)  # Plot sine angles for first 5 positions
 
    

    

if __name__ == "__main__":
    main()
