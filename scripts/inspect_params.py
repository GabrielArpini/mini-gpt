import torch
from mini_gpt.model import Transformer

# Create a small test model with same structure
mha_params = {
    'd_model': 512,
    'h': 8,
    'max_seq_len': 512,
    'dropout': 0.2
}

model = Transformer(vocab_size=20000, mha_params=mha_params, N=2, block_dropout=0.3)

print("Model Parameter Names and Shapes:")
print("=" * 80)

muon_params = []
adamw_params = []

for name, param in model.named_parameters():
    shape_str = str(tuple(param.shape))
    dims = param.ndim

    # Categorize based on the split logic
    if 'embedding' in name:
        category = "AdamW (embedding)"
        adamw_params.append((name, param))
    elif 'final_layer' in name:
        category = "AdamW (final layer)"
        adamw_params.append((name, param))
    elif 'norm' in name:
        category = "AdamW (LayerNorm)"
        adamw_params.append((name, param))
    elif 'bias' in name:
        category = "AdamW (bias)"
        adamw_params.append((name, param))
    elif param.ndim >= 2:
        category = "Muon (2D+ weights)"
        muon_params.append((name, param))
    else:
        category = "AdamW (1D other)"
        adamw_params.append((name, param))

    print(f"{name:50s} {shape_str:20s} {dims}D -> {category}")

print("\n" + "=" * 80)
print(f"\nSummary:")
print(f"  Muon parameters: {len(muon_params)}")
print(f"  AdamW parameters: {len(adamw_params)}")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Muon param count: {sum(p.numel() for _, p in muon_params):,}")
print(f"  AdamW param count: {sum(p.numel() for _, p in adamw_params):,}")
