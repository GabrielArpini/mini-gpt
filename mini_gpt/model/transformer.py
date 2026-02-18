from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from mini_gpt.model.attention import MultiHeadAttention
from mini_gpt.model.rmsnorm import RMSNorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SwiGLU(nn.Module):
    """
    A standard SwiGLU FFN implementation.
    Reference: Noam Shazeer's "GLU Variants Improve Transformer"
    (https://arxiv.org/abs/2002.05202)
    """
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        # The SwiGLU paper recommends the hidden dimension be 2/3 of the FFN dimension
        hidden_dim = int(2 * d_ffn / 3)

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        # First linear projection for the gate, activated by SiLU (Swish)
        gate = F.silu(self.w1(x))
        # Second linear projection for the data
        data = self.w2(x)
        # Element-wise multiplication, followed by the final projection
        return self.w3(gate * data)

class TransformerBlock(nn.Module):
    """Block to be iterated N times"""
    def __init__(self, mha: MultiHeadAttention, d_model: int = 128, dropout: float = 0.0):
        """
        Args:
            mha: Multi-Head Attention instance.
            d_model: embedding dimension, default 128.
            dropout: percentage of dropout, default 0.
        """
        super(TransformerBlock,self).__init__()
        self.mha = mha
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.ff = SwiGLU(d_model, 4*d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self,x, kv_cache=None, start_pos=0, use_cache=False):
        # Apply dropout to both attention and feedforward paths
        if use_cache:
            mha_norm, kv_cache = self.mha(self.norm1(x), start_pos=start_pos, kv_cache=kv_cache)
        else:
            mha_norm, _ = self.mha(self.norm1(x))
        x = x + self.dropout(mha_norm)
        x = x + self.dropout(self.ff(self.norm2(x)))
        if use_cache:
            return x,kv_cache 
        else:
            return x 

class Transformer(nn.Module):
    def __init__(self, vocab_size:int, mha_params: dict, N: int = 8, block_dropout: float = 0.0):
        """
        Args:
            N: number of times to iterate TransformerBlock.
            vocab_size: The size of the vocabulary.
            mha_params: parameters for MultiHeadAttention, it will be unpacked so careful with name.
            block_dropout: dropout to ble applied in TransformerBlock.
        """
        super(Transformer,self).__init__()
        self.d_model = mha_params['d_model']
        self.embedding = nn.Embedding(vocab_size, self.d_model).to(device)


        self.blocklist = nn.ModuleList([
            TransformerBlock(MultiHeadAttention(**mha_params),self.d_model, block_dropout).to(device)
            for _ in range(N)])

        self.final_layer = nn.Sequential(
            RMSNorm(self.d_model),
            nn.Linear(self.d_model,vocab_size)
        ).to(device)

    def forward(self, x, prev_caches=None, start_pos=0, use_cache=False):
        x = self.embedding(x).clone() # Fix torch.compile error
        
        if prev_caches and use_cache:
            caches = []
            for i,block in enumerate(self.blocklist):
                x, cache = block(x, kv_cache=prev_caches[i], start_pos=start_pos, use_cache=use_cache)
                caches.append(cache)

        else:
            caches = []
            for block in self.blocklist:
                if use_cache:
                    x, cache = block(x, use_cache=use_cache) # pos will be 0 anyways, so no need to pass param. 
                    caches.append(cache)
                else:
                    x = block(x)

        x = self.final_layer(x)
        
        if use_cache:
            return x, caches
        return x 
