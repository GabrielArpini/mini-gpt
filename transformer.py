from __future__ import annotations

import torch
import torch.nn as nn 
from MultiHeadAttention import MultiHeadAttention


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerBlock(nn.Module):
    def __init__(self, mha: MultiHeadAttention, d_model: int = 128, dropout: float = 0.0):
        super(TransformerBlock,self).__init__()
        self.mha = mha 
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model,4*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model)
        )
        self.dropout = dropout 

    def _sublayer(self,x):
        x = x + self.mha(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x
        
    
    def forward(self,x):
        x = self._sublayer(x)
        return x 


class Transformer(nn.Module):
    def __init__(self, N, vocab_size:int, mha_params: dict, block_dropout: float = 0.0):
        super(Transformer,self).__init__()
        self.d_model = mha_params["d_model"]
        self.blocklist = nn.ModuleList([
            TransformerBlock(MultiHeadAttention(**mha_params),self.d_model, block_dropout).to(device) 
            for _ in range(N)])

        self.final_layer = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model,vocab_size)
        ).to(device)
    def forward(self, x):
        for block in self.blocklist:
            x = block(x)
        x = self.final_layer(x)
        return x


