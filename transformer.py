from __future__ import annotations

import torch
import torch.nn as nn 
from MultiHeadAttention import MultiHeadAttention




class TransformerBlock(nn.Module):
    def __init__(self, mha: MultiHeadAttention, d_model: int = 128, dropout: float = 0.0):
        super(TransformerBlock,self).__init__()
        self.mha = mha 
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model,4*d_model),
            nn.GeLU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model)
        )
        self.dropout = dropout 

    def _sublayer(self,x):

        y = self.norm1(x)
        y = self.mha(y)
        z = y + x
        z = self.norm2(z)
        w_0 = self.ff(z)
        return w_0+z
        
    
    def forward(self,x):
        x = self._sublayer(x)
        return x 


class Transformer(nn.Module):
    def __init__(self, N, vocab_size:int, mha_params: dict, block_dropout: float = 0.0):
        super(Transformer,self).__init__()
        self.d_model = mha_params["d_model"]
        self.blocklist = nn.ModuleList([
            TransformerBlock(MultiHeadAttention(**mha_params),self.d_model, block_dropout) 
            for _ in range(N)])

        self.final_layer = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model,vocab_size),
            nn.Softmax(dim=-1) 
        )
    def forward(self, x):
        for block in self.blocklist:
            x = block(x)
        x = self.final_layer(x)
        return x


