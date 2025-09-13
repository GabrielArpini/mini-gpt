from __future__ import annotations

import torch
import torch.nn as nn 
from MultiHeadAttention import MultiHeadAttention

class Transformer:
    def __init__(self, dropout: float=0.0):
        super(Transformer,self).__init__()
        self.mha = MultiHeadAttention(*params)
        self.norm1 = nn.LayerNorm(d_model)
        self.sl1 = 

    def _sublayer1(self,x):
        norm = nn.LayerNorm(d_model)
        y = self.mha(x)
        z = y + x
        z = norm(z)
        return z
        
        
    def _sublayer2(self,x):
        out = nn.Sequential(
            nn.Linear(d_model,4*d_model),
            nn.GeLU(),
            nn.Dropout(self.dropout),
            nn.Linear(4*d_model, d_model)
        )
        return out(x)
