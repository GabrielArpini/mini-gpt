from __future__ import annotations

import torch
import torch.nn as nn 
from MultiHeadAttention import MultiHeadAttention




class DecodeBlock:
    def __init__(self, dropout: float=0.0):
        super(DecodeBlock,self).__init__()
        self.mha = MultiHeadAttention(*params) 

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

    def forward(self,x):
        x = self._sublayer1(x)
        x = self._sublayer2(x)


