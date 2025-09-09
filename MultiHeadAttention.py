import torch 
import torch.nn as nn 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, h: int):
        self.d_model = d_model
        self.h = h 

        # Initialize Q,K,V
        self.Q = nn.Linear(d_model,d_model)
        self.K = nn.Linear(d_model,d_model)
        self.V = nn.Linear(d_model,d_model)

        # Compute linear projections for heads. 
        self.wQ = self.Q.reshape()

