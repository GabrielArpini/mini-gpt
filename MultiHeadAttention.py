import torch 
import torch.nn as nn 
from pos_encoding import RoPE
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, h: int, dropout = 0.0,max_seq_len:int, device='cpu'):
        self.d_model = d_model
        self.h = h 
        self.head_dim = d_model // h 
        self.device = device 
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len

        # q,k,v are represented with linear projections
        self.q_proj = nn.Linear(d_model,d_model)
        self.k_proj = nn.Linear(d_model,d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.w_0 = nn.Linear(d_mode,d_model) # Output after concatenation process. 
        
        # Instantiate RoPE 
        self.rope = RoPE(d_model=self.d_model,max_seq_len=max_seq_len,device=self.device)

        
        
    def forward(self, x):
        """x: input of shape (batch_size,seq_len, d_model)"""
        # Apply linear projections of q,k,v into the input tensor
        q_w = self.q_proj(x)
        k_w = self.k_proj(x)
        v_w = self.v_proj(x)
        
        # Embedd positons into q,k  
        q_pos_encoded = self.rope(q_w)
        v_pos_encoded = self.rope(k_w)
       
        #TODO: EVERYTHING BELLOW (Scaled-dot product and MultiHeadAttention)
        # Split q,k into shape:
        # (batch_size,seq_len, num_heads, head_dim)
        q_split = q_pos_encoded.reshape((q_pos_encoded.shape[0],q_pos_encoded.shape[1],self.h,self.head_dim))
        v_split = v_pos_encoded.reshape((v_pos_encoded.shape[0],v_pos_encoded.shape[1],self.h,self.head_dim))
        k_w_T = k_w.t()

        # Scaled Dot-Product Attention 
        product = torch.matmul(q_split,k_w_T)
        scaled_product = product / torch.sqrt(head_dim)



