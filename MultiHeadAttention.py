import torch 
import torch.nn as nn 
from pos_encoding import RoPE
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, h: int, dropout = 0.0,max_seq_len:int):
        self.d_model = d_model
        self.h = h 
        self.head_dim = d_model // h 
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len

        # q,k,v are represented with linear projections
        self.q_proj = nn.Linear(d_model,d_model)
        self.k_proj = nn.Linear(d_model,d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.w_0 = nn.Linear(d_model,d_model) # Output after concatenation process. 
        
        # Instantiate RoPE 
        self.rope = RoPE(d_model=self.d_model,max_seq_len=max_seq_len,device=device=self.q_proj.weight.device)

        
        
    def forward(self, x):
        """x: input of shape (batch_size,seq_len, d_model)"""
        # Apply linear projections of q,k,v into the input tensor
        q_w = self.q_proj(x)
        k_w = self.k_proj(x)
        v_w = self.v_proj(x)
        
        # Embedd positons into q,k  
        q_pos_encoded = self.rope(q_w)
        k_pos_encoded = self.rope(k_w)
       
        # Split q,k,v into shape:
        # (batch_size,seq_len, num_heads, head_dim)

        q_split = q_pos_encoded.reshape((q_pos_encoded.shape[0],q_pos_encoded.shape[1],self.h,self.head_dim))
        k_split = k_pos_encoded.reshape((k_pos_encoded.shape[0],k_pos_encoded.shape[1],self.h,self.head_dim))

        v_split = v_w.reshape((v_w.shape[0],v_w.shape[1],self.h,self.head_dim))
        
        # For compute efficiency, we are going to transpose h dimensions earlier
        # for parallelization
        q_split = q_split.transpose(1,2)
        k_split = k_split.transpose(1,2)
        v_split = v_split.transpose(1,2)


        #Since shape is 4D and transpose is 2D, we need to specify which dimensions to transpose

        k_T = k_split.transpose(-2,-1)


        # Scaled Dot-Product Attention 
        product = torch.matmul(q_split,k_T)
        scaled_product = product / torch.sqrt(self.head_dim)
        softmax_result = nn.functional.softmax(scaled_product,dim=-1)
        softmax_result = self.dropout(softmax_result)
        attention_qkv = torch.matmul(softmax_result,v_split)

        # Concanate the last two dimensions back to d_model 
        concat_result = attention_qkv.transpose(2,1)
        concat_result = concat_result.reshape((concat_result.shape[0],concat_result.shape[1],concat_result.shape[2]*concat_result.shape[3]))
        
        x = self.w_0(concat_result)
        return x




