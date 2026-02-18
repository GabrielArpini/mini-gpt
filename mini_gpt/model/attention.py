from __future__ import annotations

import torch
import torch.nn as nn
from mini_gpt.model.rope import RoPE
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func # first one after merging qkv.

class MultiHeadAttention(nn.Module):
    """
    Computes attention using q,k,v (query,key,value) in a multi head
    fashion by splitting the input into multiple heads.
    Adapted for MQA (Multi-Query Attention), simplifying thq k,v computation.

    """
    def __init__(self, d_model:int, h: int, max_seq_len: int, dropout: float = 0.0) -> None:
        """
        Args:
        d_model (int): The embedding dimension.
        h (int): number of heads.
        dropout (float): A percentage value of chance to randomly zeroes elements
        in the input tensor. Default is 0.0.
        max_seq_len (int): The maximum length of the sequence.

        """

        super(MultiHeadAttention,self).__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, f"d_model ({d_model}) must be divisible by h ({h})"
        self.head_dim = d_model // h
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len

        # q,k,v are represented with linear projections
        self.q_proj = nn.Linear(d_model,d_model)
        self.k_proj = nn.Linear(d_model,self.head_dim)
        self.v_proj = nn.Linear(d_model, self.head_dim)
        self.w_0 = nn.Linear(d_model,d_model) # Output after the whole process.

        # Instantiate RoPE
        self.rope = RoPE(d_model=self.head_dim, max_seq_len=max_seq_len)

        # Create causal mask once and register as buffer (not a parameter)
        # Shape: (1, 1, max_seq_len, max_seq_len) - broadcasts to any batch size and num heads
        mask = torch.tril(torch.ones((1, 1, max_seq_len, max_seq_len)))
        mask = mask.masked_fill(mask == 0, float('-inf'))
        self.register_buffer('causal_mask', mask)



    def forward(self, x: torch.Tensor, start_pos:int = 0, kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        """
        Args:
        x: Tensor of the tokenized list of shape (batch_size,seq_len, d_model)
        """
        # Apply linear projections of q,k,v into the input tensor
        q_w = self.q_proj(x)
        k_w = self.k_proj(x)
        v_w = self.v_proj(x)

        # Split q,k,v d_model into shape:
        # (batch_size,seq_len, num_heads, head_dim)
        batch_size, seq_len, _ = x.shape
        q_split = q_w.reshape((batch_size,seq_len,self.h,self.head_dim))
        k_split = k_w.reshape((batch_size,seq_len,1,self.head_dim))
        v_split = v_w.reshape((batch_size,seq_len,1,self.head_dim))

        # Apply RoPE to q and k after splitting into heads
        q_split = self.rope(q_split, start_pos)
        k_split = self.rope(k_split, start_pos)

        # Start falsh attention
        # Flash attention test, everything else bellow is commented out.
        # main objective here is to see how well flash attention performs
        # Next optimization will be merging the qkv splits into one thing for better performance.
        # See: https://github.com/Dao-AILab/flash-attention
        #x = flash_attn_func(q_split, k_split, v_split, dropout_p=0.1, causal=True) # Causal true for auto regressive.
        #x = x.reshape((batch_size, seq_len, self.d_model))
        #x = self.w_0(x)
        #return x
        # End of flash attention

        # bellow is default code.
        # For compute efficiency, we are going to transpose h dimensions earlier
        # for parallelization
        q_split = q_split.transpose(1,2)
        k_split = k_split.transpose(1,2)
        v_split = v_split.transpose(1,2)
        prefill = True 
        if kv_cache is None:
            kv_cache = [k_split,v_split]
            k_full, v_full = k_split, v_split
        else:
            prefill = False 
            k_full = torch.cat([kv_cache[0], k_split], dim=2)
            v_full = torch.cat([kv_cache[1], v_split], dim=2)


        # MQA: matmul handles broadcasting efficiently for k,v with shape (batch, 1, seq, head_dim)
        # Since shape is 4D and transpose is 2D, we need to specify which dimensions to transpose
        k_T = k_full.transpose(-2,-1)

        # Scaled Dot-Product Attention
        product = torch.matmul(q_split,k_T)
        scaled_product = product / torch.sqrt(torch.tensor(self.head_dim,device=x.device))

        # Masked self-attention - use pre-created mask, slice to current seq_len
        # Shape: (1, 1, seq_len, seq_len) broadcasts to (batch, h, seq_len, seq_len)
        if prefill:
            scaled_product = scaled_product + self.causal_mask[:, :, :seq_len, :seq_len]

        softmax_result = nn.functional.softmax(scaled_product,dim=-1)
        softmax_result = self.dropout(softmax_result)
        attention_qkv = torch.matmul(softmax_result,v_full)

        # PyTorch SDPA (commented out - backend selection unpredictable for MQA)
        #k_split = k_split.expand(batch_size, self.h, seq_len, self.head_dim)
        #v_split = v_split.expand(batch_size, self.h, seq_len, self.head_dim)
        #attention_qkv = nn.functional.scaled_dot_product_attention(
        #q_split, k_split, v_split,
        #dropout_p=0.1 if self.training else 0.0,
        #is_causal=True
        #)

        # Concatenate the last two dimensions back to d_model
        concat_result = attention_qkv.transpose(1,2)
        concat_result = concat_result.reshape((concat_result.shape[0],concat_result.shape[1],concat_result.shape[2]*concat_result.shape[3]))

        # Pass concat result into final projection.
        x = self.w_0(concat_result)

        return x, kv_cache
