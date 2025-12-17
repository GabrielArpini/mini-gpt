import torch
import torch.nn as nn 
import matplotlib.pyplot as plt 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RoPE(nn.Module):
    def __init__(self,d_model: int, max_seq_len: int, base: int =10_000) -> None:
        """
        Initializes the variables and computes the sin and cos matricies.

        Args:

        d_model: A hyperparameter to define the size of vector space for each token and must be even.
        max_seq_len: A limit to memory usage, which defines the range of positional encodings RoPE computes.
        base: The base to compute theta.

        """
        super(RoPE,self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        print(d_model)
        assert d_model % 2 == 0, "d_model must be divisible by 2"

        
        pos_indices_vec = torch.arange(max_seq_len).unsqueeze(1) # shape [max_seq_len,1]

        d_dimensions_space = torch.arange(0, d_model , 2) # shape [d_model // 2],d even numbers. Since it jumps 2 in 2 the shape is d//2

        theta_i = base**(-d_dimensions_space/d_model) # shape [d_model // 2]

        angle_grid = pos_indices_vec * theta_i

        # Register as buffers so they move to device automatically with model.to(device)
        self.register_buffer('sin_angles', torch.sin(angle_grid))
        self.register_buffer('cos_angles', torch.cos(angle_grid))

    def forward(self,x):
        """
        Applies rotary positional embeddings to the input tensor.

        Args:
        x:  Input tensor of shape (batch_size, seq_len, d_model), where
            batch_size is the number of sequences, seq_len is the sequence length,
            and d_model is the embedding dimension.

        Returns:
            torch.Tensor: Rotated embeddings with the same shape as the input.
        """

        batch_size,seq_len,_ = x.shape 
        if seq_len > self.max_seq_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        # Need to extract the even and odd dimensions from d_model.
        x_even = x[:,:,0::2]
        x_odd = x[:,:,1::2]

        # Angle grid was made with max_seq_len size, so we need to account that before
        # Since sin and cos have shape (seq_len, d_model // 2)
        # and x have batch_size, we need to unsqueeze sin and cos
        sin_reduced = self.sin_angles[:seq_len,:].unsqueeze(0)
        cos_reduced = self.cos_angles[:seq_len,:].unsqueeze(0)

        # Apply rotations
        x_even_rotated = x_even*cos_reduced - x_odd*sin_reduced
        x_odd_rotated = x_even*sin_reduced + x_odd*cos_reduced
        
        # Using concatenation will not recreate original structure
        # it will have all even numbers then all odd numbers
        # we need them interleaved.
        #
        x = torch.stack((x_even_rotated,x_odd_rotated), dim=-1)
        x = x.view(batch_size,seq_len, -1)
        return x


    @staticmethod
    def plot(sin_angles, cos_angles, dim_idx=0, title="RoPE Sin/Cos Values"):
        plt.plot(sin_angles[:, dim_idx], label="sin")
        plt.plot(cos_angles[:, dim_idx], label="cos")
        plt.xlabel("Position")
        plt.ylabel(f"Value (Dimension {dim_idx})")
        plt.title(title)
        plt.legend()
        plt.show()
            

        

        




