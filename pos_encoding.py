import torch
import torch.nn as nn 
import matplotlib.pyplot as plt 

class RoPE(nn.Module):
    def __init__(self,d_model, max_seq_len, base=10_000):
        """
        d_model is a hyperparameter to define the size of vector space for each token and must be even.
        max_seq_len is a limit to memory usage, which defines the range of positional encodings RoPE computes.
        """
        super(RoPE,self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        assert d_model % 2 == 0, "d_model must be divisible by 2"

        
        pos_indices_vec = torch.arange(max_seq_len).unsqueeze(1) # shape [max_seq_len,1]

        d_dimensions_space = torch.arange(0, d_model , 2) # shape [d_model // 2],d even numbers. Since it jumos 2 in 2 the shape is d//2

        theta_i = base**(-d_dimensions_space/d_model) # shape [d_model // 2]

        angle_grid = pos_indices_vec * theta_i

        self.sin_angles = torch.sin(angle_grid)
        self.cos_angles = torch.cos(angle_grid)

    @staticmethod
    def plot(x):
        plt.plot(x)
        plt.show()


    def forward(self,x):
        """
        x: shape(batch_size,seq_len,d_model)
        """
        
        batch_size,seq_len,_ = x.shape 
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



            

        

        




