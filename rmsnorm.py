import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm is a simpler and more efficient alternative to LayerNorm that
    normalizes using only the root mean square statistic, without mean centering.

    Reference: "Root Mean Square Layer Normalization" (https://arxiv.org/abs/1910.07467)
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Args:
            d_model: The dimension of the model (feature dimension).
            eps: A small value to avoid division by zero, default 1e-6.
        """
        super().__init__()
        self.eps = eps
        # Learnable scaling parameter (gamma/g_i in the paper)
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., d_model) - corresponds to a_i in the paper

        Returns:
            Normalized tensor of the same shape as input - corresponds to ā_i in the paper
        """
        # Calculate RMS: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize and scale: ā_i = (a_i / RMS(a)) * g_i
        x_normalized = x / rms

        return self.weight * x_normalized
