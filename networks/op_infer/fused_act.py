import os

import torch
from torch import nn
from torch.nn import functional as F

class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        if bias:
            self.bias = nn.Parameter(torch.zeros(channel))

        else:
            self.bias = None

        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)

def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    """
    Inference-only version of fused bias + leaky relu + scale.
    Equivalent to the forward pass of FusedLeakyReLUFunction with act=3, grad=0.

    Args:
        input (torch.Tensor): Input tensor of shape [N, C, H, W] or [N, C].
        bias (torch.Tensor): Optional bias of shape [C].
        negative_slope (float): Slope for negative values in LeakyReLU.
        scale (float): Final scaling factor.

    Returns:
        torch.Tensor: Output tensor with applied bias, leaky relu, and scale.
    """
    if bias is not None:
        # Reshape bias to be broadcastable: [1, C, 1, 1] for 4D, [1, C] for 2D
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        bias_view = bias.view(1, bias.shape[0], *rest_dim)
        input = input + bias_view

    # Apply LeakyReLU and scale
    return F.leaky_relu(input, negative_slope=negative_slope) * scale