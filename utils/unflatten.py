import torch
import torch.nn as nn


"""
A helper function for the CVAE model.
Addresses the problem of the output of the encoder (mean, log_var) being a 2D matrix, while the decoder
(its first conv layer in particular) expects 3- or 4- D input. 
"""


class UnFlatten(nn.Module):
    def forward(self, input: torch.Tensor):
        """
        Reshapes the input tensor so that it fits the following convlayer input dimension criteria:
        [batch_size, channels, height, width]

        :param input: a torch tensor to be reshaped;
        :param size: the nunber of channels of the output tensor;
        :return: the input tensor, reshaped to the desired dimensions.
        """
        return input.view(input.size(0), input.size(1), 1, 1)
