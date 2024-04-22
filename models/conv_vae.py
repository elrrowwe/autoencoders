import torch.nn as nn


"""
A convolutional variational autoencoder model.
Conceptually the same as the vanilla VAE with the linear layers substituted for conv blocks.
"""


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 32, latent_dim: int = 128, kernel_size: int = 3, stride: int = 2):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(out_channels, out_channels * 2, kernel_size, stride),

            nn.Sigmoid(),

            nn.Flatten()
        )

        self.q_mean = nn.Linear(out_channels // 8, latent_dim)
        self.q_log_var = nn.Linear(out_channels // 8, latent_dim)

    def forward(self, inp):
        """
        The forward method of the convolutional VAE encoder.

        :param inp: an input image;
        :return: the mean and log variance of the modelled distribution q.
        """
        out = self.encoder(inp)
        mean = self.q_mean(out)
        log_var = self.q_log_var(out)

        return mean, log_var
