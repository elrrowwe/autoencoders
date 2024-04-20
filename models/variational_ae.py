import torch.nn as nn

"""
This file contains my implementation of the Variational Autoencoder model. Is meant to be trained/inferenced on CPU.
"""

# TODO: remove redundant arguments/parameters, write better docs


class Encoder(nn.Module):
    def __init__(self, inp_dim: int = 128, hidden_dim: int = 128, latent_dim: int = 512):
        """

        :param hidden_dim: the hidden dimension of the network;
        :param latent_dim: the dimension of the latent space.
        """
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim), # 28, 128 (DEFAULT VALUES)

            nn.LeakyReLU(0.001),

            nn.Linear(inp_dim, hidden_dim), # 28, 128

            nn.LeakyReLU(0.001),

            nn.Linear(inp_dim, hidden_dim), # 28, 128

            nn.LeakyReLU(0.001)
        )

        self.q_mean = nn.Linear(hidden_dim, latent_dim) # the mean of distribution q (the approximation of the actual p distribution)#
        self.q_var = nn.Linear(hidden_dim, latent_dim) # log of the variance of q

    def forward(self, inp):
        """
        the forward method of the encoder. outputs the mean and variance of the noise distribution q,
        samples from which are likely to cause the decoder to reproduce x (the input).

        :param inp: the input x;
        :return: the mean, log variance of q.
        """
        out = self.encoder(inp)

        mean = self.q_mean(out) # 128, 512
        log_var = self.q_var(out) # 128, 512

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 512, hidden_dim: int = 128, output_dim: int = 28 ):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), # 512, 128 (DEFAULT ARGUMENTS)

            nn.LeakyReLU(0.001),

            nn.Linear(hidden_dim, hidden_dim),  # 128, 128

            nn.LeakyReLU(0.001),

            nn.Linear(hidden_dim, hidden_dim), # 128, 128

            nn.LeakyReLU(0.001),

            nn.Linear(hidden_dim, output_dim),

            nn.Sigmoid() # clamps the output of the previous layer to the range [0, 1]; important to have the image normalized
        )

    def forward(self, inp):
        """
        The forward method of the encoder.

        :param inp:
        :return:
        """
        out = self.decoder(inp)

        return out

class VAE(nn.Module):
    def __init__(self, inp_dim: int = 28, encoder_hidden_dim: int = 128, encoder_latent_dim: int = 512, decoder_output_dim: int = 28):
        super(VAE, self).__init__()
