import torch
import torch.nn as nn

"""
This file contains my implementation of the Variational Autoencoder model. Is meant to be trained/inferenced on CPU.
"""

# TODO: remove redundant arguments/parameters, write better docs, explain the reparameterization trick better


class Encoder(nn.Module):
    def __init__(self, inp_dim: int = 28, hidden_dim: int = 128, latent_dim: int = 512):
        """

        :param hidden_dim: the hidden dimension of the network;
        :param latent_dim: the dimension of the latent space.
        """
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim), # 28, 128 (DEFAULT VALUES)

            # nn.Dropout(0.2),

            nn.ReLU(True),

            nn.Linear(hidden_dim, hidden_dim), # 128, 128

            # nn.Dropout(0.2),

            nn.ReLU(True),

            nn.Linear(hidden_dim, hidden_dim), # 128, 128

            nn.ReLU(True)
        )

        self.q_mean = nn.Linear(hidden_dim, latent_dim) # the mean of distribution q (an approximation of p)
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
    def __init__(self, latent_dim: int = 512, hidden_dim: int = 128, output_dim: int = 28):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), # 512, 128 (DEFAULT ARGUMENTS)

            nn.ReLU(True),

            # nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim),  # 128, 128

            # nn.Dropout(0.2, inplace=True),

            nn.ReLU(True),

            nn.Linear(hidden_dim, hidden_dim), # 128, 128

            nn.ReLU(True),

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
    def __init__(self, inp_dim: int = 28, hidden_dim: int = 128, latent_dim: int = 512):
        super(VAE, self).__init__()
        self.encoder = Encoder(inp_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, inp_dim)

    def reparameterization(self, mean, var):
        """
        The reparameterization trick, commonly used in variational autoencoders.
        Instead of sampling from N(encoder_mean, encoder_var), we sample noise from the normal distribution N(0, I),
        then multiply the noise by given encoder-generated mean and variance,
        thus allowing for cheaper encoder_mean, encoder_var optimization and addressing the bottleneck imposed
        on the network by the random node (sampling from N(encoder_mean, encoder_var) directly).

        :param mean: the mean to add to epsilon;
        :param var: the variance by which epsilon is to be multiplied;
        :return: noise z, approximately from the distribution N(encoder_mean, encoder_var).
        """
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon

        return z

    def forward(self, inp, return_encoder_output: bool = True):
        """
        The forward method of the Variational Autoencoder model.
        Returns the reconstructed input, optionally returns the ouput of the encoder (the mean, log variance of distribution q)
        :param inp: vae input;
        :param return_encoder_output: whether to return mean, log_variance from the encoder; True by default;
        :return: an image, reconstructed; optionally returns the output of the encoder.
        """
        mean, log_var = self.encoder(inp)
        var = torch.exp(0.5 * log_var) # converting the log variance, returned by the encoder, into variance (exponentiating the log)

        z = self.reparameterization(mean, var)

        out = self.decoder(z)

        if return_encoder_output:
            return mean, log_var, out
        else:
            return out

    def inference(self, inp):
        """
        Method for inferencing/testing the model. Essentially "ignores" the encoder part.

        :param inp: some decoder input of appropriate dimensions, e.g. random noise;
        :return: an image, reconstructed from the input.
        """
        out = self.decoder(inp)

        return out