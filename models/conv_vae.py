import torch.nn as nn
import torch


"""
A convolutional variational autoencoder model.
Conceptually the same as the vanilla VAE with the linear layers substituted for conv blocks.
"""

# TODO:
#  test batchnorm,
#  rewrite the docs


class Encoder(nn.Module):
    def __init__(self,
                 conv_kernel_size: int = 5,
                 pool_kernel_size: int = 2,
                 stride: int = 1,
                 pool_stride: int = 2):
        """
        The encoder part of the CVAE model.
        Based on the feature maps created by the conv layers,
        the encoder outputs the mean, log variance of the noise distribution q.

        """
        super(Encoder, self).__init__()

        # a LeNet-like convolutional encoder
        self.encoder = nn.Sequential(
            nn.LazyConv2d(32, conv_kernel_size, stride, padding=2),

            nn.ReLU(),

            nn.MaxPool2d(pool_kernel_size, pool_stride),

            nn.LazyConv2d(64, conv_kernel_size, stride),

            nn.ReLU(),

            nn.MaxPool2d(pool_kernel_size, pool_stride),

            nn.Flatten(),  # should be 1x400

            nn.LazyLinear(400),

            nn.ReLU(),

            nn.LazyLinear(400),

            nn.ReLU(),

            nn.LazyLinear(256),

            nn.ReLU(),

            nn.LazyLinear(128),
        )

        self.q_mean = nn.LazyLinear(100)
        self.q_log_var = nn.LazyLinear(100)

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


class Decoder(nn.Module):
    def __init__(self,
                 kernel_size: int = 5,
                 stride: int = 1,
                 padding: int = 0):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.LazyLinear(128),

            nn.ReLU(),

            nn.LazyLinear(256),

            nn.ReLU(),

            nn.LazyLinear(400),

            nn.ReLU(),

            nn.Unflatten(1, (16, 5, 5)),

            nn.ConvTranspose2d(16, 32, kernel_size, stride, padding),

            nn.ReLU(),

            nn.ConvTranspose2d(32, 32, kernel_size, stride, padding),

            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size, stride, padding),

            nn.ReLU(),

            nn.ConvTranspose2d(16, 16, kernel_size, stride, padding),

            nn.ReLU(),

            nn.ConvTranspose2d(16, 16, kernel_size, stride, padding),

            nn.ReLU(),

            nn.ConvTranspose2d(16, 1, kernel_size-1, stride, padding),

            nn.Sigmoid(),
        )

    def forward(self, inp):
        """
        The forward method of the CVAE decoder.

        :param inp: the input tensor;
        :return: an image, reconstructed from some input (noise).
        """
        out = self.decoder(inp)

        return out.reshape((1, 1, 28, 28))


class CVAE(nn.Module):
    def __init__(self, encoder, decoder):
        """
        The Convolutional Variational Autoencoder class.
        Accepts the two parts of the network as their respective class instances: the encoder and the decoder.
        Such construction may be subject to change in the future.

        :param encoder: the encoder part of the CVAE model;
        :param decoder: the decoder part of the CVAE model.
        """
        super(CVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder


    def reparameterization(self, mean: torch.Tensor, var: torch.Tensor):
        """
        The reparameterization trick, commonly used in variational autoencoders.
        Instead of sampling from N(encoder_mean, encoder_var), we sample noise from the normal distribution N(0, I),
        then multiply the noise by given encoder-generated mean and variance,
        thus allowing for cheaper encoder_mean, encoder_var optimization and addressing the bottleneck imposed
        on the network by the random node (sampling from N(encoder_mean, encoder_var) directly).

        :param mean: the encoder-produced mean;
        :param var: the encoder-produced log variance;
        :return: noise z.
        """
        epsilon = torch.randn_like(var)
        # z = epsilon.mul(var).add(mean)

        z = (epsilon * var) + mean

        return z

    def forward(self, inp: torch.Tensor):
        """
        The forward method of the CVAE model; conceptually the same as that of the vanilla VAE model.

        :param inp: an input tensor.
        :return: the input, reconstructed.
        """
        mean, log_var = self.encoder(inp)
        var = torch.exp(0.5 * log_var) # converting the log variance, returned by the encoder, into variance (exponentiating the log)

        z = self.reparameterization(mean, var)

        decoder_out = self.decoder(z)

        return mean, log_var,  decoder_out

    def inference(self, z):
        """
        Inference method for the CVAE model.
        Essentially drops the "encoder" part of the network,
        utilizing just the decoder to reconstruct an image from input noise.
        (should work, provided that q is close enough to N(0, I))
        Conceptually the same as that for the vanilla VAE.

        :param z: an input noise tensor, should be sampled from a normal distribution N(0, I)
        :return: an image reconstructed from input noise
        """
        decoder_out = self.decoder(z)

        return decoder_out