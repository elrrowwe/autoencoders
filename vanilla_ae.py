import torch.nn as nn

"""
Vanilla autoencoder class. The encoder and decoder parts of the network are essentially basic MLPs. (23.03.2024)
"""


class Encoder(nn.Module):
    def __init__(self, encoder_hidden_dim: int = 128, inp_dim: int = 28):
        """
        The decoder model of the Autoencoder architecture.

        :param encoder_hidden_dim: the dimension of the hidden layer of the encoder.
        :param inp_dim: the dimension of the input to the model.
        """
        super(Encoder, self).__init__()

        # the "compression" sequence, compresses the input from a 28*28 one to a 4*4 one
        self.encoder = nn.Sequential(
            nn.Linear(inp_dim * inp_dim, encoder_hidden_dim), #784, 128

            nn.LeakyReLU(0.001),

            nn.Linear(encoder_hidden_dim, encoder_hidden_dim // 2), #128, 64

            nn.LeakyReLU(0.001),

            nn.Linear(encoder_hidden_dim // 2, encoder_hidden_dim // 4), #64, 32

            nn.LeakyReLU(0.001),

            nn.Linear(encoder_hidden_dim // 4, encoder_hidden_dim // 8),  # 32, 16

        )


    def forward(self, inp_image):
        """
        The forward method of the encoder part of the Autoencoder model.
        Takes a binary image, outputs the latent representation of that image.

        :param inp: the input to the network (e.g. a 28 * 28 image); should be binary
        :return: the latent representation of the input (essentially a lossy compression of the input)
        """
        output = self.encoder(inp_image)

        return output


class Decoder(nn.Module):
    def __init__(self, encoder_hidden_dim: int = 128, decoder_output_dim: int = 28):
        """
        The decoder model of the Autoencoder architecture.

        :param encoder_hidden_dim: the dimension of the hidden layer of the encoder, used in image reconstruction (upsampling)
        :param decoder_output_dim: the dimension of the output of the decoder
        """
        super(Decoder, self).__init__()

        #  the "decompression" sequence. gradually upsamples the input from a 4*4 one to a 28*28 one.
        self.decoder = nn.Sequential(
            nn.Linear(encoder_hidden_dim // 8, encoder_hidden_dim // 4),  # 784, 128

            nn.LeakyReLU(0.001),

            nn.Linear(encoder_hidden_dim // 4, encoder_hidden_dim // 2),  # 128, 64

            nn.LeakyReLU(0.001),

            nn.Linear(encoder_hidden_dim // 2, encoder_hidden_dim),  # 64, 32

            nn.LeakyReLU(0.001),

            nn.Linear(encoder_hidden_dim, decoder_output_dim * decoder_output_dim),  # 32, 16
        )


    def forward(self, inp_latent_repr):
        """
        The forward method of the decoder part of the network.
        Takes the latent representation of an image, outputs the image, reconstructed from the representation

        :param inp_latent_repr: the latent representation of an image.
        :return: the image, reconstructed from its latent representation.
        """

        output = self.decoder(inp_latent_repr)

        return output


class VanillaAutoencoder(nn.Module):
    def __init__(self, encoder_hidden_dimension: int = 128, decoder_output_dimension: int = 28):
        """

        :param encoder_hidden_dimension:
        :param decoder_output_dimension:
        """
        super(VanillaAutoencoder, self).__init__()

        self.encoder = Encoder(encoder_hidden_dimension, decoder_output_dimension)
        self.decoder = Decoder(encoder_hidden_dimension, decoder_output_dimension)


    def forward(self, inp_image, return_compressed_input: bool = False):
        """
        The forward method of the Autoencoder.

        :param inp_image: the image to be run through the networks.
        :param return_compressed_input: whether to return the latent representation of the input (is False by default)
        :return: the image, reconstructed from its compressed representation (and the representation itself, optionally)
        """

        encoder_output = self.encoder.forward(inp_image)
        decoder_output = self.decoder.forward(encoder_output)

        if return_compressed_input:
            return encoder_output, decoder_output
        else:
            return decoder_output


