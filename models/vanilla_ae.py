import torch.nn as nn


"""
The simplest autoencoder implementation, with the encoder and decoder parts of the network being basic MLPs.
is meant to be run on CPU.
"""


class VanillaAutoencoder(nn.Module):
    def __init__(self, inp_dim: int = 28, encoder_hidden_dim: int = 128, decoder_output_dim: int = 28):
        """

        :param encoder_hidden_dim:
        :param decoder_output_dim:
        """
        super(VanillaAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(inp_dim, encoder_hidden_dim),  # 28, 128

            nn.LeakyReLU(0.001),

            nn.Dropout(0.2),  # dropout for hidden layers to prevent overfitting, though the model is simple enough to not overfit either way

            nn.Linear(encoder_hidden_dim, encoder_hidden_dim // 2),  # 128, 64

            nn.LeakyReLU(0.001),

            nn.Dropout(0.2),

            nn.Linear(encoder_hidden_dim // 2, encoder_hidden_dim // 4),  # 64, 32

            nn.LeakyReLU(0.001),

            nn.Linear(encoder_hidden_dim // 4, encoder_hidden_dim // 8),  # 32, 16

        )

        self.decoder = nn.Sequential(
            nn.Linear(encoder_hidden_dim // 8, encoder_hidden_dim // 4),  # 16, 32

            nn.LeakyReLU(0.001),

            nn.Dropout(0.2),

            nn.Linear(encoder_hidden_dim // 4, encoder_hidden_dim // 2),  # 32, 64

            nn.LeakyReLU(0.001),

            nn.Dropout(0.2),

            nn.Linear(encoder_hidden_dim // 2, encoder_hidden_dim),  # 64, 128

            nn.LeakyReLU(0.001),

            nn.Linear(encoder_hidden_dim, decoder_output_dim),  # 128, 28
        )


    def forward(self, inp_image, return_compressed_input: bool = False):
        """
        The forward method of the Autoencoder.

        :param inp_image: the image to be run through the networks.
        :param return_compressed_input: whether to return the latent representation of the input (is False by default)
        :return: the image, reconstructed from its compressed representation (and the representation itself, optionally)
        """

        encoder_output = self.encoder(inp_image)
        decoder_output = self.decoder(encoder_output)

        if return_compressed_input:
            return encoder_output, decoder_output
        else:
            return decoder_output