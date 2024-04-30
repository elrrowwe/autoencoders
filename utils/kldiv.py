import torch.nn.functional as F
import torch


"""
The loss for the Variational Autoencoder model.
Apart from the binary cross entropy term, also includes the KL divergence term. 
"""

def kldiv_loss(x_pred, x_true, mean, log_var):
    """
    Computes the loss for the Variational Autoencoder model.

    :param x_pred: the reconstructed input to the VAE model;
    :param x_true: the input, untouched;
    :param mean: the mean produced by the encoder;
    :param log_var: the log variance produced by the encoder;
    :return: the loss.
    """
    bce_loss = F.binary_cross_entropy(x_pred, x_true, reduction='sum')
    kldiv = -5e-4 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
    loss = torch.mean(bce_loss + kldiv)

    # kldiv = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
    # loss = bce_loss + kldiv

    return loss
