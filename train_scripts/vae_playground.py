import torch
import matplotlib.pyplot as plt

from models.conv_vae import (
    Encoder,
    Decoder,
    CVAE
)


"""
A file dedicated to whatever I come up with for my VAE/CVAE implementation: inference, modified architecture tests etc.
"""


# TODO: pass noise sampled from the learned distribution instead of the normal N(0, I) directly

# initializing the CVAE model + the model encoder, decoder
encoder = Encoder()
decoder = Decoder()

cvae = CVAE(encoder, decoder)

# loading the last model checkpoint
checkpoint = torch.load('../models/cvae_model.pt')
cvae.load_state_dict(checkpoint['model_state_dict'])

# loading the encoder model
encoder_checkpoint = torch.load('../models/encoder_model.pt')
encoder.load_state_dict(encoder_checkpoint['model_state_dict'])

# loading the decoder model
decoder_checkpoint = torch.load('../models/decoder_model.pt')
decoder.load_state_dict(decoder_checkpoint['model_state_dict'])

# testing the model on random noise
"""
The  decoder part of the network should perform well on noise sampled from the normal distribution,
since the objective pushes the modelled distribution q to be as close as possible to N(0,I).
"""
mean, log_var, _ = cvae.forward(torch.randn((1, 1, 28, 28)))
var = torch.exp(0.5 * log_var)
z = cvae.reparameterization(mean, var)

cvae.eval()
out = cvae.inference(z)

plt.imshow(out.reshape((28, 28)).cpu().detach().numpy())
plt.title('The output of the model given random noise')
plt.show()
