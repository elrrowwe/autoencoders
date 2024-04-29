import torch
import matplotlib.pyplot as plt

from models.conv_vae import (
    Encoder,
    Decoder,
    CVAE
)
from models.variational_ae import VAE


"""
A file dedicated to whatever I come up with for my VAE/CVAE implementation: inference, modified architecture tests etc.
"""


# initializing the CVAE model + the model encoder, decoder
encoder = Encoder()
decoder = Decoder()

cvae = CVAE(encoder, decoder)

# initializing the VAE model
vae = VAE(28, 512, 256)

# loading the last cvae model checkpoint
cvae_checkpoint = torch.load('../models/cvae_model.pt')
cvae.load_state_dict(cvae_checkpoint['model_state_dict'])

# loading the cvae encoder model
encoder_checkpoint = torch.load('../models/encoder_model.pt')
encoder.load_state_dict(encoder_checkpoint['model_state_dict'])

# loading the cvae decoder model
decoder_checkpoint = torch.load('../models/decoder_model.pt')
decoder.load_state_dict(decoder_checkpoint['model_state_dict'])

# loading the vanilla vae model
vae_checkpoint = torch.load('../models/vae_model.pt')
vae.load_state_dict(vae_checkpoint['model_state_dict'])

# testing the model on random noise
"""
The  decoder part of the network should perform well on noise sampled from the normal distribution,
since the objective pushes the modelled distribution q to be as close as possible to N(0,I).
"""
mean, log_var, _ = cvae.forward(torch.randn((1, 1, 28, 28)))
var = torch.exp(0.5 * log_var)
# z = cvae.reparameterization(mean, var)
# z = torch.randn(1, 128, 16, 16)
z = torch.normal(mean, var)
cvae.eval()
out = cvae.inference(z)

vae.eval()
z_vae = torch.randn((28, 256))
vae_out = vae.inference(z_vae)

# visualizing the cvae output
plt.imshow(out.reshape((28, 28)).cpu().detach().numpy())
plt.title('The output of the CVAE model given random noise')
plt.show()

# visualizing the vae output
# plt.imshow(vae_out.reshape((28, 28)).cpu().detach().numpy())
# plt.title('The output of the VAE model given random noise')
# plt.show()
