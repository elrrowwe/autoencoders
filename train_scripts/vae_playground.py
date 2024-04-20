import torch
import matplotlib.pyplot as plt

from models.variational_ae import VAE


"""
A file dedicated to whatever I come up with for my VAE implementation: inference, modified architecture tests etc.
"""

# initializing the VAE model
vae = VAE(28, 512, 128)

# loading the last model checkpoint
checkpoint = torch.load('model.pt')
vae.load_state_dict(checkpoint['model_state_dict'])

# testing the model on random noise
z = torch.randn((28, 128))

vae.eval()
out = vae.inference(z)

plt.imshow(out.reshape((28, 28)).cpu().detach().numpy())
plt.title('The output of the Variational Autoencoder given random noise')
plt.show()
