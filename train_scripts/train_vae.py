import torch.nn as nn
from torch.optim.adam import Adam
import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

from models.variational_ae import VAE
from utils.kldiv import kldiv_loss
from utils.batch import batch

"""
The training file for the Variational Autoencoder model. 
Model training and inference are run on CPU. 
"""

TRAIN_ITERS = 1000
CHECKPOINT_ITERS = 100

transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0,), (1,))
                             ])

# loading the MNIST digits dataset; simultaneously separating it into train/test portions
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# initializing the VAE model
vae = VAE(28, 128, 512)

# adam optimizer
optimizer = Adam(params=vae.parameters(), lr=0.0001)

# the training loop
losses = []
for epoch in range(TRAIN_ITERS):
    curr_batch = batch(mnist_trainset)
    optimizer.zero_grad()

    if epoch > 0 and epoch % CHECKPOINT_ITERS == 0:
        print(f'current epoch: {epoch}, average loss: {sum(losses) / len(losses)}')
        vae.eval()

        # the first image from the test set
        test_img = batch(mnist_testset, batch_size=1)[0]

        mean, log_var, test_img_decoded = vae.forward(test_img, return_encoder_output=True)

        test_img_decoded = test_img_decoded.reshape((28, 28)).detach().numpy()

        # plotting the original image
        plt.imshow(test_img.reshape((28, 28)).numpy())
        plt.title(f'The original image, current epoch: {epoch}, average loss: {sum(losses) / len(losses)}')
        plt.show()

        # printing the encoder-produced mean, log var
        print(f'mean: {mean}, log_var: {log_var}')

        # plotting the reconstructed image
        plt.imshow(test_img_decoded)
        plt.title(f'The image, reconstructed, current epoch: {epoch}, average loss: {sum(losses) / len(losses)}')
        plt.show()

    for img in curr_batch:
        inp_image = img
        mean, log_var, decoder_output = vae(inp_image)

        vae_loss = kldiv_loss(decoder_output, inp_image, mean, log_var)
        losses.append(vae_loss.item())
        vae_loss.backward()
        optimizer.step()
