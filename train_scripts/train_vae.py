import torch.nn as nn
from torch.optim.adam import Adam
import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

from models.variational_ae import VAE


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

mse_loss = nn.MSELoss()