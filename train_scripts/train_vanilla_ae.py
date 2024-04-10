import torch.nn as nn
from torch.optim.adam import Adam
import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

from models.vanilla_ae import VanillaAutoencoder
from utils.batch import batch


"""
The training file for the vanilla autoencoder model. 
Model training and inference are run on CPU. 
"""

# TODO: add model checkpoints, test on more complex data + torchvision transforms

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

# initializing the model
vanilla_ae = VanillaAutoencoder()

# initializing the MSE loss from pytorch
mse_loss = nn.MSELoss()

# Adam as the optimizer. SGD, based on experience, resulted in less training stability and ultimately poor performance, in this case
optimizer = Adam(vanilla_ae.parameters(), lr=0.0001)

#the training loop
losses = []
for epoch in range(TRAIN_ITERS):
    curr_batch = batch(mnist_trainset)
    optimizer.zero_grad()

    if epoch > 0 and epoch % CHECKPOINT_ITERS == 0:
        print(f'current epoch: {epoch}, average loss: {sum(losses) / len(losses)}')
        vanilla_ae.eval()

        # the first image from the test set
        test_img = batch(mnist_testset, batch_size=1)[0]

        test_img_encoded, test_img_decoded = vanilla_ae.forward(test_img, return_compressed_input=True)

        test_img_encoded = test_img_encoded.squeeze().detach().numpy()
        test_img_decoded = test_img_decoded.reshape((28, 28)).detach().numpy()

        # plotting the original image
        plt.imshow(test_img.reshape((28, 28)).numpy())
        plt.title(f'The original image, current epoch: {epoch}, average loss: {sum(losses) / len(losses)}')
        plt.show()

        # plotting the encoding of the image
        plt.imshow(test_img_encoded)
        plt.title(f'The image, encoded, current epoch: {epoch}, average loss: {sum(losses) / len(losses)}, encoding dimensions: {test_img_encoded.size}')
        plt.show()

        # plotting the reconstructed image
        plt.imshow(test_img_decoded)
        plt.title(f'The image, reconstructed, current epoch: {epoch}, average loss: {sum(losses) / len(losses)}')
        plt.show()

    for img in curr_batch:
        inp_image = img
        ae_output = vanilla_ae(inp_image)

        ae_loss = mse_loss(ae_output, inp_image)
        losses.append(ae_loss.item())
        ae_loss.backward()
        optimizer.step()
