import torch
from torch.optim.adam import Adam
import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

from models.conv_vae import (
    Encoder,
    Decoder,
    CVAE
)
from utils.kldiv import kldiv_loss
from utils.batch import batch


"""
The training script for the CVAE model. 
Conceptually the same as the one for the VAE model.
Is meant to be run on GPU.
"""


TRAIN_ITERS = 100
CHECKPOINT_ITERS = 100

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'device: {device}')


transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0,), (1,))
                             ])

# loading the MNIST digits dataset; simultaneously separating it into train/test portions
mnist_trainset = datasets.MNIST(root='./data_mnist', train=True, download=True, transform=transform)
mnist_testset = datasets.MNIST(root='./data_mnist', train=False, download=True, transform=transform)

# initializing the CVAE model (encoder, decoder, CVAE)
encoder = Encoder()

decoder = Decoder()

cvae = CVAE(encoder, decoder).to(device)

# adam optimizer
optimizer = Adam(params=cvae.parameters(), lr=0.0001)

# the training loop
losses = [1000] # a silly init value
for epoch in range(TRAIN_ITERS):
    curr_batch = batch(mnist_trainset, batch_size=100, cvae=True)
    optimizer.zero_grad()

    if epoch > 0 and epoch % CHECKPOINT_ITERS == 0:
        print(f'current epoch: {epoch}, average loss: {sum(losses) / len(losses)}')
        cvae.eval()

        # the first image from the test set
        test_img = batch(mnist_testset, batch_size=1, cvae=True)[0].to(device)

        mean, log_var, test_img_decoded = cvae.forward(test_img)

        test_img_decoded = (test_img_decoded.reshape((28, 28)).cpu().detach().numpy())

        # plotting the original image
        plt.imshow(test_img.reshape((28, 28)).cpu().numpy())
        plt.title(f'The original image, current epoch: {epoch}, average loss: {sum(losses) / len(losses)}')
        plt.show()

        # plotting the reconstructed image
        plt.imshow(test_img_decoded)
        plt.title(f'The image, reconstructed, current epoch: {epoch}, average loss: {sum(losses) / len(losses)}')
        plt.show()

    for img in curr_batch:
        inp_image = img.to(device)
        mean, log_var, decoder_output = cvae(inp_image)

        vae_loss = kldiv_loss(decoder_output, inp_image, mean, log_var)
        losses.append(vae_loss.item())
        vae_loss.backward()
        optimizer.step()


# plotting the loss statistics
plt.plot(losses)
plt.title(f'Loss statistics for {TRAIN_ITERS} epochs')
plt.show()

# saving the model
PATH = '../models/cvae_model.pt'
PATH_encoder = '../models/encoder_model.pt'
PATH_decoder = '../models/decoder_model.pt'

torch.save({
            'epoch': TRAIN_ITERS,
            'model_state_dict': cvae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses[-1],
            }, PATH)

# saving the encoder
torch.save({
    'epoch': TRAIN_ITERS,
    'model_state_dict': encoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, PATH_encoder)

# saving the decoder
torch.save({
    'epoch': TRAIN_ITERS,
    'model_state_dict': decoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, PATH_decoder)
