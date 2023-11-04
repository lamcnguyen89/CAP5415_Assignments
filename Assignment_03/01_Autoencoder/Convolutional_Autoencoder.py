"""

1. Implement an autoencoder using fully connected layers. 
    a. The encoder will have 2 layers (with 256, and 128 neurons)
    b. The decoder will have 2 layers (with 256 and 784 neurons)
    c. Train this network using MSE loss for 10 epochs
    d. Compare the number of parameters  in the encoder and decoder.
    e. Create a writeup:
        i. Show 20 reconstructed images from testing data (2 image for each class)
        ii. Show original images

"""

import torch.nn as nn # All the Neural network models, loss functions

class Autoencoder(nn.Module):

    def __init__(self):
        pass

    def forward(self, x):
        pass