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

class FCC_Autoencoder(nn.Module):

    def __init__(self,input_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=64),
            nn.ReLU(),
            nn.Linear(64,12),
            nn.ReLU(),
            nn.Linear(12,3)

        )

        self.decoder = nn.Sequential(
            nn.Linear(3,12),
            nn.ReLU(),
            nn.Linear(12,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
# Note: If images are in the range (-1,1) apply Tanh() activation instead of sigmoid

