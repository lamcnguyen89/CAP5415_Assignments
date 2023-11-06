"""

2. Implement an autoencoder using Convolutional layers. 
    a. The encoder will have 2 convolutional layers and 2 max pooling layers
        i. Use kernel size 3x3
        ii. reLU activation
        iii. padding of 1 to preserve the feature map.
    b. The decoder will have 3 convolutional layers
        i. kernel shape is 3x3
        ii. padding = 1
        iii. The first 3 convolutional layers will be followed by an upsampling layer.
                a. This upsampling layer will double the resolution of the feature maps using linear interpolation
    c. Train the network for 10 epochs
    d. Compare the number of parameters in the encoder and decoder.
    e. Compare the total parameters in this autoencoder with the previous autoencoder.
    f. Create Writeup:
        i. Show 20 sample reconstructed images from testing data (2 images for each class)
        ii. show original images
        iii. Compare the reconstructed results with the previous autoencoder

"""

import torch.nn as nn # All the Neural network models, loss functions
import torch.nn.functional as F

""" 

Formula for calculating size of feature map edge for Convolution:

L_out = [(L_in + 2 * padding - dilation * (kernel_size - 1) -1)/stride] + 1


Formula for calculating size of feature map edge for Transpose Convolution:

L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1

"""



class CNN_Autoencoder(nn.Module):
    def __init__(self):
        #N,1,28,28
        super(CNN_Autoencoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1), # N,16,10,10
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # N,16,5,5
            nn.Conv2d(16, 8, 3, stride=2, padding=1), # N,8,3,3
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1) # N,8,2,2
        )
        self.decoder = nn.Sequential(
        nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1), # N,16,3,3
        nn.ReLU(),
        nn.Upsample(scale_factor=2), # N, 16, 6, 6
        nn.ConvTranspose2d(16, 8, 3, stride=1, padding=1), # N,8,6,6
        nn.ReLU(),
        nn.Upsample(scale_factor=2), # N,8,12,12
        nn.ConvTranspose2d(8, 1, 3, stride=1, padding=1,dilation=2), # N,1,14,14
        nn.Tanh(),
        nn.Upsample(scale_factor=2) # N,1,28,28
    )
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

