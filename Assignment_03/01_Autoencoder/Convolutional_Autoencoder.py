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

class CNN_Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()
        # N, 1, 28, 28
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels= 40,
                      kernel_size= 3
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2
                         ),
            nn.Conv2d(in_channels=40, 
                      out_channels=40, 
                      kernel_size=3
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2
                         ),
        )
        
        # You want to play around with the hyperparameters in order to get the same output dimentsions as the input dimensions.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=40, 
                               out_channels=40, 
                               kernel_size=3,
                               padding=1
                               ), 
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(in_channels=40, 
                               out_channels=40, 
                               kernel_size=3, 
                               padding=1, 
                               ),
            nn.Upsample(scale_factor=2), 
            nn.ConvTranspose2d(in_channels=40, 
                               out_channels=40, 
                               kernel_size=3, 
                               padding=1, 
                               ),
            nn.Upsample(scale_factor=2),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x