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

    def __init__(self,input_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_size,
                        out_channels=40,
                        kernel_size=(3,3),
                        stride=(1,1),
                        padding=(1,1)
                   ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),
                            stride=(2,2)),
            nn.Conv2d(in_channels=40,
                        out_channels=40,
                        kernel_size=(3,3),
                        stride=(1,1),
                        padding=(1,1)
                   ),
            nn.MaxPool2d(kernel_size=(2,2),
                            stride=(2,2))
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=input_size,
                        out_channels=40,
                        kernel_size=(3,3),
                        stride=(1,1),
                        padding=(1,1)
                   ),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=input_size,
                        out_channels=40,
                        kernel_size=(3,3),
                        stride=(1,1),
                        padding=(1,1)
                   ),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=input_size,
                        out_channels=40,
                        kernel_size=(3,3),
                        stride=(1,1),
                        padding=(1,1)
                   ),
            nn.Upsample(scale_factor=2),

        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x