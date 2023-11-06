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

# =======================================================#
# 1. Import basic Modules and Functions and set variables
# =======================================================#

import torch
import torch.nn as nn # All the Neural network models, loss functions
import torch.optim as optim # Optimization algorithms
import torch.nn.functional as F # All functions without parameters
from torch.utils.data import DataLoader # Easier dataset management such as minibatches
import torchvision.datasets as datasets # Standard datasets that can be used as test training data
import torchvision.transforms as transforms # Transformations that can be performed on the dataset


# Import some packages for logging training and showing progress
from tqdm_loggable.auto import tqdm
from tqdm_loggable.tqdm_logging import tqdm_logging


# Hyperparameters
input_size = 28*28
hidden_size = 100
num_classes= 10
learning_rate = 0.1
batch_size = 64
num_epochs = 10
weight_decay = 1e-5
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# =======================================================#
# 2. Import Data:
# =======================================================#

train_dataset = datasets.MNIST(root='Assignment_03/01_Autoencoder/MNIST_dataset/', 
               train=True, 
               transform=transforms.ToTensor(),
               download=True
               )#Transforms transforms numpy array to tensors so that pytorch can use the data


train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True
)



# =======================================================#
# 3. Create Convolutional Autoencoder
# =======================================================#

""" 

1. Formula for calculating size of feature map edge for Convolution:

    L_out = [(L_in + 2 * padding - dilation * (kernel_size - 1) -1)/stride] + 1


2. Formula for calculating size of feature map edge for Transpose Convolution:

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

# =======================================================#
# 4. Train Covolutional Autoencoder:
# =======================================================#

model = CNN_Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)



image_array = []

for epoch in range(num_epochs):
    tqdm.write(f'Epoch:{epoch +1}')
    for batch_idx, (images, _) in enumerate(tqdm(train_loader)):
        
        # Get data to Cuda/gpu if possible. Data is the tuple of the images and labels
        # We have to reshape images because they are (10,1,28,28) when input into the network.
        # But for a fully connected, we need to have a shape (10,784)
        # 10 is the number of batches
        # images = images.reshape(-1,28*28)

        images = images.to(device=device) # Images
       # labels = labels.to(device=device) # label that classifies image

        # Forward
        outputs = model(images)
        loss = criterion(outputs, images) # Predicted outputs vs actual labels

        # Go Backward in the network:
        optimizer.zero_grad() # Empty the values in the gradient attribute
        loss.backward() # Backpropagation

        # gradient descent or adam step
        optimizer.step() # Update parameters
        image_array.append((epoch,images,outputs))