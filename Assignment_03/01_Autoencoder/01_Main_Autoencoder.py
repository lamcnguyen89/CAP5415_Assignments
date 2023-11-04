# CAP 5415 Programming Assignment 03: Autoencoder

"""
Due Date: 1ONov2023
Author: Lam Nguyen

Subject: Autoencoder [2.5 pts]

Overview:

Implement autoencoder using MNIST dataset. The input size of the images will be 28x28 with a single channel. You will implement two different variations, one with fully connected layers and the other convolutional neural network.

Tasks:

1. Implement an autoencoder using fully connected layers. 
    a. The encoder will have 2 layers (with 256, and 128 neurons)
    b. The decoder will have 2 layers (with 256 and 784 neurons)
    c. Train this network using MSE loss for 10 epochs
    d. Compare the number of parameters  in the encoder and decoder.
    e. Create a writeup:
        i. Show 20 reconstructed images from testing data (2 image for each class)
        ii. Show original images

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

Note that you can choose any optimizer. Just use the same for both variations
        

Sources:

Autoencoder in Pytorch-Theory and Implementation by Patrick Loeber: https://www.youtube.com/watch?v=zp8clK9yCro




"""

import torch
import torch.optim as optim # Optimization algorithms
import torch.nn.functional as F # All functions without parameters
import torchvision.datasets as datasets # Standard datasets that can be used as test training data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms # Transformations that can be performed on the dataset
import matplotlib.pyplot as plt

# Import some packages for logging training and showing progress
from tqdm_loggable.auto import tqdm

# Hyperparameters
input_size = 28*28
hidden_size = 100
num_classes= 10
learning_rate = 0.1
batch_size = 64
num_epochs = 10
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =======================================================#
# 1. Import Data:
# =======================================================#

train_dataset = datasets.MNIST(root='Assignment_03/MNIST_dataset/', 
               train=True, 
               transform=transforms.ToTensor(),
               download=True
               )#Transforms transforms numpy array to tensors so that pytorch can use the data


train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True
)

test_dataset = datasets.MNIST(root='Assignment_03/MNIST_dataset/', 
               train=False, 
               transform=transforms.ToTensor(),
               download=True
               )#Transforms transforms numpy array to tensors so that pytorch can use the data

test_loader = DataLoader(
    dataset= test_dataset,
    batch_size = batch_size,
    shuffle = True
)


