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

How to get info on model parameters using Torchinfo: https://pypi.org/project/torchinfo/

How to save trained model: https://wandb.ai/wandb/common-ml-errors/reports/How-to-Save-and-Load-Models-in-PyTorch--VmlldzozMjg0MTE

How to take subsets of dataset: https://discuss.pytorch.org/t/how-to-get-a-part-of-datasets/82161



"""
# ========================================================================================#
# 1. Load Modules
# ========================================================================================#

import torch
from pathlib import Path
from PIL import Image
import os
import numpy as np
from torchvision.utils import save_image
import torch.optim as optim # Optimization algorithms
import torch.nn as nn # All the Neural network models, loss functions
import torch.nn.functional as F # All functions without parameters
from torch.utils.data import DataLoader # Easier dataset management such as minibatches
import torchvision.datasets as datasets # Standard datasets that can be used as test training data
import torchvision.transforms as transforms # Transformations that can be performed on the dataset
from torchinfo import summary # provides a summary of the model architecture and it's parameters
import logging
import matplotlib.pyplot as plt

# Import some packages for logging training and showing progress
from tqdm_loggable.auto import tqdm



# Set up some basic logging to record traces of training
logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="Assignment_03/01_Autoencoder/Autoencoder_Documents/Autoencoder_Parameter_Summary.txt" # Save log to a file
    )


# Hyperparameters
input_size = 28*28
hidden_size = 100
num_classes= 10
learning_rate = 0.1
batch_size = 64
num_epochs = 10
weight_decay = 1e-5
    


# Load GPU Parameters
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



# ========================================================================================#
# 3. Import the Fully Connected Autoencoder Model and train
# ========================================================================================#
from Fully_Connected_Autoencoder import FCC_Autoencoder


model = FCC_Autoencoder(input_size=input_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


image_array = []

print("Beginning the training of the Fully Connected Autoencoder")
for epoch in range(num_epochs):
    tqdm.write(f'Epoch:{epoch +1}')
    for batch_idx, (images, _) in enumerate(tqdm(train_loader)):
        
        # Get data to Cuda/gpu if possible. Data is the tuple of the images and labels
        # We have to reshape images because they are (10,1,28,28) when input into the network.
        # But for a fully connected, we need to have a shape (10,784)
        # 10 is the number of batches
        images = images.reshape(-1,28*28)

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


# =======================================================#
# 5. Get the number of parameters for the FC-Autoencoder:
# =======================================================#

# Prints out the architecture of the trained model
FCC_Autoencoder_summary = summary(model)

logging.info(FCC_Autoencoder_summary) # Saves the parameter data file into the folder Autoencoder_Documents


for k in range(0, num_epochs, 4):
    plt.figure(figsize=(9, 2))
    plt.gray()
    imgs = image_array[k][1].cpu().detach().numpy()
    recon = image_array[k][2].cpu().detach().numpy()
    for i, item in enumerate(imgs):
        if i >= 9: break
        plt.subplot(2, 9, i+1)
        item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
        # item: 1, 28, 28
        plt.imshow(item[0])
            
    for i, item in enumerate(recon):
        if i >= 9: break
        plt.subplot(2, 9, 9+i+1) # row_length + i + 1
        item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
        # item: 1, 28, 28
        plt.imshow(item[0])


# =======================================================#
# 6. Save the trained Fully Connected Autoencoder
# =======================================================#

print("Saving the Fully Connected Autoencoder Model to the folder: Assignment_03/01_Autoencoder/Trained_Autoencoders")
torch.save(model.state_dict(),'Assignment_03/01_Autoencoder/Trained_Autoencoders/FCC_Autoencoder_Model.pth')



# ========================================================================================#
# 2. Import the Convolutional Autoencoder Model and train
# ========================================================================================#
from Convolutional_Autoencoder import CNN_Autoencoder

model = CNN_Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)



image_array = []

print("Beginning the training of the Convolutional Autoencoder")
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


# =======================================================#
# 5. Get the number of parameters for the CNN-Autoencoder:
# =======================================================#

# Prints out the architecture of the trained model
CNN_Autoencoder_summary = summary(model)

logging.info(CNN_Autoencoder_summary) # Saves the parameter data file into the folder Autoencoder_Documents

for k in range(0, num_epochs, 4):
    plt.figure(figsize=(9, 2))
    plt.gray()
    imgs = image_array[k][1].cpu().detach().numpy()
    recon = image_array[k][2].cpu().detach().numpy()
    for i, item in enumerate(imgs):
        if i >= 9: break
        plt.subplot(2, 9, i+1)
        # item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
        # item: 1, 28, 28
        plt.imshow(item[0])
            
    for i, item in enumerate(recon):
        if i >= 9: break
        plt.subplot(2, 9, 9+i+1) # row_length + i + 1
        # item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
        # item: 1, 28, 28
        plt.imshow(item[0])

# =======================================================#
# 6. Save the trained Convolutional Autoencoder
# =======================================================#

print("Saving the Convolutional Autoencoder Model to the folder: Assignment_03/01_Autoencoder/Trained_Autoencoders")
torch.save(model.state_dict(),'Assignment_03/01_Autoencoder/Trained_Autoencoders/CNN_Autoencoder_Model.pth')



# ========================================================================================#
# 3. Use the trained model to make predictions of the 10 classes in the MNIST Dataset:
# ========================================================================================#


# # Load an instance of the trained model and load the saved parameters that were previously trained

# #Image Dimensions from the MNIST Dataset
# channels = 1
# height =28
# width = 28

# # Create an instance of the models
# fcc_model = FCC_Autoencoder(input_size= (height*width*channels)).to(device)
# cnn_model = CNN_Autoencoder().to(device)


# # Load the saved parameters
# fcc_model.load_state_dict(torch.load('Assignment_03/01_Autoencoder/Trained_Autoencoders/FCC_Autoencoder_Model.pth'))
# cnn_model.load_state_dict(torch.load('Assignment_03/01_Autoencoder/Trained_Autoencoders/CNN_Autoencoder_Model.pth'))

# fcc_model.eval()
# cnn_model.eval()

# # Get the images (two from each classifier) and load them into a Tensor

# input_images_array = Path('Assignment_03/01_Autoencoder/Test_Images').glob('*png')
# for image in input_images_array:
#         # Load image, load the filename and convert the image to an array.
#         filename = os.path.basename(image).split('.',1)[0]
#         im = Image.open(image).convert("L")
#         im = np.asarray(im)
#         im = torch.from_numpy(im)
#         im = fcc_model.forward(im)
#         save_image(im, f'Assignment_03/01_Autoencoder/Processed_Images/{filename}_processed.png')

#         # Push the tensor through the trained model and save them to a numpy array


#         # Save the images in the numpy array to a folder.






