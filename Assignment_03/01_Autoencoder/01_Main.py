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





"""
# ========================================================================================#
# 1. Load Modules
# ========================================================================================#


from Convolutional_Autoencoder import CNN_Autoencoder
from Fully_Connected_Autoencoder import FCC_Autoencoder
from torchinfo import summary
import subprocess
import torch
import gc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import time


# Load GPU Parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
min_memory_available = 8*1024*1024*1024 # 8gb memory

# ========================================================================================#
# 2. Create and Train the Convolutional Autoencoder and Fully Connected Autoencoder Models
# ========================================================================================#


# Clear GPU memory after Training Each Model
def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()
    # del variables

# Wait until There is enough Memory to train
def wait_until_enough_gpu_memory(min_memory_available, max_retries=10, sleep_time=5):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    for _ in range(max_retries):
        info = nvmlDeviceGetMemoryInfo(handle)
        if info.free >= min_memory_available:
            break
        print(f"Waiting for {min_memory_available} bytes of free GPU memory. Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
    else:
        raise RuntimeError(f"Failed to acquire {min_memory_available} bytes of free GPU memory after {max_retries} retries.")
    


# Run the different Models
subprocess.run(f"python Assignment_03/01_Autoencoder/Fully_Connected_Autoencoder.py", shell=True)
clear_gpu_memory()
wait_until_enough_gpu_memory(min_memory_available)

subprocess.run(f"python Assignment_03/01_Autoencoder/Convolutional_Autoencoder.py", shell=True)
clear_gpu_memory()
wait_until_enough_gpu_memory(min_memory_available)




# ========================================================================================#
# 3. Use the trained model to make predictions of the 10 classes in the MNIST Dataset:
# ========================================================================================#


# Load an instance of the trained model and load the saved parameters that were previously trained

#Image Dimensions from the MNIST Dataset
channels = 1
height =28
width = 28

# Create an instance of the models
fcc_model = FCC_Autoencoder(input_size= (height*width*channels)).to(device)
cnn_model = CNN_Autoencoder().to(device)


# Load the saved parameters
fcc_model.load_state_dict(torch.load('Assignment_03/01_Autoencoder/Trained_Autoencoders/FCC_Autoencoder_Model.pth'))
cnn_model.load_state_dict(torch.load('Assignment_03/01_Autoencoder/Trained_Autoencoders/CNN_Autoencoder_Model.pth'))


# Get the images (two from each classifier) and load them into a Tensor

# Push the tensor through the trained model and save them to a numpy array

# Save the images in the numpy array to a folder.






