from Convolutional_Autoencoder import CNN_Autoencoder
from Fully_Connected_Autoencoder import FCC_Autoencoder
from torchinfo import summary
import subprocess
import torch
import gc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import time
from pathlib import Path
from PIL import Image
import os
import numpy as np
from torchvision.utils import save_image
from torchvision import transforms
import matplotlib.pyplot as plt


# Load GPU Parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
min_memory_available = 8*1024*1024*1024 # 8gb memory



channels = 1
height =28
width = 28

# Create an instance of the models
fcc_model = FCC_Autoencoder(input_size= (height*width*channels))
cnn_model = CNN_Autoencoder()


# Load the saved parameters
fcc_model.load_state_dict(torch.load('Assignment_03/01_Autoencoder/Trained_Autoencoders/FCC_Autoencoder_Model.pth'))
cnn_model.load_state_dict(torch.load('Assignment_03/01_Autoencoder/Trained_Autoencoders/CNN_Autoencoder_Model.pth'))

fcc_model.eval()
cnn_model.eval()


img = Image.open("Assignment_03/01_Autoencoder/Test_Images/0_3.png").convert("L")
convert_tensor = transforms.ToTensor()
tensor_img = convert_tensor(img)

processed_img= cnn_model.forward(tensor_img)

save_image(processed_img, 'Assignment_03/01_Autoencoder/Processed_Images/process.jpg')





