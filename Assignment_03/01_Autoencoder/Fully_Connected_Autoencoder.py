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
from torchinfo import summary # provides a summary of the model architecture and it's parameters
import logging

# Import some packages for logging training and showing progress
from tqdm_loggable.auto import tqdm
from tqdm_loggable.tqdm_logging import tqdm_logging

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
# 3. Create Fully Connected Autoencoder:
# =======================================================#

class FCC_Autoencoder(nn.Module):

    def __init__(self,input_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=128),


        )

        self.decoder = nn.Sequential(
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
# Note: If images are in the range (-1,1) apply Tanh() activation instead of sigmoid



# =======================================================#
# 4. Train Fully Connected Autoencoder
# =======================================================#

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

FCC_Autoencoder_summary = summary(model)

logging.info(FCC_Autoencoder_summary)


# =======================================================#
# 6. Save the trained Fully Connected Autoencoder
# =======================================================#

print("Saving the Fully Connected Autoencoder Model to the folder: Assignment_03/01_Autoencoder/Trained_Autoencoders")
torch.save(model.state_dict(),'Assignment_03/01_Autoencoder/Trained_Autoencoders/FCC_Autoencoder_Model.pth')