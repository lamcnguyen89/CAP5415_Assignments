"""
Programming Assignment_02: Convolutional NNs

Step 05:


Change the neuron numbers in FC layers from 100 to 1000.
        a. Train layer with SGD. Use dropout with a rate=0.5 and 40 epochs
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
from tqdm import tqdm # For progress bar


# Import some packages for logging training and showing progress
from tqdm_loggable.auto import tqdm
from tqdm_loggable.tqdm_logging import tqdm_logging
import datetime
import logging


# Set up some basic logging to record traces of training
logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="Assignment_02/logs/Step_05_log.txt" # Save log to a file
    )

tqdm_logging.set_level(logging.INFO)


# Hyperparameters
input_channels = 1
hidden_size = 1000
num_classes= 10
learning_rate = 0.03
batch_size = 10
num_epochs = 40
dropout_rate=0.5
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================================================================================#
# 2. Change Neurons from 100 to 1000, add dropout with a rate of 0.5 :
# ================================================================================#

class NN_5(nn.Module):
    def __init__(self,input_channels,hidden_size, num_classes):
        super(NN_5, self).__init__() # The Super keyword calls the initialization of the parent class
        
        self.conv1 = nn.Conv2d(in_channels=input_channels,
                               out_channels=40,
                               kernel_size=(5,5),
                               stride=(1,1),
                               padding=(0,0)
                               )
        self.conv2 = nn.Conv2d(in_channels=40,
                               out_channels=40,
                               kernel_size=(5,5),
                               stride=(1,1),
                               padding=(0,0)
                               )
        self.pool = nn.MaxPool2d(kernel_size=(2,2),
                                  stride=(2,2)
        ) 
        self.drop=nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(40*4*4, hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size) 
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def num_flat_features(self,x):
        size = x.size()[1:] # all dimensions except batch dimension
        num_features = 1
        for s in size:
            num_features *=s

        return num_features

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop(x)
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    


# =======================================================#
# 3. Train the Convolutional Neural:
# =======================================================#


# Prepare the data for processing through the Network:

train_dataset = datasets.MNIST(root='Assignment_02/dataset/', 
               train=True, 
               transform=transforms.ToTensor(),
               download=True
               )#Transforms transforms numpy array to tensors so that pytorch can use the data


train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True
)

test_dataset = datasets.MNIST(root='Assignment_02/dataset/', 
               train=False, 
               transform=transforms.ToTensor(),
               download=True
               )#Transforms transforms numpy array to tensors so that pytorch can use the data

test_loader = DataLoader(
    dataset= test_dataset,
    batch_size = batch_size,
    shuffle = True
)

#Initialize Model
model = NN_5(
    input_channels=input_channels,
    hidden_size=hidden_size,
    num_classes=num_classes
).to(device)

logging.info(f"Begin Training MNIST dataset with this model: {model}")

# Define the loss function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), 
                      lr=learning_rate)

epoch_counter= 0
# Train Network
for epoch in range(num_epochs):
    tqdm.write(f"Step 5/5 Training Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):

        
        # Get data to Cuda/gpu if possible
        data = data.to(device=device)
        targets = targets = targets.to(device=device)
    
       # print(data.shape) 

        # Forward
        scores = model(data)
        loss = criterion(scores, targets)

        # Go Backward in the network:
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
        
        #logging.info("Training single layer Neural Network ")
        if epoch > epoch_counter+4:
            logging.info(f"Training Epoch: {epoch}")

            epoch_counter = epoch


epoch_counter = 0

# =========================================================================#
# 4. Check Accuracy on training and test to see the accuracy of the model:
# =========================================================================#


def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
        logging.info("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
        logging.info("Checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad(): # No gradients have to be calculated
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)


            scores = model(x)
            _,predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples) * 100:.2f}')
        logging.info(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples) * 100:.2f}')

    model.train()
    acc = num_correct/num_samples
    return acc

check_accuracy(train_loader,model)
check_accuracy(test_loader,model)

