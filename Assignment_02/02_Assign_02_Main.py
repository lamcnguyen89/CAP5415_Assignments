# CAP 5415 Programming Assignment 02

"""
Due Date: 23Oct2023
Author: Lam Nguyen

Subject: Convolutional Neural Network (CNN) for Classification [5 pts]

Tasks:

    1. Implement ConvNET using Pytorch for digit classification. 
    
    2. Sample code is given in the attachment. Fill the parts indicated clearly in the code.

    3. Output should be saved as output.txt

    4. When asked to include the convolutional layer, don't forget to include max pooling or average pooling layer(s) as well.

    5. Create a short write-up about your implementation with results and your observations from each training:

        5a. Note that in each step you will train the corresponding architecture and report the accuracy on the test data.

        5b. Also show how training/test loss and accuracy is varying with each iteration during training using plots.


Assignment Details: 

    1. First create a fully connected (FC) hidden layer with:
        a. 100 neurons
        b. Sigmoid Activation function.
        c. Train Layer with SGD with a learning rate=0.1, epoch=60, mini-batch size = 10, no regularization

    2. Next insert two Convolutional Laters to the network built in Step 1. 
        a. For each CNN layer, include a pooling layer and Sigmoid Activation. 
        b. Pool over 2x2 regions, 40 kernels, stride=1, kernel_size=5x5.
        c. Train with SGD with a learning rate=0.1, epoch=60, mini-batch size = 10, no regularization

    3. For the network created in Step 2, replace Sigmoid with ReLU.
        a. Train the model with a new Learning_rate=0.03.

    4. Add another fully connected (FC) layer.
        a. This new FC layer should have 100 neurons
        b. Train with the same setup as Step 3. meaning you use ReLU activation function and a learning_rate=0.03

    5. Change the neuron numbers in FC layers from 100 to 1000.
        a. Train layer with SGD. Use dropout with a rate=0.5 and 40 epochs

    6. The traces from running testCNN.py <mode> for each of the 5 steps should be saved in output.txt. Each Step is 1 point
        
 
Sources:

https://medium.com/@shashankshankar10/introduction-to-neural-networks-build-a-single-layer-perceptron-in-pytorch-c22d9b412ccf

https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118

https://www.kaggle.com/code/justuser/mnist-with-pytorch-fully-connected-network

https://pypi.org/project/tqdm-loggable/
https://www.youtube.com/watch?v=urrfJgHwIJA&t=322s



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

from tqdm_loggable.auto import tqdm
from tqdm_loggable.tqdm_logging import tqdm_logging
import datetime
import logging
import time
import io

# # Class for logging progress of training
# class TqdmLoggingHandler(logging.Handler):
#     def __init__(self, level=logging.NOTSET):
#         super().__init__(level)

#     def emit(self, record):
#         try:
#             msg = self.format(record)
#            # tqdm.tqdm.write(msg)
#             self.flush()
#         except Exception:
#             self.handleError(record) 

# logging.basicConfig(level=logging.INFO, filename="log.txt", filemode="w")
# log = logging.getLogger(__name__)
# log.addHandler(TqdmLoggingHandler())

# Set up logging to write progress of training to text file
logging.basicConfig(level=logging.INFO, 
                    filename="log.txt", 
                    filemode="w",
                    format="%(asctime)s - %(levelname)s - %(message)s"
                    )

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =======================================================#
# 2a. Create Fully Connected hidden lanyer
# =======================================================#

class NN(nn.Module):
    def __init__(self,input_size, num_classes):
        super(NN, self).__init__() # The Super keyword calls the initialization of the parent class
        self.fc1 = nn.Linear(input_size, 100) # Create a small NN
        self.fc2 = nn.Linear(100, num_classes) 

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = NN(784, 10) # Check the model to see if it runs correctly by using some random test data. 10 is for the number of digits. We want 10 values for each of the 784 images
x = torch.randn(64, 784) # Number of examples to run simultaneously
print(model(x).shape) # Returns the shape of the model

# =======================================================#
# 2b. Train the Fully Connected Hidden Layer:
# =======================================================#
 
# Hyperparameters
input_size = 28*28
num_classes= 10
learning_rate = 0.1
batch_size = 10
num_epochs = 64
    


# Prepare the data for processing through the Network:

train_dataset = datasets.MNIST(root='dataset/', 
               train=True, 
               transform=transforms.ToTensor(),
               download=True
               )#Transforms transforms numpy array to tensors so that pytorch can use the data


train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True
)

test_dataset = datasets.MNIST(root='dataset/', 
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
model = NN(
    input_size=input_size,
    num_classes=num_classes
).to(device)


# Define the loss function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), 
                      lr=learning_rate)


# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        
        # Get data to Cuda/gpu if possible
        data = data.to(device=device)
        targets = targets = targets.to(device=device)

        data = data.reshape(data.shape[0], -1) # The -1 unrolls the image to single dimension. Why? Not Sure
    
       # print(data.shape) 

        # Forward
        scores = model(data)
        loss = criterion(scores, targets)

        # Go Backward in the network:
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()


# Check Accuracy on training and test to see the accuracy of the model
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad(): # No gradients have to be calculated
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1) # Have to reshape data. Why? Let me figure it out.

            scores = model(x)
            _,predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples) * 100:.2f}')

    model.train()
    acc = num_correct/num_samples
    return acc

check_accuracy(train_loader,model)
check_accuracy(test_loader,model)


# ================================================================================#
# 3. Next insert two Convolutional Laters to the network built in Step 1 and train
# ================================================================================#



# ================================================================================#
# 4. For the Network created in the previous steps, change the type of activation function used and train it
# ================================================================================#



# =====================================#
# 5. Add another Fully Connected Layer
# =====================================#



#==========================================================================================#
# 6. Change the neuron numbers from the previous model in step 5 from 100 to 1000. Train it
#==========================================================================================#







