"""
Programming Assignment_02: Convolutional NNs

Step 01:

First create a fully connected (FC) hidden layer with:
        a. 100 neurons
        b. Sigmoid Activation function.
        c. Train Layer with SGD with a learning rate=0.1, epoch=60, mini-batch size = 10, no regularization
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
import datetime
import logging


# Set up some basic logging to record traces of training
logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="Assignment_02/output/Step_01_trace.txt" # Save log to a file
    )

tqdm_logging.set_level(logging.INFO)


# Hyperparameters
input_size = 28*28
hidden_size = 100
num_classes= 10
learning_rate = 0.1
batch_size = 10
num_epochs = 60
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =======================================================#
# 2. Create Fully Connected hidden lanyer
# =======================================================#

class NN(nn.Module):
    def __init__(self,input_size, hidden_size, num_classes):
        super(NN, self).__init__() # The Super keyword calls the initialization of the parent class
        self.fc1 = nn.Linear(input_size, hidden_size) # Create a small NN
        self.fc2 = nn.Linear(hidden_size, num_classes) 

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x
    


# =======================================================#
# 3. Train the Fully Connected Hidden Layer:
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
model = NN(
    input_size=input_size,
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
    tqdm.write(f"Step 1/5 Training Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
        
        
        
        # Get data to Cuda/gpu if possible. Data is the tuple of the images and labels
        # We have to reshape images because they are (10,1,28,28) when input into the network.
        # But for a fully connected, we need to have a shape (10,784)
        # 10 is the number of batches
        images = images.reshape(-1,28*28)

        images = images.to(device=device) # Images
        labels = labels.to(device=device) # label that classifies image



        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels) # Predicted outputs vs actual labels

        # Go Backward in the network:
        optimizer.zero_grad() # Empty the values in the gradient attribute
        loss.backward() # Backpropagation

        # gradient descent or adam step
        optimizer.step() # Update parameters
        
        #logging.info("Training single layer Neural Network ")
        if epoch > epoch_counter+4:
            logging.info(f"Training Epoch: {epoch}, loss = {loss.item():.4f}")


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
            x = x.reshape(x.shape[0], -1) # Have to reshape data. Why? Let me figure it out.

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












