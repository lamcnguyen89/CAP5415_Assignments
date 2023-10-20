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


# Set up some basic logging to record traces of training
logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="NN_Training_Log_2.log" # Save log to a file
    )

tqdm_logging.set_level(logging.INFO)
tqdm_logging.set_log_rate(datetime.timedelta(seconds=3600))  


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================================================================================#
# 3a. Next insert two Convolutional Laters to the network built in Step 1 and train
# ================================================================================#

class NN_2(nn.Module):
    def __init__(self,input_size, num_classes):
        super(NN_2, self).__init__() # The Super keyword calls the initialization of the parent class
        self.fc1 = nn.Linear(input_size, 100) # Create a small NN
        self.conv1 = nn.Conv2d(in_channels=100,
                               out_channels=8,
                               kernel_size=(3,3),
                               stride=(1,1),
                               padding=(1,1)
                               )
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),
                                  stride=(2,2)
        ) 
        self.conv2 = nn.Conv2d(in_channels=8,
                               out_channels=8,
                               kernel_size=(3,3),
                               stride=(1,1),
                               padding=(1,1)
                               ) 
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2),
                                  stride=(2,2)
        )
        self.fc2 = nn.Linear(8, num_classes)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.conv1(x))
        x = self.pool1(x)
        x = F.sigmoid(self.conv2(x))
        x = self.pool2(x)
        x = self.fc2(x)
        return x
    


# =======================================================#
# 3b. Train the Convolutional Neural:
# =======================================================#
 
# Hyperparameters
input_size = 28*28*1
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
model = NN_2(
    input_size=input_size,
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
        
        #logging.info("Training single layer Neural Network ")
        if epoch > epoch_counter+4:
            logging.info(f"Training Epoch: {epoch}")

            epoch_counter = epoch


epoch_counter = 0

# Check Accuracy on training and test to see the accuracy of the model
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
