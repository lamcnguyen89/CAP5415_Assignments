import torch
import torch.nn as nn # All the Neural network models, loss functions
import torch.optim as optim # Optimization algorithms
import torch.nn.functional as F # All functions without parameters
from torch.utils.data import DataLoader # Easier dataset management such as minibatches
import torchvision.datasets as datasets # Standard datasets that can be used as test training data
import torchvision.transforms as transforms # Transformations that can be performed on the dataset



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
        x = x.view()
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
    

#Initialize Model
model = NN_2(
    input_size=input_size,
    num_classes=num_classes
)