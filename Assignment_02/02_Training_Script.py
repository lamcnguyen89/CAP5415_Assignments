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

import subprocess


# Loop through folder with the different Neural Networks created for this assignment,implement training and log results in the logs folder:
for i in range(6):
    subprocess.run(f"python Assignment_02/Step_0{i+1}.py", shell=True)




