# CAP 5415 Programming Assignment 03: Nearest Neighbor Classification

"""
Due Date: 1ONov2023
Author: Lam Nguyen

Subject: Nearest Neighbor Classification [2.5 pts]

Overview:

Implement the Nearest Neighbor Classifier for digit classification. We will use the digit dataset available from the sklearn library. 

Tasks:

1. Import and process the dataset.
    a. There are around 1800 images 
    b. 10 digit classes
    c. Each image is 8x8 single channel.
    d. Split the dataset into training and testing, keep 500 images for testing
        i. Choose randomly with 50 images per class

2. Implement Neighbor Classification using pixels as features. Test the method for classification accuracy.

3. Implement a k-nearest neighbor classifier using pixels as features.
    a. Test method for k=3,5, and 7 and compute classification accuracy.

4. Create a short writeup about implementation with results:
    1. Accuracy scores for all the variations
    2. Compare the variations using accuracy scores.
    3. Comment of how the accuracy changes when you increase the value of k

Note: You can use L2-Norm for distance between 2 samples.
        

Sources:

Train Test Split: https://www.geeksforgeeks.org/how-to-split-the-dataset-with-scikit-learns-train_test_split-function/

"""


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


digits = load_digits()

print(digits.data.shape)


image_data = digits.data
image_targets = digits.target

features_train,features_test, labels_train,labels_test = train_test_split(
image_data, image_targets, test_size=500, random_state=42
)

print(features_train)




