# ==========================================================#
# 2. Create 1-dimensional Gaussian Filter Mask and Apply it #
# ==========================================================#
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

def gaussian_filter_2D(standard_deviation):
    
   
    # Create empty matrix in order to iterate through and use a Gaussian Function
    x_filter_size = 2 * int(4 * standard_deviation + 0.5) + 1
    y_filter_size = 2 * int(4 * standard_deviation + 0.5) + 1
    gaussian_mask = np.zeros((x_filter_size, y_filter_size), np.float32)

    # Iterate through the empty matrix to create a gaussian mask
    m = x_filter_size // 2
    n = y_filter_size // 2

    for x in range(-m,m+1):
        for y in range(-n, n+1):

            a = 1/(2*np.pi*(standard_deviation**2))
            b = np.exp(-(x**2+y**2)/(2*standard_deviation**2))
            # Iterate over each index array and implement gaussian distribution:
            gaussian_mask[x+m, y+n] = a * b
    
    return gaussian_mask


def X_gaussian_filter_1D(standard_deviation):
    size=10
    # Create an array of integers from -(size//2) to size//2
    x = np.arange(-(size // 2), (size // 2) + 1)
    # Compute the Gaussian function for each value in the array
    a= (1 / (np.sqrt(2 * np.pi * standard_deviation**2)))
    b= np.exp(-(x**2) / (2 * standard_deviation**2))
    gaussian_mask = a*b
    # Normalize the filter to ensure its sum equals 1
    gaussian_mask /= np.sum(gaussian_mask)
    return gaussian_mask


def Y_gaussian_filter_1D(standard_deviation):
    size=10
    # Create an array of integers from -(size//2) to size//2
    y = np.arange(-(size // 2), (size // 2) + 1)
    y = np.vstack(y)
    # Compute the Gaussian function for each value in the array
    a= (1 / (np.sqrt(2 * np.pi * standard_deviation**2)))
    b= np.exp(-(y**2) / (2 * standard_deviation**2))
    gaussian_mask = a*b
    # Normalize the filter to ensure its sum equals 1
    gaussian_mask /= np.sum(gaussian_mask)
    return gaussian_mask



