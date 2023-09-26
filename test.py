import numpy as np
import matplotlib.pyplot as plt

def gaussian_function(x, sigma):
    return (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-(x**2) / (2 * sigma**2))

def create_gaussian_filter(size, sigma):
    # Create an array of integers from -(size//2) to size//2
    x = np.arange(-(size // 2), (size // 2) + 1)
    # Compute the Gaussian function for each value in the array
    g_filter = gaussian_function(x, sigma)
    # Normalize the filter to ensure its sum equals 1
    g_filter /= np.sum(g_filter)
    return g_filter



