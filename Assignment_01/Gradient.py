# ==================================================================================#
# 3. Create 1-dimensional  1st derivative Gaussian Filter Mask in X and Y direction #
# ==================================================================================#
import numpy as np

def gaussian_gradient_x_1D(standard_deviation):
    
    size=10
    # Create an array of integers from -(size//2) to size//2
    x = np.arange(-(size // 2), (size // 2) + 1)
    # Compute the Gaussian function for each value in the array
    a = 1/((2*np.pi*standard_deviation)**.5)
    b = np.exp(-(x**2)/(2*standard_deviation**2))
    c = -x/(standard_deviation**2)
    gradient_mask = a*b*c
    # Normalize the filter to ensure its sum equals 1
    gradient_mask /= np.sum(gradient_mask)
    return gradient_mask



def gaussian_gradient_y_1D(standard_deviation):
    
    size=10
    # Create an array of integers from -(size//2) to size//2
    y = np.arange(-(size // 2), (size // 2) + 1)
    y = np.vstack(y)
    # Compute the Gaussian function for each value in the array
    a = 1/((2*np.pi*standard_deviation)**.5)
    b = np.exp(-(y**2)/(2*standard_deviation**2))
    c = -y/(standard_deviation**2)
    gradient_mask = a*b*c
    # Normalize the filter to ensure its sum equals 1
    gradient_mask /= np.sum(gradient_mask)
    return gradient_mask



def gaussian_gradient_x_2D(standard_deviation):
     
    # Create empty matrix in order to iterate through and use a Gaussian Function
    x_filter_size = 2 * int(4 * standard_deviation + 0.5) + 1
    y_filter_size = 2 * int(4 * standard_deviation + 0.5) + 1
    gaussian_derivative_mask_x = np.zeros((x_filter_size, y_filter_size), np.float32)

    # Iterate through the empty matrix to create a gaussian mask
    m = x_filter_size // 2
    n = y_filter_size // 2

    for x in range(-m,m+1):
        for y in range(-n, n+1):
            a = 1/(2*np.pi*(standard_deviation**2))
            b = np.exp(-(x**2+y**2)/(2*standard_deviation**2))
            c = -x/(standard_deviation**2)
            # Iterate over each index array and implement derivative of gaussian distribution:
            gaussian_derivative_mask_x[x+m, y+n] = a*b*c

    return gaussian_derivative_mask_x
            


def gaussian_gradient_y_2D(standard_deviation):

    # Create empty matrix in order to iterate through and use a Gaussian Function
    x_filter_size = 2 * int(4 * standard_deviation + 0.5) + 1
    y_filter_size = 2 * int(4 * standard_deviation + 0.5) + 1
    gaussian_derivative_mask_y = np.zeros((x_filter_size, y_filter_size), np.float32)

    # Iterate through the empty matrix to create a gaussian mask
    m = x_filter_size // 2
    n = y_filter_size // 2

    for x in range(-m,m+1):
        for y in range(-n, n+1):
            a = 1/(2*np.pi*(standard_deviation**2))
            b = np.exp(-(x**2+y**2)/(2*standard_deviation**2))
            c = -y/(standard_deviation**2)
            # Iterate over each index array and implement derivative of gaussian distribution:
            gaussian_derivative_mask_y[x+m, y+n] = a*b*c

    return gaussian_derivative_mask_y
            
    




