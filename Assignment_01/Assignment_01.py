# CAP 5415 Programming Assignment 01

"""
Due Date: 29Sep2023
Author: Lam Nguyen

Subject: Canny Edge Detection Implementation [5 pts]

Tasks:
    1. Choose three example gray-scale images from Berkeley Segmentation Dataset (Training Images). When executed, your algorithm should plot intermediate and final results of Canny Edge Detection process as similar to the figure illustrated above.

    2. Please show the effect of threshold(sigma) in edge detection by choosing three different threshold(sigma) values when smoothing. Note that you need to indicate which threshold works best as a comment in your assignment.


Theory: 

    1. Noise Reduction
    2. Gradient Calculation
    3. Non-maximum suppression
    4. Double threshold
    Edge Tracking by Hysteresis


Assignment Details:

# 1. Read a grey scale image that can be found from the Berkeley Segmentation Dataset, Training Images, and store it as a matric named I
        # Link: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/dataset/images.html

# Source: https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123

# 2. Noise Reduction: Create a one-dimensional Gaussian mask G to convolve with I. The standard Deviation(s) of this Gaussian is a parameter to the edge detector (call it std > 0)

# 3. Gradient: Create a one-dimensional mask for the first derivative of the Gaussian in the x and y directions; call these G_x and G_y. The same std > 0 value is used as in step 2

# 4. Convolve I_x with G_x to give I'_x, the x component of I convolved with the derivative of the Gaussian, and convolve I_y with G_y  to give the I'_y component of I convolved with the derivative of the Gaussian

# 5. Compute the magnitude of the edge response by combining the x and y components. The magnitude of the result can be computed at each pixel (x,y) as M(x,y) = (I'_y**.5 + I'x**.5)**.5

# 6. Implement the non-maximimum suppression algorithm that we discussed in the lecture. Pixels that are not local maxima should be removed with this method. In other words, not all the pixels indicating strong magnitude are edges in fact We need to remove false positive edge locations from the image.

# 7. Apply Hysteresis thresholding to obtain the final edge-map. You may use any existing library function to compute connected components.

"""

# import required modules
from pathlib import Path
import matplotlib.image as mpimg
import numpy as np

# ========================================================#
#           1. Import Images, and append to Array          #
# ========================================================#


 
# get the path/directory
folder_dir = 'Assignment_01/Images'
 
# iterate over files in that directory and add to array

image_array = []
images = Path(folder_dir).glob('*.jpg')
for image in images:
    im = mpimg.imread(image) # Convert Image to matrix/array
    image_array.append(im) 

# print(image_array)


# ========================================================#
#      2. Create 1-dimensional Gaussian Filter Mask       #
# ========================================================#



def gaussian_filter(standard_deviation):

    # Create empty matrix in order to iterate through and use a Gaussian Function
    x_filter_size = 1
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



print(f'Gaussian Filter: {gaussian_filter(1)}')


# =============================================================================================#
#      3. Create 1-dimensional  1st derivative Gaussian Filter Mask in X and Y direction       #
# =============================================================================================#

def gaussian_filter_partial_derivative_x(standard_deviation):
    # Create empty matrix in order to iterate through and use a Gaussian Function
    x_filter_size = 1
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
            

print(gaussian_filter_partial_derivative_x(1))



def gaussian_filter_partial_derivative_y(standard_deviation):
    # Create empty matrix in order to iterate through and use a Gaussian Function
    x_filter_size = 1
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
            

print(gaussian_filter_partial_derivative_y(1))