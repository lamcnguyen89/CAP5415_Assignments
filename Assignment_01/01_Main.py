# CAP 5415 Programming Assignment 01

"""
Due Date: 29Sep2023
Author: Lam Nguyen

Subject: Canny Edge Detection Implementation [5 pts]

Tasks:
    1. Choose three example gray-scale images from Berkeley Segmentation Dataset (Training Images). When executed, your algorithm should plot intermediate and final results of Canny Edge Detection process as similar to the figure illustrated above.
        # Image Source: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/dataset/images.html

    2. Please show the effect of threshold(sigma) in edge detection by choosing three different threshold(sigma) values when smoothing. Note that you need to indicate which threshold works best as a comment in your assignment.


Assignment Details: 

    1. Implement Noise Reduction using the Gaussian Blur Filter
    2. Implement a Gradient Filter Using the Derivative of the Gaussian Blur Filter
    3. Calculate the Magnitude and Orientation of the Images which have been blurred and had the Gradient Filter Applied
    4. Implement a Non-maximum suppression Filter using the Magnitude and Orientation of the Images
    5. Implement the Double threshold Algorithm
    6. Implement Edge Tracking by Hysteresis 

"""

# =======================================================#
# 1. Import basic Modules and Functions and set variables
# =======================================================#

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


from Convolution import convolution

standard_dev_1 = 2
standard_dev_2 = 4
standard_dev_3 = 6



# ==========================================================#
# 2. Create 1-dimensional Gaussian Filter Mask and Apply it #
# ==========================================================#

from Gaussian import X_gaussian_filter_1D,Y_gaussian_filter_1D

input_folder = 'Assignment_01/Input_Images'
images = Path(input_folder).glob('*.jpg')
index = 0

# Apply Gaussian Filter to Images
print("Applying Gaussian Filter")
for image in images:
    index += 1
    im = Image.open(image).convert("L")
    im = np.asarray(im)
    im_blurred = np.zeros_like(im, dtype=np.float32)
    im_filtered_y = np.zeros_like(im, dtype=np.float32)
    im_filtered_x = np.zeros_like(im, dtype=np.float32)
    im_blurred = convolution(im, Y_gaussian_filter_1D(standard_deviation=standard_dev_2))

    # Save the images into the folder:
    plt.imsave(f"Assignment_01/Output_Images/01_Gaussian_Blur/{index}_blurred.jpg",
               im_blurred.astype(np.uint8),
               cmap=plt.cm.Greys_r
            )



# ========================================================================#
# 3. Create 1-dimensional Filter Mask in X and Y direction and apply them #
# ========================================================================#

from Gradient import gaussian_gradient_x_1D
from Gradient import gaussian_gradient_y_1D

input_folder = 'Assignment_01/Input_Images'
images = Path(input_folder).glob('*.jpg')
index=0

# Apply Gradient Filters
print("Applying Gradient Filters")
for image in images:
    index += 1
    im = Image.open(image).convert("L")
    im = np.asarray(im)
    im_filtered_y = np.zeros_like(im, dtype=np.float32)
    im_filtered_x = np.zeros_like(im, dtype=np.float32)
    im_filtered_y = convolution(im, gaussian_gradient_y_1D(standard_deviation=standard_dev_2))
    im_filtered_x = convolution(im, gaussian_gradient_x_1D(standard_deviation=standard_dev_2))

    # Save the images into the folder:
    plt.imsave(f"Assignment_01/Output_Images/03_Gradient_Edge_Detection_Y/{index}_edged_y.jpg",
               im_filtered_y.astype(np.uint8),
               cmap=plt.cm.Greys_r
            )
    plt.imsave(f"Assignment_01/Output_Images/02_Gradient_Edge_Detection_X/{index}_edged_x.jpg",
               im_filtered_x.astype(np.uint8),
               cmap=plt.cm.Greys_r
            )



# =======================================================================================#
# 5. Compute the Magnitude and Orientation of Images
# =======================================================================================#

from Magnitude_Orientation import imageMagnitudeandOrientation
from Magnitude_Orientation import sobelFilter

input_folder = 'Assignment_01/Input_Images'
images = Path(input_folder).glob('*.jpg')
index = 0

# Applying Magnitude and Orientation Filters
print("Applying Magnitude and Orientation Filters")
for im in images:
     index += 1
     magnitude,orientation = imageMagnitudeandOrientation(im,standard_deviation=standard_dev_2)
     plt.imsave(f"Assignment_01/Output_Images/04_Magnitude/{index}_magnitude.png",
                magnitude.astype(np.uint8),
                cmap=plt.cm.Greys_r
            )
     plt.imsave(f"Assignment_01/Output_Images/05_Orientation/{index}_orientation.png",
                orientation.astype(np.uint8),
                cmap=plt.cm.Greys_r
            )

print("Applying Sobel Filters to Determine Magnitude and Orientation")
input_folder = 'Assignment_01/Input_Images'
images = Path(input_folder).glob('*.jpg')
index = 0

for im in images:
     index += 1
     magnitude,orientation = sobelFilter(im)
     plt.imsave(f"Assignment_01/Output_Images/06_Magnitude_Sobel/{index}_magnitude_sobel.png",
                magnitude.astype(np.uint8),
                cmap=plt.cm.Greys_r
            )
     plt.imsave(f"Assignment_01/Output_Images/07_Orientation_Sobel/{index}_orientation_sobel.png",
                orientation.astype(np.uint8),
                cmap=plt.cm.Greys_r
            )

# ===============================================#
# 7. Implement Non-Maximum Suppression Algorithm #
# ===============================================#

"""
Look in local neighborhood and for each pixel, you want to calculate whether the magnitude of the partial derivative of the gradients on this edge pixel is greater then the neigboring pixels. If the magnitude is greater, it is a peak and edge. If not, it is not an edge.

The result of applying this algorithm is that the edges will be thinner. This is good for cluttered images where you need to have more precise descrimination of objects. It's also good for getting rid of flase edges.

"""

from Non_Max_Suppression import non_max_suppression

# Apply to the Images
input_folder = 'Assignment_01/Input_Images'
images = Path(input_folder).glob('*.jpg')
index = 0

print("Applying Non-Maximum Suppression Filters")
for im in images:
     index += 1
     im_non_max_suppression = non_max_suppression(im,standard_deviation=standard_dev_2)
     plt.imsave(f"Assignment_01/Output_Images/08_Non_Max_Suppression/{index}_NMS.png",
                im_non_max_suppression.astype(np.uint8),
                cmap=plt.cm.Greys_r
            )


# =================================#
# 6. Apply Hysteresis Thresholding on images pushed through the Non-Max Suppression Algorithm #
# =================================#

"""

Uses a high and low threshold value to suppress the detection of false edges.

1. Look at the lower threshold. Any pixel with a magnitude of the partial derivative lower then the threshold is not an endge and set to black.

2. Then look at the higher threshold. If the magnitude of the partial derivatives is greater then the threshold value, then it is an edge.

3. If the magnitude is between the higher threshold value and the low threshold value, you check to see if any neighboring pixel is conncted to another pixel with a magnitude greater then the high threshold value.

    3a. To determine what counts as connected, you can do 4 connected pixels, 6 or 8 connected pixels. Meaning that the pixel we are testing needs to be connected to a certain amount of other pixels with a high threshold value or connected to a pixel that is connected to another pixel.

"""

from Hysteresis_Thresholding import hysteresis

input_folder = 'Assignment_01/Input_Images'
images = Path(input_folder).glob('*.jpg')
index = 0

print("Applying the last step of Canny Filter Detection: Hysteresis Thresholding")
for im in images:
     index += 1
     im_hysteresis = hysteresis(image=im,
                                standard_deviation=standard_dev_2,
                                low_threshold=0.05,
                                high_threshold=0.15,
                                weak_pixel=25,
                                strong_pixel=255
                            )
     plt.imsave(f"Assignment_01/Output_Images/09_Hysteresis/{index}_Hysteresis.png",
                im_hysteresis.astype(np.uint8),
                cmap=plt.cm.Greys_r
            )