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
import os

# The convolution function will be used to combine a kernel with the input images to produce an output image
from Convolution import convolution

# Standard Deviation Values to Test out on images:
standard_deviation = [1,4,8] 
# For the standard deviations, you don't want to make it too low because you can have too many lines in the final image. 
# But the higher standard deviation seems to give a point of diminishing returns. A Standard Deviation of 4 seems to give the best results


# Folder that contains images that we will be filtering:
input_folder = 'Assignment_01/Input_Images'



# ==========================================================#
# 2. Create 1-dimensional Gaussian Filter Mask and Apply it #
# ==========================================================#

# Import Filter(s)
from Gaussian import X_gaussian_filter_1D,Y_gaussian_filter_1D

print("Applying Gaussian Filter")
input_images_array= Path(input_folder).glob('*.jpg')

for image in input_images_array:
    for st_dev in standard_deviation:
        # Load image, load the filename and convert the image to an array.
        filename = os.path.basename(image).split('.',1)[0]
        im = Image.open(image).convert("L")
        im = np.asarray(im)

        # Create an empty matrix and convolve the original image with the filter.
        im_blurred1 = np.zeros_like(im, dtype=np.float32)
        im_blurred1= convolution(im, Y_gaussian_filter_1D(standard_deviation= st_dev))
        
        
        # Save the Blurred images into the folder directory named "01_Gaussian_Blur"
        plt.imsave(f"Assignment_01/Output_Images/01_Gaussian_Blur/{filename}_blurred_stdev_{st_dev}_.png",
                im_blurred1.astype(np.uint8),
                cmap=plt.cm.Greys_r
                )



# ========================================================================#
# 3. Create 1-dimensional Filter Mask in X and Y direction and apply them #
# ========================================================================#

# Import Filter(s)
from Gradient import gaussian_gradient_x_1D
from Gradient import gaussian_gradient_y_1D


print("Applying Gradient Filters")
input_images_array= Path(input_folder).glob('*.jpg')

for image in input_images_array:
    for st_dev in standard_deviation:
        # Load image, load the filename and convert the image to an array.
        filename = os.path.basename(image).split('.',1)[0]
        im = Image.open(image).convert("L")
        im = np.asarray(im)

        # Create an empty matrix and convolve the original image with the filter.
        im_filtered_y = np.zeros_like(im, dtype=np.float32)
        im_filtered_x = np.zeros_like(im, dtype=np.float32)
        im_filtered_y = convolution(im, gaussian_gradient_y_1D(standard_deviation=st_dev))
        im_filtered_x = convolution(im, gaussian_gradient_x_1D(standard_deviation=st_dev))

        
        # Save the Edged X-Direction images into the folder directory named "02_Gradient_Edge_Detection_X"  
        plt.imsave(f"Assignment_01/Output_Images/02_Gradient_Edge_Detection_X/{filename}_edged_x_stdev_{st_dev}.png",
                im_filtered_x.astype(np.uint8),
                cmap=plt.cm.Greys_r
            )

        # Save the Edged Y-Direction images into the folder directory named "03_Gradient_Edge_Detection_Y"
        plt.imsave(f"Assignment_01/Output_Images/03_Gradient_Edge_Detection_Y/{filename}_edged_y_stdev_{st_dev}.png",
                im_filtered_y.astype(np.uint8),
                cmap=plt.cm.Greys_r
             )




# =======================================================================================#
# 5. Compute the Magnitude and Orientation of Images
# =======================================================================================#

# Import Filter(s)
from Magnitude_Orientation import imageMagnitudeandOrientation
from Magnitude_Orientation import sobelFilter

input_images_array= Path(input_folder).glob('*.jpg')
print("Applying Magnitude and Orientation Filters")

for image in input_images_array:
     for st_dev in standard_deviation:
        # Extract the name of the file to use in output image
        filename = os.path.basename(image).split('.',1)[0]

        # Convolve the original image with the filter.
        magnitude,orientation = imageMagnitudeandOrientation(image,standard_deviation=st_dev)

        # Save the Magnitude Images to a folder named "04_Magnitude"
        plt.imsave(f"Assignment_01/Output_Images/04_Magnitude/{filename}_magnitude_stdev_{st_dev}.png",
                    magnitude.astype(np.uint8),
                    cmap=plt.cm.Greys_r
                )
        # Save the Magnitude Images to a folder named "05_Orientation"
        plt.imsave(f"Assignment_01/Output_Images/05_Orientation/{filename}_orientation_stdev_{st_dev}.png",
                    orientation.astype(np.uint8),
                    cmap=plt.cm.Greys_r
                )


# Applying Sobel Filters to get Magnitude and Orientation of Image.
# This isn't part of the assignment but I wanted to compare the results.
# Sobel Filters are like handcrafted filters that allow one to quickly get the derivative of an image without doing as many calculations
print("Applying Sobel Filters to Determine Magnitude and Orientation")
input_images_array= Path(input_folder).glob('*.jpg')


for image in input_images_array:
        # Extract the name of the file to use in output image
        filename = os.path.basename(image).split('.',1)[0]
        # Convolve the original image with the filter.
        magnitude,orientation = sobelFilter(image)
        
        # Save the Magnitude images created with a sobel filter into a folder named: 06_Magnitude_Sobel
        plt.imsave(f"Assignment_01/Output_Images/06_Magnitude_Sobel/{filename}_magnitude_sobel_.png",
                    magnitude.astype(np.uint8),
                    cmap=plt.cm.Greys_r
                )
        # Save the Orientation images created with a sobel filter into a folder named: 07_Orientation_Sobel
        plt.imsave(f"Assignment_01/Output_Images/07_Orientation_Sobel/{filename}_orientation_sobel.png",
                    orientation.astype(np.uint8),
                    cmap=plt.cm.Greys_r
                )


# ===============================================#
# 7. Implement Non-Maximum Suppression Algorithm #
# ===============================================#


# Import Filter(s)
from Non_Max_Suppression import non_max_suppression


print("Applying Non-Maximum Suppression Filters")
input_images_array= Path(input_folder).glob('*.jpg')

for image in input_images_array:
     for st_dev in standard_deviation:
        # Extract the name of the file to use in output image
        filename = os.path.basename(image).split('.',1)[0]
        # Convolve the original image with the filter.
        im_non_max_suppression = non_max_suppression(image,standard_deviation=st_dev)

        # Save the NMS images created with an NMS filter into a folder named: 08_Non_Max_Suppression
        plt.imsave(f"Assignment_01/Output_Images/08_Non_Max_Suppression/{filename}_NMS_stdev_{st_dev}.png",
                    im_non_max_suppression.astype(np.uint8),
                    cmap=plt.cm.Greys_r
                )


# =================================#
# 6. Apply Hysteresis Thresholding on images pushed through the Non-Max Suppression Algorithm #
# =================================#

# Import Filter(s)
from Hysteresis_Thresholding import hysteresis

print("Applying the last step of Canny Filter Detection: Hysteresis Thresholding")
input_images_array= Path(input_folder).glob('*.jpg')

for image in input_images_array:
     for st_dev in standard_deviation:
        # Extract the name of the file to use in output image
        filename = os.path.basename(image).split('.',1)[0]

        # Input the original image with the filter.
        im_hysteresis = hysteresis(image=image,
                                    standard_deviation=st_dev,
                                    low_threshold=0.05,
                                    high_threshold=0.09,
                                    weak_pixel=125,
                                    strong_pixel=255
                                )
        
        # Save the Hysteresis images created with a Hysteresis filter into a folder named: 09_Hysteresis
        plt.imsave(f"Assignment_01/Output_Images/09_Hysteresis/{filename}_Hysteresis_stdev_{st_dev}.png",
                    im_hysteresis.astype(np.uint8),
                    cmap=plt.cm.Greys_r
                )   
