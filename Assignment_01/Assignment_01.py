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

"""


# 1. Read a grey scale image that can be found from the Berkeley Segmentation Dataset, Training Images, and store it as a matric named I
        # Link: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/dataset/images.html


# import required module
from pathlib import Path
 
# get the path/directory
folder_dir = 'Assignment_01/Images'
 
# iterate over files in that directory and add to array

image_array = []
images = Path(folder_dir).glob('*.jpg')
for image in images:
    image_array.append(str(image)) # convert image path to string
print(image_array)



# 2. Create a one-dimensional Gaussian mask G to convolve with I. The standard Deviation(s) of this Gaussian is a parameter to the edge detector (call it std > 0)




# 3. Create a one-dimensional mask for the first derivative of the Gaussian in the x and y directions; call these G_x and G_y. The same std > 0 value is used as in step 2



# 4. Convolve I_x with G_x to give I'_x, the x component of I convolved with the derivative of the Gaussian, and convolve I_y with G_y  to give the I'_y component of I convolved with the derivative of the Gaussian



# 5. Compute the magnitude of the edge response by combining the x and y components. The magnitude of the result can be computed at each pixel (x,y) as M(x,y) = (I'_y**.5 + I'x**.5)**.5



# 6. Implement the non-maximimum suppression algorithm that we discussed in the lecture. Pixels that are not local maxima should be removed with this method. In other words, not all the pixels indicating strong magnitude are edges in fact We need to remove false positive edge locations from the image.


# 7. Apply Hysteresis thresholding to obtain the final edge-map. You may use any existing library function to compute connected components.

