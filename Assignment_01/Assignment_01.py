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

# ===============================================#
# 1. Import Modules  and create utility functions#
# ===============================================#


from pathlib import Path
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Create Convolution Function:
def convolution(oldimage, kernel):
    image_h = oldimage.shape[0]
    image_w = oldimage.shape[1]
    
    
    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]
    
    if(len(oldimage.shape) == 3):
        image_pad = np.pad(oldimage, pad_width=((kernel_h // 2, kernel_h // 2),(kernel_w // 2, kernel_w // 2),(0,0)), mode='constant', constant_values=0).astype(np.float32)
    elif(len(oldimage.shape) == 2):
        image_pad = np.pad(oldimage, pad_width=((kernel_h // 2, kernel_h // 2),(kernel_w // 2, kernel_w // 2)), mode='constant', constant_values=0).astype(np.float32)
    
    
    h = kernel_h // 2
    w = kernel_w // 2
    
    image_conv = np.zeros(image_pad.shape)
    
    for i in range(h, image_pad.shape[0]-h):
        for j in range(w, image_pad.shape[1]-w):
            #sum = 0
            x = image_pad[i-h:i-h+kernel_h, j-w:j-w+kernel_w]
            x = x.flatten()*kernel.flatten()
            
            
#             for m in range(kernel_h):
#                 for n in range(kernel_w):
#                     sum += kernel[m][n] * image_pad[i-h+m][j-w+n]
            
            image_conv[i][j] = x.sum()
    h_end = -h
    w_end = -w
    
    if(h == 0):
        return image_conv[h:,w:w_end]
    if(w == 0):
        return image_conv[h:h_end,w:]

    return image_conv[h:h_end,w:w_end]


# ==========================================================#
# 2. Create 1-dimensional Gaussian Filter Mask and Apply it #
# ==========================================================#

def gaussian_filter(standard_deviation):

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



# # Apply Gaussian Filter to Images
# input_folder = 'Assignment_01/Images'
# images = Path(input_folder).glob('*.jpg')
# index = 0
# for image in images:
#     index += 1
#     im = Image.open(image)
#     im = np.asarray(im)
#     im_blurred = np.zeros_like(im, dtype=np.float32)
#     im_filtered_y = np.zeros_like(im, dtype=np.float32)
#     im_filtered_x = np.zeros_like(im, dtype=np.float32)
#     for c in range(3):
#         im_blurred[:, :, c] = convolution(im[:, :, c], gaussian_filter(2))

#     # Save the images into the folder:
#     plt.imsave(f"Assignment_01/Blurred_Images/{index}_blurred.jpg",im_blurred.astype(np.uint8))



# ==================================================================================#
# 3. Create 1-dimensional  1st derivative Gaussian Filter Mask in X and Y direction #
# ==================================================================================#

def gaussian_filter_partial_derivative_x(standard_deviation):
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
            

def gaussian_filter_partial_derivative_y(standard_deviation):
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
            
    

# ===========================================================================#
# 4. Convolve the Image with the Partial Derivatives of the Gaussian Function #
# ===========================================================================#

# # Get images and apply partial derivatives of the gaussian function and save to output folder
# input_folder = 'Assignment_01/Images'
# images = Path(input_folder).glob('*.jpg')
# index = 0
# for image in images:
#     index += 1
#     im = Image.open(image)
#     im = np.asarray(im)
#     im_filtered_y = np.zeros_like(im, dtype=np.float32)
#     im_filtered_x = np.zeros_like(im, dtype=np.float32)
#     for c in range(3):
#         im_filtered_y[:, :, c] = convolution(im[:, :, c], gaussian_filter_partial_derivative_y(2))
#         im_filtered_x[:, :, c] = convolution(im[:, :, c], gaussian_filter_partial_derivative_x(2))
#     # Save the images into the folder:
#     plt.imsave(f"Assignment_01/Edged_Images_y/{index}_edged_y.jpg",im_filtered_y.astype(np.uint8))
#     plt.imsave(f"Assignment_01/Edged_Images_x/{index}_edged_x.jpg",im_filtered_x.astype(np.uint8))



    
# =======================================================================================#
# 5. Compute the Magnitude and Orientation of Images
# =======================================================================================#

def imageMagnitudeandOrientation(image,standard_deviation):
      image = Image.open(image)
      image = np.asarray(image)
      
      im_filtered_x = np.zeros_like(image, dtype=np.float32)
      im_filtered_y = np.zeros_like(image, dtype=np.float32)

      x_filter = gaussian_filter_partial_derivative_x(standard_deviation)
      y_filter = gaussian_filter_partial_derivative_y(standard_deviation)

      for c in range(3):
         im_filtered_y[:, :, c] = convolution(image[:, :, c], y_filter)
         im_filtered_x[:, :, c] = convolution(image[:, :, c], x_filter)

      magnitude = 20*np.sqrt(im_filtered_x**2 + im_filtered_y**2)
      orientation = np.arctan(im_filtered_y,im_filtered_x)

      
      return (magnitude.astype(np.uint8),orientation.astype(np.uint8))

input_folder = 'Assignment_01/Images'
images = Path(input_folder).glob('*.jpg')
index = 0

# for im in images:
#      index += 1
#      magnitude,orientation = imageMagnitudeandOrientation(im,4)
#      plt.imsave(f"Assignment_01/Magnitude_Images/{index}_magnitude.png",magnitude.astype(np.uint8))
#      plt.imsave(f"Assignment_01/Orientation_Images/{index}_orientation.png",orientation.astype(np.uint8))
     


# Magnitude and Orientation using Sobel Filter

def sobelImage(images):
     
     image = Image.open(images)
     image = np.asarray(image)

     sobelFilterX = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
     sobelFilterY = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

     im_filtered_y = np.zeros_like(image, dtype=np.float32)
     im_filtered_x = np.zeros_like(image, dtype=np.float32)

     for c in range(3):
         im_filtered_y[:, :, c] = convolution(image[:, :, c], sobelFilterY)
         im_filtered_x[:, :, c] = convolution(image[:, :, c], sobelFilterX)

     magnitude = np.sqrt(im_filtered_x**2 + im_filtered_y**2)
     orientation = np.arctan(im_filtered_y,im_filtered_x)

     return(magnitude.astype(np.uint8), orientation.astype(np.uint8))

input_folder = 'Assignment_01/Images'
images = Path(input_folder).glob('*.jpg')
index = 0

# for im in images:
#      index += 1
#      magnitude,orientation = sobelImage(im)
#      plt.imsave(f"Assignment_01/Magnitude_Sobel_Images/{index}_magnitude_sobel.png",magnitude.astype(np.uint8))
#      plt.imsave(f"Assignment_01/Orientation_Sobel_Images/{index}_orientation_sobel.png",orientation.astype(np.uint8))
     
     

# ===============================================#
# 7. Implement Non-Maximum Suppression Algorithm #
# ===============================================#

"""
Look in local neighborhood and for each pixel, you want to calculate whether the magnitude of the partial derivative of the gradients on this edge pixel is greater then the neigboring pixels. If the magnitude is greater, it is a peak and edge. If not, it is not an edge.

The result of applying this algorithm is that the edges will be thinner. This is good for cluttered images where you need to have more precise descrimination of objects. It's also good for getting rid of flase edges.

"""


def non_max_suppression(image,standard_deviation):
        
        # For Non-Max Suppression, we need the Magnitude of the Image and it's Orientation:

        img_magnitude,img_orientation = imageMagnitudeandOrientation(image,standard_deviation)

        img_magnitude =img_magnitude[:,:,0]
        img_orientation=img_orientation[:,:,0]

        M,N = img_magnitude.shape

        im_nms = np.zeros((M,N), dtype=np.float32)
        angle = img_orientation * 180. / np.pi
        angle[angle < 0] += 180


        for i in range(1,M-1):
            for j in range(1,N-1):
                try:
                    q = 255
                    r = 255

                   #angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = img_magnitude[i, j+1]
                        r = img_magnitude[i, j-1]
                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = img_magnitude[i+1, j-1]
                        r = img_magnitude[i-1, j+1]
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = img_magnitude[i+1, j]
                        r = img_magnitude[i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = img_magnitude[i-1, j-1]
                        r = img_magnitude[i+1, j+1]

                    if (img_magnitude[i,j] >= q) and (img_magnitude[i,j] >= r):
                        im_nms[i,j] = img_magnitude[i,j]
                    else:
                        im_nms[i,j] = 0


                except IndexError as e:
                    pass

        return im_nms.astype(np.uint8)

    
# Apply to the Images



input_folder = 'Assignment_01/Images'
images = Path(input_folder).glob('*.jpg')
index = 0

for im in images:
     index += 1
     im_non_max_suppression = non_max_suppression(im,4)
     plt.imsave(f"Assignment_01/Non_Max_Suppression_Images/{index}_NMS.png",im_non_max_suppression.astype(np.uint8))



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



