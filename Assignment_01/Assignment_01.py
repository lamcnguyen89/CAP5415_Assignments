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



# Apply Gaussian Filter to Images
input_folder = 'Assignment_01/Images'
images = Path(input_folder).glob('*.jpg')
index = 0
for image in images:
    index += 1
    im = Image.open(image).convert("L")
    im = np.asarray(im)
    im_blurred = np.zeros_like(im, dtype=np.float32)
    im_filtered_y = np.zeros_like(im, dtype=np.float32)
    im_filtered_x = np.zeros_like(im, dtype=np.float32)
    im_blurred = convolution(im, gaussian_filter(2))

    # Save the images into the folder:
    plt.imsave(f"Assignment_01/Output_Images/01_Gaussian_Blur/{index}_blurred.jpg",
               im_blurred.astype(np.uint8),
               cmap=plt.cm.Greys_r
            )



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

# Get images and apply partial derivatives of the gaussian function and save to output folder
input_folder = 'Assignment_01/Images'
images = Path(input_folder).glob('*.jpg')
index = 0
for image in images:
    index += 1
    im = Image.open(image).convert("L")
    im = np.asarray(im)
    im_filtered_y = np.zeros_like(im, dtype=np.float32)
    im_filtered_x = np.zeros_like(im, dtype=np.float32)
    for c in range(3):
        im_filtered_y = convolution(im, gaussian_filter_partial_derivative_y(2))
        im_filtered_x = convolution(im, gaussian_filter_partial_derivative_x(2))
    # Save the images into the folder:
    plt.imsave(f"Assignment_01/Output_Images/03_Gradient_Edge_Detection_Y/{index}_edged_y.jpg",
               im_filtered_y.astype(np.uint8),
               cmap=plt.cm.Greys_r
            )
    plt.imsave(f"Assignment_01/Output_Images/03_Gradient_Edge_Detection_X/{index}_edged_x.jpg",
               im_filtered_x.astype(np.uint8),
               cmap=plt.cm.Greys_r
            )



    
# =======================================================================================#
# 5. Compute the Magnitude and Orientation of Images
# =======================================================================================#

def imageMagnitudeandOrientation(image,standard_deviation):
      image = Image.open(image).convert("L")
      image = np.asarray(image)
      
      im_filtered_x = np.zeros_like(image, dtype=np.float32)
      im_filtered_y = np.zeros_like(image, dtype=np.float32)

      x_filter = gaussian_filter_partial_derivative_x(standard_deviation)
      y_filter = gaussian_filter_partial_derivative_y(standard_deviation)

    
      im_filtered_y = convolution(image, y_filter)
      im_filtered_x = convolution(image, x_filter)

      magnitude = 20*np.sqrt(im_filtered_x**2 + im_filtered_y**2)
      orientation = np.arctan(im_filtered_y,im_filtered_x)

      
      return (magnitude,orientation)

input_folder = 'Assignment_01/Images'
images = Path(input_folder).glob('*.jpg')
index = 0

for im in images:
     index += 1
     magnitude,orientation = imageMagnitudeandOrientation(im,4)
     plt.imsave(f"Assignment_01/Output_Images/04_Magnitude/{index}_magnitude.png",
                magnitude.astype(np.uint8),
                cmap=plt.cm.Greys_r
            )
     plt.imsave(f"Assignment_01/Output_Images/05_Orientation/{index}_orientation.png",
                orientation.astype(np.uint8),
                cmap=plt.cm.Greys_r
            )
     


# Magnitude and Orientation using Sobel Filter

def sobelFilter(images):
     
     image = Image.open(images).convert("L")
     image = np.asarray(image)

     sobelFilterX = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
     sobelFilterY = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

     im_filtered_y = np.zeros_like(image, dtype=np.float32)
     im_filtered_x = np.zeros_like(image, dtype=np.float32)

     im_filtered_y = convolution(image, sobelFilterY)
     im_filtered_x = convolution(image, sobelFilterX)

     magnitude = np.sqrt(im_filtered_x**2 + im_filtered_y**2)
     orientation = np.arctan(im_filtered_y,im_filtered_x)

     return(magnitude, orientation)

input_folder = 'Assignment_01/Images'
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


def non_max_suppression(image,standard_deviation):
        
        # For Non-Max Suppression, we need the Magnitude of the Image and it's Orientation:

        img_magnitude,img_orientation = imageMagnitudeandOrientation(image,standard_deviation)

        m,n = img_magnitude.shape

        im_nms = np.zeros((m,n), dtype=np.float32)
        angle = img_orientation * 180. / np.pi
        angle[angle < 0] += 180


        for i in range(1,m-1):
            for j in range(1,n-1):
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

        return im_nms

    
# Apply to the Images
input_folder = 'Assignment_01/Images'
images = Path(input_folder).glob('*.jpg')
index = 0

for im in images:
     index += 1
     im_non_max_suppression = non_max_suppression(im,4)
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

def threshold(image, standard_deviation,low_threshold,high_threshold,weak_pixel,strong_pixel):

    non_max_suppression_image = non_max_suppression(image,standard_deviation)

    high_threshold = non_max_suppression_image.max() * high_threshold;
    low_threshold = high_threshold * low_threshold;

    m, n = non_max_suppression_image.shape
    img_threshold = np.zeros((m,n), dtype=np.int32)

    weak = np.int32(weak_pixel)
    strong = np.int32(strong_pixel)

    strong_i, strong_j = np.where(non_max_suppression_image >= high_threshold)
    zeros_i, zeros_j = np.where(non_max_suppression_image < low_threshold)

    weak_i, weak_j = np.where((non_max_suppression_image <= high_threshold) & (non_max_suppression_image >= low_threshold))

    img_threshold[strong_i, strong_j] = strong
    img_threshold[weak_i, weak_j] = weak

    return img_threshold



def hysteresis(image, standard_deviation,low_threshold, high_threshold, weak_pixel,strong_pixel):

    hysteresis_img = threshold(image,standard_deviation,low_threshold,high_threshold,weak_pixel,strong_pixel)

    m, n = hysteresis_img.shape
    weak = weak_pixel
    strong = strong_pixel

    for i in range(1, m-1):
        for j in range(1, n-1):
            if (hysteresis_img[i,j] == weak):
                try:
                    if ((hysteresis_img[i+1, j-1] == strong) or (hysteresis_img[i+1, j] == strong) or (hysteresis_img[i+1, j+1] == strong)
                        or (hysteresis_img[i, j-1] == strong) or (hysteresis_img[i, j+1] == strong)
                        or (hysteresis_img[i-1, j-1] == strong) or (hysteresis_img[i-1, j] == strong) or (hysteresis_img[i-1, j+1] == strong)):
                            hysteresis_img[i, j] = strong
                    else:
                            hysteresis_img[i, j] = 0
                except IndexError as e:
                        pass

    return hysteresis_img

# Apply to the Images
input_folder = 'Assignment_01/Images'
images = Path(input_folder).glob('*.jpg')
index = 0

for im in images:
     index += 1
     im_hysteresis = hysteresis(image=im,
                                standard_deviation=4,
                                low_threshold=0.05,
                                high_threshold=0.15,
                                weak_pixel=25,
                                strong_pixel=255
                            )
     plt.imsave(f"Assignment_01/Output_Images/09_Hysteresis/{index}_Hysteresis.png",
                im_hysteresis.astype(np.uint8),
                cmap=plt.cm.Greys_r
            )