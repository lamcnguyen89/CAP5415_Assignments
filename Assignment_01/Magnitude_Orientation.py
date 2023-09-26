# =======================================================================================#
# 5. Compute the Magnitude and Orientation of Images
# =======================================================================================#
from PIL import Image
import numpy as np
from Gradient import gaussian_filter_partial_derivative_x,gaussian_filter_partial_derivative_y
from Convolution import convolution

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


     