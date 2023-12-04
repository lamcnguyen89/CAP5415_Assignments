# ===============================================#
# 7. Implement Non-Maximum Suppression Algorithm #
# ===============================================#

"""
Look in local neighborhood and for each pixel, you want to calculate whether the magnitude of the partial derivative of the gradients on this edge pixel is greater then the neigboring pixels. If the magnitude is greater, it is a peak and edge. If not, it is not an edge.

The result of applying this algorithm is that the edges will be thinner. This is good for cluttered images where you need to have more precise descrimination of objects. It's also good for getting rid of flase edges.

"""
from Magnitude_Orientation import imageMagnitudeandOrientation
import numpy as np



def non_max_suppression(image,standard_deviation):
    
        
        # For Non-Max Suppression, we need the Magnitude of the Image and it's Orientation:
        img_magnitude,img_orientation = imageMagnitudeandOrientation(image,standard_deviation)

        # Get shape of Magnitude matrix and create empty matrix in the same shape
        m,n = img_magnitude.shape
        im_nms = np.zeros((m,n), dtype=np.float32)

        # Convert the orientation matrix from radians to degrees because it's more precise
        angle = img_orientation * 180. / np.pi
        angle[angle < 0] += 180

        # Iterate through each pixel and 
        for i in range(1,m-1):
            for j in range(1,n-1):
                    # 0-255 is associated with intensity of white on Greyscale. 
                    # q and r are the pixels right before and right after the tested pixel along the orientation angle
                    q = 255
                    r = 255

                   #If the orientation of the pixel is close to 0 degrees or 180 degrees:
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180): 
                        q = img_magnitude[i, j+1] # Get the magnitude of the pixel directly to the right
                        r = img_magnitude[i, j-1] # Get the magnitude of the pixel directly to the left
                    #If the orientation of the pixel is close to 45 degrees:
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = img_magnitude[i+1, j-1] # Get the magnitude of the pixel bottom left
                        r = img_magnitude[i-1, j+1] # Get the mangitude of the pixel top right
                    #If the orientation of the pixel is close to 90 degrees:
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = img_magnitude[i+1, j] # Get the magnitude of the pixel directly below
                        r = img_magnitude[i-1, j] # Get the magnitude of the pixel directly above
                    #If the orientation of the pixel is close to 135 degrees:
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = img_magnitude[i-1, j-1] # Get the magnitude of the pixel bottom right
                        r = img_magnitude[i+1, j+1] # Get the magnitude of the pixel top left

                    # Compare the magnitude of the pixel at our index with the pixels before and after it along the orientation
                    # If it is greater then both (the maximum), then the pixel value in the empty NMS array at the same index is set to magnitude of the tested pixel,
                    # Otherwise the greyscale value is set to 0. Meaning it's dark.
                    if (img_magnitude[i,j] >= q) and (img_magnitude[i,j] >= r):
                        im_nms[i,j] = img_magnitude[i,j]
                    else:
                        im_nms[i,j] = 0


        return im_nms

    

