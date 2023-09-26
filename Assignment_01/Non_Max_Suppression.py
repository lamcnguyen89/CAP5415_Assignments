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

    


