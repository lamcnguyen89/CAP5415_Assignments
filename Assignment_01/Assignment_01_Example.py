import cv2
import numpy as np
from matplotlib import pyplot as plt


# Parameters 
img = cv2.imread('Assignment_01/Images/58060.jpg', 0)
threshold_01 = 100
threshold_02 = 200


canny = cv2.Canny(img,threshold_01,threshold_02)



titles = ['image', 'canny']
images = [img, canny]

for i in range(2):
    plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])


plt.show()






