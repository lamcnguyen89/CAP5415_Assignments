# PA 1: Canny Edge Detector

Canny Edge Detection is the process of doing these steps to an image:

1. Gaussian Blur Filter
2. Gradient Filter
3. Calculate the Magnitude and Orientation of the Images which have been blurred and had the Gradient Filter Applied
4. Implement a Non-maximum suppression Filter using the Magnitude and Orientation of the Images
5. Double threshold Algorithm
6. Implement Edge Tracking by Hysteresis incorporating the Double Threshold

From those steps we can take an image and process it in such a way that the edges of objects are shown. It's like making a line drawing of an image. This allows for efficient detection of objects without having all the noise of colors and shadows getting in the way. Also Edge detection can be used to produce learnable features for a Machine Learning Model. 

## Gaussian Blur Filter
As shown in the images below, it seems that there are diminishing returns to the Blur after a Standard Deviation of 4.

#### Gaussian Blur with Standard Deviation of 1
![Gaussian Blur Std 1](/Assignment_01/Output_Images/01_Gaussian_Blur/119082_blurred_stdev_1_.png)

#### Gaussian Blur with Standard Deviation of 4
![Gaussian Blur Std 4](/Assignment_01/Output_Images/01_Gaussian_Blur/119082_blurred_stdev_4_.png)

#### Gaussian Blur with Standard Deviation of 8
![Gaussian Blur Std 8](/Assignment_01/Output_Images/01_Gaussian_Blur/119082_blurred_stdev_8_.png)


## Derivative of Gaussian Filter

#### Derivative of Gaussian Filter in X Direction with Standard Deviation of 1
![Gradient Filter Std 1](/Assignment_01/Output_Images/02_Gradient_Edge_Detection_X/119082_edged_x_stdev_1.png)

#### Derivative of Gaussian Filter with Standard Deviation of 4
![Gradient Filter Std 4](/Assignment_01/Output_Images/02_Gradient_Edge_Detection_X/119082_edged_x_stdev_4.png)

#### Derivative of Gaussian Filter with Standard Deviation of 8
![Gradient Filter Std 8](/Assignment_01/Output_Images/02_Gradient_Edge_Detection_X/119082_edged_x_stdev_8.png)



## Magnitude of the Image

#### Magnitude of Image with Standard Deviation of 1
![Magnitude Filter std 1](/Assignment_01/Output_Images/04_Magnitude/119082_magnitude_stdev_1.png)

#### Magnitude of Image with Standard Deviation of 4
![Magnitude Filter std 2](/Assignment_01/Output_Images/04_Magnitude/119082_magnitude_stdev_4.png)

#### Magnitude of Image with Standard Deviation of 8
![Magnitude Filter std 3](/Assignment_01/Output_Images/04_Magnitude/119082_magnitude_stdev_8.png)


## Non-Maximum Suppression of Image

#### Non-Maximum Suppression of Image with Standard Deviation of 1
![NMS Std 1](/Assignment_01/Output_Images/08_Non_Max_Suppression/119082_NMS_stdev_1.png)

#### Non-Maximum Suppression of Image with Standard Deviation of 4
![NMS Std 1](/Assignment_01/Output_Images/08_Non_Max_Suppression/119082_NMS_stdev_4.png)

#### Non-Maximum Suppression of Image with Standard Deviation of 8
![NMS Std 1](/Assignment_01/Output_Images/08_Non_Max_Suppression/119082_NMS_stdev_8.png)


## Hysteresis after Double Threshold

#### Hysteresis after Double Threshold with Standard Deviation of 1
![Hysteresis Std 1](/Assignment_01/Output_Images/09_Hysteresis/119082_Hysteresis_stdev_1.png)

#### Hysteresis after Double Threshold with Standard Deviation of 4
![Hysteresis Std 4](/Assignment_01/Output_Images/09_Hysteresis/119082_Hysteresis_stdev_4.png)

#### Hysteresis after Double Threshold with Standard Deviation of 8
![Hysteresis Std 8](/Assignment_01/Output_Images/09_Hysteresis/119082_Hysteresis_stdev_8.png)
