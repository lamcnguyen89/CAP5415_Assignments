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

from Non_Max_Suppression import non_max_suppression
import numpy as np

def threshold(image,standard_deviation,low_threshold_ratio,high_threshold_ratio,weak_pixel,strong_pixel):

    # Before doing thresholding, the input image has to have the Non-Max Suppression Filter Applied
    non_max_suppression_image = non_max_suppression(image,standard_deviation)

    # Threshold values are between 0-255 
    # High Threshold is used to identify strong pixels. Get the largest number and multiply it by the high threshold 
    high_threshold = non_max_suppression_image.max() * high_threshold_ratio
    # Low threshold is used to identify non-relevant pixels
    low_threshold = high_threshold * low_threshold_ratio

    # Create an empty matrix in the same shape as the image being processed
    m, n = non_max_suppression_image.shape
    img_threshold = np.zeros((m,n), dtype=np.int32)

    # Weak and Strong Pixel values are set between 0-255 in accordance to greyscale spectrum. 
    # You generally want the weak pixels to have a lower value then the strong to distinguish between strong and weak.
    weak = np.int32(weak_pixel)
    strong = np.int32(strong_pixel)

    # Compare the pixel value brightness of the NMS image with the defined threshold value 
    strong_i, strong_j = np.where(non_max_suppression_image >= high_threshold)
    weak_i, weak_j = np.where((non_max_suppression_image <= high_threshold) & (non_max_suppression_image >= low_threshold))

    # Classify the pixel as strong or weak
    img_threshold[strong_i, strong_j] = strong
    img_threshold[weak_i, weak_j] = weak

    return img_threshold



def hysteresis(image,standard_deviation,low_threshold,high_threshold,weak_pixel,strong_pixel):

    # Import unprocessed image and perform thresholding filter to classify pixels as strong or weak
    hysteresis_img = threshold(image,standard_deviation,low_threshold,high_threshold,weak_pixel,strong_pixel)

    # Get the shape of the image so that we can iterate over to perform the pixel classification
    m, n = hysteresis_img.shape

    # Set the strong and weak pixel values (0-255)
    weak = weak_pixel
    strong = strong_pixel

    # Iterate through each pixel in the Threshold image
    for i in range(1, m-1):
        for j in range(1, n-1):

            # If pixel being checked at the index is weak check if any of the adjacent pixels are strong. 
            # If none of the adjacent pixels is strong, then the pixel is classified as weak and its pixel brightness is set to 0 which means it won't be shown in the final image
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