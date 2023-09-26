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