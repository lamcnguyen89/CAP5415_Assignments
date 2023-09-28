#=================================#
# Create the Convolution Function #
#=================================#


# This is a key function for combining images with a filter. This function will be used many times for the different types of filters


import numpy as np


def convolution(oldimage, kernel):    
    
    # Get the height and width of the kernel
    try:
        kernel_h = kernel.shape[0]
    except:
        kernel_h = 1

    try:
        kernel_w = kernel.shape[1]
    except:
        kernel_w = 1
    
    # Add a certiain amount of padding based on the kernel dimensions and image dimensions to ensure that there that the filter can hit every pixel
    if(len(oldimage.shape) == 3):
        image_pad = np.pad(oldimage, pad_width=((kernel_h // 2, kernel_h // 2),(kernel_w // 2, kernel_w // 2),(0,0)), mode='constant', constant_values=0).astype(np.float32)
    elif(len(oldimage.shape) == 2):
        image_pad = np.pad(oldimage, pad_width=((kernel_h // 2, kernel_h // 2),(kernel_w // 2, kernel_w // 2)), mode='constant', constant_values=0).astype(np.float32)
    
    
    h = kernel_h // 2
    w = kernel_w // 2
    
    image_conv = np.zeros(image_pad.shape)
    
    # Iterate through the dimensions of the empty matrix, convolve (element-wise multiplication) values and add them to the particular index of that empty matrix
    for i in range(h, image_pad.shape[0]-h):
        for j in range(w, image_pad.shape[1]-w):
            #sum = 0
            x = image_pad[i-h:i-h+kernel_h, j-w:j-w+kernel_w]

            # collapse the arrays at a particular index in both the image and filter and multiply the matrices together
            x = x.flatten()*kernel.flatten()
            
            # Add all the values together to show the convolved value
            image_conv[i][j] = x.sum()
   
    h_end = -h
    w_end = -w

    # Error detection to make sure the correct matrix format is output by preventin
    if(h == 0):
        return image_conv[h:,w:w_end]
    if(w == 0):
        return image_conv[h:h_end,w:]

    return image_conv[h:h_end,w:w_end]