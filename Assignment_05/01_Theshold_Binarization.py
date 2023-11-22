input_images_array= Path(input_folder).glob('*.jpg')

for image in input_images_array:
    for st_dev in standard_deviation:
        # Load image, load the filename and convert the image to an array.
        filename = os.path.basename(image).split('.',1)[0]
        im = Image.open(image).convert("L")
        im = np.asarray(im)

        # Create an empty matrix and convolve the original image with the filter.
        im_filtered_y = np.zeros_like(im, dtype=np.float32)
        im_filtered_x = np.zeros_like(im, dtype=np.float32)
        im_filtered_y = convolution(im, gaussian_gradient_y_1D(standard_deviation=st_dev))
        im_filtered_x = convolution(im, gaussian_gradient_x_1D(standard_deviation=st_dev))

        
        # Save the Edged X-Direction images into the folder directory named "02_Gradient_Edge_Detection_X"  
        plt.imsave(f"Assignment_01/Output_Images/02_Gradient_Edge_Detection_X/{filename}_edged_x_stdev_{st_dev}.png",
                im_filtered_x.astype(np.uint8),
                cmap=plt.cm.Greys_r
            )

        # Save the Edged Y-Direction images into the folder directory named "03_Gradient_Edge_Detection_Y"
        plt.imsave(f"Assignment_01/Output_Images/03_Gradient_Edge_Detection_Y/{filename}_edged_y_stdev_{st_dev}.png",
                im_filtered_y.astype(np.uint8),
                cmap=plt.cm.Greys_r
             )