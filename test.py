from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

input_folder = 'Assignment_01/Input_Images'

input_images_array= Path(input_folder).glob('*.jpg')

for image in input_images_array:
    filename = os.path.basename(image).split('.',1)[0]
    print(filename)
