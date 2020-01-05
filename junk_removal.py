import os
from PIL import Image

directory = './Data/Original/Head'
image_names = os.listdir(directory)

for image_name in image_names:
    split = image_name.split('_')
    if 'Mask.png' in split:
        path = os.path.join(directory, image_name)
        os.remove(path)