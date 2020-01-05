import os
import tensorflow

keras = tensorflow.keras
load_img = keras.preprocessing.image.load_img
img_to_array = keras.preprocessing.image.img_to_array
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator

from numpy import expand_dims
from PIL import Image

source_dir = './Data/Converted/'
dest_dir = './Data/Augmented/'

classes = os.listdir(source_dir)
total_images = 1000

if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)

for c in classes:
    src = os.path.join(source_dir, c)
    dest = os.path.join(dest_dir, c)

    if not os.path.exists(dest):
        os.mkdir(dest)

    n_avail_images = len(os.listdir(src))
    aug_per_image = round((total_images - n_avail_images) / n_avail_images)

    images = [os.path.join(src, f) for f in os.listdir(src)]

    current = 0
    for image in images:
        img = load_img(image)
        data = img_to_array(img)

        Image.fromarray(data.astype('uint8'), 'RGB').save(os.path.join(dest, f'{c}_{current:05d}.jpg'))

        sample = expand_dims(data, 0)
        datagen = ImageDataGenerator(width_shift_range=[-10,10], rotation_range=30, 
                                        horizontal_flip=True, fill_mode='nearest')
        it = datagen.flow(sample, batch_size=1)

        current += 1
        for j in range(0, aug_per_image + 1):
            if current >= total_images: break
            batch = it.next()
            img = Image.fromarray(batch[0].astype('uint8'), 'RGB')
            img.save(os.path.join(dest, f'{c}_{current:05d}.jpg'))
            current += 1

        if current >= total_images: break