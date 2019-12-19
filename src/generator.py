import numpy as np

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

def normalize(batch):
    images, labels = batch

    for i in range(len(images)):
        average = np.average(images[i])
        st_dev = np.std(images[i])

        images[i] = (images[i] - average) / st_dev

    return (images, labels)

class Generator:
    def __init__(self, data_generator, path, target_size=(200, 200), batch_size=40, class_mode='categorical', color_mode='grayscale'):
        self.generator = data_generator.flow_from_directory(
            path, target_size=target_size, batch_size=batch_size, 
            class_mode=class_mode, color_mode=color_mode,
            seed=42
        )

    def get_image_batch(self):
        while True:
            yield( normalize(self.generator.next()) )


# datagenerator = ImageDataGenerator()
# image_datagen = get_image_batch(datagenerator, './Data/Split/train')
# print(next(image_datagen))