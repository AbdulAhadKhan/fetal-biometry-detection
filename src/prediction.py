import os
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from generator import normalize
from PIL import Image

alex_net_1 = tf.keras.models.load_model('alex_net_1.h5')
alex_net_1_custom = tf.keras.models.load_model('alex_net_1_custom.h5')

alex_prediction = []
custom_prediction = []

path = './Data/Split/test/Head'
image_names = os.listdir(path)

for i in range(10):
    image = Image.open(os.path.join(path, image_names[i])).convert('L')
    image = np.array(image)
    image = np.expand_dims(image, -1)
    image = np.expand_dims(image, 0)

    predict_1 = alex_net_1.predict(image)
    predict_2 = alex_net_1_custom.predict(image)

    value_1 = np.where(predict_1 == np.amax(predict_1))
    value_2 = np.where(predict_2 == np.amax(predict_2))

    alex_prediction.append(value_1)
    custom_prediction.append(value_2)

    print('Predicted image', i+1, 'of 10')

print(alex_prediction)
print(custom_prediction)