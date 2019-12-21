import os
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from generator import normalize
from PIL import Image

model_path = './models'
data_path = './Data/Split/test/Head'

alex_basic_path = os.path.join(model_path, 'alex_net_basic.h5')
alex_normal_path = os.path.join(model_path, 'alex_net_normalised.h5')
alex_rescale_path = os.path.join(model_path, 'alex_net_rescaled.h5')

image_names = os.listdir(data_path)

alex_net_basic = tf.keras.models.load_model(alex_basic_path)
alex_net_normal = tf.keras.models.load_model(alex_normal_path)
alex_net_rescaled = tf.keras.models.load_model(alex_rescale_path)

basic_prediction = []
normal_prediction = []
rescale_prediction = []

for i in range(10):
    image = Image.open(os.path.join(data_path, image_names[i])).convert('L')
    image = np.array(image)
    image = np.expand_dims(image, -1)
    image = np.expand_dims(image, 0)

    predict_1 = alex_net_basic.predict(image)
    predict_2 = alex_net_normal.predict(image)
    predict_3 = alex_net_rescaled.predict(image)

    value_1 = np.where(predict_1 == np.amax(predict_1))
    value_2 = np.where(predict_2 == np.amax(predict_2))
    value_3 = np.where(predict_3 == np.amax(predict_3))

    basic_prediction.append(value_1[1][0])
    normal_prediction.append(value_2[1][0])
    rescale_prediction.append(value_3[1][0])

    print('Predicted image', i+1, 'of 10')

print(basic_prediction)
print(normal_prediction)
print(rescale_prediction)