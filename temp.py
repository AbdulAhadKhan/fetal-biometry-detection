import cv2
import pickle

import tensorflow as tf

from predict import predict_confidence

def prepare(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return img.reshape(1, 200, 200, 1)

image = prepare('./Data/Split/test/Head/Head_00800.jpg')
model = tf.keras.models.load_model('./models/alex_net_rescaled.h5')

labels = {}

with open('classes.pickle', 'rb') as fp:
    labels = pickle.load(fp)

predict_confidence(model, image, labels)