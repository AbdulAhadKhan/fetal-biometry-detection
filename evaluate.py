import os
import pickle

import tensorflow as tf
import pandas as pd
import numpy as np

from PIL import Image
from predict import predict_class
from sklearn import metrics
from plot import plot_confusion
from generator import Generator

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

data_dir = './Data/Split/test'
models_path = './models'

models = os.listdir(models_path)
models = [os.path.join(models_path, model) for model in models]

classes = {}

with open('classes.pickle', 'rb') as fp:
    classes = pickle.load(fp)

labels = [names for index, names in classes.items()]

def get_confusion_evaluation(model, generator):
    predictions = []
    ground_truth = []

    i = 1
    for x, y in generator:
        key = np.where(y == np.amax(y))[1][0]
        label = classes.get(key)
        ground_truth.append(label)

        predictions += [predict_class(model, x, classes)]
        
        i += 1
        if i > 600: break

    predictions = np.array(predictions)

    confusion = metrics.confusion_matrix(ground_truth, predictions)
    confusion = np.around(100 * confusion.astype('float') /
                          confusion.sum(axis=1)[:, np.newaxis], decimals=2)

    evaluation = metrics.classification_report(
        ground_truth, predictions, digits=3)
    
    return confusion, evaluation

# TODO: Optimize to remove redundancy

for m in models:
    model = tf.keras.models.load_model(m)

    model_name = m.split('/')[-1]
    model_name = model_name.split('.')[0]
    model_name = model_name.replace('_', ' ').title()

    model_type = model_name.split(' ')[-1]

    b_datagen = ImageDataGenerator()
    r_datagen = ImageDataGenerator(1/.255)
    n_datagen = Generator(b_datagen, data_dir, batch_size=1)
    nr_datagen = Generator(r_datagen, data_dir, batch_size=1)

    b_generator = b_datagen.flow_from_directory(
        data_dir, target_size=(200, 200), batch_size=1, shuffle=False,
        class_mode='categorical', color_mode='grayscale'
    )

    r_generator = r_datagen.flow_from_directory(
        data_dir, target_size=(200, 200), batch_size=1, shuffle=False,
        class_mode='categorical', color_mode='grayscale'
    )

    n_generator = n_datagen.get_image_batch()
    nr_generator = nr_datagen.get_image_batch()

    confusion, evaluation = get_confusion_evaluation(model, b_generator)
    report = './reports/basic/{name}.txt'.format(name = model_name)
    with open(report, 'w') as text_file: text_file.write(evaluation)
    plot_confusion(labels, confusion, 'basic', model_name, 'show')

    # confusion, evaluation = get_confusion_evaluation(model, r_generator)
    # report = './reports/rescaled/{name}.txt'.format(name = model_name)
    # with open(report, 'w') as text_file: text_file.write(evaluation)
    # plot_confusion(labels, confusion, 'rescaled', model_name, 'save')

    # confusion, evaluation = get_confusion_evaluation(model, n_generator)
    # report = './reports/normalised/{name}.txt'.format(name = model_name)
    # with open(report, 'w') as text_file: text_file.write(evaluation)
    # plot_confusion(labels, confusion, 'normalised', model_name, 'save')

    # confusion, evaluation = get_confusion_evaluation(model, nr_generator)
    # report = './reports/rescaled_normalised/{name}.txt'.format(name = model_name)
    # with open(report, 'w') as text_file: text_file.write(evaluation)
    # plot_confusion(labels, confusion, 'rescaled_normalised', model_name, 'save')