import os

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

test_dir = './Data/Split/test'
classes = os.listdir(test_dir)

classes = [os.path.join(test_dir, _class) for _class in classes]
print(classes)

models_path = './models'
models = os.listdir(models_path)