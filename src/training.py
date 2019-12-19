import os
import tensorflow
import plot

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from generator import Generator
from models import naive_model_1, alex_net, sononet_32

keras = tensorflow.keras
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
plot_model = keras.utils.plot_model
optimizer = keras.optimizers

train_dir = './Data/Split/train'
validation_dir = './Data/Split/validation'

train_abdomen_dir = os.path.join(train_dir, 'Abdomen')
train_femur_dir = os.path.join(train_dir, 'Femur')
train_head_dir = os.path.join(train_dir, 'Head')

adam = optimizer.Adam(learning_rate=0.0003)

alex_net = alex_net(optimizer=adam)
naive_model_1 = naive_model_1(optimizer=adam)
sononet_32 = sononet_32(optimizer=adam)

model_details = []

# plot_model(naive_model_1, show_shapes=True, show_layer_names=False, to_file='naive_model_1.png')
# plot_model(alex_net, show_shapes=True, show_layer_names=False, to_file='alex_net.png')
# plot_model(sononet_32, show_shapes=True, show_layer_names=False, to_file='sononet_32.png')

train_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(200, 200), batch_size=40, 
    class_mode='categorical', color_mode='grayscale',
    seed=42
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=(200, 200), batch_size=20, 
    class_mode='categorical', color_mode='grayscale',
    seed=42
)

train_datagen_custom = Generator(train_datagen, train_dir)
train_generator_custom = train_datagen_custom.get_image_batch()

validation_datagen_custom = Generator(validation_datagen, validation_dir)
validation_generator_custom = validation_datagen_custom.get_image_batch()

alex_net_basic = (alex_net, "alex_net_basic", train_generator)
model_details.append(alex_net_basic)



# history_alex_net = alex_net.fit_generator(
#     train_generator, steps_per_epoch=20, epochs=100, verbose=1,
#     validation_data=validation_generator, validation_steps=5
# )

# plot.plot_cv(history_alex_net, 'Alex Net 1', 'save')
# alex_net.save('alex_net_1.h5')

# history_alex_net_custom = alex_net.fit_generator(
#     train_generator_custom, steps_per_epoch=20, epochs=100, verbose=1,
#     validation_data=validation_generator_custom, validation_steps=5
# )

# plot.plot_cv(history_alex_net_custom, 'Alex Net 2', 'save')
# alex_net.save('alex_net_1_custom.h5')

# history_naive_model = alex_net.fit_generator(
#     train_generator, steps_per_epoch=20, epochs=100, verbose=1,
#     validation_data=validation_generator, validation_steps=5
# )

# plot.plot_cv(history_naive_model, 'Naive Model', 'save')
# alex_net.save('naive_model.h5')

# history_sononet_32 = sononet_32.fit_generator(
#     train_generator, steps_per_epoch=20, epochs=100, verbose=1,
#     validation_data=validation_generator, validation_steps=5
# )

# plot.plot_cv(history_sononet_32, 'SonoNet', 'save')
# alex_net.save('sononet_32.h5')

# alex_net.save('alex_net_1.h5')
# naive_model_1.save('naive_model_1_1.h5')
# sononet_32.save('sononet_32.h5')

# plot.plot_cv(history_alex_net, 'Alex Net')
# plot.plot_cv(history_naive_model, 'Naive Model')
# plot.plot_cv(sononet_32, 'SonoNet 32')

for model_detail in model_details:
    plot_model(naive_model_1, show_shapes=True, show_layer_names=False, to_file='naive_model_1.png')

    model, model_name, generator = model_detail

    history_model = model.fit_generator(
        generator, steps_per_epoch=20, epochs=1, verbose=1,
        validation_data=validation_generator_custom, validation_steps=5
    )

    plot.plot_cv(history_model, model_name, 'show')