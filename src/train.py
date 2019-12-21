# Imports

## Essentials

import os
import tensorflow

## Customs

from plot import plot_cv, PlotLosses
from generator import Generator
from models import naive_model_1, alex_net, sononet_32

# Module Aliases

keras = tensorflow.keras
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
plot_model = keras.utils.plot_model
optimizer = keras.optimizers

# Directory Aliases

train_dir = './Data/Split/train'
validation_dir = './Data/Split/validation'

model_dir = './models'
structure_dir = './diagrams/structures'

# Required Details, Variables and Hyperparameters

epochs = 100
batch_size = 18
val_batch_size = 8
adam = optimizer.Adam(lr=3e-4, decay=1e-6)
model_details = []

# Model Initialisations

alex_net = alex_net(optimizer=adam)
naive_model = naive_model_1(optimizer=adam)
sononet_32 = sononet_32(optimizer=adam)

# Data Generators

## Initialisations

### Original

# train_datagen = ImageDataGenerator()
# validation_datagen = ImageDataGenerator()

### Rescaled

train_datagen = ImageDataGenerator(rescale=1/.255)
validation_datagen = ImageDataGenerator(rescale=1/.255)

## Actual Generators

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(200, 200), batch_size=batch_size, 
    class_mode='categorical', color_mode='grayscale'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=(200, 200), batch_size=val_batch_size, 
    class_mode='categorical', color_mode='grayscale'
)

## Normalised Initialisations

train_datagen_custom = Generator(train_datagen, train_dir, batch_size=batch_size)
validation_datagen_custom = Generator(validation_datagen, validation_dir, batch_size=val_batch_size)

## Normalised Generators

train_generator_custom = train_datagen_custom.get_image_batch()
validation_generator_custom = validation_datagen_custom.get_image_batch()

# Model Setup

### NOTE: Tuple takes data in following order
###       ("Actual Model", "Name of the Model", "Training Data Generator", "Validation Data Generator")

## Alex Net

# alex_net_basic = (alex_net, 'Alex Net Basic', train_generator, validation_generator)
# model_details.append(alex_net_basic)

# alex_net_normal = (alex_net, 'Alex Net Normalised', train_generator_custom, validation_generator_custom)
# model_details.append(alex_net_normal)

alex_net_rescaled = (alex_net, 'Alex Net Rescaled', train_generator, validation_generator)
model_details.append(alex_net_rescaled)

## Naive Model

# naive_model_basic = (naive_model, 'Naive Model Basic', train_generator, validation_generator)
# model_details.append(naive_model_basic)

# naive_model_normal = (naive_model, 'Naive Model Normal', train_generator_custom, validation_generator_custom)
# model_details.append(naive_model_normal)

naive_model_rescaled = (naive_model, 'Naive Model Rescaled', train_generator, validation_generator)
model_details.append(naive_model_rescaled)

## Sono Net 32

# sononet_32_basic = (sononet_32, 'SonoNet 32 Basic', train_generator, validation_generator)
# model_details.append(sononet_32_basic)

# sononet_32_normal = (sononet_32, 'SonoNet 32 Normalised', train_generator_custom, validation_generator_custom)
# model_details.append(sononet_32_normal)

sononet_32_rescaled = (sononet_32, 'SonoNet 32 Rescaled', train_generator, validation_generator)
model_details.append(sononet_32_rescaled)

# Structure Diagram, Training, History Plot

for model_detail in model_details:
    model, model_title, train, validate = model_detail

    model_name = model_title.lower().replace(' ', '_')
    structure_name = '{}.png'.format(model_name)
    final_model = '{}.h5'.format(model_name)

    structure_name = os.path.join(structure_dir, structure_name)
    final_model = os.path.join(model_dir, final_model)

    plot_loss_real = PlotLosses()

    print(model_title)

    plot_model(model, show_shapes=True, show_layer_names=False, to_file=structure_name)

    history_model = model.fit_generator(
        train, epochs=epochs, verbose=1, validation_data=validate,
        callbacks=[plot_loss_real], steps_per_epoch=20, validation_steps=5
    )

    model.save(final_model)

    plot_cv(history_model, model_title, 'save')