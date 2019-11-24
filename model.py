import os
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model

# from NaiveModel import naive_model_1
from AlexNet import alex_net

train_dir = './data/split/train'
validation_dir = './data/split/validation'

train_abdomen_dir = os.path.join(train_dir, 'Abdomen')
train_femur_dir = os.path.join(train_dir, 'Femur')
train_head_dir = os.path.join(train_dir, 'Head')

model = alex_net()
# model.summary()

# plot_model(model, show_shapes=True, show_layer_names=False, to_file='alex_net.png')

train_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(200, 200), batch_size=20, 
    class_mode='categorical', color_mode='grayscale'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=(200, 200), batch_size=5, 
    class_mode='categorical', color_mode='grayscale'
)

history = model.fit_generator(
    train_generator, steps_per_epoch=15, epochs=20, verbose=1,
    validation_data=validation_generator, validation_steps=15
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training Accury')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training And Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validatoin Loss')
plt.title('Training And Validation Loss')
plt.legend()

plt.show()

model.save('temp1.h5')