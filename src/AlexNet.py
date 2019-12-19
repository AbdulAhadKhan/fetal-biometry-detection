import tensorflow

def alex_net(input_shape=(200, 200, 1), number_of_classes=3, optimizer='adam'):
    layers = tensorflow.keras.layers
    Model = tensorflow.keras.models.Model

    input_layer = layers.Input(input_shape)
    conv1 = layers.Conv2D(96, 11, strides=4, activation='relu')(input_layer)
    pool1 = layers.MaxPool2D(3, 2)(conv1)

    conv2 = layers.Conv2D(256, 5, strides=1, padding='same', activation='relu')(pool1)
    pool2 = layers.MaxPool2D(3, 2)(conv2)

    conv3 = layers.Conv2D(384, 3, strides=1, padding='same', activation='relu')(pool2)
    conv4 = layers.Conv2D(256, 3, strides=1, padding='same', activation='relu')(conv3)
    pool3 = layers.MaxPool2D(3, 2)(conv4)

    flattened = layers.Flatten()(pool3)
    dense1 = layers.Dense(4096, activation='relu')(flattened)
    drop1 = layers.Dropout(0.5)(dense1)
    dense2 = layers.Dense(4096, activation='relu')(drop1)
    drop2 = layers.Dropout(0.5)(dense2)

    preds = layers.Dense(number_of_classes, activation='softmax')(drop2)

    model = Model(input_layer, preds)
    model.compile(
        loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']
    )

    return model