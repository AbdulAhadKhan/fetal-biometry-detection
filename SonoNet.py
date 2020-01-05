from tensorflow import keras

layers = keras.layers
models = keras.models
Model = keras.Model

def SonoNet32(input_shape=(200, 200, 1), num_classes=3):
    input_layer = layers.Input(input_shape)

    conv1_1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    b_norm1_1 = layers.BatchNormalization()(conv1_1)
    conv1_2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(b_norm1_1)
    b_norm1_2 = layers.BatchNormalization()(conv1_2)
    max_pool1_1 = layers.MaxPool2D((2, 2))(b_norm1_2)

    conv2_1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(max_pool1_1)
    b_norm2_1 = layers.BatchNormalization()(conv2_1)
    conv2_2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(b_norm2_1)
    b_norm2_2 = layers.BatchNormalization()(conv2_2)
    max_pool2_1 = layers.MaxPool2D((2, 2))(b_norm2_2)

    conv3_1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(max_pool2_1)
    b_norm3_1 = layers.BatchNormalization()(conv3_1)
    conv3_2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(b_norm3_1)
    b_norm3_2 = layers.BatchNormalization()(conv3_2)
    conv3_3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(b_norm3_2)
    b_norm3_3 = layers.BatchNormalization()(conv3_3)
    max_pool3_1 = layers.MaxPool2D((2, 2))(b_norm3_3)

    conv4_1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(max_pool3_1)
    b_norm4_1 = layers.BatchNormalization()(conv4_1)
    conv4_2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(b_norm4_1)
    b_norm4_2 = layers.BatchNormalization()(conv4_2)
    conv4_3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(b_norm4_2)
    b_norm4_3 = layers.BatchNormalization()(conv4_3)
    max_pool4_1 = layers.MaxPool2D((2, 2))(b_norm4_3)

    conv5_1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(max_pool4_1)
    b_norm5_1 = layers.BatchNormalization()(conv5_1)
    conv5_2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(b_norm5_1)
    b_norm5_2 = layers.BatchNormalization()(conv5_2)
    conv5_3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(b_norm5_2)

    conv6_1 = layers.Conv2D(128, (1, 1), activation='relu', padding='same')(conv5_3)
    conv6_2 = layers.Conv2D(3, (1, 1), activation='relu')(conv6_1)
    g_avg_pool = layers.GlobalAveragePooling2D()(conv6_2)
    softmax = layers.Softmax()(g_avg_pool)

    model = Model(input_layer, softmax)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model

model = SonoNet32()
model.summary()