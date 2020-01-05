import tensorflow

layers = tensorflow.keras.layers
Model = tensorflow.keras.models.Model
regularizers = tensorflow.keras.regularizers

def alex_net(input_shape=(200, 200, 1), number_of_classes=3, optimizer='adam'):
    input_layer = layers.Input(input_shape)
    conv1 = layers.Conv2D(96, 11, strides=4, activation='relu', kernel_regularizer=regularizers.l2(l=0.001))(input_layer)
    pool1 = layers.MaxPool2D(3, 2)(conv1)

    conv2 = layers.Conv2D(256, 5, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(l=0.001))(pool1)
    pool2 = layers.MaxPool2D(3, 2)(conv2)

    conv3 = layers.Conv2D(384, 3, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(l=0.001))(pool2)
    conv4 = layers.Conv2D(256, 3, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(l=0.001))(conv3)
    pool3 = layers.MaxPool2D(3, 2)(conv4)

    flattened = layers.Flatten()(pool3)
    dense1 = layers.Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(l=0.001))(flattened)
    dropout1 = layers.Dropout(0.5)(dense1)

    dense2 = layers.Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(l=0.001))(dropout1)
    drop2 = layers.Dropout(0.5)(dense2)

    preds = layers.Dense(number_of_classes, activation='softmax')(drop2)

    loss = 'binary_crossentropy' if number_of_classes == 2 else 'categorical_crossentropy'

    model = Model(input_layer, preds)
    model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])

    return model

def naive_model_1(input_shape=(200, 200, 1), number_of_classes=3, optimizer='adam'):
    input_layer = layers.Input(input_shape)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(l=0.001))(input_layer)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l=0.001))(pool1)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    conv3 = layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l=0.001))(pool2)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)

    conv4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l=0.001))(pool3)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)

    flatten = layers.Flatten()(pool4)
    dropout1 = layers.Dropout(0.5)(flatten)

    dense1 = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l=0.001))(dropout1)
    dropout2 = layers.Dropout(0.5)(dense1)

    pred = layers.Dense(number_of_classes, activation='softmax')(dropout2)

    loss = 'binary_crossentropy' if number_of_classes == 2 else 'categorical_crossentropy'

    model = Model(input_layer, pred)
    model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])

    return model

def sononet_32(input_shape=(200, 200, 1), number_of_classes=3, optimizer='adam'):
    input_layer = layers.Input(input_shape)

    conv1_1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l=0.001))(input_layer)
    b_norm1_1 = layers.BatchNormalization()(conv1_1)
    conv1_2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l=0.001))(b_norm1_1)
    b_norm1_2 = layers.BatchNormalization()(conv1_2)
    max_pool1_1 = layers.MaxPool2D((2, 2))(b_norm1_2)

    conv2_1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l=0.001))(max_pool1_1)
    b_norm2_1 = layers.BatchNormalization()(conv2_1)
    conv2_2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l=0.001))(b_norm2_1)
    b_norm2_2 = layers.BatchNormalization()(conv2_2)
    max_pool2_1 = layers.MaxPool2D((2, 2))(b_norm2_2)

    conv3_1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l=0.001))(max_pool2_1)
    b_norm3_1 = layers.BatchNormalization()(conv3_1)
    conv3_2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l=0.001))(b_norm3_1)
    b_norm3_2 = layers.BatchNormalization()(conv3_2)
    conv3_3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l=0.001))(b_norm3_2)
    b_norm3_3 = layers.BatchNormalization()(conv3_3)
    max_pool3_1 = layers.MaxPool2D((2, 2))(b_norm3_3)

    conv4_1 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l=0.001))(max_pool3_1)
    b_norm4_1 = layers.BatchNormalization()(conv4_1)
    conv4_2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l=0.001))(b_norm4_1)
    b_norm4_2 = layers.BatchNormalization()(conv4_2)
    conv4_3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l=0.001))(b_norm4_2)
    b_norm4_3 = layers.BatchNormalization()(conv4_3)
    max_pool4_1 = layers.MaxPool2D((2, 2))(b_norm4_3)

    conv5_1 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l=0.001))(max_pool4_1)
    b_norm5_1 = layers.BatchNormalization()(conv5_1)
    conv5_2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l=0.001))(b_norm5_1)
    b_norm5_2 = layers.BatchNormalization()(conv5_2)
    conv5_3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l=0.001))(b_norm5_2)

    flatten = layers.Flatten()(conv5_3)
    dropout1 = layers.Dropout(0.5)(flatten)
    
    dense1 = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l=0.001))(dropout1)
    dropout2 = layers.Dropout(0.5)(dense1)

    pred = layers.Dense(number_of_classes, activation='softmax')(dropout2)

    model = Model(input_layer, pred)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    return model