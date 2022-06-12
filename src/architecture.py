from tensorflow import keras


def cnn_model():
    """
    Simple Convolutional Neural Network to determine three continuous variables, in this case 2D eyeball center
    coordinates and gaze direction. Three blocks of convolutional layers are followed by three fully connected
    layers.
    """

    inputs = keras.Input(shape=(43, 43, 1))

    # First Convolution Layer Block
    conv1 = keras.layers.Conv2D(16, (5, 5), padding='same')(inputs)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation('relu')(conv1)
    conv1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    # Second Convolution Layer Block
    conv2 = keras.layers.Conv2D(32, (3, 3), padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)
    conv2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Third Convolution Layer Block
    conv3 = keras.layers.Conv2D(64, (3, 3), padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)
    conv3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Fully-connected dense layers
    flat0 = keras.layers.Flatten()(conv3)
    flat1 = keras.layers.Dense(units=512, activation='relu')(flat0)
    flat2 = keras.layers.Dense(units=128, activation='relu')(flat1)
    flat3 = keras.layers.Dense(units=32, activation='relu')(flat2)

    # Output with a linear activation function to allow values greater than 1
    outputs = keras.layers.Dense(units=3, activation='linear')(flat3)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Use Adam optimization algorithm and define MSE as loss metric
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss='mean_squared_error')

    return model
