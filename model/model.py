import tensorflow as tf

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization, ReLU

from model.settings import HEIGHT, WIDTH, N_CATEGORIES

############################################################################################################################################

with tf.device("/GPU:0"):
    model = Sequential()

    model.add(Input(shape=(HEIGHT, WIDTH, 1)))

    # Conv Blocks (Added more filters)

    model.add(Conv2D(filters=16, kernel_size=7, padding = 'same', kernel_initializer = 'he_normal'))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=32, kernel_size=5, padding = 'same', kernel_initializer = 'he_normal'))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=64, kernel_size=3, padding = 'same', kernel_initializer = 'he_normal'))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=2))

    # Flatten and Fully Connected Layers
    model.add(Flatten())

    for neurons in [128]:
        model.add(Dense(neurons, activation='relu', kernel_initializer='he_normal'))
        # model.add(Dropout(0.15))

    # Output layer
    model.add(Dense(N_CATEGORIES, activation='softmax', kernel_initializer='he_normal'))