import tensorflow as tf

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

from model.settings import HEIGHT, WIDTH, N_CATEGORIES

############################################################################################################################################

with tf.device("/GPU:0"):
    model = Sequential()

    model.add(Input(shape=(HEIGHT, WIDTH, 1)))

    # Conv Blocks (Added more filters)
    for filters in [32, 64, 128, 256]:
        model.add(Conv2D(filters=filters, kernel_size=3, padding='same', kernel_initializer='he_normal'))
        model.add(tf.keras.layers.ReLU())
        model.add(MaxPooling2D(pool_size=2))

    # Flatten and Fully Connected Layers
    model.add(Flatten())

    for neurons in [512, 256]:
        model.add(Dense(neurons, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.15))

    # Output layer
    model.add(Dense(N_CATEGORIES, activation='softmax', kernel_initializer='he_normal'))