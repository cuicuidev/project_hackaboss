import numpy as np
import pandas as pd
import csv
import os
import glob
import tensorflow as tf

from PIL import Image

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.models import Model

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from keras.optimizers import Adam
from keras import backend as K

###########################################################################################################################################

path = 'car_make_images/'
training_path = path + 'train'
testing_path = path + 'test'
validation_path = path + 'val'


training_data_generator = ImageDataGenerator(rescale = 1./255,
                              rotation_range = 359,
                              shear_range = 0.2,
                              width_shift_range = 0.2,
                              height_shift_range = 0.2,
                              zoom_range = 0.2,
                              horizontal_flip = True,
                              vertical_flip = True,
                              preprocessing_function = None)

validation_data_generator = ImageDataGenerator(rescale = 1./255)
test_data_generator = ImageDataGenerator(rescale = 1./255)

height = 280
width = 470
batch_size = 8  
num_classes = 39


training_generator = training_data_generator.flow_from_directory(training_path,
                                                                 target_size = (height, width),
                                                                 batch_size = 30,
                                                                 class_mode = "categorical",
                                                                 color_mode = 'grayscale',
                                                                 )

validation_generator = validation_data_generator.flow_from_directory(validation_path,
                                                                     target_size = (height, width),
                                                                     batch_size = 1,
                                                                     class_mode = "categorical",
                                                                     color_mode = 'grayscale',
                                                                     )

test_generator = test_data_generator.flow_from_directory(testing_path,
                                                         target_size = (height, width),
                                                         batch_size = 1,
                                                         class_mode = "categorical",
                                                         color_mode = 'grayscale',
                                                         )




with tf.device("/GPU:0"):
    model = Sequential()

    model.add(Input(shape=(height, width, 1)))

    # First Conv Block
    model.add(Conv2D(filters=32, kernel_size=5, padding='same', kernel_initializer='he_normal'))
    # model.add(BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(MaxPooling2D(pool_size=2))

    # Second Conv Block
    model.add(Conv2D(filters=64, kernel_size=5, padding='same', kernel_initializer='he_normal'))
    # model.add(BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(MaxPooling2D(pool_size=2))

    # Third Conv Block
    model.add(Conv2D(filters=128, kernel_size=5, padding='same', kernel_initializer='he_normal'))
    # model.add(BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(MaxPooling2D(pool_size=2))

    # Fourth Conv Block
    model.add(Conv2D(filters=256, kernel_size=5, padding='same', kernel_initializer='he_normal'))
    # model.add(BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(MaxPooling2D(pool_size=2))

    # Fifth Conv Block
    model.add(Conv2D(filters=512, kernel_size=5, padding='same', kernel_initializer='he_normal'))
    # model.add(BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(MaxPooling2D(pool_size=2))

    # Flatten and Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.15))
    model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer='he_normal'))

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.001,
                decay_steps=10000,
                decay_rate=0.9)
    
    opt = Adam(learning_rate=lr_schedule)