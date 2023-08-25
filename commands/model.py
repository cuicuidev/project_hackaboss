import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from keras.optimizers import Adam

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

height = 400
width = 400
batch_size = 30
num_classes = 39


training_generator = training_data_generator.flow_from_directory(training_path,
                                                                 target_size = (height, width),
                                                                 batch_size = batch_size,
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
    model.add(Dense(num_classes, activation='softmax', kernel_initializer='he_normal'))

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.90)
    opt = Adam(learning_rate=lr_schedule)