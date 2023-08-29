import tensorflow as tf
from keras.optimizers import Adam

VERSION = 'version-0.3'

SCALE = 0.8
HEIGHT = round(560 * SCALE)
WIDTH = round(950 * SCALE)
N_CATEGORIES = 39
BATCH_SIZE = 16
MIN_BATCH_SIZE = 8
MAX_BATCH_SIZE = 16


SCHEDULE = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.90)

OPTIMIZER = Adam(learning_rate=SCHEDULE)

DATA_PATH = 'car_make_images/'
TRAINING_PATH = DATA_PATH + 'train'
TESTING_PATH = DATA_PATH + 'test'
VALIDATION_PATH = DATA_PATH + 'val'
