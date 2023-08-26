import tensorflow as tf
from keras.optimizers import Adam

HEIGHT = 20 # round(560 / 2)
WIDTH = 20 # round(950 / 2)
N_CATEGORIES = 39


SCHEDULE = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.90)

OPTIMIZER = Adam(learning_rate=SCHEDULE)

DATA_PATH = 'car_make_images/'
TRAINING_PATH = DATA_PATH + 'train'
TESTING_PATH = DATA_PATH + 'test'
VALIDATION_PATH = DATA_PATH + 'val'
