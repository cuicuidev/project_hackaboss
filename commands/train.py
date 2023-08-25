import tensorflow as tf
from commands.train_and_save import train_and_save, f1_score
from model.model import model
from model.settings import OPTIMIZER


def run(epochs, save_interval = None, push = False):
    print("Training...")
    # Train the model and save it
    with tf.device("/GPU:0"):
        train_and_save(
            model = model,
            epochs = epochs, 
            save_interval = save_interval,
            custom_metrics = [f1_score],
            custom_optimizer = OPTIMIZER,
            push = push
        )