import tensorflow as tf
from commands.train_and_save import train_and_save, f1_score
from commands.model import model, training_generator, validation_generator, opt


def run(epochs, save_interval = None, push = False):
    print("Training...")
    # Train the model and save it
    with tf.device("/GPU:0"):
        train_and_save(
            model, 
            training_generator, 
            validation_generator, 
            epochs=epochs, 
            save_interval=save_interval, 
            model_save_path="models", 
            history_save_path="history.csv", 
            custom_metrics=[f1_score],
            custom_optimizer=opt,
            push = push
        )