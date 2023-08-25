import os
import csv
import tensorflow as tf
import glob
from keras import backend as K

from commands.auto_git import git_auto_commit
from commands.plot import update_readme

def train_and_save(model, train_data, val_data, epochs, save_interval, model_save_path, history_save_path, custom_metrics=None, custom_optimizer=None, push = False):
    """
    Train a TensorFlow model and save it along with its history.
    """
    os.makedirs(model_save_path, exist_ok=True)

    if custom_optimizer:
        optimizer = custom_optimizer
    else:
        optimizer = 'adam'

    if custom_metrics:
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'] + custom_metrics)
    else:
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    # Initialize variables
    initial_epoch = 0
    temp_history_data = []

    # Check if history file exists, if not create it
    if not os.path.exists(history_save_path):
        with open(history_save_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            columns = ['Epoch', 'Loss', 'Accuracy', 'Val_Loss', 'Val_Accuracy']
            if custom_metrics:
                for metric in custom_metrics:
                    metric_name = metric.__name__
                    columns.append(metric_name)
                    columns.append("Val_" + metric_name)
            csv_writer.writerow(columns)
    else:
        with open(history_save_path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            last_row = None
            for row in csv_reader:
                last_row = row
            if last_row:
                initial_epoch = int(last_row[0])

    latest_model_file = max(glob.glob(f"{model_save_path}/model_e*.h5"), default=None, key=os.path.getctime)
    if latest_model_file is not None:
        print(f"Resuming from {latest_model_file}")
        model = tf.keras.models.load_model(latest_model_file, custom_objects={metric.__name__: metric for metric in custom_metrics})

    for epoch in range(initial_epoch + 1, epochs + initial_epoch + 1):
        print(f"Epoch {epoch}/{epochs + initial_epoch}")

        history = model.fit(train_data, validation_data=val_data)
        history_data = [epoch] + [history.history[key][0] for key in history.history]
        temp_history_data.append(history_data)

        if epoch % save_interval == 0 or epoch == epochs + initial_epoch:
            model_file_path = os.path.join(model_save_path, f"model_e{epoch}.h5")
            model.save(model_file_path)

            # Append to CSV at checkpoints
            with open(history_save_path, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                for row in temp_history_data:
                    csv_writer.writerow(row)

            # Clear temporary history data
            temp_history_data.clear()

            print(f"Saved model and history at epoch {epoch}")

            if push:
                print("Updating README.md")
                update_readme()
                print("Pushing changes to GitHub")
                git_auto_commit('0.0-testing', epoch)
                print("Done")

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    actual_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (actual_positives + K.epsilon())
    
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val
