import os
import csv
import tensorflow as tf
import glob
from keras import backend as K

from commands.git import git_auto_commit
from commands.history_plot import update_readme
from model.data import flow_from_directory

def read_history(hist_path):
    with open(hist_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        return [row for row in csv_reader]
    
def create_history(hist_path, custom_metrics):
    with open(hist_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        columns = ['Epoch', 'Batch_Size', 'Loss', 'Accuracy', 'Val_Loss', 'Val_Accuracy']
        if custom_metrics:
            for metric in custom_metrics:
                metric_name = metric.__name__
                columns.append(metric_name)
                columns.append("Val_" + metric_name)
        csv_writer.writerow(columns)
    return list()

def train_and_save(model, epochs, save_interval = None, custom_metrics = None, custom_optimizer = None, push = None):

    hist_path = 'history.csv'

    os.makedirs('models', exist_ok=True)

    if os.path.exists(hist_path):
        history = read_history(hist_path)
        initial_epoch = int(history[-1][0])
        batch_size = int(history[-1][1])

        latest_model = max(glob.glob("models/model_e*.h5"), default=None, key=os.path.getctime)

        if latest_model is not None:
            print(f"Resuming from {latest_model}")
            model = tf.keras.models.load_model(latest_model, custom_objects={metric.__name__: metric for metric in custom_metrics})
        else: raise Exception('No models found in the model directory')

    else:
        history = create_history(hist_path, custom_metrics)
        initial_epoch = 0
        batch_size = 512

        model = compile_model(model, custom_metrics, custom_optimizer)

    temp_hist_data = []

    def save():
        try:
            return epoch % save_interval == 0
        except:
            return epoch == epochs + initial_epoch

    for epoch in range(initial_epoch + 1, epochs + initial_epoch + 1):
        
        total_history = history + temp_hist_data

        training_generator, validation_generator, _, new_batch_size = flow_from_directory(
            batch_size = batch_size,
            history = total_history,
            n_last_epochs = 20
        )

        batch_size = new_batch_size

        hist = model.fit(training_generator, validation_data = validation_generator)
        hist_data = [epoch, batch_size] + [hist.history[key][0] for key in hist.history]

        temp_hist_data.append(hist_data)
        total_history = history + temp_hist_data
            
        if save():
            model_path = f"models/model_e{epoch}.h5"
            model.save(model_path)
        
            with open(hist_path, 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    for row in temp_hist_data:
                        csv_writer.writerow(row)

            temp_hist_data.clear()
            history = read_history(hist_path)

            print(f"Saved model and history at epoch {epoch}")
            update_readme()

            if push:
                git_auto_commit('0.0-testing', epoch)



def compile_model(model, custom_metrics = None, custom_optimizer = None):
    if custom_optimizer:
        optimizer = custom_optimizer
    else:
        optimizer = 'adam'

    metrics = ['accuracy']
    if custom_metrics:
        metrics = metrics + custom_metrics
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=metrics,
                  )
    
    return model

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    actual_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (actual_positives + K.epsilon())
    
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val
