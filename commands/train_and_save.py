import os
import csv
import tensorflow as tf
import glob
from keras import backend as K

from commands.git import git_auto_commit
from commands.history_plot import update_readme
from model.data import flow_from_directory

def train_and_save(model, epochs, save_interval = None,custom_metrics = None, custom_optimizer = None, push = None):

    initial_epoch = 0

    hist_path = 'history.csv'
    all_history = []
    batch_size = 32

    if not os.path.exists(hist_path):
        os.makedirs('models', exist_ok=True)
        with open(hist_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            columns = ['Epoch', 'Loss', 'Accuracy', 'Val_Loss', 'Val_Accuracy']
            if custom_metrics:
                for metric in custom_metrics:
                    metric_name = metric.__name__
                    columns.append(metric_name)
                    columns.append("Val_" + metric_name)
            csv_writer.writerow(columns)
    else:
        with open(hist_path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            last_row = None
            for row in csv_reader:
                last_row = row
                all_history.append(row)
            if last_row:
                initial_epoch = int(last_row[0])
    
    latest_model = max(glob.glob("models/model_e*.h5"), default=None, key=os.path.getctime)

    if latest_model is not None:
        print(f"Resuming from {latest_model}")
        model = tf.keras.models.load_model(latest_model, custom_objects={metric.__name__: metric for metric in custom_metrics})
    else:
        model = compile_model(model, custom_metrics, custom_optimizer)

    temp_hist_data = []

    def save():
        try:
            return epoch % save_interval == 0
        except:
            return epoch == epochs + initial_epoch

    for epoch in range(initial_epoch + 1, epochs + initial_epoch + 1):
        
        total_history = all_history + temp_hist_data

        training_generator, validation_generator, _, new_batch_size = flow_from_directory(
            batch_size=batch_size,
            history=total_history,
            n_last_epochs=5
        )

        batch_size = new_batch_size

        hist = model.fit(training_generator, validation_data=validation_generator)
        hist_data = [epoch] + [hist.history[key][0] for key in hist.history]

        temp_hist_data.append(hist_data)
            
        if save():
            model_path = f"models/model_e{epoch}.h5"
            model.save(model_path)
        
            with open(hist_path, 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    for row in temp_hist_data:
                        csv_writer.writerow(row)

            all_history.clear()
            
            with open(hist_path, 'r') as csvfile:
                csv_reader = csv.reader(csvfile)
                for row in csv_reader:
                    all_history.append(row)
        
        # Clear temporary history data
        temp_hist_data.clear()
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
