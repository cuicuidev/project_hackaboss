import numpy as np


def stabilize_batch_size(converging, batch_size, min_batch_size = 32, max_batch_size = 512):

    can_increase = batch_size < max_batch_size
    can_decrease = batch_size > min_batch_size

    if converging and can_increase:
        batch_size = batch_size * 2

    if not converging and can_decrease:
        batch_size = batch_size // 2
    
    print(f'TRAINING WITH BATCH SIZE = {batch_size}=========================================================================')
    return batch_size

def is_converging(history, n_last_epochs, epsilon=0.001):
    if len(history) < n_last_epochs:
        return True
    
    last_n_epochs = history[-n_last_epochs:]
    
    # Extract just the accuracy from the last `n_last_epochs` entries.
    # Assuming accuracy is the third item in each row
    last_n_accuracies = [float(row[2]) for row in last_n_epochs]
    
    max_acc = max(last_n_accuracies)
    min_acc = min(last_n_accuracies)
    
    return max_acc - min_acc < epsilon


