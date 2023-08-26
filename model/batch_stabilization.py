import numpy as np
import math



def stabilize_batch_size(converging, batch_size, min_batch_size = 16, max_batch_size = 32):

    can_increase = batch_size < max_batch_size
    can_decrease = batch_size > min_batch_size

    if converging and can_increase:
        batch_size = batch_size + 1

    if not converging and can_decrease:
        batch_size = batch_size - 1
    
    print(f'TRAINING WITH BATCH SIZE = {batch_size}=========================================================================')
    return batch_size

def is_converging(history, n_last_epochs, degree=10):
    angle_in_radians = math.radians(degree)
    
    epsilon = math.tan(angle_in_radians)
    
    if len(history) < n_last_epochs + 1:
        return True
    
    last_n_epochs = history[-n_last_epochs + 1:]
    
    last_n_accuracies = [float(row[3]) for row in last_n_epochs]
    
    max_acc = max(last_n_accuracies)
    min_acc = min(last_n_accuracies)
    
    return max_acc - min_acc < epsilon




