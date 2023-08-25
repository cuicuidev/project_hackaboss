import numpy as np


def stabilize_batch_size(converging, batch_size, min_batch_size = 1, max_batch_size = 32):

    can_increase = batch_size < max_batch_size
    can_decrease = batch_size > min_batch_size

    if converging and can_increase:
        batch_size = batch_size * 2

    if not converging and can_decrease:
        batch_size = batch_size // 2
    
    print(f'TRAINING WITH BATCH SIZE = {batch_size}=========================================================================')
    return batch_size

def is_converging(history, n_last_epochs=5, metric='Val_Loss'):
    if len(history) < n_last_epochs:
        return False
    
    # Extract only the Val_Loss column from the history.
    val_loss_history = [row[history[0].index(metric)] for row in history[-n_last_epochs:]]
    
    # Check if it is converging (you can define your own conditions here).
    return all(x >= y for x, y in zip(val_loss_history, val_loss_history[1:]))

